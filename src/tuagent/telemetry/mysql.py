import os
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, get_args

from .contracts import FieldName, SCHEMA, TableName, TelemetryError


Row = dict[str, Any]
TABLES = get_args(TableName)
FIELDS = get_args(FieldName)
MAX_WINDOW_ROWS = 250_000
_TIMESTAMP = "CAST(`timestamp` AS UNSIGNED)"
_TIMESTAMP_PATTERN = "^[0-9]{1,15}$"
_WINDOW_FILTER = f"{_TIMESTAMP} >= %s AND {_TIMESTAMP} < %s AND `timestamp` REGEXP %s"


class TelemetryReader(Protocol):
    def metadata(self, table: str, profile: str) -> list[Row]: ...

    def samples(
        self, table: str, fields: tuple[str, ...], start_ms: int, end_ms: int,
        *, limit: int | None = None, newest_first: bool = False,
    ) -> Iterator[Row]: ...


class TelemetrySource(Protocol):
    def read(self) -> AbstractContextManager[TelemetryReader]: ...


@dataclass(frozen=True, slots=True)
class MySQLConfig:
    host: str
    user: str
    password: str = field(repr=False)
    database: str
    port: int
    expected_source_sha256: str | None = None

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "MySQLConfig":
        if env_file is None:
            checkout = Path(__file__).resolve().parents[3]
            root = checkout if (checkout / "pyproject.toml").is_file() else Path.cwd()
            env_file = root / ".env.local"
        values: dict[str, str] = {}
        try:
            lines = env_file.read_text(encoding="utf-8-sig").splitlines()
        except FileNotFoundError:
            lines = []
        except (OSError, UnicodeError):
            raise TelemetryError("configuration_error", "Cannot read .env.local") from None
        for line in lines:
            name, separator, value = line.strip().partition("=")
            if separator and not name.startswith("#"):
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                values[name.strip()] = value

        required = ("HOST", "MYSQL_USER", "MYSQL_PW", "DB_NAME", "PORT")
        resolved = {name: os.environ.get(name) or values.get(name, "") for name in required}
        missing = [name for name, value in resolved.items() if not value]
        if missing:
            raise TelemetryError(
                "configuration_error", f"Missing MySQL settings: {', '.join(missing)}"
            )
        try:
            port = int(resolved["PORT"])
            if not 1 <= port <= 65535:
                raise ValueError
        except ValueError:
            raise TelemetryError("configuration_error", "PORT must be in 1..65535") from None
        return cls(
            resolved["HOST"], resolved["MYSQL_USER"], resolved["MYSQL_PW"],
            resolved["DB_NAME"], port,
            os.environ.get("TELEMETRY_SOURCE_SHA256") or values.get("TELEMETRY_SOURCE_SHA256"),
        )


class MySQLTelemetrySource:
    """Read-only snapshot per tool call, opened in the tool's worker thread."""

    def __init__(self, config: MySQLConfig | None = None, *, env_file: Path | None = None):
        self._config = config
        self._env_file = env_file

    @contextmanager
    def read(self) -> Iterator["_MySQLReader"]:
        config = self._config or MySQLConfig.from_env(self._env_file)
        if config.database != SCHEMA:
            raise TelemetryError("configuration_error", "DB_NAME must be dcma")
        try:
            import pymysql
        except ImportError:
            raise TelemetryError(
                "configuration_error", 'Install the mysql extra with pip install -e ".[mysql]"'
            ) from None

        try:
            with _connect(pymysql, config) as connection:
                with connection.cursor() as cursor:
                    cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
                    cursor.execute("START TRANSACTION WITH CONSISTENT SNAPSHOT, READ ONLY")
                yield _MySQLReader(connection, pymysql.cursors, config.expected_source_sha256)
        except pymysql.MySQLError as exc:
            code = exc.args[0] if exc.args and type(exc.args[0]) is int else None
            if code in (2002, 2003, 2005, 2006, 2013):
                reason = "MySQL network connection or read failed"
            elif code in (1044, 1045, 1130, 1142, 1698, 2059, 2061):
                reason = "MySQL authentication or read permission failed"
            elif code == 3024:
                reason = "MySQL query timed out"
            else:
                reason = "MySQL read failed"
            suffix = f" (error {code})" if code is not None else ""
            raise TelemetryError("read_error", reason + suffix) from None


def _connect(driver: Any, config: MySQLConfig) -> Any:
    options = dict(
        host=config.host, port=config.port, user=config.user, password=config.password,
        database=config.database, charset="utf8mb4", autocommit=False,
        connect_timeout=5, read_timeout=12, write_timeout=5,
    )
    try:
        return driver.connect(**options)
    except driver.MySQLError as exc:
        if not exc.args or exc.args[0] not in (2002, 2003):
            raise
    # No transaction has started. Give a short connection outage one retry.
    time.sleep(0.25)
    return driver.connect(**options)


class _MySQLReader:
    def __init__(self, connection: Any, cursors: Any, expected_source_sha256: str | None):
        self._connection = connection
        self._cursors = cursors
        self.expected_source_sha256 = expected_source_sha256

    def metadata(self, table: str, profile: str) -> list[Row]:
        if table not in TABLES:
            raise TelemetryError("invalid_source", "Table is outside the telemetry allowlist")
        with self._connection.cursor(self._cursors.DictCursor) as cursor:
            cursor.execute(
                "SELECT /*+ MAX_EXECUTION_TIME(10000) */ "
                "field_name, description, quantity_kind, storage_type, is_nullable, "
                "stored_unit, display_unit, canonical_unit, scale_factor, offset_value, "
                "value_conventions, invalid_values, definition_source, scope_note, "
                "verification_status, source_sha256 FROM `signal_metadata` "
                "WHERE BINARY schema_name = %s AND BINARY table_name = %s "
                "AND BINARY profile_version = %s LIMIT 33",
                (SCHEMA, table, profile),
            )
            return list(cursor.fetchall())

    def samples(
        self, table: str, fields: tuple[str, ...], start_ms: int, end_ms: int,
        *, limit: int | None = None, newest_first: bool = False, after: tuple[int, int] | None = None,
    ) -> Iterator[Row]:
        if table not in TABLES or any(name not in FIELDS for name in fields):
            raise TelemetryError("invalid_source", "Table or telemetry field is outside the allowlist")
        if start_ms >= end_ms:
            return
        names = dict.fromkeys(("timestamp", "id", *fields))
        columns = ", ".join(f"`{name}`" for name in names)
        order = "DESC" if newest_first else "ASC"
        predicate = _WINDOW_FILTER
        params = [start_ms, end_ms, _TIMESTAMP_PATTERN]
        if after is not None:
            predicate += f" AND ({_TIMESTAMP}, `id`) > (%s, %s)"
            params.extend(after)
        params.append(limit or MAX_WINDOW_ROWS + 1)
        with self._connection.cursor(self._cursors.SSDictCursor) as cursor:
            cursor.execute(
                f"SELECT /*+ MAX_EXECUTION_TIME(10000) */ {columns} FROM `{table}` "
                f"WHERE {predicate} ORDER BY {_TIMESTAMP} {order}, `id` {order} LIMIT %s", params,
            )
            yield from cursor

    def history_page(
        self, table: str, fields: tuple[str, ...], start_ms: int, end_ms: int,
        *, after: tuple[int, int], limit: int,
    ) -> tuple[tuple[int, int], list[Row]]:
        if table not in TABLES:
            raise TelemetryError("invalid_source", "Table is outside the telemetry allowlist")
        with self._connection.cursor() as cursor:
            cursor.execute(
                f"SELECT /*+ MAX_EXECUTION_TIME(10000) */ COUNT(*), COALESCE(MAX(`id`), 0) "
                f"FROM `{table}` WHERE {_WINDOW_FILTER}", (start_ms, end_ms, _TIMESTAMP_PATTERN),
            )
            count, max_id = cursor.fetchone()
        rows = self.samples(table, fields, start_ms, end_ms, limit=limit, after=after)
        return (int(count), int(max_id)), list(rows)
