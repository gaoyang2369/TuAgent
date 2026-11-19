from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator


SCHEMA = "dcma"
PROFILE = "g120_sim_v1"
PERIOD_MS = 3000
RETENTION_MS = 7 * 24 * 3600 * 1000
MAX_TIMESTAMP_MS = 253402271999999  # Last millisecond representable in Beijing time.
MAX_BUCKETS = 2000

TableName = Literal[
    "sim_normal", "sim_f07900_blocked", "sim_f30002_overvoltage", "sim_f07011_overtemp"
]
FieldName = Literal[
    "id", "timestamp", "device_name", "inverter_name", "date", "time", "status",
    "fault_code", "alarm_code", "control_word", "status_word", "dc_voltage",
    "speed_setpoint", "speed_actual", "current_actual", "torque_setpoint",
    "torque_actual", "air_intake_temp", "motor_temp", "inverter_temp", "actual_power",
    "field_current", "torque_current", "system_run_time", "inverter_radiator_temp",
    "inverter_load_rate", "motor_load_rate", "pulse_frequency", "motor_power",
    "feedback_power", "create_time",
]
NumericField = Literal[
    "dc_voltage", "speed_setpoint", "speed_actual", "current_actual", "torque_setpoint",
    "torque_actual", "air_intake_temp", "motor_temp", "inverter_temp", "actual_power",
    "field_current", "torque_current", "inverter_radiator_temp", "inverter_load_rate",
    "motor_load_rate", "pulse_frequency", "motor_power", "feedback_power",
]
MetadataField = FieldName | Literal["actual_solution"]
CodeField = Literal["fault_code", "alarm_code"]
Metric = Literal["min", "max", "mean", "rate", "threshold_duration"]
EpochMs = Annotated[int, Field(strict=True, ge=0, le=MAX_TIMESTAMP_MS)]
Seconds = Annotated[int, Field(strict=True, ge=1, le=RETENTION_MS // 1000)]


class TelemetryError(Exception):
    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceArgs(Arguments):
    table_name: TableName
    profile_version: str = Field(default=PROFILE, min_length=1, max_length=64)


class MetadataArgs(SourceArgs):
    fields: list[MetadataField] | None = Field(default=None, min_length=1, max_length=32)


class WindowArgs(SourceArgs):
    start_ms: EpochMs | None = None
    end_ms: EpochMs | None = None
    last_seconds: Seconds | None = None

    @model_validator(mode="after")
    def check_window(self):
        if (self.start_ms is None) != (self.end_ms is None):
            raise ValueError("start_ms and end_ms must be supplied together")
        if self.start_ms is not None and self.end_ms is not None:
            if self.last_seconds is not None:
                raise ValueError("Use either start_ms/end_ms or last_seconds")
            if not 0 < self.end_ms - self.start_ms <= RETENTION_MS:
                raise ValueError("Window must be positive and at most seven days")
        return self

    def require_window(self) -> None:
        if self.start_ms is None and self.last_seconds is None:
            raise ValueError("Specify start_ms/end_ms or last_seconds")


class QueryArgs(WindowArgs):
    mode: Literal["latest", "window"] = "latest"
    fields: list[FieldName] = Field(min_length=1, max_length=12)
    limit: int = Field(default=100, strict=True, ge=1, le=1000)
    cursor: str | None = Field(default=None, min_length=1, max_length=2048)

    @model_validator(mode="after")
    def check_mode(self):
        if self.mode == "latest":
            if any(value is not None for value in (self.start_ms, self.last_seconds, self.cursor)):
                raise ValueError("Window bounds and cursor require mode='window'")
        elif self.cursor is None:
            self.require_window()
        return self


class Threshold(Arguments):
    field: NumericField
    operator: Literal[">", ">=", "<", "<="]
    value: FiniteFloat
    unit: str = Field(min_length=1, max_length=40)
    source: str = Field(min_length=1, max_length=160)


class SummaryArgs(WindowArgs):
    fields: list[NumericField] = Field(min_length=1, max_length=12)
    bucket_seconds: int = Field(strict=True, ge=3, le=RETENTION_MS // 1000)
    metrics: list[Metric] = Field(default_factory=lambda: ["min", "max", "mean", "rate"],
                                  min_length=1, max_length=5)
    thresholds: list[Threshold] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def check_metrics(self):
        self.require_window()
        threshold_fields = [item.field for item in self.thresholds]
        if len(set(threshold_fields)) != len(threshold_fields):
            raise ValueError("Supply at most one threshold per field")
        if not set(threshold_fields) <= set(self.fields):
            raise ValueError("Thresholds must refer to requested fields")
        if "threshold_duration" in self.metrics and not self.thresholds:
            raise ValueError("threshold_duration requires thresholds")
        return self


class EventArgs(WindowArgs):
    code_field: CodeField = "fault_code"
    target_code: str | None = Field(default=None, min_length=1, max_length=50)

    @model_validator(mode="after")
    def check_event_window(self):
        self.require_window()
        if self.target_code == "0":
            raise ValueError("Target code must be nonzero")
        return self
