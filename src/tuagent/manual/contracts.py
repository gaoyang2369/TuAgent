import re
from typing import Annotated
from urllib.parse import urlsplit, urlunsplit

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
)


def clean(value: str) -> str:
    return " ".join(value.split())


def validate_code(value: str) -> str:
    if not re.fullmatch(r"[FA][0-9]{5}", value):
        raise ValueError(
            "Use a complete F or A code with five digits, including leading zeros"
        )
    return value


ShortText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)
]
Code = Annotated[
    str,
    StringConstraints(strip_whitespace=True, to_upper=True, min_length=6, max_length=6),
    AfterValidator(validate_code),
]


class ManualModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ManualEdition(ManualModel):
    document_number: ShortText
    edition: ShortText
    language: ShortText
    device_series: ShortText
    control_units: tuple[ShortText, ...] = Field(min_length=1, max_length=24)
    firmware_versions: tuple[ShortText, ...] = Field(min_length=1, max_length=24)
    source_url: str = Field(min_length=1, max_length=1024)

    @field_validator("source_url")
    @classmethod
    def check_url(cls, value: str) -> str:
        parts = urlsplit(value)
        if (
            parts.scheme in ("http", "https")
            and parts.hostname
            and parts.username is None
        ):
            return urlunsplit(parts._replace(fragment=""))
        if parts.scheme == "file" and parts.path.startswith("/"):
            return urlunsplit(parts._replace(fragment=""))
        raise ValueError("Provide an HTTP(S) source URL or an absolute file URL")


class ManualScope(ManualModel):
    device_series: ShortText
    control_unit: ShortText | None = None
    firmware: ShortText | None = None
    edition: ShortText | None = None
    document_number: ShortText | None = None
    language: ShortText | None = None


class LookupArgs(ManualScope):
    code: Code


class SearchArgs(ManualScope):
    query: str = Field(min_length=1, max_length=500)
    code: Code | None = None
    section: ShortText | None = None
    limit: int = Field(default=4, strict=True, ge=1, le=8)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        value = clean(value)
        if not value:
            raise ValueError("Describe the manual evidence to find")
        return value
