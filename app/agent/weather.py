import json
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import aiohttp
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.agent.clock import timezone
from app.agent.errors import AgentError, ErrorCode
from app.agent.mcp_contracts import CityChoice, ToolFailure, WeatherFailure
from app.agent.models import WeatherArgs, WeatherResult

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def _normalized(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).casefold().replace("ё", "е")
    value = "".join(
        "" if unicodedata.combining(char) else char if char.isalnum() else " " for char in value
    )
    return " ".join(value.split())


_COUNTRY_ALIASES = {
    "RU": {
        "ru",
        "rus",
        "russia",
        "russian federation",
        "рф",
        "россия",
        "российская федерация",
    },
    "US": {
        "us",
        "usa",
        "united states",
        "united states of america",
        "сша",
        "соединенные штаты",
        "соединенные штаты америки",
    },
    "GB": {"gb", "uk", "united kingdom", "great britain", "великобритания"},
}
COUNTRY_ALIASES = {
    code: frozenset(_normalized(value) for value in aliases)
    for code, aliases in _COUNTRY_ALIASES.items()
}

PLACE_ALIAS_GROUPS = tuple(
    frozenset(_normalized(value) for value in group)
    for group in (
        {"Москва", "Moscow", "Moskva", "город Москва", "г. Москва"},
        {
            "Московская область",
            "Московская обл.",
            "Подмосковье",
            "Moscow Oblast",
            "Moscow Region",
        },
        {"Санкт-Петербург", "Санкт Петербург", "СПб", "Saint Petersburg", "St Petersburg"},
    )
)


def _country_code(value: str) -> str | None:
    normalized = _normalized(value)
    return next((code for code, aliases in COUNTRY_ALIASES.items() if normalized in aliases), None)


def _same_place(left: str, right: str) -> bool:
    left_key, right_key = _normalized(left), _normalized(right)
    if left_key == right_key:
        return True
    return any(left_key in group and right_key in group for group in PLACE_ALIAS_GROUPS)


@dataclass(frozen=True)
class LocationQuery:
    city: str
    region: str | None = None
    country: str | None = None
    qualifier: str | None = None
    country_code: str | None = None

    @classmethod
    def parse(cls, value: str) -> "LocationQuery":
        value = WeatherArgs(city=value).city
        parts = [part.strip() for part in value.split(",")]
        if not all(parts) or len(parts) > 3:
            raise AgentError(ErrorCode.INVALID_ARGUMENTS)
        if len(parts) == 1:
            return cls(city=parts[0])
        if len(parts) == 2:
            code = _country_code(parts[1])
            if code:
                return cls(city=parts[0], country=parts[1], country_code=code)
            return cls(city=parts[0], qualifier=parts[1])
        return cls(
            city=parts[0],
            region=parts[1],
            country=parts[2],
            country_code=_country_code(parts[2]),
        )


class JSONClient(Protocol):
    async def get(self, url: str, params: dict) -> dict: ...


class AiohttpJSONClient:
    def __init__(self, timeout: float, retries: int = 1):
        if not 0 < timeout <= 120 or retries not in (0, 1):
            raise ValueError("Invalid HTTP limits")
        self.timeout = timeout
        self.retries = retries

    async def get(self, url: str, params: dict) -> dict:
        if url not in {GEOCODING_URL, FORECAST_URL}:
            raise AgentError(ErrorCode.WEATHER_UNAVAILABLE)
        for attempt in range(self.retries + 1):
            try:
                async with (
                    aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=self.timeout), trust_env=False
                    ) as session,
                    session.get(url, params=params, allow_redirects=False) as response,
                ):
                    if response.status != 200:
                        if response.status >= 500 and attempt < self.retries:
                            continue
                        raise AgentError(ErrorCode.WEATHER_UNAVAILABLE)
                    content = bytearray()
                    async for chunk in response.content.iter_chunked(65536):
                        content.extend(chunk)
                        if len(content) > 1_048_576:
                            raise AgentError(ErrorCode.INVALID_RESULT)
                    result = json.loads(content)
                    if not isinstance(result, dict):
                        raise AgentError(ErrorCode.INVALID_RESULT)
                    return result
            except TimeoutError:
                if attempt == self.retries:
                    raise AgentError(ErrorCode.TIMEOUT) from None
            except (aiohttp.ClientError, OSError):
                if attempt == self.retries:
                    raise AgentError(ErrorCode.WEATHER_UNAVAILABLE) from None
            except AgentError:
                raise
            except (ValueError, UnicodeError):
                raise AgentError(ErrorCode.INVALID_RESULT) from None
        raise AgentError(ErrorCode.WEATHER_UNAVAILABLE)


class APIModel(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, hide_input_in_errors=True)


class GeoLocation(APIModel):
    name: str = Field(min_length=1, max_length=100)
    country: str = Field(min_length=1, max_length=100)
    country_code: str = Field(default="", max_length=2)
    admin1: str = Field(default="", max_length=100)
    admin2: str = Field(default="", max_length=100)
    latitude: float = Field(ge=-90, le=90, allow_inf_nan=False)
    longitude: float = Field(ge=-180, le=180, allow_inf_nan=False)


class GeoResponse(APIModel):
    results: list[GeoLocation] = Field(default_factory=list, max_length=100)


class CurrentWeather(APIModel):
    time: int = Field(gt=0)
    temperature_2m: float = Field(allow_inf_nan=False)
    weather_code: int = Field(ge=0, le=99)
    wind_speed_10m: float = Field(ge=0, allow_inf_nan=False)


class ForecastResponse(APIModel):
    timezone: str
    current: CurrentWeather
    current_units: dict[str, str]


class OpenMeteoWeather:
    def __init__(self, http: JSONClient):
        self.http = http

    async def current(self, city: str) -> WeatherResult:
        query = LocationQuery.parse(city)
        try:
            params = {
                "name": query.city,
                "count": 100,
                "language": "ru",
                "format": "json",
            }
            if query.country_code:
                params["countryCode"] = query.country_code
            raw = await self.http.get(
                GEOCODING_URL,
                params,
            )
            if raw.get("error") or ("results" not in raw and "generationtime_ms" not in raw):
                raise AgentError(ErrorCode.INVALID_RESULT)
            geo = GeoResponse.model_validate(raw)
            choices = []
            for location in geo.results:
                if not self._matches(query, location):
                    continue
                choice = CityChoice(
                    city=location.name,
                    country=location.country,
                    region=", ".join(filter(None, (location.admin1, location.admin2))),
                    latitude=location.latitude,
                    longitude=location.longitude,
                )
                if choice not in choices:
                    choices.append(choice)
            if not choices:
                raise AgentError(ErrorCode.CITY_NOT_FOUND)
            if len(choices) > 1:
                raise WeatherFailure(ToolFailure(error=ErrorCode.AMBIGUOUS_CITY, choices=choices))
            location = choices[0]
            forecast = ForecastResponse.model_validate(
                await self.http.get(
                    FORECAST_URL,
                    {
                        "latitude": location.latitude,
                        "longitude": location.longitude,
                        "current": "temperature_2m,weather_code,wind_speed_10m",
                        "timezone": "auto",
                        "timeformat": "unixtime",
                        "temperature_unit": "celsius",
                        "wind_speed_unit": "kmh",
                    },
                )
            )
            zone = timezone(forecast.timezone)
            units = forecast.current_units
            if (
                units.get("temperature_2m") != "°C"
                or units.get("wind_speed_10m") != "km/h"
                or units.get("time") != "unixtime"
            ):
                raise AgentError(ErrorCode.INVALID_RESULT)
            current = forecast.current
            return WeatherResult(
                city=location.city,
                country=location.country,
                temperature_c=current.temperature_2m,
                condition=str(current.weather_code),
                wind_speed=current.wind_speed_10m,
                wind_unit="km/h",
                observed_at=datetime.fromtimestamp(current.time, UTC).astimezone(zone),
                timezone=forecast.timezone,
            )
        except TimeoutError:
            raise AgentError(ErrorCode.TIMEOUT) from None
        except AgentError as exc:
            if exc.code == ErrorCode.INVALID_TIMEZONE:
                raise AgentError(ErrorCode.INVALID_RESULT) from None
            raise
        except (ValidationError, ValueError, TypeError, AttributeError, OverflowError):
            raise AgentError(ErrorCode.INVALID_RESULT) from None

    @staticmethod
    def _matches(query: LocationQuery, location: GeoLocation) -> bool:
        if not _same_place(query.city, location.name):
            return False
        country_matches = OpenMeteoWeather._country_matches
        region_matches = OpenMeteoWeather._region_matches
        if query.country and not country_matches(query.country, location):
            return False
        if query.region and not region_matches(query.region, location):
            return False
        if query.qualifier and not (
            country_matches(query.qualifier, location) or region_matches(query.qualifier, location)
        ):
            return False
        return True

    @staticmethod
    def _country_matches(value: str, location: GeoLocation) -> bool:
        code = _country_code(value)
        if code:
            return (
                location.country_code.upper() == code
                or _normalized(location.country) in COUNTRY_ALIASES[code]
            )
        return _same_place(value, location.country)

    @staticmethod
    def _region_matches(value: str, location: GeoLocation) -> bool:
        return any(
            candidate and _same_place(value, candidate)
            for candidate in (location.admin1, location.admin2)
        )
