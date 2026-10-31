import copy
from unittest.mock import AsyncMock, Mock

import pytest

from app.agent.errors import AgentError
from app.agent.mcp_contracts import WeatherFailure
from app.agent.weather import GEOCODING_URL, AiohttpJSONClient, OpenMeteoWeather

GEO = {
    "results": [
        {
            "name": "Москва",
            "country": "Россия",
            "country_code": "RU",
            "admin1": "Москва",
            "latitude": 55.75,
            "longitude": 37.61,
        }
    ]
}
FORECAST = {
    "timezone": "Europe/Moscow",
    "current": {
        "time": 1791806400,
        "temperature_2m": 5.0,
        "weather_code": 3,
        "wind_speed_10m": 4.0,
    },
    "current_units": {"time": "unixtime", "temperature_2m": "°C", "wind_speed_10m": "km/h"},
}


async def test_weather_normalizes_city_units_time():
    http = Mock(get=AsyncMock(side_effect=[GEO, FORECAST]))
    result = await OpenMeteoWeather(http).current(" Москва ")
    assert (result.city, result.temperature_c, result.wind_unit, result.condition) == (
        "Москва",
        5,
        "km/h",
        "3",
    )
    assert result.observed_at.utcoffset().total_seconds() == 10800
    assert http.get.await_args_list[0].args[1]["name"] == "Москва"


@pytest.mark.parametrize(
    "query",
    [
        "Москва, Россия",
        "МОСКВА, Москва, Российская Федерация",
        "москва, г. москва, РФ",
        "Moscow, Moscow, Russian Federation",
    ],
)
async def test_moscow_country_and_region_select_exact_russian_result(query):
    geo = copy.deepcopy(GEO)
    geo["results"].append(
        {
            "name": "Москва",
            "country": "Соединённые Штаты",
            "country_code": "US",
            "admin1": "Айдахо",
            "latitude": 46.73,
            "longitude": -116.99,
        }
    )
    http = Mock(get=AsyncMock(side_effect=[geo, FORECAST]))

    result = await OpenMeteoWeather(http).current(query)

    assert (result.city, result.country) == ("Москва", "Россия")
    params = http.get.await_args_list[0].args[1]
    assert params["name"].casefold() in {"москва", "moscow"}
    assert params["countryCode"] == "RU"
    assert http.get.await_args_list[1].args[1]["latitude"] == 55.75


async def test_ambiguous_city_never_chooses_first():
    geo = {
        "results": [
            {
                "name": "Спрингфилд",
                "country": "Соединённые Штаты",
                "country_code": "US",
                "admin1": "Иллинойс",
                "latitude": 39.80,
                "longitude": -89.64,
            },
            {
                "name": "Спрингфилд",
                "country": "Соединённые Штаты",
                "country_code": "US",
                "admin1": "Массачусетс",
                "latitude": 42.10,
                "longitude": -72.59,
            },
        ]
    }
    http = Mock(get=AsyncMock(return_value=geo))
    with pytest.raises(WeatherFailure) as error:
        await OpenMeteoWeather(http).current("Спрингфилд, США")
    assert len(error.value.failure.choices) == 2
    assert {choice.region for choice in error.value.failure.choices} == {
        "Иллинойс",
        "Массачусетс",
    }
    http.get.assert_awaited_once()


async def test_clarification_after_ambiguous_result_selects_matching_region():
    geo = {
        "results": [
            {
                "name": "Спрингфилд",
                "country": "Соединённые Штаты",
                "country_code": "US",
                "admin1": "Иллинойс",
                "latitude": 39.80,
                "longitude": -89.64,
            },
            {
                "name": "Спрингфилд",
                "country": "Соединённые Штаты",
                "country_code": "US",
                "admin1": "Массачусетс",
                "latitude": 42.10,
                "longitude": -72.59,
            },
        ]
    }
    http = Mock(get=AsyncMock(side_effect=[geo, geo, FORECAST]))
    weather = OpenMeteoWeather(http)
    with pytest.raises(WeatherFailure):
        await weather.current("Спрингфилд, США")

    result = await weather.current("Спрингфилд, Иллинойс, США")

    assert result.country == "Соединённые Штаты"
    assert http.get.await_args_list[1].args[1] == {
        "name": "Спрингфилд",
        "count": 100,
        "language": "ru",
        "format": "json",
        "countryCode": "US",
    }
    assert http.get.await_args_list[2].args[1]["latitude"] == 39.80


async def test_no_exact_city_or_qualifier_match_never_selects_arbitrary_result():
    prefix_only = {
        "results": [GEO["results"][0] | {"name": "Московский", "admin1": "Тюменская область"}]
    }
    with pytest.raises(AgentError, match="city_not_found"):
        await OpenMeteoWeather(Mock(get=AsyncMock(return_value=prefix_only))).current("Москва")
    with pytest.raises(AgentError, match="city_not_found"):
        await OpenMeteoWeather(Mock(get=AsyncMock(return_value=GEO))).current(
            "Москва, Ленинградская область, Россия"
        )


@pytest.mark.parametrize(
    "responses,code",
    [
        ([{"generationtime_ms": 0.1}], "city_not_found"),
        ([{}], "invalid_result"),
        ([TimeoutError("private")], "timeout"),
        ([GEO, {}], "invalid_result"),
        ([GEO, FORECAST | {"timezone": "Invalid/Secret"}], "invalid_result"),
        ([GEO, FORECAST | {"current_units": {}}], "invalid_result"),
    ],
)
async def test_weather_errors_without_network(responses, code):
    with pytest.raises(AgentError) as error:
        await OpenMeteoWeather(Mock(get=AsyncMock(side_effect=responses))).current("Москва")
    assert str(error.value) == code


@pytest.mark.parametrize(
    "status,body,expected,calls",
    [
        (500, b"private", "weather_unavailable", 2),
        (404, b"private", "weather_unavailable", 1),
        (302, b"private", "weather_unavailable", 1),
        (200, b"invalid-json-private", "invalid_result", 1),
        (200, b"[]", "invalid_result", 1),
    ],
)
async def test_http_boundary_errors_and_bounded_retries(monkeypatch, status, body, expected, calls):
    from app.agent import weather

    async def chunks(size):
        yield body

    response = Mock(status=status, content=Mock(iter_chunked=chunks))
    response_cm = AsyncMock()
    response_cm.__aenter__.return_value = response
    session = Mock(get=Mock(return_value=response_cm))
    session_cm = AsyncMock()
    session_cm.__aenter__.return_value = session
    factory = Mock(return_value=session_cm)
    monkeypatch.setattr(weather.aiohttp, "ClientSession", factory)
    with pytest.raises(AgentError, match=expected):
        await AiohttpJSONClient(2, 1).get(GEOCODING_URL, {})
    assert factory.call_count == calls
    assert factory.call_args.kwargs["timeout"].total == 2
    assert session.get.call_args.kwargs["allow_redirects"] is False
