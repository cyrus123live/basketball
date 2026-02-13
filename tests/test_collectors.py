"""Tests for data collectors.

These tests validate parsing logic without making network calls.
Network-dependent tests should be run manually.
"""

import pytest

from src.data.barttorvik_collector import _extract_value
from src.data.cbbpy_collector import _date_to_season, _names_match
from src.data.odds_api_collector import american_to_implied_prob
from src.data.sbro_loader import (
    _extract_season_from_filename,
    _parse_date,
    _parse_spread,
)


class TestBarttorvik:
    def test_extract_value_with_rank(self):
        assert _extract_value("127.2 5") == 127.2

    def test_extract_value_plain(self):
        assert _extract_value("98.5") == 98.5

    def test_extract_value_none(self):
        assert _extract_value(None) is None

    def test_extract_value_nan(self):
        import pandas as pd
        assert _extract_value(pd.NA) is None

    def test_extract_value_barthag(self):
        assert _extract_value(".9818 1") == 0.9818


class TestCbbpyCollector:
    def test_date_to_season_november(self):
        assert _date_to_season("2025-11-15") == 2026

    def test_date_to_season_december(self):
        assert _date_to_season("2025-12-01") == 2026

    def test_date_to_season_january(self):
        assert _date_to_season("2026-01-15") == 2026

    def test_date_to_season_march(self):
        assert _date_to_season("2026-03-20") == 2026

    def test_names_match_exact(self):
        assert _names_match("Duke Blue Devils", "Duke Blue Devils")

    def test_names_match_contains(self):
        assert _names_match("Duke", "Duke Blue Devils")

    def test_names_match_contains(self):
        # "Michigan" is contained in "Michigan State Spartans" — correctly matches
        # (this is a known limitation; the crosswalk handles disambiguation)
        assert _names_match("Michigan Wolverines", "Michigan State Spartans")
        assert _names_match("Houston Cougars", "Houston Baptist Huskies")

    def test_names_no_match(self):
        assert not _names_match("Duke", "UNC")


class TestOddsApi:
    def test_american_to_implied_negative(self):
        # -110 → ~52.4%
        prob = american_to_implied_prob(-110)
        assert abs(prob - 0.5238) < 0.001

    def test_american_to_implied_positive(self):
        # +150 → 40%
        prob = american_to_implied_prob(150)
        assert abs(prob - 0.400) < 0.001

    def test_american_to_implied_even(self):
        # +100 → 50%
        prob = american_to_implied_prob(100)
        assert abs(prob - 0.500) < 0.001

    def test_american_to_implied_heavy_favorite(self):
        # -300 → 75%
        prob = american_to_implied_prob(-300)
        assert abs(prob - 0.750) < 0.001


class TestSbroLoader:
    def test_parse_spread_normal(self):
        assert _parse_spread("-3.5") == -3.5

    def test_parse_spread_pick(self):
        assert _parse_spread("pk") == 0.0
        assert _parse_spread("PK") == 0.0

    def test_parse_spread_half(self):
        assert _parse_spread("-3½") == -3.5

    def test_parse_spread_none(self):
        assert _parse_spread(None) is None

    def test_extract_season_range(self):
        assert _extract_season_from_filename("ncaa basketball 2025-26.xlsx") == 2026

    def test_extract_season_single(self):
        assert _extract_season_from_filename("ncaab_2026.xlsx") == 2026

    def test_extract_season_short_range(self):
        assert _extract_season_from_filename("cbb_25-26.xlsx") == 2026

    def test_parse_date_numeric(self):
        assert _parse_date(1104, 2026) == "2025-11-04"
        assert _parse_date(215, 2026) == "2026-02-15"

    def test_parse_date_string(self):
        from datetime import datetime
        d = datetime(2026, 1, 15)
        assert _parse_date(d, 2026) == "2026-01-15"
