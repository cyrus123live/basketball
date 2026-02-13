"""Tests for feature computation."""

import pytest

from src.features.compute import compute_game_stats, possessions


class TestPossessions:
    def test_basic(self):
        # FGA=60, OREB=10, TOV=12, FTA=15
        # 60 - 10 + 12 + 0.40 * 15 = 68.0
        assert possessions(60, 10, 12, 15) == 68.0

    def test_zero_fta(self):
        assert possessions(60, 10, 12, 0) == 62.0

    def test_no_oreb(self):
        # More possessions when no OREBs (each OREB is a continuation)
        assert possessions(60, 0, 12, 15) == 78.0


class TestComputeGameStats:
    @pytest.fixture
    def sample_game(self):
        return {
            "home_score": 80, "away_score": 75,
            "home_fgm": 30, "home_fga": 60, "home_3pm": 8, "home_3pa": 20,
            "home_ftm": 12, "home_fta": 15, "home_oreb": 10, "home_dreb": 25,
            "home_tov": 12,
            "away_fgm": 28, "away_fga": 62, "away_3pm": 7, "away_3pa": 22,
            "away_ftm": 10, "away_fta": 14, "away_oreb": 8, "away_dreb": 22,
            "away_tov": 14,
            "num_ot": 0,
        }

    def test_returns_stats(self, sample_game):
        stats = compute_game_stats(sample_game)
        assert stats is not None
        assert "pace" in stats
        assert "home_ortg" in stats
        assert "away_ortg" in stats

    def test_pace_reasonable(self, sample_game):
        stats = compute_game_stats(sample_game)
        # College basketball pace typically 60-80
        assert 50 < stats["pace"] < 90

    def test_ortg_reasonable(self, sample_game):
        stats = compute_game_stats(sample_game)
        # ORtg typically 80-130
        assert 80 < stats["home_ortg"] < 140
        assert 80 < stats["away_ortg"] < 140

    def test_home_defense_equals_away_offense(self, sample_game):
        stats = compute_game_stats(sample_game)
        assert stats["home_drtg"] == stats["away_ortg"]
        assert stats["away_drtg"] == stats["home_ortg"]

    def test_efg_pct(self, sample_game):
        stats = compute_game_stats(sample_game)
        # eFG% = (FGM + 0.5 * 3PM) / FGA
        expected = (30 + 0.5 * 8) / 60
        assert abs(stats["home_efg_pct"] - expected) < 0.001

    def test_tov_pct(self, sample_game):
        stats = compute_game_stats(sample_game)
        # TOV% = TOV / (FGA + 0.40*FTA + TOV)
        denom = 60 + 0.40 * 15 + 12
        expected = 12 / denom
        assert abs(stats["home_tov_pct"] - expected) < 0.001

    def test_orb_pct(self, sample_game):
        stats = compute_game_stats(sample_game)
        # ORB% = OREB / (OREB + Opp_DREB)
        expected = 10 / (10 + 22)  # home OREB / (home OREB + away DREB)
        assert abs(stats["home_orb_pct"] - expected) < 0.001

    def test_ftr(self, sample_game):
        stats = compute_game_stats(sample_game)
        # FTr = FTA / FGA
        expected = 15 / 60
        assert abs(stats["home_ftr"] - expected) < 0.001

    def test_missing_data_returns_none(self):
        game = {"home_score": 80}  # Missing required fields
        assert compute_game_stats(game) is None

    def test_overtime_increases_pace_denominator(self, sample_game):
        stats_reg = compute_game_stats(sample_game)
        sample_game["num_ot"] = 1
        stats_ot = compute_game_stats(sample_game)
        # OT game should have lower pace (same possessions, more minutes)
        assert stats_ot["pace"] < stats_reg["pace"]


class TestComputeTeamFeaturesAsOf:
    """Integration tests that need a populated database."""

    def test_anti_lookahead(self, tmp_path):
        """Features should only use games before the as_of_date."""
        from src.utils.db import FeatureStore
        from src.features.compute import compute_team_features_as_of

        db = FeatureStore(tmp_path / "test.db")
        t1 = db.upsert_team("Team A", 2026)
        t2 = db.upsert_team("Team B", 2026)

        # Insert 3 games on different dates
        for i, date in enumerate(["2026-01-01", "2026-01-03", "2026-01-05"]):
            db.upsert_game({
                "game_id": f"test_{i}", "season": 2026, "game_date": date,
                "home_team_id": t1, "away_team_id": t2,
                "home_team_name": "Team A", "away_team_name": "Team B",
                "home_score": 80 + i * 5, "away_score": 70,
                "home_fgm": 30, "home_fga": 60, "home_3pm": 8, "home_3pa": 20,
                "home_ftm": 12, "home_fta": 15, "home_oreb": 10, "home_dreb": 25,
                "home_tov": 12,
                "away_fgm": 26, "away_fga": 58, "away_3pm": 6, "away_3pa": 18,
                "away_ftm": 12, "away_fta": 16, "away_oreb": 7, "away_dreb": 20,
                "away_tov": 15,
            })

        # Features as of Jan 3 should only use game on Jan 1
        features_jan3 = compute_team_features_as_of(db, t1, "2026-01-03")
        assert features_jan3 is not None
        assert features_jan3["games_played"] == 1

        # Features as of Jan 5 should use games on Jan 1 and Jan 3
        features_jan5 = compute_team_features_as_of(db, t1, "2026-01-05")
        assert features_jan5 is not None
        assert features_jan5["games_played"] == 2

        # Features as of Jan 1 should have no data
        features_jan1 = compute_team_features_as_of(db, t1, "2026-01-01")
        assert features_jan1 is None
