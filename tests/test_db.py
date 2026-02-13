"""Tests for the FeatureStore database layer."""

import tempfile
from pathlib import Path

import pytest

from src.utils.db import FeatureStore


@pytest.fixture
def db(tmp_path):
    """Create a temporary FeatureStore for testing."""
    return FeatureStore(tmp_path / "test.db")


@pytest.fixture
def populated_db(db):
    """DB with sample teams and games across 3 dates."""
    t1 = db.upsert_team("Duke", 2026, "ACC")
    t2 = db.upsert_team("UNC", 2026, "ACC")
    t3 = db.upsert_team("Kentucky", 2026, "SEC")

    games = [
        _make_game("g1", "2026-01-01", t1, t2, "Duke", "UNC", 80, 75),
        _make_game("g2", "2026-01-03", t1, t3, "Duke", "Kentucky", 85, 70),
        _make_game("g3", "2026-01-03", t2, t3, "UNC", "Kentucky", 90, 88),
        _make_game("g4", "2026-01-05", t3, t1, "Kentucky", "Duke", 72, 78),
        _make_game("g5", "2026-01-05", t2, t1, "UNC", "Duke", 65, 70),
    ]
    for g in games:
        db.upsert_game(g)

    return db, {"duke": t1, "unc": t2, "kentucky": t3}


def _make_game(gid, date, home_id, away_id, home_name, away_name, h_score, a_score):
    return {
        "game_id": gid, "season": 2026, "game_date": date,
        "home_team_id": home_id, "away_team_id": away_id,
        "home_team_name": home_name, "away_team_name": away_name,
        "home_score": h_score, "away_score": a_score,
        "home_fgm": 30, "home_fga": 60, "home_3pm": 8, "home_3pa": 20,
        "home_ftm": 12, "home_fta": 15, "home_oreb": 10, "home_dreb": 25,
        "home_reb": 35, "home_ast": 15, "home_tov": 12, "home_stl": 5,
        "home_blk": 3, "home_pf": 18,
        "away_fgm": 28, "away_fga": 62, "away_3pm": 7, "away_3pa": 22,
        "away_ftm": 10, "away_fta": 14, "away_oreb": 8, "away_dreb": 22,
        "away_reb": 30, "away_ast": 12, "away_tov": 14, "away_stl": 4,
        "away_blk": 2, "away_pf": 20,
    }


class TestSchemaCreation:
    def test_creates_tables(self, db):
        with db._connect() as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            names = {t["name"] for t in tables}
            assert "games" in names
            assert "teams" in names
            assert "team_features" in names
            assert "predictions" in names
            assert "barttorvik_ratings" in names
            assert "historical_odds" in names
            assert "live_odds" in names


class TestTeamOperations:
    def test_upsert_and_get(self, db):
        tid = db.upsert_team("Duke", 2026, "ACC")
        assert tid is not None
        assert db.get_team_id("Duke", 2026) == tid

    def test_upsert_idempotent(self, db):
        tid1 = db.upsert_team("Duke", 2026)
        tid2 = db.upsert_team("Duke", 2026)
        assert tid1 == tid2

    def test_different_seasons(self, db):
        tid1 = db.upsert_team("Duke", 2025)
        tid2 = db.upsert_team("Duke", 2026)
        assert tid1 != tid2

    def test_nonexistent_team(self, db):
        assert db.get_team_id("Nonexistent", 2026) is None


class TestTemporalQueries:
    def test_games_before_strict(self, populated_db):
        db, teams = populated_db
        # Duke games before Jan 5: should be g1, g2 only (not g4, g5)
        games = db.get_games_before(teams["duke"], "2026-01-05")
        assert len(games) == 2
        assert set(games["game_id"].tolist()) == {"g1", "g2"}

    def test_games_before_first_date(self, populated_db):
        db, teams = populated_db
        # No games before Jan 1
        games = db.get_games_before(teams["duke"], "2026-01-01")
        assert len(games) == 0

    def test_games_before_all(self, populated_db):
        db, teams = populated_db
        # Duke games before Jan 10: should be all 4 Duke games
        games = db.get_games_before(teams["duke"], "2026-01-10")
        assert len(games) == 4

    def test_games_before_middle(self, populated_db):
        db, teams = populated_db
        # Duke games before Jan 2: only g1
        games = db.get_games_before(teams["duke"], "2026-01-02")
        assert len(games) == 1
        assert games.iloc[0]["game_id"] == "g1"


class TestGameOperations:
    def test_get_games_for_date(self, populated_db):
        db, _ = populated_db
        games = db.get_games_for_date("2026-01-03")
        assert len(games) == 2

    def test_game_exists(self, populated_db):
        db, _ = populated_db
        assert db.game_exists("g1")
        assert not db.game_exists("nonexistent")

    def test_count_games(self, populated_db):
        db, _ = populated_db
        assert db.count_games() == 5
        assert db.count_games(2026) == 5
        assert db.count_games(2025) == 0

    def test_upsert_replaces(self, populated_db):
        db, teams = populated_db
        # Update score
        db.upsert_game({
            "game_id": "g1", "season": 2026, "game_date": "2026-01-01",
            "home_team_id": teams["duke"], "away_team_id": teams["unc"],
            "home_team_name": "Duke", "away_team_name": "UNC",
            "home_score": 100, "away_score": 90,
        })
        games = db.get_games_for_date("2026-01-01")
        assert len(games) == 1
        assert games.iloc[0]["home_score"] == 100


class TestFeatureStore:
    def test_upsert_and_get_features(self, populated_db):
        db, teams = populated_db
        features = {
            "team_id": teams["duke"], "team_name": "Duke",
            "season": 2026, "as_of_date": "2026-01-03",
            "games_played": 1, "pace": 70.0,
            "ortg": 110.0, "drtg": 95.0, "net_rtg": 15.0,
            "efg_pct": 0.55, "efg_pct_d": 0.45,
            "tov_pct": 0.15, "tov_pct_d": 0.18,
            "orb_pct": 0.32, "orb_pct_d": 0.28,
            "ftr": 0.35, "ftr_d": 0.30,
            "rest_days": 2,
        }
        db.upsert_team_features(features)

        result = db.get_team_features(teams["duke"], "2026-01-03")
        assert result is not None
        assert result["pace"] == 70.0
        assert result["net_rtg"] == 15.0

    def test_features_anti_lookahead(self, populated_db):
        db, teams = populated_db
        # Insert features for two dates
        for date, pace in [("2026-01-02", 68.0), ("2026-01-04", 72.0)]:
            db.upsert_team_features({
                "team_id": teams["duke"], "team_name": "Duke",
                "season": 2026, "as_of_date": date,
                "games_played": 1, "pace": pace,
                "ortg": 110.0, "drtg": 95.0, "net_rtg": 15.0,
            })

        # As of Jan 3, should get Jan 2 features (not Jan 4)
        result = db.get_team_features(teams["duke"], "2026-01-03")
        assert result["pace"] == 68.0

        # As of Jan 5, should get Jan 4 features (latest <= date)
        result = db.get_team_features(teams["duke"], "2026-01-05")
        assert result["pace"] == 72.0

    def test_matchup_features(self, populated_db):
        db, teams = populated_db
        for tid, name, nr in [
            (teams["duke"], "Duke", 15.0),
            (teams["unc"], "UNC", 5.0),
        ]:
            db.upsert_team_features({
                "team_id": tid, "team_name": name,
                "season": 2026, "as_of_date": "2026-01-05",
                "games_played": 2, "pace": 70.0,
                "ortg": 105.0 + nr/2, "drtg": 105.0 - nr/2, "net_rtg": nr,
            })

        matchup = db.get_matchup_features(
            teams["duke"], teams["unc"], "2026-01-05"
        )
        assert matchup is not None
        assert matchup["net_rtg_diff"] == 10.0  # 15 - 5
        assert matchup["home_net_rtg"] == 15.0
        assert matchup["away_net_rtg"] == 5.0


class TestPredictions:
    def test_upsert_and_get_predictions(self, db):
        db.upsert_prediction({
            "game_id": "test1", "game_date": "2026-01-05",
            "model_name": "test_model",
            "home_team_name": "Duke", "away_team_name": "UNC",
            "predicted_home_prob": 0.65, "market_home_prob": 0.55,
            "market_spread": -3.5, "edge": 0.10,
            "bet_recommended": 1, "actual_home_win": 1, "clv": 0.05,
        })

        preds = db.get_predictions("test_model")
        assert len(preds) == 1
        assert preds.iloc[0]["predicted_home_prob"] == 0.65
