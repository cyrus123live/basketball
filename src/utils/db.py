"""SQLite FeatureStore — single database for all basketball data.

All temporal queries enforce `as_of_date` constraints to prevent lookahead bias.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from src.utils.config import DB_PATH

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS teams (
    team_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    conference  TEXT,
    season      INTEGER NOT NULL,
    UNIQUE(name, season)
);

CREATE TABLE IF NOT EXISTS games (
    game_id         TEXT PRIMARY KEY,   -- ESPN game ID or synthetic key
    season          INTEGER NOT NULL,
    game_date       TEXT NOT NULL,      -- YYYY-MM-DD
    home_team_id    INTEGER REFERENCES teams(team_id),
    away_team_id    INTEGER REFERENCES teams(team_id),
    home_team_name  TEXT NOT NULL,
    away_team_name  TEXT NOT NULL,
    home_score      INTEGER,
    away_score      INTEGER,
    num_ot          INTEGER DEFAULT 0,
    -- Home box score
    home_fgm    INTEGER, home_fga    INTEGER,
    home_3pm    INTEGER, home_3pa    INTEGER,
    home_ftm    INTEGER, home_fta    INTEGER,
    home_oreb   INTEGER, home_dreb   INTEGER,
    home_reb    INTEGER,
    home_ast    INTEGER, home_tov    INTEGER,
    home_stl    INTEGER, home_blk    INTEGER,
    home_pf     INTEGER,
    -- Away box score
    away_fgm    INTEGER, away_fga    INTEGER,
    away_3pm    INTEGER, away_3pa    INTEGER,
    away_ftm    INTEGER, away_fta    INTEGER,
    away_oreb   INTEGER, away_dreb   INTEGER,
    away_reb    INTEGER,
    away_ast    INTEGER, away_tov    INTEGER,
    away_stl    INTEGER, away_blk    INTEGER,
    away_pf     INTEGER
);
CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date);
CREATE INDEX IF NOT EXISTS idx_games_season ON games(season);
CREATE INDEX IF NOT EXISTS idx_games_home ON games(home_team_id, game_date);
CREATE INDEX IF NOT EXISTS idx_games_away ON games(away_team_id, game_date);

CREATE TABLE IF NOT EXISTS barttorvik_ratings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    team_name   TEXT NOT NULL,
    team_id     INTEGER REFERENCES teams(team_id),
    season      INTEGER NOT NULL,
    as_of_date  TEXT NOT NULL,      -- YYYY-MM-DD
    adj_o       REAL,
    adj_d       REAL,
    adj_t       REAL,
    barthag     REAL,
    efg_pct     REAL,
    efg_pct_d   REAL,
    tov_pct     REAL,
    tov_pct_d   REAL,
    orb_pct     REAL,
    orb_pct_d   REAL,
    ftr         REAL,
    ftr_d       REAL,
    UNIQUE(team_name, season, as_of_date)
);
CREATE INDEX IF NOT EXISTS idx_bart_date ON barttorvik_ratings(as_of_date);

CREATE TABLE IF NOT EXISTS historical_odds (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    season          INTEGER NOT NULL,
    game_date       TEXT NOT NULL,      -- YYYY-MM-DD
    home_team_name  TEXT,
    away_team_name  TEXT,
    game_id         TEXT REFERENCES games(game_id),
    spread_open     REAL,   -- Home spread (negative = home favored)
    spread_close    REAL,
    total_open      REAL,
    total_close     REAL,
    home_ml         INTEGER,
    away_ml         INTEGER,
    UNIQUE(season, game_date, home_team_name, away_team_name)
);

CREATE TABLE IF NOT EXISTS live_odds (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id         TEXT NOT NULL,
    snapshot_time   TEXT NOT NULL,      -- ISO 8601
    bookmaker       TEXT NOT NULL,
    market_type     TEXT NOT NULL,      -- h2h, spreads, totals
    outcome_name    TEXT NOT NULL,      -- team name or Over/Under
    price           REAL NOT NULL,      -- American odds
    point           REAL,              -- spread or total line
    UNIQUE(game_id, snapshot_time, bookmaker, market_type, outcome_name)
);
CREATE INDEX IF NOT EXISTS idx_live_odds_game ON live_odds(game_id);

CREATE TABLE IF NOT EXISTS team_features (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id         INTEGER NOT NULL REFERENCES teams(team_id),
    team_name       TEXT NOT NULL,
    season          INTEGER NOT NULL,
    as_of_date      TEXT NOT NULL,      -- YYYY-MM-DD (features computed BEFORE this date's games)
    games_played    INTEGER,
    pace            REAL,
    ortg            REAL,
    drtg            REAL,
    net_rtg         REAL,
    efg_pct         REAL,
    efg_pct_d       REAL,
    tov_pct         REAL,
    tov_pct_d       REAL,
    orb_pct         REAL,
    orb_pct_d       REAL,
    ftr             REAL,
    ftr_d           REAL,
    rest_days       INTEGER,
    UNIQUE(team_id, as_of_date)
);
CREATE INDEX IF NOT EXISTS idx_features_team_date ON team_features(team_id, as_of_date);

CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id         TEXT NOT NULL,
    game_date       TEXT NOT NULL,
    model_name      TEXT NOT NULL,
    home_team_name  TEXT,
    away_team_name  TEXT,
    predicted_home_prob REAL,
    market_home_prob    REAL,
    market_spread       REAL,
    edge                REAL,       -- predicted_prob - market_prob
    bet_recommended     INTEGER,    -- 0/1
    actual_home_win     INTEGER,    -- 0/1/NULL (filled after game)
    clv                 REAL,       -- closing line value (filled later)
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(game_id, model_name)
);
"""


class FeatureStore:
    """SQLite-backed store with anti-lookahead temporal queries."""

    def __init__(self, db_path: Path | str = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self):
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── Team operations ────────────────────────────────────────────────

    def upsert_team(self, name: str, season: int, conference: str = None) -> int:
        """Insert or get team, return team_id."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT team_id FROM teams WHERE name = ? AND season = ?",
                (name, season),
            ).fetchone()
            if row:
                return row["team_id"]
            cursor = conn.execute(
                "INSERT INTO teams (name, season, conference) VALUES (?, ?, ?)",
                (name, season, conference),
            )
            return cursor.lastrowid

    def get_team_id(self, name: str, season: int) -> int | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT team_id FROM teams WHERE name = ? AND season = ?",
                (name, season),
            ).fetchone()
            return row["team_id"] if row else None

    # ── Game operations ────────────────────────────────────────────────

    def upsert_game(self, game: dict):
        """Insert or replace a game record."""
        cols = [
            "game_id", "season", "game_date",
            "home_team_id", "away_team_id",
            "home_team_name", "away_team_name",
            "home_score", "away_score", "num_ot",
            "home_fgm", "home_fga", "home_3pm", "home_3pa",
            "home_ftm", "home_fta", "home_oreb", "home_dreb",
            "home_reb", "home_ast", "home_tov", "home_stl",
            "home_blk", "home_pf",
            "away_fgm", "away_fga", "away_3pm", "away_3pa",
            "away_ftm", "away_fta", "away_oreb", "away_dreb",
            "away_reb", "away_ast", "away_tov", "away_stl",
            "away_blk", "away_pf",
        ]
        values = [game.get(c) for c in cols]
        placeholders = ",".join(["?"] * len(cols))
        col_names = ",".join(cols)
        with self._connect() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO games ({col_names}) VALUES ({placeholders})",
                values,
            )

    def upsert_games_bulk(self, games: list[dict]):
        """Bulk insert games."""
        for game in games:
            self.upsert_game(game)

    def get_games_before(self, team_id: int, before_date: str) -> pd.DataFrame:
        """Get all games for a team strictly before a date (anti-lookahead).

        Returns games where the team played as either home or away.
        """
        sql = """
            SELECT * FROM games
            WHERE game_date < ?
              AND (home_team_id = ? OR away_team_id = ?)
            ORDER BY game_date
        """
        with self._connect() as conn:
            return pd.read_sql_query(sql, conn, params=(before_date, team_id, team_id))

    def get_games_for_date(self, game_date: str) -> pd.DataFrame:
        """Get all games on a specific date."""
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM games WHERE game_date = ?",
                conn, params=(game_date,),
            )

    def get_games_for_season(self, season: int) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM games WHERE season = ?",
                conn, params=(season,),
            )

    def game_exists(self, game_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM games WHERE game_id = ?", (game_id,)
            ).fetchone()
            return row is not None

    # ── Feature operations ─────────────────────────────────────────────

    def upsert_team_features(self, features: dict):
        """Insert or replace team features."""
        cols = [
            "team_id", "team_name", "season", "as_of_date", "games_played",
            "pace", "ortg", "drtg", "net_rtg",
            "efg_pct", "efg_pct_d", "tov_pct", "tov_pct_d",
            "orb_pct", "orb_pct_d", "ftr", "ftr_d", "rest_days",
        ]
        values = [features.get(c) for c in cols]
        placeholders = ",".join(["?"] * len(cols))
        col_names = ",".join(cols)
        with self._connect() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO team_features ({col_names}) VALUES ({placeholders})",
                values,
            )

    def get_team_features(self, team_id: int, as_of_date: str) -> dict | None:
        """Get the most recent features for a team as of a date (anti-lookahead)."""
        sql = """
            SELECT * FROM team_features
            WHERE team_id = ? AND as_of_date <= ?
            ORDER BY as_of_date DESC
            LIMIT 1
        """
        with self._connect() as conn:
            row = conn.execute(sql, (team_id, as_of_date)).fetchone()
            return dict(row) if row else None

    def get_matchup_features(
        self, home_team_id: int, away_team_id: int, game_date: str
    ) -> dict | None:
        """Build feature vector for a matchup using pre-game data only."""
        home = self.get_team_features(home_team_id, game_date)
        away = self.get_team_features(away_team_id, game_date)
        if home is None or away is None:
            return None
        return {
            "game_date": game_date,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "home_pace": home["pace"],
            "away_pace": away["pace"],
            "home_ortg": home["ortg"],
            "home_drtg": home["drtg"],
            "home_net_rtg": home["net_rtg"],
            "away_ortg": away["ortg"],
            "away_drtg": away["drtg"],
            "away_net_rtg": away["net_rtg"],
            "home_efg_pct": home["efg_pct"],
            "home_efg_pct_d": home["efg_pct_d"],
            "away_efg_pct": away["efg_pct"],
            "away_efg_pct_d": away["efg_pct_d"],
            "home_tov_pct": home["tov_pct"],
            "home_tov_pct_d": home["tov_pct_d"],
            "away_tov_pct": away["tov_pct"],
            "away_tov_pct_d": away["tov_pct_d"],
            "home_orb_pct": home["orb_pct"],
            "home_orb_pct_d": home["orb_pct_d"],
            "away_orb_pct": away["orb_pct"],
            "away_orb_pct_d": away["orb_pct_d"],
            "home_ftr": home["ftr"],
            "home_ftr_d": home["ftr_d"],
            "away_ftr": away["ftr"],
            "away_ftr_d": away["ftr_d"],
            "home_rest_days": home["rest_days"],
            "away_rest_days": away["rest_days"],
            "home_games_played": home["games_played"],
            "away_games_played": away["games_played"],
            "net_rtg_diff": home["net_rtg"] - away["net_rtg"],
        }

    # ── Barttorvik operations ──────────────────────────────────────────

    def upsert_barttorvik_ratings(self, ratings: list[dict]):
        """Bulk insert Barttorvik ratings."""
        cols = [
            "team_name", "team_id", "season", "as_of_date",
            "adj_o", "adj_d", "adj_t", "barthag",
            "efg_pct", "efg_pct_d", "tov_pct", "tov_pct_d",
            "orb_pct", "orb_pct_d", "ftr", "ftr_d",
        ]
        with self._connect() as conn:
            for r in ratings:
                values = [r.get(c) for c in cols]
                placeholders = ",".join(["?"] * len(cols))
                col_names = ",".join(cols)
                conn.execute(
                    f"INSERT OR REPLACE INTO barttorvik_ratings ({col_names}) VALUES ({placeholders})",
                    values,
                )

    # ── Odds operations ────────────────────────────────────────────────

    def upsert_historical_odds(self, odds: list[dict]):
        """Bulk insert historical odds (SBRO)."""
        cols = [
            "season", "game_date", "home_team_name", "away_team_name",
            "game_id", "spread_open", "spread_close",
            "total_open", "total_close", "home_ml", "away_ml",
        ]
        with self._connect() as conn:
            for o in odds:
                values = [o.get(c) for c in cols]
                placeholders = ",".join(["?"] * len(cols))
                col_names = ",".join(cols)
                conn.execute(
                    f"INSERT OR REPLACE INTO historical_odds ({col_names}) VALUES ({placeholders})",
                    values,
                )

    def upsert_live_odds(self, odds: list[dict]):
        """Bulk insert live odds snapshots."""
        cols = [
            "game_id", "snapshot_time", "bookmaker",
            "market_type", "outcome_name", "price", "point",
        ]
        with self._connect() as conn:
            for o in odds:
                values = [o.get(c) for c in cols]
                placeholders = ",".join(["?"] * len(cols))
                col_names = ",".join(cols)
                conn.execute(
                    f"INSERT OR REPLACE INTO live_odds ({col_names}) VALUES ({placeholders})",
                    values,
                )

    # ── Predictions ────────────────────────────────────────────────────

    def upsert_prediction(self, pred: dict):
        cols = [
            "game_id", "game_date", "model_name",
            "home_team_name", "away_team_name",
            "predicted_home_prob", "market_home_prob", "market_spread",
            "edge", "bet_recommended", "actual_home_win", "clv",
        ]
        values = [pred.get(c) for c in cols]
        placeholders = ",".join(["?"] * len(cols))
        col_names = ",".join(cols)
        with self._connect() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO predictions ({col_names}) VALUES ({placeholders})",
                values,
            )

    def get_predictions(self, model_name: str = None) -> pd.DataFrame:
        with self._connect() as conn:
            if model_name:
                return pd.read_sql_query(
                    "SELECT * FROM predictions WHERE model_name = ? ORDER BY game_date",
                    conn, params=(model_name,),
                )
            return pd.read_sql_query(
                "SELECT * FROM predictions ORDER BY game_date", conn
            )

    # ── Utility ────────────────────────────────────────────────────────

    def count_games(self, season: int = None) -> int:
        with self._connect() as conn:
            if season:
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM games WHERE season = ?", (season,)
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) as cnt FROM games").fetchone()
            return row["cnt"]

    def game_counts_by_season(self) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT season, COUNT(*) as game_count FROM games GROUP BY season ORDER BY season",
                conn,
            )
