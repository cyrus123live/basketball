"""Load historical odds from Sports Book Review Online (SBRO) Excel files.

SBRO provides free historical closing lines (spreads, totals, ML) back to ~2007-08.
Files are downloaded manually and placed in data/odds/sbro/.
"""

import logging
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from thefuzz import fuzz

from src.utils.config import CROSSWALK_PATH, SBRO_DIR

logger = logging.getLogger(__name__)

# SBRO Excel files typically have columns like:
# Date, Rot, VH, Team, 1st, 2nd, Final, Open, Close, ML, 2H
# Where VH = V (visitor) or H (home)


def load_sbro_file(filepath: Path | str) -> pd.DataFrame:
    """Load and parse a single SBRO Excel file.

    Args:
        filepath: Path to the SBRO .xlsx or .xls file.

    Returns:
        DataFrame with parsed odds data.
    """
    filepath = Path(filepath)
    logger.info("Loading SBRO file: %s", filepath.name)

    try:
        if filepath.suffix == ".xlsx":
            df = pd.read_excel(filepath, engine="openpyxl")
        else:
            df = pd.read_excel(filepath)
    except Exception as e:
        logger.error("Failed to load %s: %s", filepath, e)
        return pd.DataFrame()

    # Normalize column names
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Common SBRO column name variations
    col_map = {
        "date": "date",
        "rot": "rot",
        "vh": "vh",
        "team": "team",
        "1st": "first_half",
        "2nd": "second_half",
        "final": "final_score",
        "open": "spread_open",
        "close": "spread_close",
        "ml": "ml",
        "2h": "second_half_line",
    }

    rename = {}
    for orig, target in col_map.items():
        for col in df.columns:
            if col == orig or col.startswith(orig):
                rename[col] = target
                break
    df = df.rename(columns=rename)

    return df


def parse_sbro_season(filepath: Path | str, season: int) -> list[dict]:
    """Parse an SBRO file into structured odds records.

    SBRO format: rows alternate between visitor and home team.
    The date appears on the visitor row; the home row has no date.

    Args:
        filepath: Path to the SBRO Excel file.
        season: Season end year (e.g. 2026).

    Returns:
        List of odds dicts ready for FeatureStore.upsert_historical_odds().
    """
    df = load_sbro_file(filepath)
    if df.empty:
        return []

    odds_records = []
    current_date = None
    visitor_row = None

    for _, row in df.iterrows():
        # Skip non-data rows
        team = str(row.get("team", "")).strip()
        if not team or team.lower() in ("team", "nan", ""):
            continue

        # Check if this row has a date (visitor row)
        date_val = row.get("date")
        if pd.notna(date_val):
            current_date = _parse_date(date_val, season)
            visitor_row = row
            continue

        # This is a home row (no date)
        if visitor_row is None or current_date is None:
            continue

        home_row = row
        away_team = str(visitor_row.get("team", "")).strip()
        home_team = str(home_row.get("team", "")).strip()

        # Parse the odds values
        record = {
            "season": season,
            "game_date": current_date,
            "away_team_name": away_team,
            "home_team_name": home_team,
            "game_id": None,  # Will be matched later
            "spread_open": _parse_spread(home_row.get("spread_open")),
            "spread_close": _parse_spread(home_row.get("spread_close")),
            "total_open": _parse_total(visitor_row.get("spread_open")),
            "total_close": _parse_total(visitor_row.get("spread_close")),
            "home_ml": _safe_int(home_row.get("ml")),
            "away_ml": _safe_int(visitor_row.get("ml")),
        }

        odds_records.append(record)
        visitor_row = None

    logger.info(
        "Parsed %d odds records from %s (season %d)",
        len(odds_records), Path(filepath).name, season,
    )
    return odds_records


def load_all_sbro(sbro_dir: Path = SBRO_DIR) -> list[dict]:
    """Load all SBRO files from the sbro directory.

    Expects filenames to contain the season year, e.g.
    'ncaa basketball 2025-26.xlsx' or 'ncaab_2026.xlsx'.
    """
    all_records = []
    files = sorted(sbro_dir.glob("*.xls*"))

    if not files:
        logger.warning("No SBRO files found in %s", sbro_dir)
        return []

    for filepath in files:
        season = _extract_season_from_filename(filepath.name)
        if season is None:
            logger.warning("Cannot determine season from filename: %s", filepath.name)
            continue

        records = parse_sbro_season(filepath, season)
        all_records.extend(records)

    logger.info("Loaded %d total odds records from %d SBRO files", len(all_records), len(files))
    return all_records


def match_odds_to_games(
    odds_records: list[dict], db
) -> list[dict]:
    """Match SBRO odds to games in the database using fuzzy name matching.

    Updates each odds record's game_id field.

    Args:
        odds_records: List of odds dicts from parse_sbro_season.
        db: FeatureStore instance.

    Returns:
        Updated odds records with game_id filled where matched.
    """
    crosswalk = _load_crosswalk()
    matched = 0
    total = len(odds_records)

    for record in odds_records:
        game_date = record["game_date"]
        if game_date is None:
            continue

        # Get games on this date
        games_df = db.get_games_for_date(game_date)
        if games_df.empty:
            continue

        # Try to match by team names
        sbro_home = record["home_team_name"]
        sbro_away = record["away_team_name"]

        best_match = None
        best_score = 0

        for _, game in games_df.iterrows():
            db_home = game["home_team_name"]
            db_away = game["away_team_name"]

            # Try crosswalk first
            mapped_home = crosswalk.get(sbro_home, sbro_home)
            mapped_away = crosswalk.get(sbro_away, sbro_away)

            score = (
                fuzz.token_sort_ratio(mapped_home, db_home)
                + fuzz.token_sort_ratio(mapped_away, db_away)
            ) / 2

            if score > best_score:
                best_score = score
                best_match = game["game_id"]

        if best_score >= 70:  # Threshold for accepting a match
            record["game_id"] = best_match
            matched += 1

    match_rate = matched / total * 100 if total > 0 else 0
    logger.info(
        "Matched %d/%d odds records (%.1f%%)", matched, total, match_rate
    )
    return odds_records


def _load_crosswalk() -> dict:
    """Load team name crosswalk CSV mapping SBRO names to ESPN names."""
    if not CROSSWALK_PATH.exists():
        logger.info("No crosswalk file at %s, using raw names", CROSSWALK_PATH)
        return {}

    df = pd.read_csv(CROSSWALK_PATH)
    crosswalk = {}
    if "sbro_name" in df.columns and "espn_name" in df.columns:
        for _, row in df.iterrows():
            crosswalk[row["sbro_name"]] = row["espn_name"]
    return crosswalk


def _parse_date(val, season: int) -> str | None:
    """Parse SBRO date values to YYYY-MM-DD string."""
    if pd.isna(val):
        return None

    # SBRO dates are often numeric (e.g., 1104 for Nov 4) or strings
    val_str = str(val).strip()

    # Handle datetime objects
    if isinstance(val, datetime):
        return val.strftime("%Y-%m-%d")

    # Handle numeric dates like 1104, 104, etc.
    try:
        num = int(float(val_str))
        if num > 1231:  # Might be a full date
            return None

        month = num // 100
        day = num % 100

        if month < 1 or month > 12 or day < 1 or day > 31:
            return None

        # Determine year from season and month
        if month >= 9:  # Sep-Dec
            year = season - 1
        else:  # Jan-Apr
            year = season

        return f"{year}-{month:02d}-{day:02d}"
    except (ValueError, TypeError):
        pass

    # Try common string date formats
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(val_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    return None


def _parse_spread(val) -> float | None:
    """Parse spread value, handling various SBRO formats."""
    if pd.isna(val):
        return None
    try:
        val_str = str(val).strip()
        # Remove 'pk' (pick'em = 0)
        if val_str.lower() in ("pk", "pk'", "pick"):
            return 0.0
        # Handle half-points like -3½ or -3.5
        val_str = val_str.replace("½", ".5")
        return float(val_str)
    except (ValueError, TypeError):
        return None


def _parse_total(val) -> float | None:
    """Parse total value (same format as spread)."""
    return _parse_spread(val)


def _safe_int(val) -> int | None:
    if pd.isna(val):
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _extract_season_from_filename(filename: str) -> int | None:
    """Extract season year from SBRO filename.

    Examples:
        'ncaa basketball 2025-26.xlsx' → 2026
        'ncaab_2026.xlsx' → 2026
        'cbb_25-26.xlsx' → 2026
    """
    # Try 4-digit year range: "2025-26" or "2025-2026"
    match = re.search(r"(\d{4})-(\d{2,4})", filename)
    if match:
        end_year = match.group(2)
        if len(end_year) == 2:
            return int(match.group(1)[:2] + end_year)
        return int(end_year)

    # Try 2-digit year range: "25-26"
    match = re.search(r"(\d{2})-(\d{2})", filename)
    if match:
        return 2000 + int(match.group(2))

    # Try single 4-digit year (assumed to be end year)
    match = re.search(r"(\d{4})", filename)
    if match:
        return int(match.group(1))

    return None
