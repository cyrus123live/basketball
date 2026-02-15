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

    SBRO format: rows alternate between visitor (VH=V) and home (VH=H) team.
    Every row has a date and a VH indicator.

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
    visitor_row = None

    for _, row in df.iterrows():
        # Skip non-data rows
        team = str(row.get("team", "")).strip()
        if not team or team.lower() in ("team", "nan", ""):
            continue

        vh = str(row.get("vh", "")).strip().upper()

        if vh == "V":
            visitor_row = row
            continue

        if vh == "H" and visitor_row is not None:
            home_row = row
            game_date = _parse_date(home_row.get("date"), season)
            if game_date is None:
                game_date = _parse_date(visitor_row.get("date"), season)

            away_team = str(visitor_row.get("team", "")).strip()
            home_team = str(home_row.get("team", "")).strip()

            # In SBRO, the visitor row's Open/Close are the total,
            # and the home row's Open/Close are the spread.
            record = {
                "season": season,
                "game_date": game_date,
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


def _normalize_name(name: str) -> str:
    """Normalize a team name for matching.

    Handles both ESPN format ('Miami (OH) RedHawks') and
    SBRO format ('MiamiOhio') by stripping to a common form.
    """
    import unicodedata

    s = name.strip()

    # Remove parenthetical qualifiers like (OH), (FL), (NY)
    s = re.sub(r"\([^)]*\)", "", s)

    # Split camelCase: 'MiamiOhio' -> 'Miami Ohio'
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    # Split letter-to-digit: 'A&M' stays, but 'NC2' etc.
    s = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", s)

    # Normalize unicode
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()

    # Lowercase, strip punctuation except &
    s = s.lower()
    s = re.sub(r"[^a-z0-9& ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Exact whole-name substitutions (before abbreviation expansion)
    exact_subs = {
        "app state": "appalachian state",
        "nc wilmington": "unc wilmington",
        "ncwilmington": "unc wilmington",
        "nc state": "nc state",
        "nc greensboro": "unc greensboro",
        "nc asheville": "unc asheville",
        "nc charlotte": "charlotte",
        "utarlington": "ut arlington",
        "utsa": "ut san antonio",
        "utep": "ut el paso",
        "ut rio grande valley": "ut rio grande valley",
        "fla atlantic": "florida atlantic",
        "fla gulf coast": "florida gulf coast",
        "ark pine bluff": "arkansas pine bluff",
        "ark little rock": "little rock",
        "arkansas lr": "little rock",
        "southern cal": "usc",
        "ncstate": "nc state",
        "illinois chicago": "uic",
        "md baltimore co": "umbc",
        "mdbaltimore co": "umbc",
        "wisc green bay": "green bay",
        "wisc milwaukee": "milwaukee",
        "siuedwardsville": "siu edwardsville",
        "utrio grande valley": "ut rio grande valley",
        "state peter s": "saint peters",
        "state peter": "saint peters",
        "florida am": "florida a&m",
        "texas am": "texas a&m",
        "prairie view am": "prairie view a&m",
        "st josephs": "saint josephs",
        "st johns": "saint johns",
        "st louis": "saint louis",
        "st marys": "saint marys",
        "st bonaventure": "saint bonaventure",
        "st peters": "saint peters",
        "st thomas": "saint thomas",
        "st francis": "saint francis",
        "miami ohio": "miami oh",
        "miami fl": "miami",
    }
    for old, new in exact_subs.items():
        if s == old or s.startswith(old + " "):
            s = new + s[len(old):]
            break

    # Abbreviation expansions (word boundaries)
    abbrevs = [
        (r"\bst\b", "state"),
        (r"\bso\b", "south"),
        (r"\bse\b", "southeast"),
        (r"\bsw\b", "southwest"),
        (r"\btenn\b", "tennessee"),
        (r"\bokla\b", "oklahoma"),
        (r"\bmiss\b", "mississippi"),
        (r"\bfla\b", "florida"),
        (r"\bark\b", "arkansas"),
        (r"\bwis\b", "wisconsin"),
        (r"\bconn\b", "connecticut"),
    ]
    for pattern, repl in abbrevs:
        s = re.sub(pattern, repl, s)

    # Expand UNC -> unc (keep as-is, ESPN uses it)
    # Don't expand NC at start — too ambiguous (NC State vs North Carolina)

    # Remove common mascot/suffix words (ESPN includes these, SBRO doesn't)
    # Use multi-word mascots first, then single-word
    multi_mascots = [
        "fighting illini", "tar heels", "nittany lions", "scarlet knights",
        "crimson tide", "yellow jackets", "demon deacons", "blue devils",
        "golden gophers", "horned frogs", "red raiders", "red wolves",
        "red hawks", "sun devils", "golden lions", "delta devils",
        "blue jays", "golden flashes", "thundering herd", "golden griffins",
        "purple eagles", "red foxes", "river hawks", "rainbow warriors",
        "ragin cajuns", "mean green", "black knights", "blue hose",
        "great danes", "running rebels", "fighting irish", "fighting hawks",
        "runnin bulldogs",
    ]

    for mascot in multi_mascots:
        if s.endswith(" " + mascot):
            s = s[: -(len(mascot) + 1)]
            break

    single_mascots = {
        "wildcats", "bulldogs", "eagles", "tigers", "bears", "lions",
        "panthers", "hawks", "cougars", "huskies", "cavaliers", "knights",
        "cardinals", "hornets", "aggies", "warriors", "pirates", "rebels",
        "falcons", "owls", "rams", "spartans", "wolverines", "bruins",
        "gators", "sooners", "longhorns", "buckeyes", "hoosiers",
        "jayhawks", "razorbacks", "mountaineers", "volunteers",
        "commodores", "gamecocks", "seminoles", "hurricanes",
        "wolfpack", "orange", "hokies", "terrapins", "boilermakers",
        "hawkeyes", "badgers", "cornhuskers", "cyclones", "cowboys",
        "bison", "zips", "redhawks", "bearcats", "bobcats", "broncos",
        "musketeers", "flyers", "friars", "hoyas", "colonials",
        "explorers", "dukes", "spiders", "governors", "racers",
        "skyhawks", "buccaneers", "mocs", "paladins", "catamounts",
        "keydets", "seahawks", "chanticleers", "phoenix", "flames",
        "rockets", "chippewas", "redbirds", "salukis", "sycamores",
        "braves", "leathernecks", "penguins", "vikings", "mastodons",
        "jaguars", "dolphins", "ospreys", "hatters", "blazers",
        "49ers", "monarchs", "highlanders", "peacocks", "jaspers",
        "gaels", "bonnies", "griffs", "stags", "blackbirds",
        "seawolves", "retrievers", "minutemen", "lumberjacks",
        "antelopes", "toreros", "dons", "waves", "pilots", "zags",
        "broncs", "saints", "utes", "buffaloes", "lobos", "aztecs",
        "matadors", "gauchos", "mustangs", "roadrunners", "miners",
        "warhawks", "trojans", "thunderbirds", "lopes", "rattlers",
        "terriers", "hounds", "mavericks", "ramblers",
    }

    words = s.split()
    while len(words) > 1 and words[-1] in single_mascots:
        words.pop()

    s = " ".join(words)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def match_odds_to_games(
    odds_records: list[dict], db
) -> list[dict]:
    """Match SBRO odds to games in the database using normalized name matching.

    Updates each odds record's game_id field.

    Args:
        odds_records: List of odds dicts from parse_sbro_season.
        db: FeatureStore instance.

    Returns:
        Updated odds records with game_id filled where matched.
    """
    matched = 0
    total = len(odds_records)

    # Cache: group games by date for fast lookup
    games_cache: dict[str, pd.DataFrame] = {}
    # Cache normalized ESPN names
    espn_norm_cache: dict[str, str] = {}

    for record in odds_records:
        game_date = record["game_date"]
        if game_date is None:
            continue

        if game_date not in games_cache:
            games_cache[game_date] = db.get_games_for_date(game_date)
        games_df = games_cache[game_date]
        if games_df.empty:
            continue

        sbro_home_norm = _normalize_name(record["home_team_name"])
        sbro_away_norm = _normalize_name(record["away_team_name"])

        best_match = None
        best_score = 0

        for _, game in games_df.iterrows():
            db_home = game["home_team_name"]
            db_away = game["away_team_name"]

            if db_home not in espn_norm_cache:
                espn_norm_cache[db_home] = _normalize_name(db_home)
            if db_away not in espn_norm_cache:
                espn_norm_cache[db_away] = _normalize_name(db_away)

            db_home_norm = espn_norm_cache[db_home]
            db_away_norm = espn_norm_cache[db_away]

            score = (
                fuzz.token_sort_ratio(sbro_home_norm, db_home_norm)
                + fuzz.token_sort_ratio(sbro_away_norm, db_away_norm)
            ) / 2

            if score > best_score:
                best_score = score
                best_match = game["game_id"]

        if best_score >= 75:
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
