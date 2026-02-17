"""Fuzzy-match Barttorvik team names to ESPN team_ids.

Barttorvik uses short names ("Alabama", "Michigan St.") while ESPN uses
full names ("Alabama Crimson Tide", "Michigan State Spartans"). This module
bridges the gap using normalization + fuzzy matching.
"""

import logging
import re
import unicodedata

from thefuzz import fuzz, process

logger = logging.getLogger(__name__)

# Exact substitutions applied before fuzzy matching.
# Map Barttorvik quirks to forms closer to ESPN normalized names.
EXACT_SUBS = {
    "n.c. state": "nc state",
    "uconn": "connecticut",
    "unlv": "unlv",
    "ucf": "ucf",
    "lsu": "lsu",
    "smu": "smu",
    "vcu": "vcu",
    "fiu": "fiu",
    "liu": "liu",
    "umbc": "umbc",
    "uic": "uic",
    "unc": "north carolina",
    "unc wilmington": "unc wilmington",
    "unc greensboro": "unc greensboro",
    "unc asheville": "unc asheville",
    "utep": "utep",
    "utsa": "ut san antonio",
    "ut arlington": "ut arlington",
    "ut martin": "ut martin",
    "ut rio grande valley": "ut rio grande valley",
    "usc": "usc",
    "ole miss": "ole miss",
    "miami fl": "miami",
    "miami oh": "miami oh",
    "saint mary's": "saint marys",
    "saint joseph's": "saint josephs",
    "saint peter's": "saint peters",
    "saint john's": "saint johns",
    "saint francis": "saint francis",
    "saint bonaventure": "saint bonaventure",
    "saint louis": "saint louis",
    "saint thomas": "saint thomas",
    "loyola chicago": "loyola chicago",
    "texas a&m": "texas a&m",
    "florida a&m": "florida a&m",
    "prairie view a&m": "prairie view a&m",
    "north carolina a&t": "north carolina a&t",
    "alabama a&m": "alabama a&m",
    "corpus christi": "texas a&m corpus christi",
    "little rock": "little rock",
    "app state": "appalachian state",
    "southeastern la": "southeastern louisiana",
    "northwestern la": "northwestern state",
    "mcneese st.": "mcneese",
    "nicholls st.": "nicholls",
    "green bay": "green bay",
    "milwaukee": "milwaukee",
    "siu edwardsville": "siu edwardsville",
    "umass lowell": "umass lowell",
    "umass": "umass",
    "pitt": "pittsburgh",
    "cal st. bakersfield": "cal state bakersfield",
    "cal st. fullerton": "cal state fullerton",
    "cal st. northridge": "cal state northridge",
    "cal poly": "cal poly",
}


def normalize_name(name: str) -> str:
    """Normalize a team name for fuzzy matching.

    Strips punctuation, expands abbreviations, removes mascots.
    Works on both Barttorvik and ESPN name formats.
    """
    s = name.strip()

    # Normalize unicode
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()

    # Lowercase
    s = s.lower()

    # Strip Barttorvik tournament suffixes like "1 seed, Final Four"
    s = re.sub(r"\s+\d+\s+seed.*$", "", s)

    # Remove parenthetical qualifiers like (OH), (FL)
    s = re.sub(r"\([^)]*\)", "", s)

    # Strip possessive 's before general punct removal
    s = re.sub(r"'s\b", "s", s)

    # Strip punctuation except & and .
    s = re.sub(r"[^a-z0-9&. ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Apply exact substitutions (longest match first)
    for old, new in sorted(EXACT_SUBS.items(), key=lambda x: -len(x[0])):
        if s == old or s.startswith(old + " "):
            s = new + s[len(old):]
            break

    # Expand abbreviations
    abbrevs = [
        (r"\bst\.\b", "state"),
        (r"\bst\b", "state"),
        (r"\bso\.\b", "south"),
        (r"\bso\b", "south"),
        (r"\bse\b", "southeast"),
        (r"\bsw\b", "southwest"),
        (r"\bn\.\b", "north"),
        (r"\bs\.\b", "south"),
        (r"\be\.\b", "east"),
        (r"\bw\.\b", "west"),
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

    # Remove dots and extra spaces
    s = s.replace(".", "")
    s = re.sub(r"\s+", " ", s).strip()

    # Remove common mascot/suffix words (ESPN includes these, Barttorvik doesn't)
    multi_mascots = [
        "fighting illini", "tar heels", "nittany lions", "scarlet knights",
        "crimson tide", "yellow jackets", "demon deacons", "blue devils",
        "golden gophers", "horned frogs", "red raiders", "red wolves",
        "red hawks", "sun devils", "golden lions", "delta devils",
        "blue jays", "golden flashes", "thundering herd", "golden griffins",
        "purple eagles", "red foxes", "river hawks", "rainbow warriors",
        "ragin cajuns", "mean green", "black knights", "blue hose",
        "great danes", "running rebels", "fighting irish", "fighting hawks",
        "runnin bulldogs", "golden eagles", "red storm", "blue demons",
        "wolf pack", "golden hurricane", "golden bears",
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
        "bluejays", "ducks", "beavers", "coyotes", "thunderbolts",
        "shockers", "billikens", "hoosier", "bulls", "crimson",
        "anteaters", "raiders", "tribe", "midshipmen", "crusaders",
    }
    words = s.split()
    while len(words) > 1 and words[-1] in single_mascots:
        words.pop()
    s = " ".join(words)

    return s.strip()


def resolve_barttorvik_team_ids(db, ratings: list[dict], season: int) -> dict:
    """Fuzzy-match Barttorvik names to ESPN team_ids for a given season.

    For season 2015, matches against 2016 ESPN teams (since our DB starts at 2016).
    Mutates each rating dict in-place by setting 'team_id'.

    Args:
        db: FeatureStore instance.
        ratings: List of Barttorvik rating dicts (with 'team_name').
        season: Barttorvik season year.

    Returns:
        Dict with match summary: {matched, total, unmatched_names}.
    """
    # For season 2015, no ESPN teams exist — match against season 2016
    lookup_season = season + 1 if season == 2015 else season

    # Build lookup: normalized ESPN name -> team_id
    with db._connect() as conn:
        teams = conn.execute(
            "SELECT team_id, name FROM teams WHERE season = ?",
            (lookup_season,),
        ).fetchall()

    if not teams:
        logger.warning("No ESPN teams found for season %d", lookup_season)
        return {"matched": 0, "total": len(ratings), "unmatched_names": []}

    espn_choices = {}
    for t in teams:
        norm = normalize_name(t["name"])
        espn_choices[norm] = t["team_id"]

    choice_list = list(espn_choices.keys())
    matched = 0
    unmatched = []

    for r in ratings:
        bart_norm = normalize_name(r["team_name"])

        # Try exact match first
        if bart_norm in espn_choices:
            r["team_id"] = espn_choices[bart_norm]
            matched += 1
            continue

        # Fuzzy match
        result = process.extractOne(
            bart_norm, choice_list, scorer=fuzz.token_sort_ratio
        )
        if result and result[1] >= 80:
            r["team_id"] = espn_choices[result[0]]
            matched += 1
        else:
            r["team_id"] = None
            unmatched.append(r["team_name"])
            if result:
                logger.debug(
                    "No match for '%s' (best: '%s' @ %d)",
                    r["team_name"], result[0], result[1],
                )

    logger.info(
        "Season %d: matched %d/%d Barttorvik names (%.1f%%)",
        season, matched, len(ratings), matched / len(ratings) * 100,
    )
    if unmatched:
        logger.debug("Unmatched names: %s", unmatched[:20])

    return {
        "matched": matched,
        "total": len(ratings),
        "unmatched_names": unmatched,
    }
