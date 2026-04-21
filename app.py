import json
import math
from collections import Counter
from functools import lru_cache
# Required imports
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import dash
from dash import dcc, html, dash_table, Input, Output, State
import plotly.express as px
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# File path constants
BOUNDARY_FILE = "NGA_LGA_Boundaries_2_7839795478074887427.geojson"
PROJECT_DATA_FILE = "nigeria_project_data.csv"
PROJECT_LOCATION_FILE = "PROJECT_GEOGRAPHIC_LOCATION_V2_NG.csv"

# Risk and scoring constants
TARGET_RISK_SCORES = {
    "None of these": 0,
    "Civilians": 4,
    "Military": 3,
    "Police": 3,
    "Government (Employees and offices)": 2,
    "Other": 1,
}
SEVERITY_SCORES = {
    "Unknown": 1,
    "Low": 2,
    "Medium": 3,
    "High": 4,
}
EXPOSURE_RISK_THRESHOLD = 4
IMPACT_BINS = [-float("inf"), 25, 50, 100, float("inf")]
LIKELIHOOD_BAND_MAP = {
    "Very Unlikely": 0,
    "Unlikely": 1,
    "Likely": 2,
    "Highly Likely": 3,
}
RISK_MATRIX = {
    (0, 0): 0, (0, 1): 1, (0, 2): 1, (0, 3): 2,
    (1, 0): 0, (1, 1): 1, (1, 2): 2, (1, 3): 2,
    (2, 0): 1, (2, 1): 2, (2, 2): 2, (2, 3): 3,
    (3, 0): 1, (3, 1): 2, (3, 2): 3, (3, 3): 3,
}
MAP_CENTER = {"lat": 9.082, "lon": 8.6753}  # Nigeria center


def load_lga_boundaries():
    lga_gdf = gpd.read_file(BOUNDARY_FILE)
    if lga_gdf.crs is None:
        lga_gdf = lga_gdf.set_crs(epsg=4326)
    else:
        lga_gdf = lga_gdf.to_crs(epsg=4326)

    lga_gdf["lga_key"] = (
        lga_gdf["statename"].astype(str).str.strip()
        + " | "
        + lga_gdf["lganame"].astype(str).str.strip()
    )
    return lga_gdf


def spatial_join_events(armed_df, lga_gdf):
    armed_points = gpd.GeoDataFrame(
        armed_df,
        geometry=gpd.points_from_xy(armed_df["longitude"], armed_df["latitude"]),
        crs="EPSG:4326",
    )

    armed_joined = gpd.sjoin(
        armed_points,
        lga_gdf[["lga_key", "lganame", "statename", "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"], errors="ignore")

    armed_joined = armed_joined.dropna(subset=["lga_key"]).copy()
    if armed_joined.empty:
        raise ValueError("No Armed clash events were spatially matched to an LGA.")

    armed_joined["year_month"] = armed_joined["event_date"].dt.to_period("M").astype(str)
    return armed_joined


def load_project_locations(lga_gdf):
    def read_csv_with_fallback(path, **kwargs):
        # Source extracts may have mixed encodings and a few malformed rows.
        last_exc = None
        for encoding in [None, "utf-8-sig", "cp1252", "latin-1"]:
            for engine in ["c", "python"]:
                read_kwargs = dict(kwargs)
                if encoding is not None:
                    read_kwargs["encoding"] = encoding
                read_kwargs["engine"] = engine

                # Python engine can tolerate occasional bad rows in vendor extracts.
                if engine == "python":
                    # low_memory is only valid for the C engine.
                    read_kwargs.pop("low_memory", None)
                    read_kwargs.setdefault("on_bad_lines", "skip")

                try:
                    return pd.read_csv(path, **read_kwargs)
                except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
                    last_exc = exc

        if last_exc is not None:
            raise last_exc
        raise ValueError(f"Unable to read CSV file: {path}")

    projects = read_csv_with_fallback(PROJECT_DATA_FILE, low_memory=False)
    if "PROJ_ID" not in projects.columns:
        raise ValueError("Missing PROJ_ID in nigeria_project_data.csv")

    # Keep one row of metadata per project ID for hover details.
    project_cols = [
        "PROJ_ID",
        "PROJ_DISPLAY_NAME",
        "PROJ_LGL_NAME",
        "PROJ_STAT_NAME",
        "PROJ_APPRVL_FY",
    ]
    available_project_cols = [col for col in project_cols if col in projects.columns]
    project_details = projects[available_project_cols].copy()
    project_details["PROJ_ID"] = project_details["PROJ_ID"].astype(str).str.strip()
    project_details = project_details.drop_duplicates(subset=["PROJ_ID"])
    valid_project_ids = set(project_details["PROJ_ID"].tolist())

    required_geo_cols = ["PROJ_ID", "ISO_CNTRY_CODE", "GEO_LOC_NME", "GEO_LATITUDE_NBR", "GEO_LONGITUDE_NBR"]

    # Support both raw source exports (header starts after 4 metadata rows)
    # and pre-filtered CSVs that already start with the true header at row 1.
    geo = read_csv_with_fallback(PROJECT_LOCATION_FILE, low_memory=False)
    geo.columns = [str(col).strip().strip('"') for col in geo.columns]
    missing_geo_cols = [col for col in required_geo_cols if col not in geo.columns]
    if missing_geo_cols:
        geo = read_csv_with_fallback(PROJECT_LOCATION_FILE, skiprows=4, low_memory=False)
        geo.columns = [str(col).strip().strip('"') for col in geo.columns]
        missing_geo_cols = [col for col in required_geo_cols if col not in geo.columns]
        if missing_geo_cols:
            raise ValueError(f"Missing required columns in {PROJECT_LOCATION_FILE}: {missing_geo_cols}")

    geo["PROJ_ID"] = geo["PROJ_ID"].astype(str).str.strip()
    geo["ISO_CNTRY_CODE"] = geo["ISO_CNTRY_CODE"].astype(str).str.strip().str.upper()

    geo = geo[geo["PROJ_ID"].isin(valid_project_ids)].copy()
    geo = geo[geo["ISO_CNTRY_CODE"].eq("NG")].copy()

    geo["GEO_LATITUDE_NBR"] = pd.to_numeric(geo["GEO_LATITUDE_NBR"], errors="coerce")
    geo["GEO_LONGITUDE_NBR"] = pd.to_numeric(geo["GEO_LONGITUDE_NBR"], errors="coerce")
    geo = geo.dropna(subset=["GEO_LATITUDE_NBR", "GEO_LONGITUDE_NBR"]).copy()

    if geo.empty:
        return pd.DataFrame(
            columns=[
                "PROJ_ID",
                "GEO_LOC_NME",
                "GEO_LATITUDE_NBR",
                "GEO_LONGITUDE_NBR",
                "PROJ_DISPLAY_NAME",
                "PROJ_LGL_NAME",
                "PROJ_STAT_NAME",
                "PROJ_APPRVL_FY",
                "project_title",
                "lga_key",
            ]
        )

    geo = geo.merge(project_details, on="PROJ_ID", how="left")
    geo["project_title"] = (
        geo.get("PROJ_DISPLAY_NAME", pd.Series(index=geo.index, dtype=object)).fillna("").astype(str).str.strip()
    )
    if "PROJ_LGL_NAME" in geo.columns:
        geo["project_title"] = geo["project_title"].mask(
            geo["project_title"].eq(""),
            geo["PROJ_LGL_NAME"].fillna("").astype(str).str.strip(),
        )
    geo["project_title"] = geo["project_title"].mask(geo["project_title"].eq(""), geo["PROJ_ID"])

    project_points = gpd.GeoDataFrame(
        geo,
        geometry=gpd.points_from_xy(geo["GEO_LONGITUDE_NBR"], geo["GEO_LATITUDE_NBR"]),
        crs="EPSG:4326",
    )
    project_joined = gpd.sjoin(
        project_points,
        lga_gdf[["lga_key", "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"], errors="ignore")

    return pd.DataFrame(project_joined)


def calculate_impact_score(target, severity_type):
    """Calculate impact score as target_score × severity_score."""
    target_score = TARGET_RISK_SCORES.get(str(target).strip() if pd.notna(target) else "None of these", 0)
    severity_score = SEVERITY_SCORES.get(str(severity_type).strip() if pd.notna(severity_type) else "Unknown", 0)
    return target_score * severity_score


def parse_multi_value(cell):
    if pd.isna(cell):
        return []
    text = str(cell).strip()
    if not text:
        return []
    return [item.strip() for item in text.split(";") if item and item.strip()]


def indicator_counts(df, column_name):
    total_counter = Counter()

    for _, raw in df[column_name].items():
        indicators = parse_multi_value(raw)
        if indicators:
            total_counter.update(indicators)

    if not total_counter:
        return pd.DataFrame(columns=["indicator", "occurrences"])

    return pd.DataFrame(
        {
            "indicator": list(total_counter.keys()),
            "occurrences": [total_counter[key] for key in total_counter.keys()],
        }
    ).sort_values("occurrences", ascending=False)


def build_actor_list(lga_df, actor1_col, actor2_col):
    actor_frames = [lga_df[actor1_col].dropna()]
    if actor2_col and actor2_col in lga_df.columns:
        actor_frames.append(lga_df[actor2_col].dropna())

    actor_series = pd.concat(actor_frames, ignore_index=True) if actor_frames else pd.Series(dtype=object)
    return (
        actor_series.astype(str)
        .str.strip()
        .loc[lambda series: series.ne("")]
        .loc[lambda series: ~series.str.contains("civilian", case=False, na=False)]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )


def summarize_lga(lga_key, filtered_df, actor1_col, actor2_col):
    lga_df = filtered_df[filtered_df["lga_key"].eq(lga_key)].copy()

    total_deaths = pd.to_numeric(lga_df["fatalities"], errors="coerce").fillna(0).sum()
    civilian_deaths = pd.to_numeric(lga_df["civilian_fatalities"], errors="coerce").fillna(0).sum()
    
    # Calculate impact/exposure/severity scores for selected LGA.
    total_impact_score = lga_df["impact_score"].sum() if "impact_score" in lga_df.columns else 0
    # Map total impact score to bucket
    if total_impact_score < 25:
        impact_bucket = "Insignificant"
        impact_range = "<25"
    elif total_impact_score < 50:
        impact_bucket = "Minor"
        impact_range = "25–49"
    elif total_impact_score < 100:
        impact_bucket = "Severe"
        impact_range = "50–99"
    else:
        impact_bucket = "Critical"
        impact_range = "100+"
    print(lga_df.head())
    total_exposure_score = lga_df["target_score"].sum() if "target_score" in lga_df.columns else 0
    severity_score = lga_df["severity_score"].sum() if "severity_score" in lga_df.columns and not lga_df.empty else 0
    event_count = len(lga_df)
    avg_impact_score = total_impact_score / event_count if event_count > 0 else 0

    return {
        "event_count": event_count,
        "total_deaths": int(total_deaths),
        "civilian_deaths": int(civilian_deaths),
        "total_impact_score": round(total_impact_score, 2),
            "impact_bucket": impact_bucket,
            "impact_range": impact_range,
        "total_exposure_score": round(total_exposure_score, 2),
        "severity_score": int(severity_score),
        "avg_impact_score": round(avg_impact_score, 2),
        "high_impact_event_count": int((pd.to_numeric(lga_df["impact_score"], errors="coerce").fillna(0) >= EXPOSURE_RISK_THRESHOLD).sum()) if "impact_score" in lga_df.columns else 0,
        "actors": build_actor_list(lga_df, actor1_col, actor2_col),
        "knowledge_df": indicator_counts(lga_df, "knowledge"),
        "resources_df": indicator_counts(lga_df, "resources"),
        "expectation_df": indicator_counts(lga_df, "expectation"),
    }


def classify_lga_exposure(lga_key, filtered_df, end_label):
    lga_df = filtered_df[filtered_df["lga_key"].eq(lga_key)].copy()
    if lga_df.empty:
        return {
            "exposure": "N/A",
            "descriptor": "No events recorded",
            "count_3m": 0,
            "count_12m": 0,
            "count_24m": 0,
        }

    lga_df["impact_score"] = pd.to_numeric(lga_df.get("impact_score"), errors="coerce").fillna(0)
    qualifying = lga_df[lga_df["impact_score"] >= EXPOSURE_RISK_THRESHOLD].copy()

    if qualifying.empty:
        return {
            "exposure": "N/A",
            "descriptor": "No qualifying events (impact score >= 4)",
            "count_3m": 0,
            "count_12m": 0,
            "count_24m": 0,
        }

    end_period = pd.Period(end_label, freq="M")
    end_ts = end_period.to_timestamp(how="end")

    def count_in_last_months(months):
        start_period = end_period - (months - 1)
        start_ts = start_period.to_timestamp(how="start")
        return int(((qualifying["event_date"] >= start_ts) & (qualifying["event_date"] <= end_ts)).sum())

    count_3m = count_in_last_months(3)
    count_12m = count_in_last_months(12)
    count_24m = count_in_last_months(24)

    if count_3m >= 1:
        return {
            "exposure": "Highly Likely",
            "descriptor": "Could occur within days or weeks",
            "count_3m": count_3m,
            "count_12m": count_12m,
            "count_24m": count_24m,
        }
    if count_12m >= 1:
        return {
            "exposure": "Likely",
            "descriptor": "Could occur within a year or so",
            "count_3m": count_3m,
            "count_12m": count_12m,
            "count_24m": count_24m,
        }
    if count_24m >= 1:
        return {
            "exposure": "Unlikely",
            "descriptor": "Could occur within 24+ months",
            "count_3m": count_3m,
            "count_12m": count_12m,
            "count_24m": count_24m,
        }

    return {
        "exposure": "N/A",
        "descriptor": "No qualifying events in lookback windows",
        "count_3m": count_3m,
        "count_12m": count_12m,
        "count_24m": count_24m,
    }


def calculate_lga_likelihood(lga_key, end_period_label, data_df):
    """Calculate likelihood classification for LGA based on event frequency in lookback windows from end_period."""
    lga_df = data_df[data_df["lga_key"].eq(lga_key)].copy()
    
    if lga_df.empty:
        return "Very Unlikely"
    
    end_period = pd.Period(end_period_label, freq="M")
    end_ts = end_period.to_timestamp(how="end")
    
    # Check 3 months back
    start_3m = (end_period - 2).to_timestamp(how="start")
    if ((lga_df["event_date"] >= start_3m) & (lga_df["event_date"] <= end_ts)).any():
        return "Highly Likely"
    
    # Check 12 months back
    start_12m = (end_period - 11).to_timestamp(how="start")
    if ((lga_df["event_date"] >= start_12m) & (lga_df["event_date"] <= end_ts)).any():
        return "Likely"
    
    # Check 24 months back
    start_24m = (end_period - 23).to_timestamp(how="start")
    if ((lga_df["event_date"] >= start_24m) & (lga_df["event_date"] <= end_ts)).any():
        return "Unlikely"
    
    return "Very Unlikely"


def dataframe_records(df):
    if df.empty:
        return []
    return df.to_dict("records")

# Load and preprocess armed clash data
def load_armed_clash_data():
    events = pd.read_csv("acled_armed_clash.csv", low_memory=False)
    # Ensure date and coordinates are present
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
    events = events.dropna(subset=["event_date", "latitude", "longitude"]).copy()
    # Only keep Armed clash events
    armed = events[events["sub_event_type"] == "Armed clash"].copy()
    if armed.empty:
        raise ValueError("No rows found where sub_event_type == 'Armed clash'.")
    # Determine actor columns
    actor1_col = "actor1" if "actor1" in armed.columns else "actor_1" if "actor_1" in armed.columns else None
    actor2_col = "actor2" if "actor2" in armed.columns else "actor_2" if "actor_2" in armed.columns else None
    return armed, actor1_col, actor2_col

armed, ACTOR1_COL, ACTOR2_COL = load_armed_clash_data()


## armed, ACTOR1_COL, ACTOR2_COL = load_data()  # Removed: replaced by load_armed_clash_data()
LGA_GDF = load_lga_boundaries()
ARMED_JOINED = spatial_join_events(armed, LGA_GDF)
PROJECT_LOCATIONS = load_project_locations(LGA_GDF)

# Calculate impact scores
ARMED_JOINED["impact_score"] = ARMED_JOINED.apply(
    lambda row: calculate_impact_score(row.get("llm_target"), row.get("llm_impact_type")),
    axis=1
)

ARMED_JOINED["target_score"] = ARMED_JOINED["llm_target"].apply(
    lambda t: TARGET_RISK_SCORES.get(str(t).strip() if pd.notna(t) else "None of these", 0)
)
ARMED_JOINED["severity_score"] = ARMED_JOINED["llm_impact_type"].apply(
    lambda i: SEVERITY_SCORES.get(str(i).strip() if pd.notna(i) else "Unknown", 0)
)

PERIOD_LABELS = sorted(ARMED_JOINED["year_month"].unique().tolist())
LGA_BASE = LGA_GDF[["lga_key", "lganame", "statename", "geometry"]].copy()
LGA_GEOJSON = json.loads(LGA_BASE[["lga_key", "lganame", "statename", "geometry"]].to_json())
LGA_OPTIONS = [
    {"label": f"{row.lganame} ({row.statename})", "value": row.lga_key}
    for row in LGA_BASE[["lga_key", "lganame", "statename"]]
    .drop_duplicates()
    .sort_values(["lganame", "statename"])
    .itertuples(index=False)
]
VALID_LGA_VALUES = {option["value"] for option in LGA_OPTIONS}
DEFAULT_LGA = LGA_OPTIONS[0]["value"] if LGA_OPTIONS else None
LGA_KEYS = LGA_BASE["lga_key"].drop_duplicates().tolist()

# Precompute cumulative monthly counts for fast map updates.
MONTHLY_LGA_COUNTS = (
    ARMED_JOINED.groupby(["year_month", "lga_key"]).size().unstack(fill_value=0)
    .reindex(index=PERIOD_LABELS, fill_value=0)
    .reindex(columns=LGA_KEYS, fill_value=0)
)
MONTHLY_TOTAL_COUNTS = ARMED_JOINED.groupby("year_month").size().reindex(PERIOD_LABELS, fill_value=0)
CUM_LGA_COUNTS = MONTHLY_LGA_COUNTS.cumsum()
CUM_TOTAL_COUNTS = MONTHLY_TOTAL_COUNTS.cumsum()

# Precompute cumulative monthly impact scores for fast map updates.
MONTHLY_LGA_IMPACT_SCORES = (
    ARMED_JOINED.groupby(["year_month", "lga_key"])["impact_score"].sum().unstack(fill_value=0)
    .reindex(index=PERIOD_LABELS, fill_value=0)
    .reindex(columns=LGA_KEYS, fill_value=0)
)
MONTHLY_TOTAL_IMPACT_SCORES = ARMED_JOINED.groupby("year_month")["impact_score"].sum().reindex(PERIOD_LABELS, fill_value=0)
CUM_LGA_IMPACT_SCORES = MONTHLY_LGA_IMPACT_SCORES.cumsum()
CUM_TOTAL_IMPACT_SCORES = MONTHLY_TOTAL_IMPACT_SCORES.cumsum()

# Precompute cumulative monthly target scores for exposure overlay.
MONTHLY_LGA_TARGET_SCORES = (
    ARMED_JOINED.groupby(["year_month", "lga_key"])["target_score"].sum().unstack(fill_value=0)
    .reindex(index=PERIOD_LABELS, fill_value=0)
    .reindex(columns=LGA_KEYS, fill_value=0)
)
CUM_LGA_TARGET_SCORES = MONTHLY_LGA_TARGET_SCORES.cumsum()

# Precompute cumulative monthly severity scores for severity overlay.
MONTHLY_LGA_SEVERITY_SCORES = (
    ARMED_JOINED.groupby(["year_month", "lga_key"])["severity_score"].sum().unstack(fill_value=0)
    .reindex(index=PERIOD_LABELS, fill_value=0)
    .reindex(columns=LGA_KEYS, fill_value=0)
)
CUM_LGA_SEVERITY_SCORES = MONTHLY_LGA_SEVERITY_SCORES.cumsum()

# Precompute likelihood classifications for each period (for fast likelihood map updates).
LIKELIHOOD_BY_PERIOD = {}
for period_label in PERIOD_LABELS:
    LIKELIHOOD_BY_PERIOD[period_label] = {
        lga_key: calculate_lga_likelihood(lga_key, period_label, ARMED_JOINED)
        for lga_key in LGA_KEYS
    }


def build_slider_marks(period_labels, interval=6):
    marks = {}
    for index, label in enumerate(period_labels):
        if index == 0 or index == len(period_labels) - 1 or index % interval == 0:
            marks[index] = label
    return marks


SLIDER_MARKS = build_slider_marks(PERIOD_LABELS)


def filter_data(range_values):
    if not range_values or len(range_values) != 2:
        start_index, end_index = 0, len(PERIOD_LABELS) - 1
    else:
        # Dash slider values can arrive as numeric types; normalize safely for indexing.
        start_index = int(round(range_values[0]))
        end_index = int(round(range_values[1]))

    start_index = max(0, min(start_index, len(PERIOD_LABELS) - 1))
    end_index = max(0, min(end_index, len(PERIOD_LABELS) - 1))
    if end_index < start_index:
        start_index, end_index = end_index, start_index

    start_label = PERIOD_LABELS[start_index]
    end_label = PERIOD_LABELS[end_index]
    mask = (ARMED_JOINED["year_month"] >= start_label) & (ARMED_JOINED["year_month"] <= end_label)
    return ARMED_JOINED.loc[mask].copy(), start_label, end_label


def normalize_range_indices(range_values):
    if not range_values or len(range_values) != 2:
        start_index, end_index = 0, len(PERIOD_LABELS) - 1
    else:
        start_index = int(round(range_values[0]))
        end_index = int(round(range_values[1]))

    start_index = max(0, min(start_index, len(PERIOD_LABELS) - 1))
    end_index = max(0, min(end_index, len(PERIOD_LABELS) - 1))
    if end_index < start_index:
        start_index, end_index = end_index, start_index
    return start_index, end_index


def extract_lga_from_click(click_data):
    if not click_data or not click_data.get("points"):
        return None

    point = click_data["points"][0]
    clicked_lga = point.get("location")
    if clicked_lga in VALID_LGA_VALUES:
        return clicked_lga

    custom = point.get("customdata")
    if isinstance(custom, (list, tuple)):
        for value in custom:
            if isinstance(value, str) and value in VALID_LGA_VALUES:
                return value

    return None


def build_map_dataframe(filtered_df):
    counts = (
        filtered_df.groupby("lga_key").size().rename("events_in_range").reset_index()
        if not filtered_df.empty
        else pd.DataFrame(columns=["lga_key", "events_in_range"])
    )
    map_df = LGA_BASE.drop(columns=["geometry"]).merge(counts, on="lga_key", how="left")
    map_df["events_in_range"] = map_df["events_in_range"].fillna(0).astype(int)
    return map_df


def get_range_counts(start_index, end_index):
    if start_index == 0:
        lga_counts = CUM_LGA_COUNTS.iloc[end_index]
        total_events = int(CUM_TOTAL_COUNTS.iloc[end_index])
    else:
        lga_counts = CUM_LGA_COUNTS.iloc[end_index] - CUM_LGA_COUNTS.iloc[start_index - 1]
        total_events = int(CUM_TOTAL_COUNTS.iloc[end_index] - CUM_TOTAL_COUNTS.iloc[start_index - 1])
    return lga_counts, total_events


def get_range_impact_scores(start_index, end_index):
    if start_index == 0:
        lga_impact_scores = CUM_LGA_IMPACT_SCORES.iloc[end_index]
        total_impact = float(CUM_TOTAL_IMPACT_SCORES.iloc[end_index])
    else:
        lga_impact_scores = CUM_LGA_IMPACT_SCORES.iloc[end_index] - CUM_LGA_IMPACT_SCORES.iloc[start_index - 1]
        total_impact = float(CUM_TOTAL_IMPACT_SCORES.iloc[end_index] - CUM_TOTAL_IMPACT_SCORES.iloc[start_index - 1])
    return lga_impact_scores, total_impact


def build_map_dataframe_from_indices(start_index, end_index, metric="threat"):
    if metric == "impact":
        lga_values, _ = get_range_impact_scores(start_index, end_index)
        value_col_name = "total_impact_score"
    else:
        lga_values, _ = get_range_counts(start_index, end_index)
        value_col_name = "events_in_range"

    # Convert Series to DataFrame, handling the index carefully
    values_df = lga_values.to_frame(name=value_col_name).reset_index()
    values_df.columns = ["lga_key", value_col_name]

    map_df = LGA_BASE.drop(columns=["geometry"]).merge(values_df, on="lga_key", how="left")
    map_df[value_col_name] = map_df[value_col_name].fillna(0).astype(float)
    return map_df


def build_exposure_map_dataframe(start_index, end_index):
    """Sum of llm_target associated scores per LGA for the selected time range."""
    if start_index == 0:
        lga_target_scores = CUM_LGA_TARGET_SCORES.iloc[end_index]
    else:
        lga_target_scores = CUM_LGA_TARGET_SCORES.iloc[end_index] - CUM_LGA_TARGET_SCORES.iloc[start_index - 1]

    values_df = lga_target_scores.to_frame(name="exposure_score").reset_index()
    values_df.columns = ["lga_key", "exposure_score"]

    map_df = LGA_BASE.drop(columns=["geometry"]).merge(values_df, on="lga_key", how="left")
    map_df["exposure_score"] = map_df["exposure_score"].fillna(0).astype(float)
    return map_df


def build_severity_map_dataframe(start_index, end_index):
    """Sum of severity_score per LGA for the selected time range."""
    if start_index == 0:
        lga_severity_scores = CUM_LGA_SEVERITY_SCORES.iloc[end_index]
    else:
        lga_severity_scores = CUM_LGA_SEVERITY_SCORES.iloc[end_index] - CUM_LGA_SEVERITY_SCORES.iloc[start_index - 1]

    values_df = lga_severity_scores.to_frame(name="severity_score").reset_index()
    values_df.columns = ["lga_key", "severity_score"]

    map_df = LGA_BASE.drop(columns=["geometry"]).merge(values_df, on="lga_key", how="left")
    map_df["severity_score"] = map_df["severity_score"].fillna(0).astype(float)
    return map_df


def build_risk_map_dataframe(start_index, end_index):
    """Combine impact band and likelihood band via RISK_MATRIX to produce per-LGA risk level."""
    end_label = PERIOD_LABELS[end_index]

    # Impact scores for the range
    if start_index == 0:
        lga_impact = CUM_LGA_IMPACT_SCORES.iloc[end_index]
    else:
        lga_impact = CUM_LGA_IMPACT_SCORES.iloc[end_index] - CUM_LGA_IMPACT_SCORES.iloc[start_index - 1]

    impact_series = lga_impact.reindex(LGA_KEYS, fill_value=0)
    impact_bands = pd.cut(impact_series, bins=IMPACT_BINS, labels=[0, 1, 2, 3], right=False).astype(int)

    likelihood_labels = LIKELIHOOD_BY_PERIOD[end_label]
    likelihood_bands = pd.Series(
        {lga_key: LIKELIHOOD_BAND_MAP.get(likelihood_labels.get(lga_key, "Very Unlikely"), 0) for lga_key in LGA_KEYS},
        dtype=int,
    )

    risk_numeric = pd.Series(
        {lga_key: RISK_MATRIX[(likelihood_bands[lga_key], impact_bands[lga_key])] for lga_key in LGA_KEYS},
        dtype=int,
    )
    risk_labels_map = {0: "Low", 1: "Medium", 2: "High", 3: "Very High"}
    risk_label_series = risk_numeric.map(risk_labels_map)

    values_df = pd.DataFrame({
        "lga_key": LGA_KEYS,
        "risk_band": risk_numeric.values,
        "risk_label": risk_label_series.values,
    })
    map_df = LGA_BASE.drop(columns=["geometry"]).merge(values_df, on="lga_key", how="left")
    map_df["risk_band"] = map_df["risk_band"].fillna(0).astype(int)
    return map_df


def build_likelihood_map_dataframe(end_index):
    """Build map with likelihood classifications based on lookback windows from end_period."""
    end_label = PERIOD_LABELS[end_index]
    likelihood_scores = LIKELIHOOD_BY_PERIOD[end_label]
    
    # Map classification to numeric score for coloring
    likelihood_numeric_map = {
        "Very Unlikely": 0,
        "Unlikely": 1,
        "Likely": 2,
        "Highly Likely": 3,
    }
    
    values_df = pd.DataFrame([
        {
            "lga_key": lga_key,
            "likelihood": likelihood_scores.get(lga_key, "Very Unlikely"),
            "likelihood_numeric": likelihood_numeric_map.get(likelihood_scores.get(lga_key, "Very Unlikely"), 0),
        }
        for lga_key in LGA_KEYS
    ])

    map_df = LGA_BASE.drop(columns=["geometry"]).merge(values_df, on="lga_key", how="left")
    return map_df


@lru_cache(maxsize=128)
def build_base_map_figure(start_index, end_index, metric="threat"):
    range_color = None
    if metric == "impact":
        map_df = build_map_dataframe_from_indices(start_index, end_index, metric="impact")
        map_df["impact_band"] = pd.cut(
            map_df["total_impact_score"],
            bins=[-float("inf"), 25, 50, 100, float("inf")],
            labels=[0, 1, 2, 3],
            right=False,
        ).astype(int)
        color_col = "impact_band"
        title_text = "Impact"
        hover_data = {
            "statename": True,
            "total_impact_score": True,
            "impact_band": False,
            "lga_key": False,
        }
        color_scale = [
            (0.0, "#fce7f3"),
            (0.33, "#f9a8d4"),
            (0.66, "#ec4899"),
            (1.0, "#9d174d"),
        ]
        range_color = [0, 3]
        colorbar = {
            "title": title_text,
            "tickmode": "array",
            "tickvals": [0, 1, 2, 3],
            "ticktext": ["Insignificant (<25)", "Minor (25–49)", "Severe (50–99)", "Critical (100+)"],
        }
    elif metric == "exposure":
        map_df = build_exposure_map_dataframe(start_index, end_index)
        color_col = "exposure_score"
        title_text = "Exposure"
        hover_data = {
            "statename": True,
            color_col: True,
            "lga_key": False,
        }
        color_scale = [[0.0, "#ffffff"], [1.0, "#7c3aed"]]
        colorbar = {"title": title_text}
    elif metric == "severity":
        map_df = build_severity_map_dataframe(start_index, end_index)
        color_col = "severity_score"
        title_text = "Severity"
        hover_data = {
            "statename": True,
            color_col: True,
            "lga_key": False,
        }
        color_scale = [
            (0.0, "#dcfce7"),
            (0.5, "#16a34a"),
            (1.0, "#14532d"),
        ]
        colorbar = {"title": title_text}
    elif metric == "likelihood":
        map_df = build_likelihood_map_dataframe(end_index)
        color_col = "likelihood_numeric"
        title_text = "Likelihood"
        hover_data = {
            "statename": True,
            "likelihood": True,
            "likelihood_numeric": False,
            "lga_key": False,
        }
        color_scale = [
            (0.0, "#e0f2fe"),   # very light blue
            (0.33, "#7dd3fc"),   # light blue
            (0.66, "#38bdf8"),   # medium blue
            (1.0, "#0369a1"),    # darker blue
        ]
        range_color = [0, 3]
        colorbar = {
            "title": title_text,
            "tickmode": "array",
            "tickvals": [0, 1, 2, 3],
            "ticktext": ["Very Unlikely", "Unlikely", "Likely", "Highly Likely"],
        }
    elif metric == "risk":
        map_df = build_risk_map_dataframe(start_index, end_index)
        color_col = "risk_band"
        title_text = "Risk"
        hover_data = {
            "statename": True,
            "risk_label": True,
            "risk_band": False,
            "lga_key": False,
        }
        # Explicit color scale for risk bands: 0=Low (green), 1=Medium (yellow), 2=High (orange), 3=Very High (red)
        color_scale = [
            [0.0, "#00B050"],   # Low (green)
            [0.33, "#FFFF00"],  # Medium (yellow)
            [0.66, "#FF8C00"],  # High (orange)
            [1.0, "#FF0000"],   # Very High (red)
        ]
        range_color = [0, 3]
        colorbar = {
            "title": title_text,
            "tickmode": "array",
            "tickvals": [0, 1, 2, 3],
            "ticktext": ["Low", "Medium", "High", "Very High"],
        }
    else:  # threat
        map_df = build_map_dataframe_from_indices(start_index, end_index, metric="threat")
        color_col = "events_in_range"
        title_text = "Threat"
        hover_data = {
            "statename": True,
            color_col: True,
            "lga_key": False,
        }
        color_scale = "Reds"
        colorbar = {"title": title_text}

    figure = px.choropleth_map(
        map_df,
        geojson=LGA_GEOJSON,
        locations="lga_key",
        featureidkey="properties.lga_key",
        color=color_col,
        color_continuous_scale=color_scale,
        range_color=range_color,
        hover_name="lganame",
        hover_data=hover_data,
        custom_data=["lga_key", "statename", "lganame"],
        center=MAP_CENTER,
        zoom=5,
        map_style="carto-positron",
        opacity=0.7,
    )
    figure.update_traces(marker_line_width=0.5, marker_line_color="#222222")

    figure.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        clickmode="event",
        coloraxis_colorbar=colorbar,
        showlegend=True,
        legend={"orientation": "h", "y": 0.99, "x": 0.01, "bgcolor": "rgba(255,255,255,0.90)"},
    )

    if not PROJECT_LOCATIONS.empty:
        project_hover = PROJECT_LOCATIONS.copy()
        for col in ["PROJ_STAT_NAME", "PROJ_APPRVL_FY", "GEO_LOC_NME"]:
            if col not in project_hover.columns:
                project_hover[col] = ""
            project_hover[col] = project_hover[col].fillna("")
        project_hover["location_hover"] = project_hover["GEO_LOC_NME"].astype(str).str.strip().map(
            lambda value: f"<br>Location: {value}" if value else ""
        )
        project_custom = project_hover[["PROJ_ID", "project_title", "PROJ_STAT_NAME", "PROJ_APPRVL_FY", "location_hover", "lga_key"]].to_numpy()

        figure.add_trace(
            go.Scattermap(
                lat=project_hover["GEO_LATITUDE_NBR"],
                lon=project_hover["GEO_LONGITUDE_NBR"],
                mode="markers",
                marker={"size": 11, "opacity": 0.9, "color": "#2563eb", "symbol": "circle"},
                customdata=project_custom,
                name="Projects (blue dots)",
                showlegend=True,
                hovertemplate=(
                    "<b>Project: %{customdata[0]}</b><br>"
                    "%{customdata[1]}<br>"
                    "Status: %{customdata[2]}<br>"
                    "Approval FY: %{customdata[3]}<br>"
                    "%{customdata[4]}"
                    "<extra></extra>"
                ),
            )
        )

    return figure.to_dict()


def build_map_figure(range_values, selected_lga=None, metric="threat"):
    start_index, end_index = normalize_range_indices(range_values)
    figure = go.Figure(build_base_map_figure(start_index, end_index, metric=metric))

    if selected_lga and selected_lga in VALID_LGA_VALUES:
        selected_shape = LGA_BASE[LGA_BASE["lga_key"].eq(selected_lga)]
        if not selected_shape.empty:
            boundary = selected_shape.iloc[0].geometry.boundary

            def add_boundary_line(coords, color, width):
                lon_vals = [coord[0] for coord in coords]
                lat_vals = [coord[1] for coord in coords]
                figure.add_trace(
                    go.Scattermap(
                        lon=lon_vals,
                        lat=lat_vals,
                        mode="lines",
                        line={"color": color, "width": width},
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

            if boundary.geom_type == "LineString":
                coords_seq = list(boundary.coords)
                add_boundary_line(coords_seq, "rgba(34,197,94,0.35)", 12)
                add_boundary_line(coords_seq, "#111111", 3)
            else:
                for segment in boundary.geoms:
                    coords_seq = list(segment.coords)
                    add_boundary_line(coords_seq, "rgba(34,197,94,0.35)", 12)
                    add_boundary_line(coords_seq, "#111111", 3)

    return figure


def build_empty_events_map(message):
    figure = go.Figure()
    figure.update_layout(
        map_style="carto-positron",
        map_center=MAP_CENTER,
        map_zoom=5,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )
    figure.add_annotation(
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        text=message,
        showarrow=False,
        font={"size": 14, "color": "#475569"},
    )
    return figure


def build_lga_events_figure(filtered_df, selected_lga):
    if not selected_lga:
        return build_empty_events_map("Select an LGA to view event locations.")

    lga_df = filtered_df[filtered_df["lga_key"] == selected_lga].copy()
    if lga_df.empty:
        return build_empty_events_map("No events found in this LGA for the selected period.")

    lga_df["event_date_label"] = pd.to_datetime(lga_df["event_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    lga_df["civilian_fatalities"] = pd.to_numeric(lga_df["civilian_fatalities"], errors="coerce").fillna(0).astype(int)
    lga_df["fatalities"] = pd.to_numeric(lga_df["fatalities"], errors="coerce").fillna(0).astype(int)

    notes_col = "notes" if "notes" in lga_df.columns else "note" if "note" in lga_df.columns else None
    if notes_col:
        lga_df["event_note_short"] = (
            lga_df[notes_col]
            .fillna("")
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.slice(0, 180)
        )
    else:
        lga_df["event_note_short"] = ""

    # Fit map to selected LGA bounds with padding so full boundary is visible.
    selected_shape = LGA_BASE[LGA_BASE["lga_key"].eq(selected_lga)]
    if not selected_shape.empty:
        geom = selected_shape.iloc[0].geometry
        centroid = geom.centroid
        minx, miny, maxx, maxy = geom.bounds
        lon_span = max(float(maxx - minx), 1e-6)
        lat_span = max(float(maxy - miny), 1e-6)
        padded_span = max(lon_span, lat_span) * 1.5
        dynamic_zoom = float(max(5.3, min(9.5, 8.0 - math.log(padded_span, 2))))
        map_center = {"lat": float(centroid.y), "lon": float(centroid.x)}
    else:
        mean_lat = float(pd.to_numeric(lga_df["latitude"], errors="coerce").mean())
        mean_lon = float(pd.to_numeric(lga_df["longitude"], errors="coerce").mean())
        map_center = MAP_CENTER if pd.isna(mean_lat) or pd.isna(mean_lon) else {"lat": mean_lat, "lon": mean_lon}
        dynamic_zoom = 9.0

    # Apply small deterministic jitter so stacked events are individually clickable.
    lga_df = lga_df.reset_index(drop=True)
    lga_df["lat_plot"] = lga_df["latitude"]
    lga_df["lon_plot"] = lga_df["longitude"]
    jitter_radius = 0.003
    for (lat, lon), group in lga_df.groupby(["latitude", "longitude"]):
        if len(group) <= 1:
            continue
        n = len(group)
        for rank, idx in enumerate(group.index):
            angle = 2 * math.pi * rank / n
            lga_df.at[idx, "lat_plot"] = lat + jitter_radius * math.sin(angle)
            lga_df.at[idx, "lon_plot"] = lon + jitter_radius * math.cos(angle)

    figure = px.scatter_map(
        lga_df,
        lat="lat_plot",
        lon="lon_plot",
        center=map_center,
        zoom=dynamic_zoom,
        map_style="open-street-map",
        hover_data={"lat_plot": False, "lon_plot": False},
    )

    # Draw selected LGA boundary as explicit map lines for reliable visibility.
    if not selected_shape.empty:
        boundary = selected_shape.iloc[0].geometry.boundary

        def add_boundary_line(coords):
            lon_vals = [coord[0] for coord in coords]
            lat_vals = [coord[1] for coord in coords]
            figure.add_trace(
                go.Scattermap(
                    lon=lon_vals,
                    lat=lat_vals,
                    mode="lines",
                    line={"color": "#111111", "width": 4},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        if boundary.geom_type == "LineString":
            add_boundary_line(list(boundary.coords))
        else:
            for segment in boundary.geoms:
                add_boundary_line(list(segment.coords))

    custom_data = lga_df[["sub_event_type", "event_date_label", "civilian_fatalities", "fatalities", "event_note_short"]].to_numpy()
    figure.update_traces(name="Events", selector={"type": "scattermap", "mode": "markers"})
    figure.update_traces(
        marker={
            "size": 16,
            "opacity": 1.0,
            "color": "#e11d48",
            "symbol": "circle",
        },
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Date: %{customdata[1]}<br>"
            "Deaths: %{customdata[3]} (%{customdata[2]} civilian)"
            "<extra></extra>"
        ),
        hoverlabel={
            "bgcolor": "rgba(15,23,42,0.90)",
            "font": {"size": 12, "color": "#f8fafc"},
            "align": "left",
            "namelength": 0,
        },
        showlegend=True,
        customdata=custom_data,
        selector={"type": "scattermap", "name": "Events", "mode": "markers"},
    )

    if not PROJECT_LOCATIONS.empty:
        project_lga = PROJECT_LOCATIONS[PROJECT_LOCATIONS["lga_key"].eq(selected_lga)].copy()
        if not project_lga.empty:
            for col in ["PROJ_STAT_NAME", "PROJ_APPRVL_FY", "GEO_LOC_NME"]:
                if col not in project_lga.columns:
                    project_lga[col] = ""
                project_lga[col] = project_lga[col].fillna("")
            project_lga["location_hover"] = project_lga["GEO_LOC_NME"].astype(str).str.strip().map(
                lambda value: f"<br>Location: {value}" if value else ""
            )
            project_custom = project_lga[["PROJ_ID", "project_title", "PROJ_STAT_NAME", "PROJ_APPRVL_FY", "location_hover"]].to_numpy()

            figure.add_trace(
                go.Scattermap(
                    lat=project_lga["GEO_LATITUDE_NBR"],
                    lon=project_lga["GEO_LONGITUDE_NBR"],
                    mode="markers",
                    marker={"size": 16, "opacity": 0.95, "color": "#2563eb", "symbol": "circle"},
                    customdata=project_custom,
                    name="Projects",
                    showlegend=True,
                    hovertemplate=(
                        "<b>Project: %{customdata[0]}</b><br>"
                        "%{customdata[1]}<br>"
                        "Status: %{customdata[2]}<br>"
                        "Approval FY: %{customdata[3]}<br>"
                        "%{customdata[4]}"
                        "<extra></extra>"
                    ),
                )
            )

    figure.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        showlegend=True,
        legend={"orientation": "h", "y": 0.99, "x": 0.01, "bgcolor": "rgba(255,255,255,0.90)"},
        hoverdistance=20,
    )
    return figure


INITIAL_RANGE = [0, len(PERIOD_LABELS) - 1]
INITIAL_FILTERED_DF, INITIAL_START_LABEL, INITIAL_END_LABEL = filter_data(INITIAL_RANGE)
INITIAL_STATUS_TEXT = (
    f"Month-Year range: {INITIAL_START_LABEL} to {INITIAL_END_LABEL} | "
    f"Total Armed clash events: {len(INITIAL_FILTERED_DF)}"
)
INITIAL_MAP_FIGURE = build_map_figure(INITIAL_RANGE, DEFAULT_LGA, metric="risk")
INITIAL_EVENTS_MAP_FIGURE = build_lga_events_figure(INITIAL_FILTERED_DF, DEFAULT_LGA)
INITIAL_EVENTS_MAP_TITLE = "Events in selected LGA"
INITIAL_EVENT_CLICK_TEXT = "Click an event dot on the map to view details, or select a row in the table below."


def metric_card(title, value, description=None):
    children = [
        html.Div(title, className="metric-label"),
        html.Div(str(value), className="metric-value"),
    ]
    if description:
        children.append(html.Div(description, className="metric-description"))
    return html.Div(children, className="metric-card")


def build_lga_event_records(filtered_df, selected_lga):
    """Return all events in the selected LGA as a list of dicts for the events table."""
    if not selected_lga:
        return []

    lga_df = filtered_df[filtered_df["lga_key"].eq(selected_lga)].copy()
    if lga_df.empty:
        return []

    lga_df["event_date"] = pd.to_datetime(lga_df["event_date"], errors="coerce")
    lga_df["event_date_label"] = lga_df["event_date"].dt.strftime("%Y-%m-%d")
    lga_df["civilian_fatalities"] = pd.to_numeric(lga_df["civilian_fatalities"], errors="coerce").fillna(0).astype(int)
    lga_df["fatalities"] = pd.to_numeric(lga_df["fatalities"], errors="coerce").fillna(0).astype(int)
    lga_df["impact_score"] = pd.to_numeric(lga_df.get("impact_score"), errors="coerce").fillna(0).round(1)

    notes_col = "notes" if "notes" in lga_df.columns else "note" if "note" in lga_df.columns else None
    lga_df["event_note"] = (
        lga_df[notes_col].fillna("").astype(str).str.replace(r"\s+", " ", regex=True)
        if notes_col else ""
    )

    actor1_col = "actor_1" if "actor_1" in lga_df.columns else "actor1" if "actor1" in lga_df.columns else None
    lga_df["actor"] = lga_df[actor1_col].fillna("").astype(str).str.strip() if actor1_col else ""

    lga_df = lga_df.sort_values("event_date", ascending=False)
    return [
        {
            "event_date": row["event_date_label"],
            "sub_event_type": row.get("sub_event_type", ""),
            "actor": row["actor"],
            "civilian_fatalities": int(row["civilian_fatalities"]),
            "fatalities": int(row["fatalities"]),
            "impact_score": float(row["impact_score"]),
            "note": row["event_note"],
        }
        for _, row in lga_df.iterrows()
    ]


def table_block(title, records, empty_message):
    table = (
        dash_table.DataTable(
            data=records,
            columns=[
                {"name": "Indicator", "id": "indicator"},
                {"name": "Occurrences", "id": "occurrences"},
            ],
            style_as_list_view=True,
            style_cell={
                "padding": "10px 12px",
                "fontFamily": "Segoe UI, sans-serif",
                "fontSize": "14px",
                "textAlign": "left",
                "border": "none",
            },
            style_header={
                "backgroundColor": "#f8fafc",
                "fontWeight": "600",
                "borderBottom": "1px solid #e2e8f0",
            },
            style_data={
                "backgroundColor": "white",
                "borderBottom": "1px solid #eef2f7",
            },
            page_action="none",
            fill_width=True,
        )
        if records
        else html.Div(empty_message, className="empty-note")
    )
    return html.Div([html.H4(title, className="section-title"), table], className="table-card")


app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(
    [
        html.Div(
            [
                html.H1("Nigeria Armed Clash Explorer", className="page-title"),
                html.P(
                    "Click an LGA on the map to see detailed information about armed clashes in that area, including involved actors and key indicators.",
                    className="page-subtitle",
                ),
                html.P(
                    "Risk* = Impact × Likelihood. Impact* = Exposure × Severity.",
                    className="risk-definition-note",
                ),
            ],
            className="hero",
        ),
        # Controls and main map at the top
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Month-Year range", className="control-label"),
                        dcc.RangeSlider(
                            id="month-range",
                            min=0,
                            max=len(PERIOD_LABELS) - 1,
                            value=INITIAL_RANGE,
                            marks=SLIDER_MARKS,
                            allowCross=False,
                            updatemode="mouseup",
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ],
                    className="control-card control-card-wide",
                ),
                html.Div(
                    [
                        html.Label("Selected LGA", className="control-label"),
                        dcc.Dropdown(
                            id="lga-dropdown",
                            options=LGA_OPTIONS,
                            value=DEFAULT_LGA,
                            clearable=False,
                        ),
                    ],
                    className="control-card",
                ),
            ],
            className="controls-grid",
            style={"display": "flex", "flexDirection": "column", "gap": "16px", "width": "100%"},
        ),
        html.Div(INITIAL_STATUS_TEXT, id="status-bar", className="status-bar"),
        html.Div([
            dcc.Loading(
                id="lga-map-loading",
                type="circle",
                color="#334155",
                delay_show=250,
                overlay_style={"visibility": "visible", "filter": "none", "backgroundColor": "rgba(255,255,255,0.20)"},
                children=dcc.Graph(
                    id="lga-map",
                    figure=INITIAL_MAP_FIGURE,
                    className="map-panel",
                    config={"displayModeBar": False},
                ),
            ),
            html.Div(
                [
                    html.Label("Map Overlay", className="control-label"),
                    dcc.RadioItems(
                        id="metric-selector",
                        options=[
                            {
                                "label": html.Span([
                                    "Threat ",
                                    html.Span("i", className="info-icon", **{"data-tooltip": "Count of armed clash events in the selected period"}),
                                ], className="overlay-option-label"),
                                "value": "threat",
                            },
                            {
                                "label": html.Span([
                                    "Exposure ",
                                    html.Span("i", className="info-icon", **{"data-tooltip": "Sum of target scores of all events for given time period"}),
                                ], className="overlay-option-label"),
                                "value": "exposure",
                            },
                            {
                                "label": html.Span([
                                    "Severity ",
                                    html.Span("i", className="info-icon", **{"data-tooltip": "Severity of event"}),
                                ], className="overlay-option-label"),
                                "value": "severity",
                            },
                            {
                                "label": html.Span([
                                    "Likelihood ",
                                    html.Span("i", className="info-icon", **{"data-tooltip": "Probability of event in next week, month, year, or 2 years based on historical frequency"}),
                                ], className="overlay-option-label"),
                                "value": "likelihood",
                            },
                            {
                                "label": html.Span([
                                    "Impact* ",
                                    html.Span("i", className="info-icon", **{"data-tooltip": "Exposure × Severity. The total score is mapped into 4 buckets: Insignificant (<25), Minor (25–49), Severe (50–99), Critical (100+)"}),
                                ], className="overlay-option-label"),
                                "value": "impact",
                            },
                            {
                                "label": html.Span([
                                    "Risk ",
                                    html.Span("i", className="info-icon", **{"data-tooltip": "Combined Impact × Likelihood risk rating"}),
                                ], className="overlay-option-label"),
                                "value": "risk",
                            },
                        ],
                        value="risk",
                        inline=False,
                        labelStyle={"display": "block", "marginBottom": "8px", "cursor": "pointer", "width": "100%"},
                        inputStyle={"marginRight": "8px", "cursor": "pointer"},
                    ),
                ],
                className="control-card",
                style={"minWidth": "260px", "maxWidth": "340px", "marginLeft": "24px"},
            ),
        ], style={"display": "flex", "alignItems": "flex-start", "gap": "0", "width": "100%"}),
        # Details section for selected LGA (with risk score)
        html.Div([
            html.Div(id="details-section"),
        ], className="details-panel"),
        # LGA events map and indicators below details
        html.Div(INITIAL_EVENTS_MAP_TITLE, id="events-map-title", className="status-bar"),
        dcc.Loading(
            id="events-map-loading",
            type="circle",
            color="#334155",
            delay_show=250,
            overlay_style={"visibility": "visible", "filter": "none", "backgroundColor": "rgba(255,255,255,0.20)"},
            children=dcc.Graph(
                id="events-map",
                figure=INITIAL_EVENTS_MAP_FIGURE,
                className="events-map-panel",
                config={"displayModeBar": False},
            ),
        ),
        html.Div(id="actors-section"),
        html.Div(
            [
                html.Span("Bottom map legend:", style={"fontWeight": "700", "marginRight": "12px"}),
                html.Span("\u25cf", style={"color": "#2563eb", "fontSize": "16px", "marginRight": "6px"}),
                html.Span("Projects", style={"marginRight": "16px"}),
                html.Span("\u25cf", style={"color": "#e11d48", "fontSize": "16px", "marginRight": "6px"}),
                html.Span("Events (jittered if overlapping — see table for full notes)"),
            ],
            className="status-bar",
            style={"marginTop": "-6px", "marginBottom": "10px"},
        ),
        html.Div(INITIAL_EVENT_CLICK_TEXT, id="event-click-card", className="status-bar"),
        html.Div(
            [
                html.H4("All events in selected LGA", id="lga-events-title", className="section-title"),
                dash_table.DataTable(
                    id="lga-events-table",
                    data=[],
                    columns=[
                        {"name": "Date", "id": "event_date"},
                        {"name": "Type", "id":  "sub_event_type"},
                        {"name": "Actor", "id": "actor"},
                        {"name": "Civ. Deaths", "id": "civilian_fatalities"},
                        {"name": "Total Deaths", "id": "fatalities"},
                        {"name": "Impact Score", "id": "impact_score"},
                        {"name": "Note", "id": "note"},
                    ],
                    sort_action="native",
                    style_as_list_view=True,
                    style_cell={
                        "padding": "10px 12px",
                        "fontFamily": "Segoe UI, sans-serif",
                        "fontSize": "13px",
                        "textAlign": "left",
                        "border": "none",
                        "whiteSpace": "normal",
                        "height": "auto",
                    },
                    style_cell_conditional=[
                        {"if": {"column_id": ["event_date", "sub_event_type", "civilian_fatalities", "fatalities", "impact_score"]}, "whiteSpace": "nowrap", "width": "auto"},
                        {"if": {"column_id": "note"}, "minWidth": "240px", "maxWidth": "520px", "whiteSpace": "normal"},
                        {"if": {"column_id": "actor"}, "minWidth": "140px", "maxWidth": "260px", "whiteSpace": "normal"},
                    ],
                    style_header={
                        "backgroundColor": "#f8fafc",
                        "fontWeight": "600",
                        "borderBottom": "1px solid #e2e8f0",
                    },
                    style_data={
                        "backgroundColor": "white",
                        "borderBottom": "1px solid #eef2f7",
                    },
                    style_data_conditional=[
                        {
                            "if": {"state": "selected"},
                            "backgroundColor": "#e2e8f0",
                            "border": "1px solid #94a3b8",
                        }
                    ],
                    page_size=10,
                ),
            ],
            className="details-card",
        ),
        html.Div([
            html.Div(id="knowledge-table"),
            html.Div(id="resources-table"),
            html.Div(id="expectation-table"),
        ], className="detail-grid"),
        html.Div(id="details-bottom-panel", className="details-panel"),
    ],
    className="page-shell",
)


app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                margin: 0;
                background: linear-gradient(180deg, #f5f7fb 0%, #edf2f7 100%);
                color: #0f172a;
                font-family: Segoe UI, sans-serif;
            }
            .page-shell {
                max-width: 1440px;
                margin: 0 auto;
                padding: 24px;
            }
            .hero {
                padding: 24px 28px;
                border-radius: 18px;
                background: linear-gradient(135deg, #ffffff 0%, #e7eef7 100%);
                box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
                margin-bottom: 20px;
            }
            .page-title {
                margin: 0;
                font-size: 34px;
                line-height: 1.1;
            }
            .page-subtitle {
                margin: 10px 0 0;
                font-size: 16px;
                color: #334155;
            }
            .risk-definition-note {
                /* ...existing styles... */
            }
            .impact-bucket-card {
                display: flex;
                flex-direction: column;
                align-items: flex-start;
                background: #ede9fe;
                border-radius: 14px;
                padding: 18px 22px 14px 22px;
                margin-bottom: 18px;
                box-shadow: 0 4px 16px rgba(124, 58, 237, 0.07);
                min-width: 180px;
                max-width: 320px;
            }
            .impact-bucket-value {
                font-size: 2.1rem;
                font-weight: 700;
                color: #5b21b6;
                margin-bottom: 2px;
            }
            .impact-bucket-desc {
                font-size: 1.05rem;
                color: #3730a3;
                font-weight: 500;
                margin-bottom: 0;
            }
                margin: 8px 0 0;
                font-size: 13px;
                font-weight: 700;
                color: #7c3aed;
                letter-spacing: 0.02em;
            }
            .controls-grid {
                display: grid;
                grid-template-columns: 2fr 1fr 0.8fr;
                gap: 16px;
                margin-bottom: 16px;
            }
            .control-card {
                background: #ffffff;
                border-radius: 16px;
                padding: 18px 18px 22px;
                box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
            }
            .control-card-wide {
                overflow-x: auto;
            }
            .control-label {
                display: block;
                font-size: 13px;
                font-weight: 700;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                color: #475569;
                margin-bottom: 12px;
            }
            .overlay-option-label {
                display: inline-flex;
                align-items: center;
                gap: 6px;
            }
            .info-icon {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                position: relative;
                width: 16px;
                height: 16px;
                border-radius: 999px;
                border: 1px solid #64748b;
                color: #334155;
                font-size: 10px;
                font-weight: 700;
                line-height: 1;
                background: #ffffff;
                cursor: help;
                user-select: none;
            }
            .info-icon::after {
                content: attr(data-tooltip);
                position: absolute;
                left: 50%;
                top: calc(100% + 10px);
                transform: translateX(-50%);
                background: #0f172a;
                color: #f8fafc;
                font-size: 12px;
                font-weight: 500;
                line-height: 1.3;
                padding: 8px 10px;
                border-radius: 8px;
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.25);
                width: 220px;
                max-width: 220px;
                min-width: 220px;
                white-space: normal;
                word-break: normal;
                overflow-wrap: normal;
                text-align: left;
                opacity: 0;
                visibility: hidden;
                pointer-events: none;
                z-index: 20;
            }
            .info-icon::before {
                content: "";
                position: absolute;
                left: 50%;
                top: calc(100% + 4px);
                transform: translateX(-50%);
                border-left: 6px solid transparent;
                border-right: 6px solid transparent;
                border-bottom: 6px solid #0f172a;
                opacity: 0;
                visibility: hidden;
                pointer-events: none;
                z-index: 20;
            }
            .info-icon:hover::after,
            .info-icon:hover::before {
                opacity: 1;
                visibility: visible;
            }
            .status-bar {
                margin: 10px 0 16px;
                padding: 12px 16px;
                border-radius: 14px;
                background: rgba(255, 255, 255, 0.85);
                box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
                font-size: 14px;
                color: #334155;
            }
            .map-panel {
                background: white;
                border-radius: 18px;
                overflow: hidden;
                box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
                margin-bottom: 18px;
                height: 68vh;
            }
            .events-map-panel {
                background: white;
                border-radius: 18px;
                overflow: hidden;
                box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
                margin-bottom: 18px;
                height: 45vh;
            }
            .details-panel {
                gap: 16px;
                margin-bottom: 20px;
            }
            .details-panel-inline {
                display: block;
                min-width: 220px;
                max-width: 340px;
                flex: 0 0 300px;
            }
            .details-card {
                background: #ffffff;
                border-radius: 18px;
                padding: 20px;
                box-shadow: 0 16px 36px rgba(15, 23, 42, 0.07);
            }
            .details-title {
                margin: 0 0 16px;
                font-size: 24px;
            }
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 14px;
                margin-bottom: 18px;
            }
            .metric-card {
                padding: 16px;
                border-radius: 16px;
                background: linear-gradient(180deg, #fff7ed 0%, #ffffff 100%);
                border: 1px solid #fed7aa;
            }
            .metric-label {
                font-size: 12px;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                color: #9a3412;
                margin-bottom: 8px;
            }
            .metric-value {
                font-size: 30px;
                font-weight: 700;
                color: #7c2d12;
            }
            .metric-description {
                font-size: 12px;
                color: #64748b;
                margin-top: 6px;
                font-weight: 500;
            }
            .exposure-card {
                border-radius: 14px;
                border: 1px solid #cbd5e1;
                background: #f8fafc;
                padding: 14px 16px;
                margin-bottom: 16px;
            }
            .exposure-title {
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0.05em;
                text-transform: uppercase;
                color: #475569;
                margin-bottom: 6px;
            }
            .exposure-value {
                font-size: 24px;
                font-weight: 800;
                color: #0f172a;
                line-height: 1.1;
            }
            .exposure-desc {
                margin-top: 4px;
                font-size: 14px;
                color: #334155;
            }
            .actors-card, .table-card {
                background: #ffffff;
                border-radius: 16px;
                padding: 18px;
                border: 1px solid #e2e8f0;
            }
            .actors-list {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 10px;
                padding: 0;
                margin: 0;
                list-style: none;
            }
            .actor-pill {
                padding: 10px 12px;
                border-radius: 999px;
                background: #f8fafc;
                border: 1px solid #cbd5e1;
            }
            .detail-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 16px;
            }
            .section-title {
                margin: 0 0 12px;
                font-size: 18px;
            }
            .empty-note {
                color: #64748b;
                font-size: 14px;
            }
            @media (max-width: 900px) {
                .controls-grid,
                .detail-grid,
                .metrics-grid {
                    grid-template-columns: 1fr;
                }
                .page-shell {
                    padding: 14px;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


@app.callback(
    Output("status-bar", "children"),
    Output("lga-map", "figure"),
    Input("month-range", "value"),
    Input("lga-dropdown", "value"),
    Input("metric-selector", "value"),
    Input("lga-map", "clickData"),
)
def update_map(range_values, selected_lga, metric, click_data):
    start_index, end_index = normalize_range_indices(range_values)
    start_label = PERIOD_LABELS[start_index]
    end_label = PERIOD_LABELS[end_index]
    _, total_events = get_range_counts(start_index, end_index)

    active_lga = selected_lga
    clicked_lga = extract_lga_from_click(click_data)
    if clicked_lga:
        active_lga = clicked_lga

    status_text = (
        f"Month-Year range: {start_label} to {end_label} | "
        f"Total Armed clash events: {total_events}"
    )
    return status_text, build_map_figure(range_values, active_lga, metric=metric)


@app.callback(
    Output("lga-dropdown", "value"),
    Input("lga-map", "clickData"),
    State("lga-dropdown", "value"),
    prevent_initial_call=True,
)
def sync_dropdown_from_map(click_data, current_value):
    clicked_lga = extract_lga_from_click(click_data)
    if clicked_lga:
        return clicked_lga
    return current_value


@app.callback(
    Output("events-map-title", "children"),
    Output("events-map", "figure"),
    Input("month-range", "value"),
    Input("lga-dropdown", "value"),
    Input("lga-map", "clickData"),
)
def update_events_map(range_values, selected_lga, click_data):
    filtered_df, start_label, end_label = filter_data(range_values)
    active_lga = selected_lga
    clicked_lga = extract_lga_from_click(click_data)
    if clicked_lga:
        active_lga = clicked_lga

    if not active_lga:
        title_text = f"Events in selected LGA | Range: {start_label} to {end_label}"
        return title_text, build_empty_events_map("Select an LGA to view event locations.")

    state_name, lga_name = active_lga.split(" | ", 1)
    title_text = f"Event locations in {lga_name} ({state_name}) | Range: {start_label} to {end_label}"
    return title_text, build_lga_events_figure(filtered_df, active_lga)


@app.callback(
    Output("lga-events-title", "children"),
    Output("lga-events-table", "data"),
    Input("month-range", "value"),
    Input("lga-dropdown", "value"),
    Input("lga-map", "clickData"),
)
def update_lga_events_table(range_values, selected_lga, lga_click_data):
    filtered_df, _, _ = filter_data(range_values)
    active_lga = selected_lga
    clicked_lga = extract_lga_from_click(lga_click_data)
    if clicked_lga:
        active_lga = clicked_lga

    if not active_lga:
        return "All events in selected LGA", []

    records = build_lga_event_records(filtered_df, active_lga)
    state_name, lga_name = active_lga.split(" | ", 1)
    title = f"{lga_name} ({state_name}) — {len(records)} event(s) in selected range"
    return title, records


@app.callback(
    Output("event-click-card", "children"),
    Input("events-map", "clickData"),
)
def update_event_click_card(click_data):
    if click_data and click_data.get("points"):
        point = click_data["points"][0]
        custom = point.get("customdata")
        if custom and len(custom) >= 5:
            return (
                f"Event: {custom[0]} | Date: {custom[1]} | Civilian deaths: {custom[2]} | "
                f"Total fatalities: {custom[3]} | Note: {custom[4]}"
            )

    return "Click an event dot on the map to view details, or sort/view the table below."





# Callback to populate details and indicator tables for selected LGA
@app.callback(
    Output("details-section", "children"),
    Output("knowledge-table", "children"),
    Output("resources-table", "children"),
    Output("expectation-table", "children"),
    Input("month-range", "value"),
    Input("lga-dropdown", "value"),
    Input("lga-map", "clickData"),
)
def update_details_section(range_values, selected_lga, click_data):
    filtered_df, _, _ = filter_data(range_values)
    active_lga = selected_lga
    clicked_lga = extract_lga_from_click(click_data)
    if clicked_lga:
        active_lga = clicked_lga
    if not active_lga:
        return "", "", "", ""
    summary = summarize_lga(active_lga, filtered_df, ACTOR1_COL, ACTOR2_COL)
    # Calculate risk score for the selected LGA and period
    # Risk = Impact bucket × Likelihood bucket (from LIKELIHOOD_BY_PERIOD)
    # We'll use the current period (end of selected range)
    _, _, end_label = filter_data(range_values)
    impact_bucket = summary['impact_bucket']
    # Get likelihood label
    end_period = end_label
    likelihood_label = LIKELIHOOD_BY_PERIOD.get(end_period, {}).get(active_lga, "Very Unlikely")
    # Map impact and likelihood to numeric for risk matrix
    impact_band_map = {"Insignificant": 0, "Minor": 1, "Severe": 2, "Critical": 3}
    likelihood_band_map = {"Very Unlikely": 0, "Unlikely": 1, "Likely": 2, "Highly Likely": 3}
    impact_band = impact_band_map.get(impact_bucket, 0)
    likelihood_band = likelihood_band_map.get(likelihood_label, 0)
    risk_numeric = RISK_MATRIX.get((likelihood_band, impact_band), 0)
    risk_label_map = {0: "Low", 1: "Medium", 2: "High", 3: "Very High"}
    risk_label = risk_label_map.get(risk_numeric, "Low")
    # Metrics - new order and style
    metrics = html.Div([
        html.Div([
            metric_card("Risk Score", risk_label, description=f"Impact: {impact_bucket}, Likelihood: {likelihood_label}"),
        ], style={"marginBottom": "18px"}),
        html.Div([
            metric_card("Total Deaths", summary["total_deaths"]),
            metric_card("Civilian Deaths", summary["civilian_deaths"]),
        ], style={"display": "flex", "gap": "18px", "marginBottom": "18px"}),
        html.Div([
            metric_card("Threat (Events)", summary["event_count"]),
            metric_card("Exposure Score", summary["total_exposure_score"]),
            metric_card("Severity Score", summary["severity_score"]),
            metric_card("Impact Score", summary["total_impact_score"]),
            metric_card("Likelihood", likelihood_label),
        ], className="metrics-grid"),
    ], className="details-card")
    # Indicator tables
    knowledge_table = table_block("Knowledge Indicators", dataframe_records(summary["knowledge_df"]), "No knowledge indicators.")
    resources_table = table_block("Resource Indicators", dataframe_records(summary["resources_df"]), "No resource indicators.")
    expectation_table = table_block("Expectation Indicators", dataframe_records(summary["expectation_df"]), "No expectation indicators.")
    # Prominent actors section (below LGA events map, not in details)
    actors_section = html.Div([
        html.H4("Active Actors in this LGA", className="section-title"),
        html.Ul([
            html.Li(actor, className="actor-pill") for actor in summary["actors"]
        ], className="actors-list") if summary["actors"] else html.Div("No active actors found.", className="empty-note")
    ], className="actors-card")
    return metrics, knowledge_table, resources_table, expectation_table, actors_section


if __name__ == "__main__":
    app.run(debug=True)
