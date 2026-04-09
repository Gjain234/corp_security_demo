import json
import math
from collections import Counter
from functools import lru_cache

import dash
import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, dash_table, dcc, html


DATA_FILE = "acled_kri_enriched_llm_targets.xlsx"
BOUNDARY_FILE = "NGA_LGA_Boundaries_2_7839795478074887427.geojson"
PROJECT_DATA_FILE = "nigeria_project_data.csv"
PROJECT_LOCATION_FILE = "PROJECT_GEOGRAPHIC_LOCATION_V2.csv"
MAP_CENTER = {"lat": 9.082, "lon": 8.6753}
LIKELIHOOD_RISK_THRESHOLD = 4

# Risk Score Mappings
TARGET_RISK_SCORES = {
    "Military (Troops and infrastructure)": 2,
    "Police (Troops and infrastructure)": 2,
    "Government (Employees and offices)": 3,
    "Local leaders/Administrators": 3,
    "Foreign Expatriates": 5,
    "Students": 3,
    "Local NGOs": 4,
    "Political Candidates": 3,
    "UN AFPs (Not including DPKO)": 5,
    "State Development Agencies": 5,
    "Diplomatic Missions": 4,
    "INGO (Staff and Offices)": 5,
    "International Financial Institution (e.g. WBG, IMF, EBRD, ADB, IADB)": 5,
    "UN DPKO Mission": 3,
    "Militant Leaders": 1,
    "Militant Troops/Infrastructure": 1,
    "Critical Civilian (Energy, water, road, train, airport, hospital)": 4,
    "Information (Radio/TV/Broadcasters)": 2,
    "Private Business or Residence": 3,
    "Vehicle/Transport": 2,
    "None of these": 0,
}

IMPACT_RISK_SCORES = {
    "Low/Moderate": 2,
    "Substantial/High": 4,
    "Unknown": 0,
}


def load_data():
    events = pd.read_excel(DATA_FILE)

    actor1_col = "actor_1" if "actor_1" in events.columns else "actor1"
    actor2_col = "actor_2" if "actor_2" in events.columns else "actor2" if "actor2" in events.columns else None

    required_cols = [
        "sub_event_type",
        "event_date",
        "latitude",
        "longitude",
        "fatalities",
        "civilian_fatalities",
        "knowledge",
        "resources",
        "expectation",
        actor1_col,
    ]
    missing_cols = [column for column in required_cols if column not in events.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
    events = events.dropna(subset=["event_date", "latitude", "longitude"]).copy()

    armed = events[events["sub_event_type"].eq("Armed clash")].copy()
    if armed.empty:
        raise ValueError("No rows found where sub_event_type == 'Armed clash'.")

    return armed, actor1_col, actor2_col


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

    geo = read_csv_with_fallback(PROJECT_LOCATION_FILE, skiprows=4, low_memory=False)
    geo.columns = [str(col).strip().strip('"') for col in geo.columns]

    required_geo_cols = ["PROJ_ID", "ISO_CNTRY_CODE", "GEO_LOC_NME", "GEO_LATITUDE_NBR", "GEO_LONGITUDE_NBR"]
    missing_geo_cols = [col for col in required_geo_cols if col not in geo.columns]
    if missing_geo_cols:
        raise ValueError(f"Missing required columns in PROJECT_GEOGRAPHIC_LOCATION_V2.csv: {missing_geo_cols}")

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


def calculate_risk_score(target, impact_type):
    """Calculate risk score as target_score × impact_score."""
    target_score = TARGET_RISK_SCORES.get(str(target).strip() if pd.notna(target) else "None of these", 0)
    impact_score = IMPACT_RISK_SCORES.get(str(impact_type).strip() if pd.notna(impact_type) else "Unknown", 0)
    return target_score * impact_score


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
    
    # Calculate risk scores
    total_risk_score = lga_df["risk_score"].sum() if "risk_score" in lga_df.columns else 0
    event_count = len(lga_df)
    avg_risk_score = total_risk_score / event_count if event_count > 0 else 0

    return {
        "event_count": event_count,
        "total_deaths": int(total_deaths),
        "civilian_deaths": int(civilian_deaths),
        "total_risk_score": round(total_risk_score, 2),
        "avg_risk_score": round(avg_risk_score, 2),
        "high_risk_event_count": int((pd.to_numeric(lga_df["risk_score"], errors="coerce").fillna(0) >= LIKELIHOOD_RISK_THRESHOLD).sum()) if "risk_score" in lga_df.columns else 0,
        "actors": build_actor_list(lga_df, actor1_col, actor2_col),
        "knowledge_df": indicator_counts(lga_df, "knowledge"),
        "resources_df": indicator_counts(lga_df, "resources"),
        "expectation_df": indicator_counts(lga_df, "expectation"),
    }


def classify_lga_likelihood(lga_key, filtered_df, end_label):
    lga_df = filtered_df[filtered_df["lga_key"].eq(lga_key)].copy()
    if lga_df.empty:
        return {
            "likelihood": "N/A",
            "descriptor": "No events recorded",
            "count_3m": 0,
            "count_12m": 0,
            "count_24m": 0,
        }

    lga_df["risk_score"] = pd.to_numeric(lga_df.get("risk_score"), errors="coerce").fillna(0)
    qualifying = lga_df[lga_df["risk_score"] >= LIKELIHOOD_RISK_THRESHOLD].copy()

    if qualifying.empty:
        return {
            "likelihood": "N/A",
            "descriptor": "No qualifying events (risk score >= 4)",
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
            "likelihood": "Highly Likely",
            "descriptor": "Could occur within days or weeks",
            "count_3m": count_3m,
            "count_12m": count_12m,
            "count_24m": count_24m,
        }
    if count_12m >= 1:
        return {
            "likelihood": "Likely",
            "descriptor": "Could occur within a year or so",
            "count_3m": count_3m,
            "count_12m": count_12m,
            "count_24m": count_24m,
        }
    if count_24m >= 1:
        return {
            "likelihood": "Unlikely",
            "descriptor": "Could occur within 24+ months",
            "count_3m": count_3m,
            "count_12m": count_12m,
            "count_24m": count_24m,
        }

    return {
        "likelihood": "N/A",
        "descriptor": "No qualifying events in lookback windows",
        "count_3m": count_3m,
        "count_12m": count_12m,
        "count_24m": count_24m,
    }


def dataframe_records(df):
    if df.empty:
        return []
    return df.to_dict("records")


armed, ACTOR1_COL, ACTOR2_COL = load_data()
LGA_GDF = load_lga_boundaries()
ARMED_JOINED = spatial_join_events(armed, LGA_GDF)
PROJECT_LOCATIONS = load_project_locations(LGA_GDF)

# Calculate risk scores
ARMED_JOINED["risk_score"] = ARMED_JOINED.apply(
    lambda row: calculate_risk_score(row.get("llm_target"), row.get("llm_impact_type")),
    axis=1
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

# Precompute cumulative monthly risk scores for fast map updates.
MONTHLY_LGA_RISK_SCORES = (
    ARMED_JOINED.groupby(["year_month", "lga_key"])["risk_score"].sum().unstack(fill_value=0)
    .reindex(index=PERIOD_LABELS, fill_value=0)
    .reindex(columns=LGA_KEYS, fill_value=0)
)
MONTHLY_TOTAL_RISK_SCORES = ARMED_JOINED.groupby("year_month")["risk_score"].sum().reindex(PERIOD_LABELS, fill_value=0)
CUM_LGA_RISK_SCORES = MONTHLY_LGA_RISK_SCORES.cumsum()
CUM_TOTAL_RISK_SCORES = MONTHLY_TOTAL_RISK_SCORES.cumsum()


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


def get_range_risk_scores(start_index, end_index):
    if start_index == 0:
        lga_risk_scores = CUM_LGA_RISK_SCORES.iloc[end_index]
        total_risk = float(CUM_TOTAL_RISK_SCORES.iloc[end_index])
    else:
        lga_risk_scores = CUM_LGA_RISK_SCORES.iloc[end_index] - CUM_LGA_RISK_SCORES.iloc[start_index - 1]
        total_risk = float(CUM_TOTAL_RISK_SCORES.iloc[end_index] - CUM_TOTAL_RISK_SCORES.iloc[start_index - 1])
    return lga_risk_scores, total_risk


def build_map_dataframe_from_indices(start_index, end_index, metric="events"):
    if metric == "risk_score":
        lga_values, _ = get_range_risk_scores(start_index, end_index)
        value_col_name = "total_risk_score"
    else:
        lga_values, _ = get_range_counts(start_index, end_index)
        value_col_name = "events_in_range"
    
    # Convert Series to DataFrame, handling the index carefully
    values_df = lga_values.to_frame(name=value_col_name).reset_index()
    values_df.columns = ["lga_key", value_col_name]
    
    map_df = LGA_BASE.drop(columns=["geometry"]).merge(values_df, on="lga_key", how="left")
    map_df[value_col_name] = map_df[value_col_name].fillna(0).astype(float)
    return map_df


def build_likelihood_map_dataframe(start_index, end_index):
    start_label = PERIOD_LABELS[start_index]
    end_label = PERIOD_LABELS[end_index]
    mask = (ARMED_JOINED["year_month"] >= start_label) & (ARMED_JOINED["year_month"] <= end_label)
    filtered_df = ARMED_JOINED.loc[mask].copy()

    if filtered_df.empty:
        map_df = LGA_BASE.drop(columns=["geometry"]).copy()
        map_df["likelihood"] = "N/A"
        map_df["likelihood_code"] = 0
        map_df["qualifying_events"] = 0
        return map_df

    end_period = pd.Period(end_label, freq="M")
    end_ts = end_period.to_timestamp(how="end")

    filtered_df["risk_score"] = pd.to_numeric(filtered_df["risk_score"], errors="coerce").fillna(0)
    qualifying = filtered_df[filtered_df["risk_score"] >= LIKELIHOOD_RISK_THRESHOLD].copy()

    records = []
    for lga_key in LGA_KEYS:
        lga_q = qualifying[qualifying["lga_key"].eq(lga_key)]

        if lga_q.empty:
            likelihood = "N/A"
            code = 0
            count_q = 0
        else:
            count_q = int(len(lga_q))

            def count_in_last_months(months):
                start_period = end_period - (months - 1)
                start_ts = start_period.to_timestamp(how="start")
                return int(((lga_q["event_date"] >= start_ts) & (lga_q["event_date"] <= end_ts)).sum())

            count_3m = count_in_last_months(3)
            count_12m = count_in_last_months(12)
            count_24m = count_in_last_months(24)

            if count_3m >= 1:
                likelihood = "Highly Likely"
                code = 3
            elif count_12m >= 1:
                likelihood = "Likely"
                code = 2
            elif count_24m >= 1:
                likelihood = "Unlikely"
                code = 1
            else:
                likelihood = "N/A"
                code = 0

        records.append(
            {
                "lga_key": lga_key,
                "likelihood": likelihood,
                "likelihood_code": code,
                "qualifying_events": count_q,
            }
        )

    map_df = LGA_BASE.drop(columns=["geometry"]).merge(pd.DataFrame(records), on="lga_key", how="left")
    map_df["likelihood"] = map_df["likelihood"].fillna("N/A")
    map_df["likelihood_code"] = map_df["likelihood_code"].fillna(0).astype(int)
    map_df["qualifying_events"] = map_df["qualifying_events"].fillna(0).astype(int)
    return map_df


@lru_cache(maxsize=128)
def build_base_map_figure(start_index, end_index, metric="events"):
    if metric == "risk_score":
        map_df = build_map_dataframe_from_indices(start_index, end_index, metric="risk_score")
        color_col = "total_risk_score"
        title_text = "Total Risk Score"
        hover_data = {
            "statename": True,
            color_col: True,
            "lga_key": False,
        }
        color_scale = "Reds"
        colorbar = {"title": title_text}
    elif metric == "likelihood":
        map_df = build_likelihood_map_dataframe(start_index, end_index)
        color_col = "likelihood_code"
        title_text = "Likelihood"
        hover_data = {
            "statename": True,
            "likelihood": True,
            "qualifying_events": True,
            "likelihood_code": False,
            "lga_key": False,
        }
        color_scale = [
            (0.0, "#cbd5e1"),
            (0.333333, "#facc15"),
            (0.666666, "#fb923c"),
            (1.0, "#ef4444"),
        ]
        colorbar = {
            "title": title_text,
            "tickmode": "array",
            "tickvals": [0, 1, 2, 3],
            "ticktext": ["N/A", "Unlikely", "Likely", "Highly Likely"],
        }
    else:
        map_df = build_map_dataframe_from_indices(start_index, end_index, metric="events")
        color_col = "events_in_range"
        title_text = "Events"
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


def build_map_figure(range_values, selected_lga=None, metric="events"):
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

    lga_df = filtered_df[filtered_df["lga_key"].eq(selected_lga)].copy()
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

    figure = px.scatter_map(
        lga_df,
        lat="latitude",
        lon="longitude",
        center=map_center,
        zoom=dynamic_zoom,
        map_style="carto-positron",
        hover_data={"latitude": False, "longitude": False},
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
            "size": 14,
            "opacity": 0.95,
            "color": "#111111",
            "symbol": "circle",
        },
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Date: %{customdata[1]}<br>"
            "Civilian deaths: %{customdata[2]}<br>"
            "Total fatalities: %{customdata[3]}"
            "<extra></extra>"
        ),
        hoverlabel={"bgcolor": "rgba(255,255,255,0.95)", "font": {"size": 11, "color": "#0f172a"}},
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
                    marker={"size": 13, "opacity": 0.95, "color": "#2563eb", "symbol": "circle"},
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
    )
    return figure


INITIAL_RANGE = [0, len(PERIOD_LABELS) - 1]
INITIAL_FILTERED_DF, INITIAL_START_LABEL, INITIAL_END_LABEL = filter_data(INITIAL_RANGE)
INITIAL_STATUS_TEXT = (
    f"Month-Year range: {INITIAL_START_LABEL} to {INITIAL_END_LABEL} | "
    f"Total Armed clash events: {len(INITIAL_FILTERED_DF)}"
)
INITIAL_MAP_FIGURE = build_map_figure(INITIAL_RANGE, DEFAULT_LGA)
INITIAL_EVENTS_MAP_FIGURE = build_lga_events_figure(INITIAL_FILTERED_DF, DEFAULT_LGA)
INITIAL_EVENTS_MAP_TITLE = "Events in selected LGA"
INITIAL_EVENT_CLICK_TEXT = "Click a black dot to view event details."


def metric_card(title, value):
    return html.Div(
        [
            html.Div(title, className="metric-label"),
            html.Div(str(value), className="metric-value"),
        ],
        className="metric-card",
    )


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
            ],
            className="hero",
        ),
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
                html.Div(
                    [
                        html.Label("Map Overlay", className="control-label"),
                        dcc.RadioItems(
                            id="metric-selector",
                            options=[
                                {"label": " Events", "value": "events"},
                                {"label": " Risk Score", "value": "risk_score"},
                                {"label": " Likelihood", "value": "likelihood"},
                            ],
                            value="events",
                            inline=False,
                            labelStyle={"display": "block", "marginBottom": "8px", "cursor": "pointer"},
                            inputStyle={"marginRight": "8px", "cursor": "pointer"},
                        ),
                    ],
                    className="control-card",
                ),
            ],
            className="controls-grid",
        ),
        html.Div(INITIAL_STATUS_TEXT, id="status-bar", className="status-bar"),
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
        html.Div(id="details-top-panel", className="details-panel"),
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
        html.Div(
            [
                html.Span("Bottom map legend:", style={"fontWeight": "700", "marginRight": "12px"}),
                html.Span("●", style={"color": "#2563eb", "fontSize": "16px", "marginRight": "6px"}),
                html.Span("Projects", style={"marginRight": "16px"}),
                html.Span("●", style={"color": "#111111", "fontSize": "16px", "marginRight": "6px"}),
                html.Span("Events"),
            ],
            className="status-bar",
            style={"marginTop": "-6px", "marginBottom": "10px"},
        ),
        html.Div(INITIAL_EVENT_CLICK_TEXT, id="event-click-card", className="status-bar"),
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
                display: grid;
                gap: 16px;
                margin-bottom: 20px;
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
            .likelihood-card {
                border-radius: 14px;
                border: 1px solid #cbd5e1;
                background: #f8fafc;
                padding: 14px 16px;
                margin-bottom: 16px;
            }
            .likelihood-title {
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0.05em;
                text-transform: uppercase;
                color: #475569;
                margin-bottom: 6px;
            }
            .likelihood-value {
                font-size: 24px;
                font-weight: 800;
                color: #0f172a;
                line-height: 1.1;
            }
            .likelihood-desc {
                margin-top: 4px;
                font-size: 14px;
                color: #334155;
            }
            .likelihood-meta {
                margin-top: 8px;
                font-size: 13px;
                color: #475569;
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
    Output("event-click-card", "children"),
    Input("events-map", "clickData"),
)
def update_event_click_card(click_data):
    if not click_data or not click_data.get("points"):
        return "Click a black dot to view event details."

    point = click_data["points"][0]
    custom = point.get("customdata")
    if not custom or len(custom) < 5:
        return "Click a black dot to view event details."

    return (
        f"Event: {custom[0]} | Date: {custom[1]} | Civilian deaths: {custom[2]} | "
        f"Total fatalities: {custom[3]} | Note: {custom[4]}"
    )


@app.callback(
    Output("details-top-panel", "children"),
    Output("details-bottom-panel", "children"),
    Input("month-range", "value"),
    Input("lga-dropdown", "value"),
)
def update_details(range_values, selected_lga):
    filtered_df, _, end_label = filter_data(range_values)
    if not selected_lga:
        empty = html.Div("Select an LGA to view details.", className="details-card")
        return empty, html.Div()

    summary = summarize_lga(selected_lga, filtered_df, ACTOR1_COL, ACTOR2_COL)
    likelihood = classify_lga_likelihood(selected_lga, filtered_df, end_label)
    state_name, lga_name = selected_lga.split(" | ", 1)

    actor_block = html.Div(
        [
            html.H4("Unique Actors (actor_1 and actor_2, civilians excluded)", className="section-title"),
            html.Ul(
                [html.Li(actor, className="actor-pill") for actor in summary["actors"]],
                className="actors-list",
            ) if summary["actors"] else html.Div("No non-civilian actors in this period.", className="empty-note"),
        ],
        className="actors-card",
    )

    top_panel = html.Div(
        [
            html.Div(
                [
                    html.H3(f"Details for {lga_name} ({state_name})", className="details-title"),
                    html.Div(
                        [
                            html.Div("Armed Clash Likelihood (LGA)", className="likelihood-title"),
                            html.Div(likelihood["likelihood"], className="likelihood-value"),
                            html.Div(likelihood["descriptor"], className="likelihood-desc"),
                            html.Div(
                                f"Qualifying events (risk score >= {LIKELIHOOD_RISK_THRESHOLD}) in selected range: {summary['high_risk_event_count']} | "
                                f"Last 3 months: {likelihood['count_3m']} | "
                                f"Last 12 months: {likelihood['count_12m']} | "
                                f"Last 24 months: {likelihood['count_24m']}",
                                className="likelihood-meta",
                            ),
                        ],
                        className="likelihood-card",
                    ),
                    html.Div(
                        [
                            metric_card("Armed Clash Events", summary["event_count"]),
                            metric_card("Total Risk Score", summary["total_risk_score"]),
                            metric_card("Avg Risk Score/Event", summary["avg_risk_score"]),
                        ],
                        className="metrics-grid",
                    ),
                    html.Div(
                        [
                            metric_card("Total Deaths", summary["total_deaths"]),
                            metric_card("Civilian Deaths", summary["civilian_deaths"]),
                        ],
                        className="metrics-grid",
                    ),
                ],
                className="details-card",
            ),
        ]
    )

    bottom_panel = html.Div(
        [
            actor_block,
            html.Div(
                [
                    html.Div(
                        [
                            table_block(
                                "Knowledge Indicators",
                                dataframe_records(summary["knowledge_df"]),
                                "No indicators in this period.",
                            ),
                            table_block(
                                "Expectation Indicators",
                                dataframe_records(summary["expectation_df"]),
                                "No indicators in this period.",
                            ),
                        ],
                        className="details-card",
                    ),
                    html.Div(
                        [
                            table_block(
                                "Resources Indicators",
                                dataframe_records(summary["resources_df"]),
                                "No indicators in this period.",
                            )
                        ],
                        className="details-card",
                    ),
                ],
                className="detail-grid",
            ),
        ]
    )

    return top_panel, bottom_panel


if __name__ == "__main__":
    app.run(debug=True)
