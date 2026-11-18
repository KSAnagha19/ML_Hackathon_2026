
import streamlit as st
import requests
import pandas as pd
import pydeck as pdk


API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="PathSnap | GPS Map Matching",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ PathSnap")
st.caption(
    "Intelligent GPS Map Matching using "
    "Optuna-Tuned XGBoost + HMM/Viterbi"
)

st.divider()


@st.cache_data(ttl=30)
def fetch_trips():
    response = requests.get(
        f"{API_URL}/trips",
        timeout=10
    )
    response.raise_for_status()
    return response.json()["trip_ids"]


try:
    trips = fetch_trips()

except requests.RequestException as error:
    st.error(
        "Cannot connect to the FastAPI backend. "
        "Make sure Uvicorn is running on port 8000."
    )
    st.code(str(error))
    st.stop()


with st.sidebar:
    st.header("⚙️ Map Matching")

    selected_trip = st.selectbox(
        "Select GPS trip",
        trips,
        index=0
    )

    run_button = st.button(
        "🚀 Run Map Matching",
        type="primary",
        use_container_width=True
    )

    st.divider()

    st.subheader("Model Configuration")
    st.write("**Model:** Optuna-tuned XGBoost")
    st.write("**Decoder:** Network-aware HMM/Viterbi")
    st.write("**Beta:** 10.0")
    st.write("**Transition weight:** 0.75")


if run_button:
    with st.spinner(
        "Running candidate generation, "
        "XGBoost and HMM/Viterbi..."
    ):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={"trip_id": selected_trip},
                timeout=180
            )
            response.raise_for_status()

            st.session_state["result"] = (
                response.json()
            )

        except requests.RequestException as error:
            st.error(f"Prediction failed: {error}")
            st.stop()


result = st.session_state.get("result")

if result is None:
    st.info(
        "Select a trip from the sidebar and click "
        "'Run Map Matching' to visualize predictions."
    )
    st.stop()


st.subheader(
    f"📍 Results — Trip {result['trip_id']}"
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "GPS Observations",
    f"{result['total_gps_points']:,}"
)

col2.metric(
    "Matched Observations",
    f"{result['matched_points']:,}"
)

col3.metric(
    "Coverage",
    f"{result['coverage_percent']:.2f}%"
)

col4.metric(
    "Runtime",
    f"{result['runtime_seconds']:.2f} s"
)

st.metric(
    "Predictions Changed by HMM",
    result["changed_predictions"]
)

predictions = result["predictions"]

if not predictions:
    st.warning(
        "No road predictions were returned."
    )
    st.stop()


df = pd.DataFrame(predictions)

st.divider()
st.subheader("🗺️ GPS and Matched Road Map")

gps_df = df[
    ["gps_lat", "gps_lon"]
].rename(
    columns={
        "gps_lat": "lat",
        "gps_lon": "lon"
    }
)

road_data = []

for item in predictions:
    road_data.append({
        "path": [
            item["road_start"],
            item["road_end"]
        ]
    })


gps_layer = pdk.Layer(
    "ScatterplotLayer",
    data=gps_df,
    get_position="[lon, lat]",
    get_radius=8,
    get_fill_color=[255, 170, 0, 150],
    pickable=True
)

road_layer = pdk.Layer(
    "PathLayer",
    data=road_data,
    get_path="path",
    get_color=[0, 200, 255, 220],
    get_width=5,
    width_min_pixels=2,
    pickable=True
)

center_lat = float(gps_df["lat"].mean())
center_lon = float(gps_df["lon"].mean())

view_state = pdk.ViewState(
    latitude=center_lat,
    longitude=center_lon,
    zoom=12,
    pitch=0
)

deck = pdk.Deck(
    layers=[road_layer, gps_layer],
    initial_view_state=view_state,
    map_style="light",
    tooltip={
        "text": "PathSnap GPS Map Matching"
    }
)

st.pydeck_chart(
    deck,
    use_container_width=True
)

st.caption(
    "Orange dots represent GPS observations. "
    "Blue segments represent roads selected by HMM. "
    "This is a per-observation road visualization, "
    "not a reconstructed continuous route."
)

st.divider()

st.subheader("📊 Prediction Details")

display_columns = [
    "gps_index",
    "gps_lon",
    "gps_lat",
    "xgboost_arc_id",
    "hmm_arc_id"
]

st.dataframe(
    df[display_columns],
    use_container_width=True,
    hide_index=True
)

csv_data = df.to_csv(index=False)

st.download_button(
    label="⬇️ Download Predictions CSV",
    data=csv_data,
    file_name=(
        f"pathsnap_predictions_"
        f"{result['trip_id']}.csv"
    ),
    mime="text/csv"
)

st.divider()
st.caption(
    "PathSnap | GPS Map Matching | "
    "XGBoost + Network HMM/Viterbi"
)
