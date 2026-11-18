import streamlit as st
import requests
import streamlit.components.v1 as components
from src.ui.map_utils import generate_folium_map

st.set_page_config(
    page_title="PathSnap Evaluator Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished look
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #1f77b4;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚗 PathSnap: Distributed Map-Matching & Route Reconstruction")
st.caption("PES University | UE24CS352A Machine Learning | Laptop 2 Evaluator Interface")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("🌐 System Architecture (Laptop 2)")
    st.info("This interface runs on **Laptop 2** and communicates over the local network with the FastAPI backend on **Laptop 1**.")

    st.subheader("1. Laptop 1 API Connection")
    api_url = st.text_input(
        "Backend Base URL",
        value="http://localhost:8000",
        help="Enter Laptop 1's IP (e.g., http://10.1.33.53:8000)"
    )

    # Health Check Button
    if st.button("🔌 Test Connection"):
        try:
            r = requests.get(f"{api_url}/health", timeout=3)
            if r.status_code == 200:
                st.success("✅ Connected to Laptop 1 Backend!")
            else:
                st.warning(f"Server responded with code {r.status_code}")
        except Exception as e:
            st.error(f"Cannot reach Laptop 1: {str(e)}")

    st.markdown("---")
    st.subheader("2. Evaluation Selection")

    # Fetch trips from server or fallback
    available_trips = [
        "00000000", "00000007", "00000009", "00000040", "00000052",
        "00000056", "00000065", "00000076", "00000077", "00000091"
    ]
    try:
        trips_res = requests.get(f"{api_url}/trips", timeout=2)
        if trips_res.status_code == 200:
            available_trips = trips_res.json().get("trips", available_trips)
    except Exception:
        pass

    selected_trip = st.selectbox(
        "Choose an Evaluation Trip:",
        available_trips,
        index=0
    )

    evaluate_btn = st.button("🚀 Reconstruct Route", type="primary", use_container_width=True)

# --- Main Dashboard ---
if evaluate_btn:
    with st.spinner(f"Sending Trip '{selected_trip}' inference request to Laptop 1 ({api_url})..."):
        try:
            res = requests.post(f"{api_url}/predict/{selected_trip}", timeout=15)
            
            if res.status_code == 200:
                data = res.json()
                st.success(f"Trip '{selected_trip}' successfully reconstructed and evaluated on Laptop 1!")

                # --- 4 Metrics Cards ---
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        label="Top-1 Route Accuracy",
                        value=f"{data['accuracy_score']}%",
                        help="Calculated server-side against ground-truth route"
                    )
                with col2:
                    st.metric(
                        label="Route Continuity",
                        value=f"{data['route_continuity']}%",
                        help="Topological connectivity of consecutive predicted road arcs"
                    )
                with col3:
                    st.metric(
                        label="GPS Observations",
                        value=f"{data['gps_point_count']}",
                        help="Noisy GPS coordinate points processed"
                    )
                with col4:
                    st.metric(
                        label="Inference Latency",
                        value=f"{data['inference_time_ms']} ms",
                        help="Total end-to-end model inference and decoding time"
                    )

                st.markdown("### 🗺️ Geographical Trajectory Map")
                st.caption(
                    "🔵 **Blue markers:** Raw noisy GPS observations | "
                    "🟢 **Green polyline:** Reconstructed road route | "
                    "Use layer toggle in top right corner"
                )

                # Render Folium Map reliably using components HTML (instant, never goes blank)
                folium_map = generate_folium_map(
                    gps_coords=data["gps_coordinates"],
                    predicted_path_coords=data["predicted_path_coords"],
                    trip_id=selected_trip
                )
                map_html = folium_map._repr_html_()
                components.html(map_html, height=550)

                # Route Details
                with st.expander(f"📋 View Predicted Road Arc Sequence ({data['predicted_arc_count']} arcs)"):
                    st.json({
                        "trip_id": selected_trip,
                        "total_arcs": data["predicted_arc_count"],
                        "predicted_arcs": data["predicted_arcs"][:100]
                    })

            else:
                st.error(f"Laptop 1 API Error ({res.status_code}): {res.text}")

        except requests.exceptions.ConnectionError:
            st.error(f"Could not connect to {api_url}. Is Laptop 1 running `python3 server.py`?")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")

else:
    st.info("👈 Select an evaluation trip from the sidebar and click **'Reconstruct Route'** to run live map-matching.")
    
    # Architecture Overview Panel
    st.markdown("---")
    st.subheader("System Architecture & Data Flow")
    st.markdown("""
    ```
    ┌───────────────────────────┐                 ┌────────────────────────────────────┐
    │         LAPTOP 2          │                 │              LAPTOP 1              │
    │   (Evaluator Interface)   │                 │       (Model & Evaluation API)     │
    │                           │                 │                                    │
    │  1. Evaluator Selects     │  POST /predict  │  2. Loads raw GPS observations     │
    │     Trip ID               │ ───────────────>│  3. Runs HMM/Viterbi route decoder │
    │                           │                 │  4. Evaluates accuracy vs GT       │
    │  5. Renders Folium Map &  │ <───────────────│     (Ground-truth kept server-side)│
    │     Route Metrics         │   JSON Result   │  6. Logs live request on CMD       │
    └───────────────────────────┘                 └────────────────────────────────────┘
    ```
    """)
