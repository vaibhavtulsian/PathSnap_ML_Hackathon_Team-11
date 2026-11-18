# 🚗 PathSnap: Distributed Map-Matching & Route Reconstruction

> **PES University | UE24CS352A: Machine Learning Hackathon**  
> A simple, end-to-end distributed system that reconstructs real vehicle routes from noisy GPS data.

---

## ⚡ 1-Minute Quick Start for Each Member

Find your name/role below to see **exactly where to start**:

| Team Member | What to Open / Run | What You Need to Produce |
| :--- | :--- | :--- |
| **Member 1 (Data & Features)** | Open [`notebooks/01_MEMBER_1_eda_and_features.ipynb`](notebooks/01_MEMBER_1_eda_and_features.ipynb) | Candidate table `[trip_id, obs_idx, arc_id, distance]`, miss rate, and EDA Folium map for **Report Section 2.2**. |
| **Member 2 (Models & Tuning)** | Open [`notebooks/02_MEMBER_2_models_and_tuning.ipynb`](notebooks/02_MEMBER_2_models_and_tuning.ipynb) | 5 models trained, comparison table for **Section 3.1**, Optuna search for **Section 4**, and saved weights in `artifacts/`. |
| **Member 3 (HMM & Viterbi)** | Open [`notebooks/03_MEMBER_3_hmm_route_decoder.ipynb`](notebooks/03_MEMBER_3_hmm_route_decoder.ipynb) | Road graph, Viterbi route decoder, Folium maps for **Section 5 & 6**, and give Member 4 the function: `predict_route(trip_id) -> list of arc IDs`. |
| **Member 4 (Server, UI & PDF)** | Run `python server.py` and `streamlit run ui.py` | FastAPI backend on Laptop 1, Streamlit UI on Laptop 2, Two-laptop demo, and final report assembly. |

---

## 🚀 How to Run the System (Member 4)

You do **NOT** need complex commands. The repository has 1-line root runners:

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Test the Entire Pipeline in 1 Click
```bash
python test_system.py
```
*(This loads all 10 evaluation trips, tests the evaluator logic, and confirms that accuracy calculation and maps work!)*

### Step 3: Run on Two Laptops (For Hackathon Evaluator Demo)

#### On Laptop 1 (Model & Server):
```bash
python server.py
```
- Starts FastAPI on port `8000`.
- Keep the terminal/CMD open: evaluators will stand here to check live incoming request logs!

#### On Laptop 2 (User Interface):
```bash
streamlit run ui.py
```
- Connects Laptop 2 to Laptop 1 over local Wi-Fi / hotspot.
- Evaluators select any of the 10 trips and click **"Reconstruct Route"** to see the map and accuracy scores!

---

## 📁 Simple Project Layout

```text
PathSnap/
├── server.py              # 👉 Run this on Laptop 1 (starts FastAPI server)
├── ui.py                  # 👉 Run this on Laptop 2 (starts Streamlit UI)
├── test_system.py         # 👉 1-click test to verify all 10 trips
├── requirements.txt       # Dependencies
│
├── notebooks/             # 👉 Step-by-step notebooks for each teammate:
│   ├── 01_MEMBER_1_eda_and_features.ipynb
│   ├── 02_MEMBER_2_models_and_tuning.ipynb
│   └── 03_MEMBER_3_hmm_route_decoder.ipynb
│
├── data/
│   └── student_test/      # The 10 reserved evaluation trips
│
├── src/                   # Source code
│   ├── api/               # FastAPI backend & evaluation logic (Member 4)
│   ├── ui/                # Streamlit UI & Folium mapping (Member 4)
│   ├── data_processing/   # Preprocessing (Member 1)
│   ├── models/            # Model definitions (Member 2)
│   └── hmm_decoder/       # Viterbi decoder (Member 3)
│
├── artifacts/             # Saved model weights & scalers
└── docs/
    └── report_template.md # Master template for final PDF report
```

---

## 🔒 Ground-Truth Leakage Prevention
- Ground-truth files (`.route`) reside strictly on **Laptop 1** and are evaluated **server-side only**.
- The UI on **Laptop 2** never accesses or receives ground-truth routes before prediction.
