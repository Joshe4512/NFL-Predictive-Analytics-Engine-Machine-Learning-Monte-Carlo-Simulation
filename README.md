# NFL Predictive Analytics Engine: Machine Learning & Monte Carlo Simulation

An end-to-end Python data pipeline that ingests historical NFL performance data, engineers time-series momentum features, trains an ensemble Random Forest classifier, and executes a stochastic Monte Carlo simulation with dynamic Regression to the Mean to predict the remainder of the NFL season.

**Author:** Joshua Evans  
**Technologies:** Python 3.10+, Pandas, Scikit-Learn, Matplotlib, Seaborn  
**Domain:** Applied Machine Learning, Time-Series Feature Engineering, Stochastic Modeling, Sports Analytics  

---

## 📌 Architectural Overview

Standard sports prediction models frequently fall victim to two structural flaws:
1. **The Deterministic Trap:** Models using basic `predict()` assign victory to the statistical favorite 100% of the time, resulting in unrealistic regular-season records (e.g., top contenders going 17-0 while rebuilding teams finish 0-17).
2. **Momentum Extrapolation:** Models evaluating trailing performance often extrapolate early-season 3-game hot streaks across a full 17-game schedule without accounting for performance stabilization.

This engine solves both problems by coupling a **Random Forest Classifier** with a **Stochastic Monte Carlo Simulation** and a **15% Weekly Regression-to-the-Mean Decay Loop**.

[nflverse Live Data Repository]
│
▼
[Automated Extraction & ETL] ──────► Cache Invalidation (nfl_matches.csv)
│
▼
[Time-Series Feature Engineering] ───► 3-Game Rolling Window (closed='left')
│
▼
[Supervised Classifier Training] ────► Random Forest (Optimized for Precision)
│
▼
[Dynamic Simulation Engine]
├── Weekly Momentum Decay ───► Regression to Historical Baselines (15%)
└── Monte Carlo Engine ──────► predict_proba() + Stochastic Seed Roll
│
▼
[Output Generation & Analytics] ─────► Console Feed & 2026_season_predictions.csv


---

## ⚙️ Core Engineering Methodologies

### 1. Data Pipeline & Cache Invalidation
* **Direct Server Ingestion:** Bypasses unmaintained API wrappers by pulling raw tabular game data directly from the official `nflverse` GitHub master data repository.
* **Cache Management:** Implements automated cache invalidation on local execution to clear cached CSVs and capture up-to-the-minute results from prime-time games.
* **Tidy Transformation:** Restructures single-row matchups into dual perspective rows (Home Perspective vs. Away Perspective) to double effective training observations while preserving directional target integrity.

### 2. Time-Series Engineering & Data Leakage Prevention
* **Rolling Averages:** Utilizes Pandas `.groupby('team')` and `.transform()` to generate 3-game trailing performance windows for Offensive Points Scored (`points_for_rolling`) and Defensive Points Allowed (`points_against_rolling`).
* **Zero Leakage:** Strictly enforces `closed='left'` window boundaries so feature calculations only evaluate games completed prior to kickoff, completely isolating training data from current-game ground truth.
* **Feature Imputation:** Parses 24-hour UTC game timestamps into 12-hour EST formats and heuristically imputes broadcast carriers (Prime Video, ESPN/ABC, NBC, CBS/FOX, NFL Network) based on scheduling windows.

### 3. Chronological Regression to the Mean
To prevent early-season statistical anomalies (such as a fringe team starting 3-0 with high offensive output) from artificially dominating simulated season outcomes, the simulation applies a decay formula every new week $w$:

$$\text{Momentum}_{w+1} = \text{Momentum}_w + (\text{Baseline}_{\text{Franchise}} - \text{Momentum}_w) \times 0.15$$

This pulls anomalous rolling stats 15% closer to each team's multi-year franchise baseline with each advancing week.

### 4. Stochastic Monte Carlo Inference Engine
Rather than relying on binary classification boundaries, the inference engine queries class prediction probabilities via `model.predict_proba()`:
1. Obtains the raw probability distribution for both the home and away combatants.
2. Normalizes output probabilities to ensure $P(\text{Home}) + P(\text{Away}) = 1.0$.
3. Generates a pseudo-random floating-point scalar $r \in [0.0, 1.0)$.
4. Awards the match outcome based on probability threshold boundaries, mathematically modeling real-world variance and upsets ("Any Given Sunday").

---

## 📊 Model Evaluation & Diagnostic Visualizations

The pipeline automatically generates diagnostic figures and saves them to the `assets/` directory:

| Diagnostic Artifact | Description |
|---|---|
| `points_eda.png` | Boxplot tracking the distribution of offensive point production mapped to match results. |
| `confusion_matrix.png` | Heatmap evaluating true positives vs. false positives on the chronological holdout test set (100 most recent games). |
| `feature_importance.png` | Relative weighting of features, confirming that trailing scoring momentum carries greater predictive signal than categorical venue codes. |

* **Target Evaluation Metric:** Model architecture is tuned specifically for **Precision Score** (~59%–62%) rather than raw accuracy, minimizing false-positive winner projections against Vegas market efficiency.

---

## 💻 Installation & Usage

### Setup Virtual Environment
```bash
git clone [https://github.com/yourusername/NFL-Predictive-Analytics-Engine-Machine-Learning-Monte-Carlo-Simulation.git](https://github.com/yourusername/NFL-Predictive-Analytics-Engine-Machine-Learning-Monte-Carlo-Simulation.git)
cd NFL-Predictive-Analytics-Engine-Machine-Learning-Monte-Carlo-Simulation
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
Install Dependencies
Bash
pip install pandas scikit-learn matplotlib seaborn
Execute the Pipeline
Bash
python script1.py
📋 Sample Output Feed
Plaintext
[*] Fetching real NFL data for 2022-2026 directly from source...
[*] Success! Saved 2340 game records to nfl_matches.csv
[*] Engineering features (Rolling Averages)...
[*] Training Random Forest Classifier...
[*] Model Precision Score: 60.4%
[*] Generating Data Visualizations in /assets folder...

🔮 MONTE CARLO SIMULATION: PREDICTING THE REST OF 2026 🔮
-----------------------------------------------------------------
Week 4
-----------------------------------------------------------------
DAL @ NYG @ 8:15 PM EST (Thursday on Prime Video) -- DAL wins (2-2) [DAL 58.2% | NYG 41.8%]
NO @ ATL @ 1:00 PM EST (Sunday on CBS/FOX) -- ATL wins (2-2) [NO 44.1% | ATL 55.9%]
CIN @ CAR @ 1:00 PM EST (Sunday on CBS/FOX) -- CIN wins (1-3) [CIN 63.7% | CAR 36.3%]
LAR @ CHI @ 1:00 PM EST (Sunday on CBS/FOX) -- CHI wins (2-2) [LAR 48.0% | CHI 52.0%]
MIN @ GB @ 1:00 PM EST (Sunday on CBS/FOX) -- MIN wins (4-0) [MIN 53.4% | GB 46.6%]
JAX @ HOU @ 1:00 PM EST (Sunday on CBS/FOX) -- HOU wins (3-1) [JAX 37.5% | HOU 62.5%]
KC @ LAC @ 4:25 PM EST (Sunday on CBS/FOX) -- LAC wins (3-1) [KC 62.1% | LAC 37.9%] (UPSET!)
BAL @ DAL @ 4:25 PM EST (Sunday on CBS/FOX) -- BAL wins (2-2) [BAL 61.3% | DAL 38.7%]
BUF @ BAL @ 8:20 PM EST (Sunday on NBC) -- BAL wins (3-2) [BUF 49.2% | BAL 50.8%]
TEN @ MIA @ 7:30 PM EST (Monday on ESPN/ABC) -- MIA wins (2-2) [TEN 41.0% | MIA 59.0%]

[*] Complete! Exported all batch predictions to 2026_season_predictions.csv
🤖 AI Usage & Engineering Attribution
In alignment with modern software engineering and data science workflows, artificial intelligence tools were leveraged throughout this project:

Algorithm & Pipeline Design: Brainstorming feature decay mathematics, time-series windowing functions, and stochastic probability implementations.

Debugging & Refactoring: Resolving environment compiler failures and Pandas grouped transform index behaviors.

Documentation: Structuring clear Markdown documentation, technical explanations, and portfolio presentation standards.

Verification: All data transformations, machine learning routines, and probability distributions were independently tested, executed, and verified inside a local Python runtime.

🔮 Roadmap (V2)
[ ] Expected Points Added (EPA/Play): Replace raw box-score scoring averages with down-and-distance success rates to isolate offensive efficiency from garbage-time scoring.

[ ] Dynamic Quarterback Elo Ratings: Integrate live injury and depth-chart ingestion to penalize team momentum when a backup quarterback starts.

[ ] Automated CI/CD Workflow: Schedule a GitHub Action to re-run the inference pipeline every Tuesday morning and auto-commit updated season projections.
