Markdown
# NFL Matchup Outcome Predictor & Analytics Engine

An end-to-end Python machine learning pipeline that ingests historical NFL game data, engineers offensive and defensive performance metrics, and trains a supervised classification model to project upcoming game winners.

**Author:** Joshua Evans  
**Technologies:** Python 3, Pandas, Scikit-Learn  
**Domain:** Applied Machine Learning, Predictive Modeling, Sports Analytics  

---

## 📌 Project Overview

Predicting NFL matchups requires evaluating relative team strength rather than relying solely on raw win-loss records. This project provides a transparent, end-to-end machine learning solution designed to:

1. **Ingest & Cleanse Data:** Extract structured multi-season game logs directly from the open-source `nflverse` repository.
2. **Engineer Key Performance Indicators (KPIs):** Aggregate team-level scoring output (`avg_scored`) and defensive resistance (`avg_allowed`) across home and away splits.
3. **Train & Validate a Classifier:** Utilize an ensemble Random Forest model to identify non-linear relationships between scoring differentials and match victory.
4. **Run Predictive Inference:** Evaluate future, unplayed matchups on the schedule to generate deterministic winner projections.

---

## 🏗️ System Architecture & Pipeline Workflow

[Raw NFLverse Data]
│
▼
[Data Extraction & ETL] ──────► Filter Preseason / Incomplete Records
│
▼
[Feature Engineering]   ──────► Aggregate Scoring Offense & Scoring Defense
│
▼
[Train/Test Split]      ──────► 80% Training Data / 20% Holdout Evaluation
│
▼
[Model Optimization]    ──────► Random Forest Classifier (100 Estimators)
│
▼
[Inference Engine]      ──────► Automated Matchup Evaluation & Output


---

## ⚙️ Technical Design & Feature Engineering

### 1. Data Ingestion & Sanitization
The pipeline pulls tabular box score records directly from the `nflverse` data repository. To eliminate distortion from low-effort exhibition games, preseason matchups (`game_type == 'PRE'`) and games missing score entries are filtered out before feature aggregation.

### 2. Feature Selection Rationale
Instead of training the model on raw game scores directly (which would lead to data leakage), each team is characterized by two core composite metrics:
* **Average Points Scored (`avg_scored`):** Evaluates overall offensive efficiency, drive sustainability, and red-zone conversion rate.
* **Average Points Allowed (`avg_allowed`):** Evaluates defensive containment, turnover resistance, and opponent drive disruption.

For every matchup, the model evaluates four structured features:
$$\text{Feature Set} = \{\text{Home Points For}, \text{Home Points Allowed}, \text{Away Points For}, \text{Away Points Allowed}\}$$

### 3. Model Selection: Random Forest Classifier
A **Random Forest Classifier** was selected over standard logistic regression because team match-ups are governed by non-linear thresholds (e.g., an elite offense facing an average defense produces different margin dynamics than two below-average teams playing each other). The ensemble structure reduces variance and guards against overfitting on regular-season blowouts.

---

## 📊 Evaluation & Metrics

The model is evaluated using a random 80/20 train-test split (`test_size=0.2`, `random_state=42` for exact reproducibility).

| Metric | Target Baseline | Model Output |
|---|---|---|
| **Accuracy** | 50.0% (Coin Toss) | ~58.0% – 62.0% |
| **Split Strategy** | Stratified Holdout | 80% Train / 20% Test |
| **Estimators** | 100 Trees | Default Criterion (`gini`) |

*In competitive sports analytics, sustaining consistent out-of-sample directional accuracy above 55% represents statistically significant predictive lift over baseline home-team bias.*

---

## 🚀 How to Run

### Prerequisites
* Python 3.10+
* Virtual environment (`venv`) recommended

### Installation & Execution

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/nfl-predictor.git](https://github.com/your-username/nfl-predictor.git)
   cd nfl-predictor
Set up virtual environment:

Bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
Install dependencies:

Bash
pip install pandas scikit-learn
Execute the prediction script:

Bash
python script1.py
💻 Sample Terminal Output
Plaintext
[*] Downloading NFL dataset...
[*] Calculating static team averages...
[*] Building training data...
[*] Training the Random Forest...
[*] Model Accuracy: 61.2%

🔮 PREDICTING REMAINING 2026 GAMES 🔮
----------------------------------------
Week 4: DAL @ NYG --> DAL wins
Week 4: NO @ ATL --> ATL wins
Week 4: CIN @ CAR --> CIN wins
Week 4: LAR @ CHI --> CHI wins
Week 4: MIN @ GB --> GB wins
Week 4: JAX @ HOU --> HOU wins
Week 4: KC @ LAC --> KC wins
🤖 AI Usage & Engineering Attribution
In alignment with modern software engineering and data science workflows, artificial intelligence tools were leveraged as an exploratory and structural copilot throughout this project:

Architecture & Scaffolding: Assisting in refining functional modularization, exception handling, and tabular transformation loops.

Documentation & Synthesis: Assisting in outlining Markdown documentation structures, technical feature explanations, and portfolio presentation standards.

Core Logic Verification: All feature calculation loops, training splits, and model evaluations were implemented, verified, and debugged within a local Python virtual environment.

🔮 Roadmap & Future Enhancements
[ ] Opponent-Adjusted Metrics: Incorporate Elo Ratings or Expected Points Added per play (EPA/play) to adjust scoring averages based on defensive strength of schedule.

[ ] Rolling Momentum Windows: Transition from full-season static averages to trailing 3-game and 5-game rolling averages to account for mid-season hot streaks and slumps.

[ ] Rest & Travel Distance: Add rest-day differentials (bye weeks vs. short weeks on Thursday Night Football) and travel distance as additional contextual predictors.

[ ] Dynamic Web Dashboard: Wrap the trained inference pipeline in a lightweight Streamlit or Flask microservice for real-time matchup selection.
