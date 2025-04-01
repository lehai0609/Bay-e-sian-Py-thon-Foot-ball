## **Software Design Document: Bayesian Real-Time Soccer Forecasting (Python)**

**1\. Overview**

* **Purpose:** This document outlines the design for a Python application that replicates the Bayesian ordered probit model for real-time soccer match outcome forecasting, as described in the research paper "Real-time forecasting within soccer matches through a Bayesian lens" \[cite: 873\] and inspired by the provided R implementation \[cite: 1226\].  
* **Goal:** To create a modular and understandable Python system capable of extracting data, preprocessing features, training the 90 distinct minute-by-minute Bayesian models in parallel, and evaluating their predictive performance.  
* **Scope:** This is designed as a side project, prioritizing clarity, core functionality, and reproducibility over enterprise-level complexity.

**2\. Architectural Design**

* **Style:** A modular pipeline architecture. Data flows sequentially through distinct processing stages.  
* **Core Pipeline:**  
  1. Data Extraction (SQLite \-\> Raw CSVs)  
  2. Preprocessing (Raw CSVs \-\> Feature Matrix)  
  3. Model Training (Feature Matrix \-\> Trained Models \- executed in parallel for minutes 1-90)  
  4. Prediction & Evaluation (Trained Models \+ Test Data \-\> Metrics & Results)  
* **Key Components:**  
  * **Data Layer:** Manages raw, processed, and output data persistence.  
  * **Processing Layer:** Handles data extraction and feature engineering.  
  * **Modeling Layer:** Contains the Bayesian model logic and MCMC implementation.  
  * **Analysis Layer:** Performs prediction and evaluates model performance.  
  * **Orchestration Layer:** Manages the execution flow of the pipelines.  
  * **Configuration:** External files for parameters.  
* **Technology Stack (Proposed):**  
  * **Language:** Python 3.x  
  * **Data Handling:** Pandas, NumPy  
  * **Database/XML:** sqlite3, lxml (or xml.etree.ElementTree)  
  * **Modeling:** SciPy (stats, optimize), NumPy. *Note: Building the Gibbs sampler from scratch using these is feasible for this specific model.*  
  * **Parallelism:** multiprocessing or joblib.  
  * **Configuration:** PyYAML  
  * **Utilities:** logging, pickle / joblib (for saving models)

**3\. Detailed Design**

* **Project Structure (src/ directory):**  
  src/  
  ├── data\_extraction/  
  │   └── sqlite\_extractor.py  
  ├── preprocessing/  
  │   ├── feature\_engineer.py  
  │   └── data\_cleaner.py  
  ├── modeling/  
  │   ├── priors.py  
  │   ├── bayesian\_ordered\_probit.py  
  │   └── utils.py  
  ├── analysis/  
  │   └── evaluate.py  
  ├── pipelines/  
  │   ├── training\_pipeline.py  
  │   └── prediction\_pipeline.py  
  └── utils/  
      ├── config\_loader.py  
      └── logging\_setup.py

* **Key Modules & Functionality:**  
  * sqlite\_extractor.py: Functions to connect to SQLite DB, execute queries, parse XML event data from specific columns (like goal, shoton etc. in the Match table) into structured format (e.g., list of dictionaries or DataFrame). Mirrors extract\_data\_sqlite.R \[cite: 1231\].  
  * feature\_engineer.py: Functions to:  
    * Calculate minute-by-minute cumulative counts for each event type (goal, card, shot, etc.) per team per match.  
    * Calculate average team strength based on starting player ratings (requires joining match, player, and player attribute data).  
    * Assemble the final wide-format feature matrix for a given minute k.  
  * data\_cleaner.py: Functions for handling missing values (e.g., filling event counts with 0), scaling numerical features (like team strength, potentially using sklearn.preprocessing.StandardScaler), and ensuring correct data types.  
  * priors.py: Functions to:  
    * Generate the AR(1) covariance matrix (ar1\_cor equivalent) \[cite: 1246\].  
    * Assemble the full prior mean vector (beta\_not) and covariance matrix (sigma\_not) for the model coefficients based on configuration.  
  * bayesian\_ordered\_probit.py: Core Gibbs Sampler implementation:  
    * run\_gibbs\_sampler(X, Z, priors, mcmc\_params, config): Main loop iterating niter times.  
    * Includes steps for sampling the latent variable y (using scipy.stats.truncnorm), sampling coefficients Beta (using multivariate normal properties derived from paper Eq. 2.16), and sampling cutoffs delta (using the Beta distribution approach derived from paper Eq. 2.25).  
  * modeling/utils.py: Helper for Gibbs sampler, e.g., Geweke convergence diagnostic function (geweke.diag equivalent) \[cite: 1263\].  
  * evaluate.py: Functions to calculate Brier score (paper Eq. 2.34), F1-score (paper Eq. 2.33), sensitivity/specificity (paper Eq. 2.32), using predictions and true outcomes.  
  * training\_pipeline.py: Orchestrates the process for minutes 1-90. Calls preprocessing and feature engineering for each minute k. Uses multiprocessing.Pool or joblib.Parallel to run run\_gibbs\_sampler for each minute concurrently. Saves trained model parameters (e.g., mean beta, mean delta) for each minute.  
  * prediction\_pipeline.py: Loads saved model parameters for relevant minutes, loads/preprocesses test/new data, calculates predictions and probabilities (paper Eq. 2.30).  
  * config\_loader.py: Simple function to load parameters from a YAML file.

**4\. Interface Specifications**

* **Data Flow:** Primarily file-based between major stages.  
  * **Extraction \-\> Preprocessing:** Raw data (e.g., CSVs like all\_incidents.csv, matchdetails.csv).  
  * **Preprocessing \-\> Modeling:** Processed feature DataFrames (or potentially Feather/Parquet files) containing the wide-format matrix for training/testing.  
  * **Modeling \-\> Analysis/Prediction:** Saved model parameters (e.g., Python dictionaries pickled per minute) and potentially CSV/DataFrame of posterior samples.  
  * **Prediction \-\> Evaluation:** CSV/DataFrame containing predictions and probabilities per match per minute.  
* **Configuration:** A central YAML file (config/model\_params.yaml) provides parameters (file paths, MCMC iterations, burn-in, thinning, prior settings, y\_sd, stdnorm for cutoffs) accessed via config\_loader.py.  
* **Execution:** main.py script acts as the entry point, likely using command-line arguments (e.g., python main.py train or python main.py predict) to trigger the appropriate pipeline in pipelines/.

**5\. Data Design**

* **Input Data:**  
  * database.sqlite: As provided on Kaggle. Key tables: Match, Player, Player\_Attributes, Team. Match table contains XML columns.  
  * XML Schema (within Match table columns like goal): Contains nested \<value\> tags with details like id, type, subtype, player1, team, elapsed, etc.  
* **Processed Feature Data (Conceptual final\_model\_input.csv):**  
  * Format: Pandas DataFrame (likely saved to disk).  
  * Structure: Wide format. One row per match.  
  * Columns: match\_id, home\_team\_strength, away\_team\_strength, intercept, cumulative event counts per minute (e.g., home\_goal\_cum\_1, away\_goal\_cum\_1, ... home\_card\_y\_cum\_90, away\_corner\_cum\_90), outcome (-1, 0, 1). *Alternatively, could be structured longer for easier processing per minute.*  
* **Model Output Data:**  
  * model\_minute\_k.pkl: Pickled Python dictionary containing learned parameters for minute k (e.g., { 'beta\_mean': numpy\_array, 'delta\_mean': numpy\_array, 'posterior\_samples\_beta': numpy\_array, ... }).  
  * predictions.csv: Columns: match\_id, minute, true\_outcome, predicted\_outcome, prob\_loss, prob\_draw, prob\_win.  
  * metrics.json: Dictionary storing evaluation metrics (Brier, F1, etc.) per minute.