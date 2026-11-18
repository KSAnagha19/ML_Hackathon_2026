\# PathSnap — GPS Road Map Matching Using Machine Learning



\## Project Overview



PathSnap is a machine learning-based GPS map-matching system that identifies the most likely road segments corresponding to noisy GPS observations.



The system combines candidate road generation, feature engineering, an Optuna-tuned XGBoost classifier, and network-aware Hidden Markov Model (HMM) decoding using the Viterbi algorithm.



It includes a FastAPI backend and a Streamlit frontend for interactive visualization.



\## Key Features



\- GPS-to-road candidate generation using spatial indexing.

\- Feature extraction for candidate road segments.

\- Training and comparison of multiple machine learning models.

\- Grouped cross-validation for model evaluation.

\- Bayesian hyperparameter optimization using Optuna.

\- Network-aware HMM/Viterbi decoding for consistent road predictions.

\- FastAPI endpoints for trip selection and prediction.

\- Streamlit dashboard for GPS and predicted-route visualization.



\## Machine Learning Pipeline



1\. Load GPS observations and road-network data.

2\. Generate nearby candidate road segments using a KD-tree.

3\. Extract candidate-level features.

4\. Train and evaluate multiple classification models.

5\. Optimize XGBoost hyperparameters using Optuna.

6\. Estimate candidate probabilities with the tuned XGBoost model.

7\. Apply network-aware Viterbi decoding to improve sequence consistency.

8\. Reconstruct and visualize the predicted route.



\## Models Explored



\- Logistic Regression

\- Random Forest

\- Extra Trees

\- Histogram Gradient Boosting

\- Multilayer Perceptron (MLP)

\- XGBoost



\*\*Final inference model:\*\* Optuna-tuned XGBoost with network-aware HMM/Viterbi decoding.



\## Project Structure



```text

ML\_Hackathon\_2026/

├── backend/

│   ├── api.py

│   ├── inference\_service.py

│   ├── candidate\_generation.py

│   ├── feature\_extraction.py

│   ├── road\_network.py

│   ├── viterbi\_decoder.py

│   ├── network\_viterbi\_decoder.py

│   └── ...additional training and evaluation scripts

├── frontend/

│   └── app.py

├── models/

│   └── xgboost\_tuned.joblib

├── results/

│   └── ...selected evaluation results

├── requirements.txt

├── .gitignore

└── README.md

```



\## Installation



\### 1. Clone the repository



```bash

git clone https://github.com/KSAnagha19/ML\_Hackathon\_2026.git

cd ML\_Hackathon\_2026

```



\### 2. Create a virtual environment



```bash

python -m venv .venv

```



Activate it on Windows CMD:



```cmd

.venv\\Scripts\\activate

```



\### 3. Install dependencies



```bash

pip install -r requirements.txt

```



\## Dataset Setup



The project uses road-network and GPS observation files supplied for the hackathon.



For each trip, the application expects three files:



\- `<trip\_id>.nodes` — road-network nodes.

\- `<trip\_id>.arcs` — road-network edges.

\- `<trip\_id>.track` — GPS observations.



Example:



```text

student\_test/

└── 00000011/

&#x20;   ├── 00000011.nodes

&#x20;   ├── 00000011.arcs

&#x20;   └── 00000011.track

```



Place the organizer-provided `student\_test` folder in the repository root before running trip predictions.



The dataset is not included in this repository. Training and evaluation scripts may also require the corresponding organizer-provided training data.



\## Running the Application



\### Start the FastAPI backend



From the repository root:



```bash

python -m uvicorn backend.api:app --reload

```



Backend address:



`http://127.0.0.1:8000`



API documentation:



`http://127.0.0.1:8000/docs`



\### Start the Streamlit frontend



Open a second terminal in the repository root and run:



```bash

python -m streamlit run frontend/app.py

```



Streamlit typically starts at:



`http://localhost:8501`



Keep the backend running while using the frontend.



\## Model Evaluation



The `results/` directory contains selected artifacts documenting:



\- Cross-validation comparisons.

\- XGBoost hyperparameter optimization.

\- Feature-selection analysis.

\- ROC-AUC comparisons.

\- HMM tuning.

\- Network-aware Viterbi evaluation.



The final inference configuration uses:



\- Model: Optuna-tuned XGBoost.

\- HMM beta: `10.0`.

\- Transition weight: `0.75`.



\## Technology Stack



\*\*Programming:\*\* Python



\*\*Machine Learning:\*\* scikit-learn, XGBoost, Optuna



\*\*Data Processing:\*\* NumPy, Pandas, SciPy



\*\*Backend:\*\* FastAPI, Uvicorn



\*\*Frontend:\*\* Streamlit, PyDeck



\*\*Visualization:\*\* Matplotlib



\## Notes



The repository includes the trained model and selected evaluation artifacts. Original hackathon datasets and private evaluation materials are excluded.



The application requires the appropriate dataset files for trip-level predictions. Model training requires additional data and may take time depending on the hardware.



\## Project Purpose



PathSnap demonstrates how machine learning and road-network-aware sequence decoding can improve the consistency of GPS map matching and provide interpretable visualizations of predicted road routes.


## Application Screenshots



The following screenshots demonstrate the PathSnap application and its GPS map-matching workflow.



!\[PathSnap Screenshot 1](Screenshots/ml\_1.png)



!\[PathSnap Screenshot 2](Screenshots/ml\_2.png)



!\[PathSnap Screenshot 3](Screenshots/ml\_3.png)



!\[PathSnap Screenshot 4](Screenshots/ml\_4.png)



!\[PathSnap Screenshot 5](Screenshots/ml\_5.png)



