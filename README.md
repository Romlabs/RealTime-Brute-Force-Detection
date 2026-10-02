# 🛡️ Real-Time Brute-Force Detection via Predictive Machine Learning

An interactive **Streamlit** dashboard that detects brute-force attacks
using a Random Forest classifier, SMOTE balancing, and Grid/Randomized
hyperparameter search.

## Features

- 📥 Kaggle auto-download or CSV upload
- 🧹 Automated cleaning (label encoding, dedup)
- ⚖️ SMOTE visualization
- 🌲 Train Random Forest with tunable parameters
- 🔧 Grid Search + 🎲 Randomized Search
- 📊 Feature importance charts
- 🔮 Upload new data and download predictions

## Local Setup

```bash
git clone <your-repo-url>
cd brute-force-detection
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
