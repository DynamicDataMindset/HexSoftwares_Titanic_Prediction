# 🚢 Titanic Survival Prediction

<div align="center">

**Final Portfolio Project | Hex Softwares Data Science Internship Program**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)](https://scikit-learn.org/)
[![Random Forest](https://img.shields.io/badge/Model-Random%20Forest-green.svg)](https://scikit-learn.org/stable/modules/ensemble.html#forest)

*An End-to-End Machine Learning Solution*

[📧 Email](mailto:bonifaceramushu28@gmail.com) • [💼 LinkedIn](https://www.linkedin.com/in/b-ramushu)

</div>

---

## 🎯 Project Overview

This project represents the **third and final deliverable** for the Hex Softwares Data Science Internship Program, demonstrating complete mastery of the end-to-end data science workflow. The challenge: predict passenger survival on the Titanic using advanced ML techniques.

### 🏆 Mission

Build a production-ready Random Forest Classifier capable of predicting binary survival outcomes (`Survived`: 0=Died, 1=Survived) for 418 unknown test passengers, leveraging:
- 🎫 Passenger demographics and ticket class
- 👨‍👩‍👧‍👦 Family structure and relationships
- 🏛️ Social status indicators
- 🚪 Cabin location data

---

## 🛠️ Technology Stack

```
Language       → Python 3.8+
Data Wrangling → Pandas, NumPy
Visualization  → Matplotlib, Seaborn
ML Framework   → Scikit-learn (RandomForestClassifier, accuracy_score)
Platform       → Jupyter Notebook / Google Colab
```

---

## 🔬 Data Science Pipeline

### 🎨 1. Feature Engineering
*Transforming raw data into predictive intelligence*

**Domain-Driven Feature Creation:**

- **Title Extraction** 
  - Mined honorifics from `Name` column (`Mr.`, `Mrs.`, `Master.`, `Miss.`, `Rev.`, etc.)
  - Purpose: Capture social hierarchy and improve age imputation accuracy
  - Impact: Titles revealed survival patterns tied to gender and social class

- **FamilySize Creation**
  - Formula: `SibSp` (Siblings/Spouses) + `Parch` (Parents/Children) + 1 (self)
  - Purpose: Quantify family connections aboard ship
  - Insight: Family size correlates with survival (solo travelers vs. large families)

- **Deck Extraction**
  - Parsed first letter from `Cabin` column (A, B, C, D, E, F, G)
  - Missing values coded as 'U' (Unknown)
  - Purpose: Capture proximity to lifeboats and deck-level survival differences

### 🔧 2. Intelligent Imputation
*Context-aware missing data handling*

**Age Imputation Strategy:**
- Used **median age per Title group** instead of global median
- Example: Missing age for "Master" → filled with median age of all "Master" passengers
- Result: Preserves demographic patterns and improves model accuracy

**Categorical Encoding:**
- Applied **One-Hot Encoding** via `pd.get_dummies()` to:
  - `Sex` (Male/Female)
  - `Embarked` (C/Q/S ports)
  - `Deck` (A-G, U)
  - `Title` (Mr., Mrs., Miss., Master., etc.)
  - `Pclass` (1st, 2nd, 3rd class)
- Prevents model from misinterpreting categorical variables as ordered numbers

### 🤖 3. Model Architecture

**Random Forest Classifier**
```python
RandomForestClassifier(n_estimators=100, random_state=42)
```

**Why Random Forest?**
- 🌳 Ensemble of 100 decision trees (wisdom of crowds)
- 💪 Handles non-linear relationships and feature interactions
- 🎯 Resistant to overfitting through bagging
- ⚡ Strong baseline performance with minimal hyperparameter tuning

---

## 📊 Results & Performance Metrics

### 🎓 Training Performance

| Metric | Score | Interpretation |
|:-------|:------|:--------------|
| **Training Accuracy** | **87.09%** | Model correctly classified survival outcomes for 87.09% of known training passengers |

*This demonstrates strong pattern recognition while maintaining generalization capacity.*

---

### 🔮 Test Set Predictions (Final Deliverable)

**Dataset:** 418 passengers (IDs 892–1309)  
**File:** `titanic_submission_rf_v1.csv`

| Prediction | Count | Percentage | Visual |
|:-----------|:------|:-----------|:-------|
| **Died (0)** | **267** | **64%** | ████████████████████████████████ |
| **Survived (1)** | **151** | **36%** | ██████████████████ |
| **Total** | **418** | **100%** | ██████████████████████████████████████████████████ |

**Key Insights from Predictions:**
- 64% mortality rate aligns with historical Titanic disaster statistics
- Reflects learned patterns: higher survival for women, children, and upper-class passengers
- Predictions ready for Kaggle submission and evaluation

---

## 🚀 Reproduction Guide

### Prerequisites
- Python 3.8+
- Jupyter Notebook or Google Colab
- Kaggle account (for dataset download)

### Step-by-Step Setup

**1. Download Dataset**
```
Source: kaggle.com/c/titanic/data
Files: train.csv, test.csv
```

**2. Environment Setup**
```bash
# Install required libraries
pip install pandas numpy matplotlib seaborn scikit-learn
```

**3. Run the Notebook**
```python
# Load and execute all cells in sequence
# Output: titanic_submission_rf_v1.csv
```

**4. Verify Output**
```python
import pandas as pd
submission = pd.read_csv('titanic_submission_rf_v1.csv')
print(f"Total predictions: {len(submission)}")
print(submission['Survived'].value_counts())
```

---

## 📁 Project Structure

```
HexSoftwares_Titanic_Survival_Prediction/
│
├── 📊 titanic_analysis.ipynb           # Complete ML pipeline notebook
├── 📄 titanic_submission_rf_v1.csv     # Final predictions (418 rows)
├── 📖 README.md                        # Project documentation
└── 📋 requirements.txt                 # Python dependencies
```

---

## 🎓 Key Learning Outcomes

**Technical Skills Demonstrated:**
- ✅ End-to-end ML pipeline development
- ✅ Advanced feature engineering techniques
- ✅ Context-aware data imputation strategies
- ✅ Ensemble model implementation and evaluation
- ✅ Production-ready prediction generation

**Domain Insights Discovered:**
- 👩 Gender was the strongest predictor (women-and-children-first protocol)
- 💎 Passenger class significantly impacted survival (1st > 2nd > 3rd)
- 👨‍👩‍👧 Family size showed non-linear relationship with survival
- 🚪 Cabin deck location correlated with lifeboat accessibility

---

## 📧 Contact & Links

<div align="center">

**Boniface Ramushu**

[![Email](https://img.shields.io/badge/Email-bonifaceramushu28%40gmail.com-red?style=for-the-badge&logo=gmail&logoColor=white)](mailto:bonifaceramushu28@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-b--ramushu-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/b-ramushu)

</div>

---

## 🙏 Acknowledgments

- **Hex Softwares** – For the comprehensive Data Science Internship Program
- **Kaggle** – For hosting the Titanic ML Competition dataset
- **Scikit-learn Community** – For exceptional open-source ML tools

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">

**⭐ Final Project Deliverable for Hex Softwares Data Science Internship**

*Built with 💙 using Python, Pandas, and Random Forest*

</div>
