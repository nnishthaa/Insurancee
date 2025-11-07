# 🏢 Insurance Policy Analytics Dashboard

This Streamlit Cloud application provides an **interactive data analytics dashboard** for insurance policy analysis and prediction.  
It enables visualization of policy insights, model-based predictions, and uploading of new data for automated classification.

---

## 🚀 Features

### **1. Interactive Analytics Dashboard**
- View 5 insightful and complex charts:
  - Pie chart: Policy Status Distribution  
  - Bar chart: Average Sum Assured by Occupation  
  - Line chart: Average Sum Assured by Payment Mode  
  - Box plot: Age vs Sum Assured by Gender  
  - Heatmap: Correlation among numeric variables  
- Filters available:
  - **Multi-select:** Occupation (`PI_OCCUPATION`)
  - **Slider:** Sum Assured (`SUM_ASSURED`)

---

### **2. Machine Learning Model Comparison**
- Compare predictive performance of three models:
  - Decision Tree  
  - Random Forest  
  - Gradient Boosting  
- Shows model accuracy metrics in a clean table.

---

### **3. Upload and Predict**
- Upload a new CSV dataset.
- Model automatically predicts the **Policy Status (Approved / Repudiate)**.
- Download the resulting CSV with a new prediction column for ready reference.

---

## 🧠 Technologies Used
- **Python 3**
- **Streamlit** for dashboard UI  
- **Pandas / Numpy** for data processing  
- **Plotly** for interactive visualization  
- **Scikit-learn** for machine learning models  

---

## 📦 Files in Repository
| File | Description |
|------|--------------|
| `app.py` | Main Streamlit app file |
| `Insurance.csv` | Source dataset used in analysis |
| `requirements.txt` | Python dependencies |
| `README.md` | Documentation for setup and usage |

---

## 🛠️ How to Run Locally

### Step 1: Clone Repository
```bash
git clone https://github.com/your-username/insurance-dashboard.git
cd insurance-dashboard
```

### Step 2: Install Requirements
```bash
pip install -r requirements.txt
```

### Step 3: Run Streamlit App
```bash
streamlit run app.py
```

---

## ☁️ Deploy to Streamlit Cloud
1. Push all files to your GitHub repository.  
2. Go to [Streamlit Cloud](https://share.streamlit.io/).  
3. Select **“New App” → Connect GitHub Repo**.  
4. Choose this repository and set `app.py` as the main file.  
5. Deploy and enjoy your live dashboard!

---




