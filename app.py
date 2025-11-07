
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import base64

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Insurance.csv")
    df['SUM_ASSURED'] = df['SUM_ASSURED'].replace(',', '', regex=True).astype(float)
    return df

df = load_data()

st.set_page_config(page_title="Insurance Analytics Dashboard", layout="wide")
st.title("🏢 Insurance Policy Analytics Dashboard")

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 Model Comparison", "📂 Upload & Predict"])

# ---------------------- TAB 1 ----------------------
with tab1:
    st.header("Insurance Policy Insights")
    job_roles = st.multiselect("Select Occupation(s):", options=df["PI_OCCUPATION"].unique(), default=df["PI_OCCUPATION"].unique())
    min_sa, max_sa = float(df["SUM_ASSURED"].min()), float(df["SUM_ASSURED"].max())
    sa_slider = st.slider("Select Sum Assured Range:", min_sa, max_sa, (min_sa, max_sa))

    filtered_df = df[(df["PI_OCCUPATION"].isin(job_roles)) & 
                     (df["SUM_ASSURED"].between(sa_slider[0], sa_slider[1]))]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Policy Status Distribution")
        fig1 = px.pie(filtered_df, names='POLICY_STATUS', title="Policy Status Distribution")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Average Sum Assured by Occupation")
        fig2 = px.bar(filtered_df.groupby('PI_OCCUPATION', as_index=False)['SUM_ASSURED'].mean(),
                      x='PI_OCCUPATION', y='SUM_ASSURED', color='PI_OCCUPATION', title="Average Sum Assured by Occupation")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Sum Assured by Payment Mode")
        fig3 = px.line(filtered_df.groupby('PAYMENT_MODE', as_index=False)['SUM_ASSURED'].mean(),
                       x='PAYMENT_MODE', y='SUM_ASSURED', title="Average Sum Assured by Payment Mode")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("Age vs Sum Assured (Gender-wise)")
        fig4 = px.box(filtered_df, x='PI_GENDER', y='SUM_ASSURED', color='PI_GENDER', title="Age vs Sum Assured by Gender")
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Correlation Heatmap")
    numeric_df = filtered_df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        fig5 = px.imshow(numeric_df.corr(), text_auto=True, title="Correlation Heatmap")
        st.plotly_chart(fig5, use_container_width=True)

# ---------------------- TAB 2 ----------------------
with tab2:
    st.header("Model Comparison")

    st.write("Applying Decision Tree, Random Forest, and Gradient Boosting to predict POLICY_STATUS.")
    df_model = df.copy()
    df_model = df_model.dropna(subset=['POLICY_STATUS'])
    label_enc = LabelEncoder()
    for col in df_model.select_dtypes(include='object').columns:
        df_model[col] = label_enc.fit_transform(df_model[col].astype(str))
    X = df_model.drop("POLICY_STATUS", axis=1)
    y = df_model["POLICY_STATUS"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append({"Model": name, "Accuracy": acc})
    result_df = pd.DataFrame(results)

    st.dataframe(result_df)

# ---------------------- TAB 3 ----------------------
with tab3:
    st.header("Upload New Data for Prediction")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        new_df = pd.read_csv(uploaded_file)
        df_copy = df.copy()
        label_enc = LabelEncoder()
        for col in df_copy.select_dtypes(include='object').columns:
            df_copy[col] = label_enc.fit_transform(df_copy[col].astype(str))
        model = RandomForestClassifier()
        X = df_copy.drop("POLICY_STATUS", axis=1)
        y = df_copy["POLICY_STATUS"]
        model.fit(X, y)

        new_df_encoded = new_df.copy()
        for col in new_df_encoded.select_dtypes(include='object').columns:
            if col in label_enc.classes_:
                new_df_encoded[col] = label_enc.transform(new_df_encoded[col].astype(str))
            else:
                new_df_encoded[col] = label_enc.fit_transform(new_df_encoded[col].astype(str))

        preds = model.predict(new_df_encoded)
        new_df["Predicted_POLICY_STATUS"] = preds

        st.success("Prediction Completed!")
        st.dataframe(new_df.head())

        csv = new_df.to_csv(index=False).encode()
        b64 = base64.b64encode(csv).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="Predicted_Insurance.csv">Download Predicted File</a>'
        st.markdown(href, unsafe_allow_html=True)
