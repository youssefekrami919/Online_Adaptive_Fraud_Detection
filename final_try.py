import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Page config
st.set_page_config(page_title="Adaptive Fraud Detection", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    h1 {color: #2c3e50; text-align: center;}
    h2, h3 {color: #34495e;}
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center;
        margin: 10px 0;
    }
    .fraud {color: #e74c3c; font-weight: bold;}
    .safe {color: #27ae60; font-weight: bold;}
    .stButton > button {background-color: #3498db; color: white; border-radius: 8px;}
    .stButton > button:hover {background-color: #2980b9;}
</style>
""", unsafe_allow_html=True)

st.title("🔍 Adaptive Financial Fraud Detection")
st.markdown("<p style='text-align:center; color:#7f8c8d;'>Online Learning Model that Adapts with New Data in Real-Time</p>", unsafe_allow_html=True)

# Project info
with st.expander("📋 Project Information"):
    st.markdown("""
    **FINAL PROJECT GSB-TRAINING 2025 [GSB X DIBIMBING.ID]**  
    Team: Muhammad Khayruhanif | Panji Elang Permanajati | Izzat Khalil Yassin | Aisyah Syakira Aulia  
    Dataset: PaySim Synthetic Financial Dataset (Kaggle)
    """)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\dell\Desktop\Semester_7\Advanced_ML\project\archive (6)\Synthetic_Financial_datasets_log.csv")

df = load_data()

# Feature Engineering
df_fe = df.drop(columns=['nameOrig', 'nameDest'])
X = df_fe.drop(columns=['isFraud'])
y = df_fe['isFraud']

# One-hot encoding
X = pd.get_dummies(X, columns=['type'], prefix='type')

# Train-test split (stratified to keep imbalance ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Compute fixed balanced class weights from training data
classes = np.array([0, 1])
class_weights_array = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights_array))

st.info(f"Computed balanced class weights: Non-Fraud (0): {class_weight_dict[0]:.4f}, Fraud (1): {class_weight_dict[1]:.2f}")

# Initialize model in session state
if 'model' not in st.session_state:
    st.session_state.model = SGDClassifier(
        loss='log_loss',               # for probability output
        random_state=42,
        warm_start=True
    )
    
    # Initial incremental training with sample_weight for imbalance
    with st.spinner("Training initial model incrementally (handling imbalance with sample weights)..."):
        batch_size = 10000
        for i in range(0, len(X_train), batch_size):
            end = min(i + batch_size, len(X_train))
            X_batch = X_train.iloc[i:end]
            y_batch = y_train.iloc[i:end]
            
            # Compute sample_weight for this batch
            sample_weight_batch = np.where(y_batch == 0, class_weight_dict[0], class_weight_dict[1])
            
            if i == 0:
                # First batch: provide classes
                st.session_state.model.partial_fit(X_batch, y_batch, classes=classes, sample_weight=sample_weight_batch)
            else:
                st.session_state.model.partial_fit(X_batch, y_batch, sample_weight=sample_weight_batch)
    
    st.success("Initial model trained with balanced weighting!")

# Evaluation function
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    return acc, recall, auc, y_pred, y_proba, cm

# Current performance (before any new update)
old_acc, old_recall, old_auc, _, _, old_cm = evaluate_model(st.session_state.model, X_test, y_test)

st.header("📊 Current Model Performance (Before New Data)")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div class='metric-card'><h3>Accuracy</h3><h2>{old_acc:.4f}</h2></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-card'><h3>Recall (Fraud)</h3><h2>{old_recall:.4f}</h2></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-card'><h3>ROC AUC</h3><h2>{old_auc:.4f}</h2></div>", unsafe_allow_html=True)

# Interactive new transaction
st.header("➕ Add New Transaction & Adapt Model")
col1, col2 = st.columns(2)
with col1:
    step = st.number_input('Step (hour)', min_value=1, value=1)
    amount = st.number_input('Amount', min_value=0.0, value=1000.0)
    oldbalanceOrg = st.number_input('Old Balance Origin', min_value=0.0, value=10000.0)
    newbalanceOrig = st.number_input('New Balance Origin', min_value=0.0, value=9000.0)
with col2:
    oldbalanceDest = st.number_input('Old Balance Destination', min_value=0.0, value=0.0)
    newbalanceDest = st.number_input('New Balance Destination', min_value=0.0, value=1000.0)
    isFlaggedFraud = st.selectbox('Is Flagged Fraud?', [0, 1])
    transaction_type = st.selectbox('Transaction Type', ['CASH_IN', 'PAYMENT', 'CASH_OUT', 'TRANSFER', 'DEBIT'])

# Prepare new data point
type_cols = [col for col in X.columns if col.startswith('type_')]
new_row = {
    'step': step, 'amount': amount, 'oldbalanceOrg': oldbalanceOrg,
    'newbalanceOrig': newbalanceOrig, 'oldbalanceDest': oldbalanceDest,
    'newbalanceDest': newbalanceDest, 'isFlaggedFraud': isFlaggedFraud
}
for col in type_cols:
    new_row[col] = 1 if col == f'type_{transaction_type}' else 0

new_df = pd.DataFrame([new_row])[X.columns]

# Predict button
if st.button("🔮 Predict Fraud on New Transaction"):
    pred = st.session_state.model.predict(new_df)[0]
    proba = st.session_state.model.predict_proba(new_df)[0][1]
    
    st.markdown("### Prediction Result")
    if pred == 1:
        st.markdown(f"<h2 class='fraud'>⚠️ FRAUD DETECTED</h2>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h2 class='safe'>✅ Transaction is Safe</h2>", unsafe_allow_html=True)
    
    st.markdown(f"<h3>Fraud Probability: <b>{proba:.2%}</b></h3>", unsafe_allow_html=True)

# Update model
st.markdown("---")
st.subheader("🧠 Update Model with True Label (Adaptive Learning)")
true_label = st.radio("What was the TRUE label of this transaction?", options=[0, 1], format_func=lambda x: "Not Fraud" if x == 0 else "Fraud")

if st.button("Update Model with This New Labeled Data"):
    with st.spinner("Updating model adaptively..."):
        # Compute sample_weight for the single new point
        sample_weight_new = np.array([class_weight_dict[true_label]])
        
        st.session_state.model.partial_fit(new_df, np.array([true_label]), sample_weight=sample_weight_new)
    
    # New metrics after update
    new_acc, new_recall, new_auc, _, _, new_cm = evaluate_model(st.session_state.model, X_test, y_test)
    
    st.success("Model successfully updated with new labeled data!")
    
    # Comparison
    st.header("📈 Performance Comparison: Before vs After Adaptation")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>Accuracy</h3>
            <p>Old: <b>{old_acc:.4f}</b></p>
            <p>New: <b>{new_acc:.4f}</b></p>
            <p style='color:{"green" if new_acc >= old_acc else "red"};'>
                {'↑' if new_acc >= old_acc else '↓'} {abs(new_acc - old_acc):.6f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>Recall (Fraud)</h3>
            <p>Old: <b>{old_recall:.4f}</b></p>
            <p>New: <b>{new_recall:.4f}</b></p>
            <p style='color:{"green" if new_recall >= old_recall else "red"};'>
                {'↑' if new_recall >= old_recall else '↓'} {abs(new_recall - old_recall):.6f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>ROC AUC</h3>
            <p>Old: <b>{old_auc:.4f}</b></p>
            <p>New: <b>{new_auc:.4f}</b></p>
            <p style='color:{"green" if new_auc >= old_auc else "red"};'>
                {'↑' if new_auc >= old_auc else '↓'} {abs(new_auc - old_auc):.6f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Confusion matrices
    st.subheader("Confusion Matrices: Before vs After")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(old_cm, annot=True, fmt='d', cmap='Oranges', ax=axes[0])
    axes[0].set_title('Before Update')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    sns.heatmap(new_cm, annot=True, fmt='d', cmap='Oranges', ax=axes[1])
    axes[1].set_title('After Update')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    st.pyplot(fig)

st.info("💡 The model now correctly handles class imbalance during online learning! Add more labeled transactions (especially fraud cases) to see improvements over time.")