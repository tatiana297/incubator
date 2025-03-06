import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random, string
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from cryptography.fernet import Fernet

# =============================================================================
# Page Config & CSS
# =============================================================================

st.set_page_config(
    page_title="Innovative Data Sharing",
    page_icon=":bank:",
    layout="wide"
)

custom_css = """
<style>
/* Main container width */
.block-container {
    max-width: 1200px;
}
/* Background color for main area */
.stApp {
    background-color: #F9FBFD;
}
/* Big button style for the Start button */
.big-button button {
    font-size: 20px;
    padding: 15px 30px;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# =============================================================================
# Utility Functions
# =============================================================================

def add_audit_log(event):
    """Simulate a blockchain audit log entry."""
    if "audit_log" not in st.session_state:
        st.session_state.audit_log = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.audit_log.append(f"{timestamp}: {event}")

def generate_pe_asset_data():
    """
    Generate Private Equity asset data with realistic distributions.
    """
    np.random.seed(42)
    n = 100
    assets = [f'PE Asset {i+1}' for i in range(n)]
    
    expected_return = np.clip(np.random.normal(loc=15, scale=4, size=n), 5, 25)
    volatility = np.clip(np.random.normal(loc=25, scale=5, size=n), 10, 40)
    irr = np.clip(np.random.normal(loc=20, scale=4, size=n), 10, 30)
    moic = np.clip(np.random.normal(loc=2.5, scale=0.7, size=n), 1, 4)
    
    stages = np.random.choice(['Seed', 'Early', 'Growth', 'Mature'], n, p=[0.2, 0.3, 0.3, 0.2])
    
    # Risk rating formula
    risk_rating = (volatility / 40) * 3 + ((4 - moic) / 3) * 2
    
    data = pd.DataFrame({
        'Asset': assets,
        'Expected Return (%)': expected_return,
        'Volatility (%)': volatility,
        'IRR (%)': irr,
        'MOIC': moic,
        'Investment Stage': stages,
        'Risk Rating': risk_rating
    })
    return data

def apply_differential_privacy(data, epsilon):
    """
    Apply differential privacy by adding Laplacian noise to numeric columns.
    """
    noisy_data = data.copy()
    for col in ['Expected Return (%)', 'Volatility (%)', 'IRR (%)', 'MOIC', 'Risk Rating']:
        noise = np.random.laplace(loc=0, scale=1/epsilon, size=noisy_data.shape[0])
        noisy_data[col] = noisy_data[col] + noise
    return noisy_data

def run_clustering_auto(data, cluster_range=(2, 6)):
    """
    Automatically determine the optimal number of clusters using silhouette score.
    """
    features = data[['Expected Return (%)', 'Volatility (%)']]
    best_score = -1
    best_k = cluster_range[0]
    best_kmeans = None

    for k in range(cluster_range[0], cluster_range[1] + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(features)
        score = silhouette_score(features, labels)
        if score > best_score:
            best_score = score
            best_k = k
            best_kmeans = kmeans
    
    data['Cluster'] = best_kmeans.labels_.astype(str)
    return data, best_kmeans, best_k

def train_pe_model(data, model_type="Random Forest"):
    """
    Train a predictive model to forecast IRR (%) from Volatility, Risk Rating, and MOIC.
    """
    X = data[['Volatility (%)', 'Risk Rating', 'MOIC']]
    y = data['IRR (%)']
    if model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    return model, y_pred

def plot_feature_importance(model, model_type="Random Forest"):
    """Plot feature importances."""
    features = ['Volatility (%)', 'Risk Rating', 'MOIC']
    importances = model.feature_importances_
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(features, importances, color='#FFC300')
    ax.set_title(f"{model_type} Feature Importance", fontsize=14)
    ax.set_ylabel("Importance", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

def plot_clustering(data, kmeans_model, n_clusters):
    """Scatter plot for asset clustering."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#FF6F61', '#6B5B95', '#88B04B', '#F7CAC9', '#92A8D1', '#955251']
    for i in range(n_clusters):
        subset = data[data['Cluster'] == str(i)]
        ax.scatter(
            subset["Expected Return (%)"], 
            subset["Volatility (%)"], 
            color=colors[i % len(colors)], 
            label=f"Cluster {i}", 
            s=50, alpha=0.7
        )
        if hasattr(kmeans_model, "cluster_centers_"):
            ax.scatter(
                kmeans_model.cluster_centers_[i, 0], 
                kmeans_model.cluster_centers_[i, 1],
                color='black', marker='X', s=150
            )
    ax.set_xlabel("Expected Return (%)", fontsize=12)
    ax.set_ylabel("Volatility (%)", fontsize=12)
    ax.set_title("PE Asset Clustering", fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

def plot_average_risk(data):
    """Bar chart of average risk rating per cluster."""
    avg_risk = data.groupby('Cluster')['Risk Rating'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(avg_risk['Cluster'], avg_risk['Risk Rating'], color='#74C2E1')
    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_ylabel("Average Risk Rating", fontsize=12)
    ax.set_title("Average Risk per Cluster", fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', 
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points", 
                    ha='center', 
                    fontsize=10)
    plt.tight_layout()
    return fig

def plot_irr_vs_moic(data):
    """Scatter plot showing IRR vs. MOIC."""
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(data['IRR (%)'], data['MOIC'], color='#9B59B6', alpha=0.6)
    ax.set_xlabel("IRR (%)", fontsize=12)
    ax.set_ylabel("MOIC", fontsize=12)
    ax.set_title("IRR vs MOIC", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

def plot_investment_stage_distribution(data):
    """Pie chart of the distribution of investment stages."""
    stage_counts = data['Investment Stage'].value_counts()
    fig, ax = plt.subplots(figsize=(6,6))
    ax.pie(stage_counts, labels=stage_counts.index, autopct='%1.1f%%', startangle=140)
    ax.set_title("Investment Stage Distribution", fontsize=14)
    plt.tight_layout()
    return fig

def plot_prediction_scatter(data, y_pred):
    """Scatter plot: Actual vs. Predicted IRR."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(data["Volatility (%)"], data["IRR (%)"], color='#27AE60', label='Actual IRR', alpha=0.6)
    ax.scatter(data["Volatility (%)"], y_pred, color='#E74C3C', label='Predicted IRR', alpha=0.6)
    ax.set_xlabel("Volatility (%)", fontsize=12)
    ax.set_ylabel("IRR (%)", fontsize=12)
    ax.set_title("Actual vs Predicted IRR", fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

def plot_residuals(data, y_pred):
    """Plot residuals (difference between actual and predicted IRR)."""
    fig, ax = plt.subplots(figsize=(8,5))
    residuals = data['IRR (%)'] - y_pred
    ax.scatter(y_pred, residuals, color='#34495E', alpha=0.6)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel("Predicted IRR (%)", fontsize=12)
    ax.set_ylabel("Residuals", fontsize=12)
    ax.set_title("Residual Plot", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(data):
    """Plot a heatmap of feature correlations."""
    fig, ax = plt.subplots(figsize=(6,5))
    corr = data[['Expected Return (%)', 'Volatility (%)', 'IRR (%)', 'MOIC', 'Risk Rating']].corr()
    cax = ax.matshow(corr, cmap='coolwarm')
    fig.colorbar(cax)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    ax.set_title("Correlation Heatmap", pad=20)
    plt.tight_layout()
    return fig

def encryption_demo(text):
    """Demonstrate encryption and decryption for sensitive data."""
    key = Fernet.generate_key()
    f = Fernet(key)
    token = f.encrypt(text.encode())
    decrypted = f.decrypt(token).decode()
    return key.decode(), token.decode(), decrypted

# =============================================================================
# Navigation & State Management
# =============================================================================

if "page" not in st.session_state:
    st.session_state.page = "title"
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []
if "verified" not in st.session_state:
    st.session_state.verified = False

def go_to(page_name):
    st.session_state.page = page_name
    add_audit_log(f"Navigated to {page_name} page")
    if page_name == "captcha":
        st.session_state.verified = False

def render_top_nav():
    """Render a top navigation bar with buttons for the main pages."""
    cols = st.columns(5)
    
    if cols[0].button("Onboarding"):
         go_to("onboarding")
    if cols[1].button("Analysis"):
         go_to("analysis")
    if cols[2].button("Review"):
         go_to("review")
    if cols[3].button("Code Overview"):
         go_to("code_overview")
    if cols[4].button("Conclusion"):
         go_to("conclusion")

# =============================================================================
# Pages
# =============================================================================

def title_page():
    logo_path = logo_path = "logo.jpg"
    col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 1])
    with col3:
        st.image(logo_path, width=200)
    st.markdown("<h1 style='text-align: center; color: #2C3E50;'>Innovative Data Sharing Platform</h1>", unsafe_allow_html=True)
    
    # La parte "Hello and welcome" è stata rimossa dalla Title Page.
    st.markdown("---")
    st.markdown('<div class="big-button" style="text-align: center;">', unsafe_allow_html=True)
    st.write(" ")
    col1, col2, col3, col4, col5, col6 = st.columns([1, 2, 2, 3, 3, 1])
    with col4:
        # Il pulsante ora porta alla Start Page
        if st.button("Start Now"):
            go_to("start_page")
    st.markdown('</div>', unsafe_allow_html=True)

def start_page():
    st.write("Before starting, please scan the QR Code to enjoy the experience with us.")
    logo_path = "/Users/aurora.ambrosoni/Desktop/logo.jpg"
    col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 1])
    with col3:
        st.image(logo_path, width=200)
    
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    #####to put QR Code image !!!
    col1, col2, col3, col4, col5, col6 = st.columns([1, 2, 2, 3, 2, 1])
    with col4:
        # Il pulsante ora porta alla Start Page
        if st.button("Start Next"):
            go_to("hub")
    st.markdown('</div>', unsafe_allow_html=True)
def hub_page():
    # Non viene renderizzata la top nav qui
    st.markdown(
        """
        <h2 style='text-align:center; color:#34495E;'>Hello and welcome!</h2>
        <p style='text-align:center; font-size:16px; color:#555;'>
        Welcome to our innovative platform—redefining how banks share data with institutional clients through secure, transparent methods and advanced machine learning for personalized investment insights.
        </p>
        """,
        unsafe_allow_html=True
    )
    st.write(" ")
    st.markdown("### What we provide:")
    st.markdown(
        """
        <div style="text-align: center;">
            <ul style="list-style-type: disc; display: inline-block; text-align: left;">
                <li><strong>Safety:</strong> We prioritize robust security measures including encryption, CAPTCHA verification, and secure data sharing protocols to protect your sensitive information.</li>
                <li><strong>Personalization:</strong> Our platform is tailored to your unique investment goals and risk appetite, ensuring a customized experience that meets your specific needs.</li>
                <li><strong>Advanced Machine Learning:</strong> By leveraging state-of-the-art machine learning algorithms, we deliver predictive insights and data-driven strategies for optimal decision-making.</li>
                <li><strong>Transparency:</strong> We uphold complete transparency with clear audit logs and open data practices, building trust and accountability in every interaction.</li>
            </ul>
        </div>
        """, 
        unsafe_allow_html=True
    )
    # Pulsante Next che porta alla Captcha Page
    st.write(" ")
    st.write(" ")
    st.write(" ")
    col1, col2, col3, col4, col5, col6 = st.columns([1, 2, 2, 3, 2, 1])
    with col4:
        if st.button("Next"):
            go_to("captcha")

def captcha_page():
    st.subheader("CAPTCHA Verification")
    if "captcha" not in st.session_state:
         st.session_state.captcha = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
    st.write("Please type the text below to verify you are human:")
    st.markdown(f"<h2 style='color:#E67E22;'>{st.session_state.captcha}</h2>", unsafe_allow_html=True)
    user_input = st.text_input("Enter CAPTCHA", key="captcha_input")
    
    # Se la verifica ha successo, l'utente viene indirizzato all'Onboarding.
    st.write(" ")
    st.write(" ")
    st.write(" ")
    col1, col2, col3, col4, col5, col6 = st.columns([1, 2, 2, 3, 2, 1])
    with col4:
        if st.button("Verify"):
            if user_input == st.session_state.captcha:
                st.success("Verification successful!")
                st.session_state.verified = True
                go_to("onboarding")
            else:
                st.error("Incorrect CAPTCHA. Please try again.")
                st.session_state.captcha = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))

def onboarding_page():
    st.header("Client Onboarding & Personalization")
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            user_name = st.text_input("Enter your name:", value="User")
            risk_appetite = st.slider("Set your risk appetite (0 = Low, 10 = High)", 0, 10, 5)
            investment_goal = st.selectbox("Select your investment goal", ["Aggressive Growth", "Steady Income", "Balanced"])
        with col2:
            time_horizon = st.slider("Investment time horizon (years)", 1, 30, 10)
            stage_preference = st.selectbox("Preferred Investment Stage", ["Seed", "Early", "Growth", "Mature"])
    
    st.session_state.user_name = user_name
    st.session_state.risk_appetite = risk_appetite
    st.session_state.investment_goal = investment_goal
    st.session_state.time_horizon = time_horizon
    st.session_state.stage_preference = stage_preference
    
    st.info(f"Welcome, **{user_name}**! You prefer **{investment_goal}** strategies over a **{time_horizon}-year** horizon.")
    
    data = generate_pe_asset_data()
    st.subheader("PE Asset Overview")
    clustered_data, kmeans_model, optimal_k = run_clustering_auto(data.copy())
    st.write(f"Optimal number of clusters determined: **{optimal_k}**")
    st.pyplot(plot_clustering(clustered_data, kmeans_model, optimal_k))
    st.pyplot(plot_average_risk(clustered_data))
    
    col3, col4 = st.columns(2)
    with col3:
        st.pyplot(plot_irr_vs_moic(data))
    with col4:
        st.pyplot(plot_investment_stage_distribution(data))
    
    if st.button("Next: Advanced Analysis & PE Insights"):
        go_to("analysis")

def analysis_page():
    st.header("Advanced Analysis & PE Insights")
    st.write("Simulate market conditions and forecast IRR using our predictive model. This analysis demonstrates how data sharing can drive informed decision-making for institutional clients.")
    
    model_choice = "Random Forest"
    
    if "risk_appetite" not in st.session_state:
        st.session_state.risk_appetite = 5
    
    data = generate_pe_asset_data()
    model, y_pred = train_pe_model(data, model_type=model_choice)
    
    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_prediction_scatter(data, y_pred))
    with col2:
        st.pyplot(plot_feature_importance(model, model_type=model_choice))
    
    st.subheader("Residual Analysis")
    st.pyplot(plot_residuals(data, y_pred))
    
    st.subheader("Feature Correlation")
    st.pyplot(plot_correlation_heatmap(data))
    
    if st.button("Next: Review"):
        go_to("review")

def review_page():
    st.header("Review: Stress Testing & Insights")
    st.write("In this section, you can perform a stress test of the model and view the predicted IRR in real time.")
    
    # Generate data and train the model (always using Random Forest)
    data = generate_pe_asset_data()
    model, _ = train_pe_model(data, model_type="Random Forest")
    
    st.subheader("Stress Testing & Personalized Prediction")
    user_vol = st.slider("Set Base Volatility (%) for Prediction", 10.0, 40.0, 25.0, step=0.5, key="review_vol")
    shock_factor = st.slider("Market Shock Factor", 1.0, 3.0, 1.0, step=0.1)
    adjusted_vol = user_vol * shock_factor
    risk_rating_input = st.session_state.risk_appetite * 2  # Mapping for Private Equity
    moic_input = 1 + (10 - st.session_state.risk_appetite) / 10 * 3  # Scale from 1 to 4
    input_features = np.array([[adjusted_vol, risk_rating_input, moic_input]])
    predicted_irr = model.predict(input_features)[0]
    
    st.write(f"At an adjusted volatility of **{adjusted_vol:.1f}%**, risk input **{risk_rating_input:.1f}**, and MOIC **{moic_input:.1f}**,")
    st.success(f"the predicted IRR is **{predicted_irr:.2f}%**")
    
    st.markdown("<h3 style='color:#E67E22; text-align:center;'>The numbers don't lie: invest smartly for a brighter tomorrow!</h3>", unsafe_allow_html=True)
    
    if st.button("Next: Code Overview"):
        go_to("code_overview")

def code_overview_page():
    st.header("Code Overview :page_facing_up:")
    st.write("Key code snippets driving the functionality of our platform:")
    st.subheader("1. Data Generation for Private Equity Assets")
    code_snippet1 = '''def generate_pe_asset_data():
    np.random.seed(42)
    n = 100
    ...
    return pd.DataFrame({...})'''
    st.code(code_snippet1, language='python')
    
    st.subheader("2. Training the Predictive Model")
    code_snippet2 = '''def train_pe_model(data, model_type="Random Forest"):
    X = data[['Volatility (%)', 'Risk Rating', 'MOIC']]
    y = data['IRR (%)']
    ...
    return model, y_pred'''
    st.code(code_snippet2, language='python')
    
    if st.button("Next: Conclusion & Transparency"):
        go_to("conclusion")

def conclusion_page():
    st.header("Conclusion & Transparency Dashboard")
    st.write("This platform demonstrates a secure, transparent, and personalized approach to innovative data sharing between a bank and its institutional clients.")
    st.write("**Key Features:**")
    st.markdown("""
    - **Enhanced Security:** CAPTCHA verification, simulated blockchain audit logs, and robust encryption demo.  
    - **Personalization:** Onboarding inputs tailor the platform to client-specific investment strategies.  
    - **Advanced Analytics:** Clustering, predictive modeling, and stress testing provide deep insights into Private Equity assets.
    """)
    st.subheader("Digital Consent & Encryption Demo")
    consent = st.checkbox("I hereby provide digital consent for data sharing.")
    if consent:
        key, token, decrypted = encryption_demo("Client Consent: Data Sharing Approved")
        st.write(f"**Encryption Key:** {key}")
        st.write(f"**Encrypted Token:** {token}")
        st.write(f"**Decrypted Message:** {decrypted}")
    
    if st.button("View Audit Log"):
        st.subheader("Audit Log (Simulated Blockchain)")
        for entry in st.session_state.audit_log:
            st.write(entry)
    
    if st.button("Restart"):
        go_to("title")
    
    # Aggiungi il tasto Next che porta alla pagina Thank You
    if st.button("Next"):
        go_to("thank_you")


def thank_you_page():
    # Mostra una scritta gigante Thank You
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.markdown("<h1 style='text-align: center; font-size: 64px;'>THANK YOU!</h1>", unsafe_allow_html=True)


# =============================================================================
# Main App Logic
# =============================================================================

def main():
    # Non renderizzare la top nav nelle pagine: title, start_page e hub
    if st.session_state.page not in ["title", "start_page", "hub", "captcha", "thank_you"]:
         render_top_nav()
         
    if st.session_state.page == "title":
         title_page()
    elif st.session_state.page == "start_page":
         start_page()
    elif st.session_state.page == "captcha":
         captcha_page()
    elif st.session_state.page == "hub":
         hub_page()
    elif st.session_state.page == "onboarding":
         onboarding_page()
    elif st.session_state.page == "analysis":
         analysis_page()
    elif st.session_state.page == "review":
         review_page()
    elif st.session_state.page == "code_overview":
         code_overview_page()
    elif st.session_state.page == "conclusion":
         conclusion_page()
    elif st.session_state.page == "thank_you":
        thank_you_page()

if __name__ == "__main__":
    main()
