import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from textwrap import dedent

from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag


def render_markup(markup, **kwargs):
    if kwargs.pop("unsafe_allow_html", False) and "<" in markup:
        return st.html(dedent(markup))
    return st.markdown(dedent(markup), **kwargs)


def render_sidebar_markup(markup, **kwargs):
    if kwargs.pop("unsafe_allow_html", False) and "<" in markup:
        with st.sidebar:
            return st.html(dedent(markup))
    return st.sidebar.markdown(dedent(markup), **kwargs)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

render_markup(
    """
    <style>

    :root {
        --ink: #16212b;
        --muted: #66727c;
        --paper: #f4f1eb;
        --surface: #fffdfa;
        --line: #ded9cf;
        --navy: #183247;
        --teal: #0f766e;
        --amber: #d97706;
        --coral: #c2413b;
        --shadow: 0 18px 45px rgba(31, 42, 51, 0.10);
    }

    .stApp {
        background-color: var(--paper);
        background-image: radial-gradient(#d8d1c5 0.7px, transparent 0.7px);
        background-size: 18px 18px;
        color: var(--ink);
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    .block-container {
        max-width: 1280px;
        padding: 2.75rem 3rem 3.5rem;
    }

    .main-header {
        position: relative;
        overflow: hidden;
        background: var(--navy);
        padding: 2.3rem 2.6rem;
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 6px;
        box-shadow: var(--shadow);
        margin-bottom: 1.75rem;
    }

    .main-header::after {
        content: "";
        position: absolute;
        right: -70px;
        top: -90px;
        width: 280px;
        height: 280px;
        border: 1px solid rgba(245, 158, 11, 0.45);
        border-radius: 50%;
        box-shadow: 0 0 0 18px rgba(245, 158, 11, 0.06), 0 0 0 38px rgba(245, 158, 11, 0.04);
    }

    .main-header h1 {
        color: white;
        font-size: clamp(2rem, 4vw, 3.25rem);
        line-height: 1.05;
        letter-spacing: 0;
        font-weight: 750;
        margin: 0;
        max-width: 760px;
    }

    .main-header p {
        color: #cbd8dc;
        font-size: 1rem;
        line-height: 1.6;
        margin-top: 0.9rem;
        margin-bottom: 0;
        max-width: 650px;
    }

    .section-card {
        background: rgba(255, 253, 250, 0.92);
        padding: 1.7rem 1.9rem;
        border: 1px solid var(--line);
        border-left: 5px solid var(--amber);
        border-radius: 5px;
        box-shadow: 0 10px 25px rgba(31, 42, 51, 0.06);
        margin-bottom: 1.35rem;
    }

    .section-title {
        color: var(--ink);
        font-size: 1.55rem;
        font-weight: 750;
        letter-spacing: 0;
        margin-bottom: 0.35rem;
    }

    .section-description {
        color: var(--muted);
        font-size: 0.95rem;
        line-height: 1.55;
        margin-bottom: 1rem;
    }

    .metric-card {
        background: var(--surface);
        min-height: 108px;
        padding: 1.2rem 1.35rem;
        border: 1px solid var(--line);
        border-radius: 5px;
        box-shadow: 0 8px 18px rgba(31, 42, 51, 0.05);
        text-align: center;
    }

    .metric-title {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .metric-value {
        color: var(--navy);
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 0.35rem;
    }

    [data-testid="stSidebar"] {
        background: var(--navy);
        border-right: 1px solid rgba(255,255,255,0.12);
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.16);
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        border: 1px solid rgba(255,255,255,0.16);
        border-radius: 4px;
        padding: 0.55rem 0.7rem;
        margin-bottom: 0.45rem;
        background: rgba(255,255,255,0.05);
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(245, 158, 11, 0.16);
        border-color: rgba(245, 158, 11, 0.75);
    }

    .stButton > button,
    .stFormSubmitButton > button {
        width: 100%;
        min-height: 3rem;
        border-radius: 4px;
        padding: 0.7rem 1.15rem;
        font-weight: 700;
        border: 1px solid var(--teal);
        background: var(--teal);
        color: white;
        box-shadow: 0 5px 12px rgba(15, 118, 110, 0.2);
        transition: transform 160ms ease, box-shadow 160ms ease, background 160ms ease;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background: #115e59;
        border-color: #115e59;
        transform: translateY(-1px);
        box-shadow: 0 8px 16px rgba(15, 118, 110, 0.28);
    }

    [data-testid="stNumberInput"] input {
        background: var(--surface);
        border: 1px solid #c9c3b8;
        border-radius: 4px !important;
        color: var(--ink);
        min-height: 2.75rem;
    }

    [data-testid="stNumberInput"] input:focus {
        border-color: var(--teal);
        box-shadow: 0 0 0 2px rgba(15, 118, 110, 0.16);
    }

    [data-testid="stNumberInput"] [data-testid="stWidgetLabel"] p,
    [data-testid="stNumberInput"] label p {
        color: var(--navy) !important;
        font-weight: 750 !important;
        opacity: 1 !important;
        letter-spacing: 0.01em;
    }

    .info-box {
        background: #eef7f4;
        border: 1px solid #b9ddd4;
        border-left: 4px solid var(--teal);
        padding: 0.9rem 1rem;
        border-radius: 4px;
        color: #155e58;
        line-height: 1.5;
        margin: 1rem 0 0;
    }

    .footer {
        text-align: center;
        margin-top: 2.5rem;
        padding: 1.3rem;
        border-top: 1px solid var(--line);
        color: var(--muted);
        font-size: 0.78rem;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 4px;
        overflow: hidden;
    }

    @media (max-width: 640px) {
        .block-container {
            padding: 1.35rem 1rem 2.5rem;
        }

        .main-header {
            padding: 1.65rem 1.25rem;
        }

        .main-header h1 {
            font-size: 2rem;
        }

        .section-card {
            padding: 1.25rem;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

render_markup(
    """
    <div class="main-header">

        <h1>📦 Vendor Invoice Intelligence Portal</h1>

        <p>
            AI-Driven Freight Cost Prediction & Invoice Risk Flagging
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

render_sidebar_markup(
    """
    <div style="
        text-align:center;
        padding:15px 5px 25px 5px;
    ">

        <div style="font-size:45px;">🤖</div>

        <h2 style="margin-bottom:5px;">
            Invoice AI
        </h2>

        <p style="
            color:#cbd5e1;
            font-size:13px;
        ">
            Intelligent Finance Analytics
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


render_sidebar_markup("### 🔍 Model Selection")


selected_model = st.sidebar.radio(
    "Choose Prediction Module",
    [
        "Freight Cost Prediction",
        "Invoice Manual Approval Flag"
    ]
)


# ============================================================
# SIDEBAR BUSINESS IMPACT
# ============================================================

st.sidebar.markdown("---")

render_sidebar_markup(
    """
    ### 💼 Business Impact

    **💰 Improved cost forecasting**

    **🚩 Reduced invoice anomalies**

    **⚡ Faster finance operations**

    **📊 Data-driven decisions**

    ---
    
    **Models**
    
    • Freight Regression Model  
    • Invoice Classification Model
    
    ---
    
    **Technology**
    
    Python • Pandas • Scikit-Learn • Streamlit

    """,
    unsafe_allow_html=True
)


# ============================================================
# FREIGHT COST PREDICTION
# ============================================================

if selected_model == "Freight Cost Prediction":

    render_markup(
        """
        <div class="section-card">

            <div class="section-title">
                🚚 Freight Cost Prediction
            </div>

            <div class="section-description">
                Estimate the expected freight cost of a vendor invoice
                using invoice quantity and invoice dollar value.
            </div>

            <div class="info-box">
                💡 <b>Objective:</b>
                Predict freight cost to support budgeting,
                forecasting and vendor negotiations.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # INPUT FORM
    # --------------------------------------------------------

    with st.form("freight_form"):

        col1, col2 = st.columns(2)

        with col1:

            quantity = st.number_input(
                "📦 Quantity",
                min_value=1.0,
                value=1200.0,
                step=1.0
            )

        with col2:

            dollars = st.number_input(
                "💰 Invoice Dollars",
                min_value=1.0,
                value=18500.0,
                step=100.0
            )


        st.markdown("")


        submit_freight = st.form_submit_button(
            "🔮 Predict Freight Cost"
        )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if submit_freight:

        try:

            input_data = {
                "Quantity": [quantity],
                "Dollars": [dollars]
            }


            prediction = predict_freight_cost(
                input_data
            )["Predicted_Freight"]


            predicted_value = float(
                prediction.iloc[0]
            )


            st.success(
                "✅ Prediction completed successfully."
            )


            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            col1, col2, col3 = st.columns(3)


            with col1:

                render_markup(
                    f"""
                    <div class="metric-card">

                        <div class="metric-title">
                            📦 Quantity
                        </div>

                        <div class="metric-value">
                            {quantity:,.0f}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            with col2:

                    render_markup(
                    f"""
                    <div class="metric-card">

                        <div class="metric-title">
                            💵 Invoice Value
                        </div>

                        <div class="metric-value">
                            ${dollars:,.2f}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            with col3:

                    render_markup(
                    f"""
                    <div class="metric-card">

                        <div class="metric-title">
                            🚚 Estimated Freight
                        </div>

                        <div class="metric-value">
                            ${predicted_value:,.2f}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # ------------------------------------------------
            # RESULT TABLE
            # ------------------------------------------------

            st.markdown("### 📊 Prediction Details")


            result_df = pd.DataFrame(
                {
                    "Quantity": [quantity],
                    "Invoice Dollars": [dollars],
                    "Predicted Freight": [predicted_value]
                }
            )


            st.dataframe(
                result_df,
                width="stretch",
                hide_index=True
            )


            # ------------------------------------------------
            # VISUALIZATION
            # ------------------------------------------------

            chart_df = pd.DataFrame(
                {
                    "Category": [
                        "Invoice Value",
                        "Predicted Freight"
                    ],
                    "Amount": [
                        dollars,
                        predicted_value
                    ]
                }
            )


            fig = px.bar(
                chart_df,
                x="Category",
                y="Amount",
                title="Invoice Value vs Predicted Freight",
                text_auto=".2f"
            )


            fig.update_layout(
                template="plotly_white",
                height=400,
                showlegend=False
            )


            st.plotly_chart(
                fig,
                width="stretch"
            )


        except Exception as e:

            st.error(
                f"❌ Prediction failed: {e}"
            )


# ============================================================
# INVOICE FLAG PREDICTION
# ============================================================

else:

    render_markup(
        """
        <div class="section-card">

            <div class="section-title">
                🚩 Invoice Manual Approval Prediction
            </div>

            <div class="section-description">
                Identify invoices that may require manual review
                based on invoice, freight and item-level patterns.
            </div>

            <div class="info-box">
                💡 <b>Objective:</b>
                Predict whether a vendor invoice should be
                <b>flagged for manual approval</b>.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # INPUT FORM
    # --------------------------------------------------------

    with st.form("invoice_flag_form"):

        col1, col2, col3 = st.columns(3)


        # --------------------------------------------
        # COLUMN 1
        # --------------------------------------------

        with col1:

            invoice_quantity = st.number_input(
                "📦 Invoice Quantity",
                min_value=1.0,
                value=50.0,
                step=1.0
            )


            freight = st.number_input(
                "🚚 Freight Cost",
                min_value=0.0,
                value=1.73,
                step=0.01
            )


        # --------------------------------------------
        # COLUMN 2
        # --------------------------------------------

        with col2:

            invoice_dollars = st.number_input(
                "💰 Invoice Dollars",
                min_value=1.0,
                value=352.95,
                step=10.0
            )


            total_item_quantity = st.number_input(
                "📦 Total Item Quantity",
                min_value=1.0,
                value=162.0,
                step=1.0
            )


        # --------------------------------------------
        # COLUMN 3
        # --------------------------------------------

        with col3:

            total_item_dollars = st.number_input(
                "💵 Total Item Dollars",
                min_value=1.0,
                value=2476.0,
                step=10.0
            )

            avg_receiving_delay = st.number_input(
                "⏱️ Average Receiving Delay (days)",
                min_value=0.0,
                value=5.0,
                step=1.0
            )


        st.markdown("")


        submit_flag = st.form_submit_button(
            "🚩 Evaluate Invoice Risk"
        )


    # --------------------------------------------------------
    # FLAG PREDICTION
    # --------------------------------------------------------

    if submit_flag:

        try:

            input_data = {

                "invoice_quantity": [
                    invoice_quantity
                ],

                "invoice_dollars": [
                    invoice_dollars
                ],

                "Freight": [
                    freight
                ],

                "total_item_quantity": [
                    total_item_quantity
                ],

                "total_item_dollars": [
                    total_item_dollars
                ],

                "avg_receiving_delay": [
                    avg_receiving_delay
                ]

            }


            flag_prediction = predict_invoice_flag(
                input_data
            )["Predicted_Flag"]


            is_flagged = bool(
                flag_prediction.iloc[0]
            )


            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            if is_flagged:

                st.error(
                    "🔴 **HIGH RISK — MANUAL APPROVAL REQUIRED**"
                )

                st.warning(
                    "The model has identified this invoice "
                    "as potentially requiring additional review."
                )

            else:

                st.success(
                    "🟢 **LOW RISK — AUTO-APPROVAL RECOMMENDED**"
                )

                st.info(
                    "The model did not identify this invoice "
                    "as requiring manual approval."
                )


            # ------------------------------------------------
            # RISK METRIC
            # ------------------------------------------------

            col1, col2, col3 = st.columns(3)


            with col1:

                    render_markup(
                    f"""
                    <div class="metric-card">

                        <div class="metric-title">
                            📦 Invoice Quantity
                        </div>

                        <div class="metric-value">
                            {invoice_quantity:,.0f}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            with col2:

                    render_markup(
                    f"""
                    <div class="metric-card">

                        <div class="metric-title">
                            💰 Invoice Value
                        </div>

                        <div class="metric-value">
                            ${invoice_dollars:,.2f}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            with col3:

                status = (
                    "HIGH RISK"
                    if is_flagged
                    else "LOW RISK"
                )


                render_markup(
                    f"""
                    <div class="metric-card">

                        <div class="metric-title">
                            🚦 Prediction
                        </div>

                        <div class="metric-value">
                            {status}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # ------------------------------------------------
            # RESULT TABLE
            # ------------------------------------------------

            st.markdown("### 📊 Invoice Evaluation Details")


            result_df = pd.DataFrame(
                {
                    "Invoice Quantity": [
                        invoice_quantity
                    ],

                    "Invoice Dollars": [
                        invoice_dollars
                    ],

                    "Freight": [
                        freight
                    ],

                    "Total Item Quantity": [
                        total_item_quantity
                    ],

                    "Total Item Dollars": [
                        total_item_dollars
                    ],

                    "Average Receiving Delay": [
                        avg_receiving_delay
                    ],

                    "Predicted Flag": [
                        int(is_flagged)
                    ],

                    "Decision": [
                        "Manual Approval"
                        if is_flagged
                        else "Auto Approval"
                    ]
                }
            )


            st.dataframe(
                result_df,
                width="stretch",
                hide_index=True
            )


            # ------------------------------------------------
            # VISUALIZATION
            # ------------------------------------------------

            chart_df = pd.DataFrame(
                {
                    "Metric": [
                        "Invoice Dollars",
                        "Total Item Dollars",
                        "Freight"
                    ],

                    "Amount": [
                        invoice_dollars,
                        total_item_dollars,
                        freight
                    ]
                }
            )


            fig = px.bar(
                chart_df,
                x="Metric",
                y="Amount",
                title="Invoice Cost Analysis",
                text_auto=".2f"
            )


            fig.update_layout(
                template="plotly_white",
                height=400,
                showlegend=False
            )


            st.plotly_chart(
                fig,
                width="stretch"
            )


        except Exception as e:

            st.error(
                f"❌ Invoice prediction failed: {e}"
            )


# ============================================================
# FOOTER
# ============================================================

render_markup(
    """
    <div class="footer">

        <b>Vendor Invoice Intelligence Portal</b>
        <br>

        Powered by Machine Learning • Python • Scikit-Learn • Streamlit

        <br><br>

        © 2026 Invoice Intelligence System

    </div>
    """,
    unsafe_allow_html=True
)