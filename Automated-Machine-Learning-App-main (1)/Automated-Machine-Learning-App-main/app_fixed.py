import streamlit as st
import pandas as pd
import os
from streamlit_extras.add_vertical_space import add_vertical_space
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import numpy as np
import pickle
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import json
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')


with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("🚀 Data Science Platform")
    
    # Main sections
    st.header("📊 Data Operations")
    data_ops = st.radio("Data Operations", ["📁 Upload", "🧹 Data Cleaning", "⚙️ Feature Engineering", "📈 Advanced Profiling"], key="data_ops")
    
    st.header("🤖 Machine Learning")
    ml_ops = st.radio("Machine Learning", ["🎯 Modeling", "🔍 Model Comparison", "🧩 Clustering", "📊 Model Explainability"], key="ml_ops")
    
    st.header("📈 Analytics")
    analytics_ops = st.radio("Analytics", ["📉 Statistical Testing", "⏰ Time Series", "📋 Reports"], key="analytics_ops")
    
    st.header("💾 Export")
    st.radio("Export", ["📥 Download"], key="export")
    
    # Map choices to sections
    choice_map = {
        "📁 Upload": "Upload",
        "🧹 Data Cleaning": "Data Cleaning", 
        "⚙️ Feature Engineering": "Feature Engineering",
        "📈 Advanced Profiling": "Advanced Profiling",
        "🎯 Modeling": "Modeling",
        "🔍 Model Comparison": "Model Comparison",
        "🧩 Clustering": "Clustering",
        "📊 Model Explainability": "Model Explainability",
        "📉 Statistical Testing": "Statistical Testing",
        "⏰ Time Series": "Time Series",
        "📋 Reports": "Reports",
        "📥 Download": "Download"
    }
    
    choice = choice_map[data_ops] if data_ops in choice_map else choice_map[ml_ops] if ml_ops in choice_map else choice_map[analytics_ops] if analytics_ops in choice_map else "Download"
    
    st.info("🎯 Comprehensive Data Science Platform - From Data Cleaning to Model Deployment!")
    st.write("Made with ❤️ by Data Science Team")


if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)


if choice=="Upload":
    st.title("📁 Multi-Format Data Upload & Analysis")
    
    # Data source selection
    st.subheader("🔗 Data Source Selection")
    data_source = st.radio("Choose Data Source", 
                         ["📁 File Upload", "🗄️ Database Connection", "🌐 API URL", "📝 Sample Data Generation"],
                         key="data_source")
    
    df = None
    
    if data_source == "📁 File Upload":
        st.subheader("📂 Upload Your Data")
        
        # File format selection
        file_format = st.selectbox("Select File Format", 
                                ["CSV", "Excel", "JSON", "Parquet", "SQL Database"],
                                key="file_format")
        
        if file_format == "CSV":
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            if uploaded_file:
                try:
                    # CSV options
                    separator = st.selectbox("Separator", [",",", ";", "\\t"], key="csv_sep")
                    encoding = st.selectbox("Encoding", ["utf-8", "latin-1", "iso-8859-1"], key="csv_encoding")
                    header_row = st.checkbox("First row is header", value=True, key="csv_header")
                    
                    df = pd.read_csv(uploaded_file, sep=separator, encoding=encoding, header=0 if header_row else None)
                    st.success(f"✅ CSV file loaded successfully! Shape: {df.shape}")
                    
                except Exception as e:
                    st.error(f"❌ Error loading CSV: {str(e)}")
        
        elif file_format == "Excel":
            uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx', 'xls'])
            if uploaded_file:
                try:
                    # Excel options
                    sheet_name = st.text_input("Sheet name (leave empty for first sheet)", key="excel_sheet")
                    
                    if sheet_name:
                        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    st.success(f"✅ Excel file loaded successfully! Shape: {df.shape}")
                    
                except Exception as e:
                    st.error(f"❌ Error loading Excel: {str(e)}")
        
        elif file_format == "JSON":
            uploaded_file = st.file_uploader("Upload JSON file", type=['json'])
            if uploaded_file:
                try:
                    # JSON options
                    json_format = st.selectbox("JSON Format", ["Records", "Values", "Columns", "Split"], key="json_format")
                    orient_map = {"Records": "records", "Values": "values", "Columns": "columns", "Split": "split"}
                    
                    df = pd.read_json(uploaded_file, orient=orient_map[json_format])
                    st.success(f"✅ JSON file loaded successfully! Shape: {df.shape}")
                    
                except Exception as e:
                    st.error(f"❌ Error loading JSON: {str(e)}")
        
        elif file_format == "Parquet":
            uploaded_file = st.file_uploader("Upload Parquet file", type=['parquet'])
            if uploaded_file:
                try:
                    df = pd.read_parquet(uploaded_file)
                    st.success(f"✅ Parquet file loaded successfully! Shape: {df.shape}")
                    
                except Exception as e:
                    st.error(f"❌ Error loading Parquet: {str(e)}")
        
        elif file_format == "SQL Database":
            st.subheader("🗄️ SQL Database Connection")
            
            # Database connection options
            db_type = st.selectbox("Database Type", ["SQLite", "MySQL", "PostgreSQL", "SQL Server"], key="db_type")
            
            if db_type == "SQLite":
                uploaded_db = st.file_uploader("Upload SQLite database", type=['db', 'sqlite', 'sqlite3'])
                if uploaded_db:
                    try:
                        # Save uploaded database temporarily
                        with open("temp_db.sqlite", "wb") as f:
                            f.write(uploaded_db.getbuffer())
                        
                        # Connect and list tables
                        conn = sqlite3.connect("temp_db.sqlite")
                        cursor = conn.cursor()
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                        tables = cursor.fetchall()
                        
                        if tables:
                            table_names = [table[0] for table in tables]
                            selected_table = st.selectbox("Select Table", table_names, key="sqlite_table")
                            
                            if selected_table:
                                df = pd.read_sql_query(f"SELECT * FROM {selected_table}", conn)
                                st.success(f"✅ Data loaded from SQLite table '{selected_table}'! Shape: {df.shape}")
                        
                        conn.close()
                        os.remove("temp_db.sqlite")
                        
                    except Exception as e:
                        st.error(f"❌ Error connecting to SQLite: {str(e)}")
            
            else:
                st.warning("🚧 Advanced database connections (MySQL, PostgreSQL, SQL Server) require additional configuration")
                st.info("For now, please export your data to CSV/Excel format and upload it.")
    
    elif data_source == "🗄️ Database Connection":
        st.subheader("🔌 Direct Database Connection")
        
        with st.expander("📋 Connection Parameters"):
            host = st.text_input("Host", value="localhost")
            port = st.number_input("Port", value=3306)
            database = st.text_input("Database Name")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.button("🔗 Test Connection"):
                st.info("🚧 Database connection feature requires additional libraries. Please use file upload for now.")
    
    elif data_source == "🌐 API URL":
        st.subheader("🌐 Load Data from API/URL")
        
        url = st.text_input("Enter API endpoint or data URL", placeholder="https://api.example.com/data")
        
        if st.button("📥 Fetch Data"):
            try:
                if url.startswith('http'):
                    # Try to load as JSON from API
                    response = pd.read_json(url)
                    df = pd.DataFrame(response)
                    st.success(f"✅ Data loaded from URL! Shape: {df.shape}")
                else:
                    st.error("❌ Please enter a valid URL")
            except Exception as e:
                st.error(f"❌ Error loading data from URL: {str(e)}")
    
    elif data_source == "📝 Sample Data Generation":
        st.subheader("🎲 Generate Sample Dataset")
        
        with st.expander("📊 Dataset Configuration"):
            dataset_type = st.selectbox("Dataset Type", 
                                   ["Classification", "Regression", "Clustering", "Time Series", "Mixed"])
            n_samples = st.number_input("Number of Samples", value=1000, min=100, max=10000)
            n_features = st.number_input("Number of Features", value=5, min=2, max=20)
            noise_level = st.slider("Noise Level", 0.0, 1.0, 0.1)
            
            if st.button("🎯 Generate Dataset"):
                try:
                    from sklearn.datasets import make_classification, make_regression, make_blobs
                    
                    if dataset_type == "Classification":
                        X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                                                 n_informative=n_features//2, n_redundant=0, 
                                                 n_clusters_per_class=1, random_state=42)
                        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
                        df['target'] = y
                        
                    elif dataset_type == "Regression":
                        X, y = make_regression(n_samples=n_samples, n_features=n_features, 
                                             noise=noise_level, random_state=42)
                        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
                        df['target'] = y
                        
                    elif dataset_type == "Clustering":
                        X, y = make_blobs(n_samples=n_samples, n_features=n_features, 
                                          centers=3, random_state=42)
                        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
                        df['cluster'] = y
                        
                    elif dataset_type == "Time Series":
                        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
                        trend = np.linspace(0, 100, n_samples)
                        seasonal = 10 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25)
                        noise = noise_level * np.random.normal(0, 1, n_samples)
                        values = trend + seasonal + noise
                        
                        df = pd.DataFrame({'date': dates, 'value': values})
                        
                    else:  # Mixed
                        X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                                                 n_informative=n_features//2, random_state=42)
                        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
                        df['target'] = y
                        # Add some categorical features
                        df['category_A'] = np.random.choice(['A', 'B', 'C'], n_samples)
                        df['category_B'] = np.random.choice(['X', 'Y'], n_samples)
                    
                    st.success(f"✅ {dataset_type} dataset generated! Shape: {df.shape}")
                    
                except Exception as e:
                    st.error(f"❌ Error generating dataset: {str(e)}")
    
    # Display loaded data
    if df is not None:
        st.subheader("📋 Data Preview & Analysis")
        
        # Data overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        with col4:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        # Data preview
        st.write("**📊 Data Preview:**")
        st.dataframe(df.head(10))
        
        # Column information
        st.write("**📋 Column Information:**")
        col_info = []
        for col in df.columns:
            col_info.append({
                'Column': col,
                'Data Type': str(df[col].dtype),
                'Non-Null Count': df[col].count(),
                'Unique Values': df[col].nunique(),
                'Missing %': (df[col].isnull().sum() / len(df)) * 100
            })
        
        col_info_df = pd.DataFrame(col_info)
        st.dataframe(col_info_df)
        
        # Data type summary
        st.write("**📈 Data Type Summary:**")
        type_counts = df.dtypes.value_counts()
        fig = px.pie(values=type_counts.values, names=type_counts.index, 
                     title="Data Types Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        # Quick insights
        st.write("**💡 Quick Insights:**")
        
        # Identify potential issues
        issues = []
        if df.isnull().sum().sum() > 0:
            issues.append(f"🔍 {df.isnull().sum().sum()} missing values found")
        
        if df.duplicated().sum() > 0:
            issues.append(f"🔄 {df.duplicated().sum()} duplicate rows found")
        
        constant_cols = [col for col in df.columns if df[col].nunique() == 1]
        if constant_cols:
            issues.append(f"📌 {len(constant_cols)} constant columns found")
        
        if issues:
            for issue in issues:
                st.warning(issue)
        else:
            st.success("✅ No major data quality issues detected!")
        
        # Save the data
        df.to_csv("sourcedata.csv", index=False)
        st.success("💾 Data saved successfully for analysis!")
        
        # Data processing options
        st.subheader("⚙️ Quick Processing Options")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧹 Quick Clean"):
                # Basic cleaning
                df_cleaned = df.drop_duplicates()
                df_cleaned = df_cleaned.fillna(df_cleaned.mean(numeric_only=True))
                df = df_cleaned
                st.success("✅ Quick cleaning completed!")
                st.rerun()
        
        with col2:
            if st.button("📊 Basic Stats"):
                st.subheader("📈 Basic Statistics")
                st.dataframe(df.describe(include='all'))
    
    else:
        st.info("👆 Please select a data source to begin analysis")


# Continue with the rest of the original app sections...
if choice=="Profiling":
    st.title("Automated Exploratory Data Analysis")
    
    if 'df' in locals():
        st.subheader("Dataset Overview")
        st.write(f"Shape: {df.shape}")
        st.write(f"Columns: {list(df.columns)}")
        
        st.subheader("First 5 Rows")
        st.dataframe(df.head())
        
        st.subheader("Data Types")
        st.write(df.dtypes)
        
        st.subheader("Missing Values")
        missing_values = df.isnull().sum()
        st.write(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values")
        
        st.subheader("Statistical Summary")
        st.dataframe(df.describe())
        
        # Visualizations for numeric columns
        numeric_columns = df.select_dtypes(include=['number']).columns
        if len(numeric_columns) > 0:
            st.subheader("Distribution of Numeric Columns")
            for col in numeric_columns[:4]:  # Show max 4 columns
                fig, ax = plt.subplots()
                sns.histplot(df[col], kde=True, ax=ax)
                ax.set_title(f'Distribution of {col}')
                st.pyplot(fig)
                plt.close()
    else:
        st.warning("Please upload data first!")


if choice=="Modeling":
    if 'df' in locals():
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        
        if st.button('Run Modelling'): 
            st.info("Training Model...")
            
            # Prepare data
            X = df.drop(columns=[chosen_target])
            y = df[chosen_target]
            
            # Handle categorical variables
            X = pd.get_dummies(X, drop_first=True)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Determine if it's classification or regression
            is_classification = False
            if y.dtype == 'object' or y.dtype.name == 'category':
                is_classification = True
            elif len(np.unique(y)) <= 20:  # If few unique values, treat as classification
                is_classification = True
            
            # Train appropriate model
            if is_classification:
                st.write("🎯 **Classification Task Detected**")
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Show classification results
                st.subheader("Model Performance")
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Accuracy: {accuracy:.4f}")
                
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())
            else:
                st.write("📈 **Regression Task Detected**")
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Show regression results
                st.subheader("Model Performance")
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.write(f"Mean Squared Error: {mse:.4f}")
                st.write(f"R² Score: {r2:.4f}")
                
                # Show actual vs predicted
                st.subheader("Actual vs Predicted")
                comparison_df = pd.DataFrame({
                    'Actual': y_test,
                    'Predicted': y_pred
                })
                st.dataframe(comparison_df.head(10))
            
            # Save model
            with open('best_model.pkl', 'wb') as f:
                pickle.dump(model, f)
            
            st.success("Model trained and saved successfully!")
    else:
        st.warning("Please upload data first!")


if choice=="Download":
    st.title("💾 Download & Export")
    
    # Download trained model
    st.subheader("📥 Download Trained Model")
    if os.path.exists('best_model.pkl'):
        with open('best_model.pkl', 'rb') as f: 
            st.download_button('Download Trained Model', f, file_name="trained_model.pkl")
    else:
        st.warning("No trained model found. Please run modeling first!")
    
    # Download processed data
    st.subheader("📊 Download Processed Data")
    if 'df' in locals():
        # Download cleaned data
        csv = df.to_csv(index=False)
        st.download_button("Download Current Dataset", csv, "processed_data.csv", "text/csv")
