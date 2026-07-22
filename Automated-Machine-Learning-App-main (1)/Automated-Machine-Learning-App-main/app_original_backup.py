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
                    separator = st.selectbox("Separator", [",",", ";", "\t"], key="csv_sep")
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


if choice=="Modelling":
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


if choice=="Data Cleaning":
    st.title("🧹 Data Cleaning & Preprocessing")
    
    if 'df' in locals():
        st.subheader("Data Quality Overview")
        
        # Show basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        # Missing values analysis
        st.subheader("🔍 Missing Values Analysis")
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        missing_df = pd.DataFrame({'Missing Count': missing_data, 'Missing %': missing_percent})
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
        
        if len(missing_df) > 0:
            st.dataframe(missing_df)
            
            # Missing values visualization
            fig = px.bar(missing_df, x=missing_df.index, y='Missing %', 
                        title="Missing Values Percentage by Column")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values found! ✅")
        
        # Data Cleaning Options
        st.subheader("⚙️ Data Cleaning Options")
        
        # Handle missing values
        if len(missing_df) > 0:
            st.write("**Handle Missing Values:**")
            missing_strategy = st.selectbox("Choose Strategy", 
                                          ["Drop Rows", "Drop Columns", "Fill with Mean", "Fill with Median", "Fill with Mode", "Forward Fill"])
            
            if st.button("Apply Missing Values Strategy"):
                if missing_strategy == "Drop Rows":
                    df_cleaned = df.dropna()
                elif missing_strategy == "Drop Columns":
                    df_cleaned = df.dropna(axis=1)
                elif missing_strategy == "Fill with Mean":
                    df_cleaned = df.fillna(df.mean())
                elif missing_strategy == "Fill with Median":
                    df_cleaned = df.fillna(df.median())
                elif missing_strategy == "Fill with Mode":
                    df_cleaned = df.fillna(df.mode().iloc[0])
                elif missing_strategy == "Forward Fill":
                    df_cleaned = df.fillna(method='ffill')
                
                st.success(f"Applied {missing_strategy} strategy!")
                st.dataframe(df_cleaned.head())
                
                # Update the main dataframe
                df = df_cleaned
                df.to_csv("sourcedata.csv", index=False)
        
        # Remove duplicates
        st.write("**Remove Duplicates:**")
        duplicate_count = df.duplicated().sum()
        st.write(f"Found {duplicate_count} duplicate rows")
        
        if duplicate_count > 0 and st.button("Remove Duplicates"):
            df_cleaned = df.drop_duplicates()
            st.success(f"Removed {duplicate_count} duplicate rows!")
            st.dataframe(df_cleaned.head())
            df = df_cleaned
            df.to_csv("sourcedata.csv", index=False)
        
        # Outlier detection
        st.subheader("🎯 Outlier Detection")
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) > 0:
            outlier_col = st.selectbox("Select Column for Outlier Detection", numeric_columns)
            
            if st.button("Detect Outliers"):
                Q1 = df[outlier_col].quantile(0.25)
                Q3 = df[outlier_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[outlier_col] < lower_bound) | (df[outlier_col] > upper_bound)]
                
                st.write(f"Found {len(outliers)} outliers in {outlier_col}")
                
                # Visualize outliers
                fig = px.box(df, y=outlier_col, title=f"Boxplot for {outlier_col}")
                st.plotly_chart(fig, use_container_width=True)
                
                if st.button("Remove Outliers"):
                    df_cleaned = df[(df[outlier_col] >= lower_bound) & (df[outlier_col] <= upper_bound)]
                    st.success(f"Removed {len(outliers)} outliers from {outlier_col}")
                    st.dataframe(df_cleaned.head())
                    df = df_cleaned
                    df.to_csv("sourcedata.csv", index=False)
        
        # Data type conversion
        st.subheader("🔄 Data Type Conversion")
        st.write("**Convert Data Types:**")
        
        for col in df.columns:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"Column: {col} (Current: {df[col].dtype})")
            with col2:
                new_type = st.selectbox(f"Convert {col}", 
                                      ["Keep Current", "int", "float", "object", "datetime"], 
                                      key=f"type_{col}")
                
                if new_type != "Keep Current" and st.button(f"Convert {col}", key=f"convert_{col}"):
                    try:
                        if new_type == "datetime":
                            df[col] = pd.to_datetime(df[col])
                        else:
                            df[col] = df[col].astype(new_type)
                        st.success(f"Converted {col} to {new_type}")
                        df.to_csv("sourcedata.csv", index=False)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error converting {col}: {str(e)}")
        
        # Cleaned data summary
        st.subheader("📊 Cleaned Data Summary")
        st.dataframe(df.describe(include='all'))
        
        if st.button("Download Cleaned Data"):
            csv = df.to_csv(index=False)
            st.download_button("Download Cleaned CSV", csv, "cleaned_data.csv", "text/csv")
            
    else:
        st.warning("Please upload data first!")


if choice=="Feature Engineering":
    st.title("⚙️ Feature Engineering")
    
    if 'df' in locals():
        st.subheader("🔧 Feature Engineering Tools")
        
        # Create a copy for feature engineering
        df_fe = df.copy()
        
        # 1. Encoding Categorical Variables
        st.subheader("🏷️ Categorical Encoding")
        categorical_cols = df_fe.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) > 0:
            encoding_method = st.selectbox("Choose Encoding Method", 
                                         ["Label Encoding", "One-Hot Encoding", "Target Encoding"])
            
            selected_cat_col = st.selectbox("Select Categorical Column", categorical_cols)
            
            if st.button("Apply Encoding"):
                if encoding_method == "Label Encoding":
                    le = LabelEncoder()
                    df_fe[selected_cat_col + '_encoded'] = le.fit_transform(df_fe[selected_cat_col].astype(str))
                    st.success(f"Label encoded {selected_cat_col}")
                    
                elif encoding_method == "One-Hot Encoding":
                    dummies = pd.get_dummies(df_fe[selected_cat_col], prefix=selected_cat_col)
                    df_fe = pd.concat([df_fe, dummies], axis=1)
                    st.success(f"One-hot encoded {selected_cat_col}")
                    
                elif encoding_method == "Target Encoding":
                    # Simple target encoding (mean encoding)
                    if 'target_column' in st.session_state:
                        target_col = st.session_state.target_column
                        mean_encoding = df_fe.groupby(selected_cat_col)[target_col].mean()
                        df_fe[selected_cat_col + '_target_encoded'] = df_fe[selected_cat_col].map(mean_encoding)
                        st.success(f"Target encoded {selected_cat_col}")
                    else:
                        st.warning("Please select a target column first in the Modeling section")
                
                st.dataframe(df_fe.head())
        else:
            st.info("No categorical columns found")
        
        # 2. Feature Scaling
        st.subheader("📏 Feature Scaling")
        numeric_cols = df_fe.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            scaling_method = st.selectbox("Choose Scaling Method", 
                                        ["StandardScaler", "MinMaxScaler", "RobustScaler"])
            
            selected_num_cols = st.multiselect("Select Numeric Columns to Scale", numeric_cols)
            
            if st.button("Apply Scaling") and selected_num_cols:
                if scaling_method == "StandardScaler":
                    scaler = StandardScaler()
                    df_fe[selected_num_cols] = scaler.fit_transform(df_fe[selected_num_cols])
                elif scaling_method == "MinMaxScaler":
                    from sklearn.preprocessing import MinMaxScaler
                    scaler = MinMaxScaler()
                    df_fe[selected_num_cols] = scaler.fit_transform(df_fe[selected_num_cols])
                elif scaling_method == "RobustScaler":
                    from sklearn.preprocessing import RobustScaler
                    scaler = RobustScaler()
                    df_fe[selected_num_cols] = scaler.fit_transform(df_fe[selected_num_cols])
                
                st.success(f"Applied {scaling_method} to selected columns")
                st.dataframe(df_fe.head())
        
        # 3. Feature Creation
        st.subheader("🔨 Feature Creation")
        
        # Polynomial Features
        st.write("**Create Polynomial Features:**")
        poly_col = st.selectbox("Select Column for Polynomial Features", numeric_cols)
        poly_degree = st.slider("Polynomial Degree", 2, 4, 2)
        
        if st.button("Create Polynomial Features"):
            for degree in range(2, poly_degree + 1):
                df_fe[f"{poly_col}_poly_{degree}"] = df_fe[poly_col] ** degree
            st.success(f"Created polynomial features up to degree {poly_degree}")
            st.dataframe(df_fe.head())
        
        # Interaction Features
        st.write("**Create Interaction Features:**")
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                interaction_col1 = st.selectbox("First Column", numeric_cols, key="int1")
            with col2:
                interaction_col2 = st.selectbox("Second Column", numeric_cols, key="int2")
            
            if st.button("Create Interaction Feature"):
                df_fe[f"{interaction_col1}_x_{interaction_col2}"] = df_fe[interaction_col1] * df_fe[interaction_col2]
                df_fe[f"{interaction_col1}_div_{interaction_col2}"] = df_fe[interaction_col1] / (df_fe[interaction_col2] + 1e-8)
                st.success("Created interaction features")
                st.dataframe(df_fe.head())
        
        # 4. Feature Selection
        st.subheader("🎯 Feature Selection")
        
        if st.checkbox("Enable Feature Selection"):
            # Need a target column for feature selection
            target_col = st.selectbox("Select Target Column for Feature Selection", df_fe.columns)
            st.session_state.target_column = target_col
            
            selection_method = st.selectbox("Feature Selection Method", 
                                         ["Univariate Selection", "Correlation Analysis", "Variance Threshold"])
            
            if selection_method == "Univariate Selection":
                k_best = st.slider("Number of Best Features", 5, min(20, len(df_fe.columns)), 10)
                
                if st.button("Run Univariate Selection"):
                    X = df_fe.drop(columns=[target_col])
                    y = df_fe[target_col]
                    
                    # Handle categorical target
                    if y.dtype == 'object':
                        le = LabelEncoder()
                        y = le.fit_transform(y.astype(str))
                    
                    selector = SelectKBest(score_func=f_classif if len(np.unique(y)) < 20 else f_regression, k=k_best)
                    X_new = selector.fit_transform(X.select_dtypes(include=[np.number]), y)
                    
                    selected_features = X.select_dtypes(include=[np.number]).columns[selector.get_support()]
                    
                    st.success(f"Selected {k_best} best features")
                    st.write("**Selected Features:**")
                    for feature in selected_features:
                        st.write(f"- {feature}")
                    
                    # Show feature scores
                    scores = selector.scores_
                    feature_scores = pd.DataFrame({
                        'Feature': X.select_dtypes(include=[np.number]).columns,
                        'Score': scores
                    }).sort_values('Score', ascending=False)
                    
                    st.write("**Feature Scores:**")
                    st.dataframe(feature_scores.head(10))
            
            elif selection_method == "Correlation Analysis":
                if st.button("Run Correlation Analysis"):
                    corr_matrix = df_fe.corr()
                    
                    # Plot correlation heatmap
                    fig = px.imshow(corr_matrix, 
                                   title="Feature Correlation Matrix",
                                   color_continuous_scale="RdBu",
                                   aspect="auto")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show highly correlated features
                    threshold = st.slider("Correlation Threshold", 0.7, 0.95, 0.8)
                    
                    high_corr_pairs = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            if abs(corr_matrix.iloc[i, j]) > threshold:
                                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
                    
                    if high_corr_pairs:
                        st.write(f"Highly correlated pairs (threshold > {threshold}):")
                        for col1, col2, corr in high_corr_pairs:
                            st.write(f"- {col1} ↔ {col2}: {corr:.3f}")
                    else:
                        st.info("No highly correlated features found")
        
        # 5. Feature Importance
        st.subheader("📊 Feature Importance")
        
        if st.checkbox("Show Feature Importance") and 'target_column' in st.session_state:
            target_col = st.session_state.target_column
            
            if st.button("Calculate Feature Importance"):
                X = df_fe.drop(columns=[target_col])
                y = df_fe[target_col]
                
                # Handle categorical target
                if y.dtype == 'object':
                    le = LabelEncoder()
                    y = le.fit_transform(y.astype(str))
                    model = RandomForestClassifier(random_state=42)
                else:
                    model = RandomForestRegressor(random_state=42)
                
                # Use only numeric features for importance calculation
                X_numeric = X.select_dtypes(include=[np.number])
                
                if len(X_numeric.columns) > 0:
                    model.fit(X_numeric, y)
                    importance = model.feature_importances_
                    
                    feature_importance = pd.DataFrame({
                        'Feature': X_numeric.columns,
                        'Importance': importance
                    }).sort_values('Importance', ascending=False)
                    
                    # Plot feature importance
                    fig = px.bar(feature_importance.head(15), 
                               x='Importance', y='Feature',
                               title="Top 15 Feature Importance",
                               orientation='h')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(feature_importance)
                else:
                    st.warning("No numeric features available for importance calculation")
        
        # Save engineered features
        st.subheader("💾 Save Engineered Features")
        
        if st.button("Save Engineered Dataset"):
            df_fe.to_csv("engineered_data.csv", index=False)
            st.success("Engineered dataset saved as 'engineered_data.csv'")
            
            # Update main dataframe
            df = df_fe
            df.to_csv("sourcedata.csv", index=False)
            
            # Show download button
            csv = df_fe.to_csv(index=False)
            st.download_button("Download Engineered Data", csv, "engineered_features.csv", "text/csv")
        
        # Show final dataset info
        st.subheader("📋 Final Dataset Info")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Features", len(df.columns))
        with col2:
            st.metric("Engineered Features", len(df_fe.columns))
        with col3:
            st.metric("New Features Added", len(df_fe.columns) - len(df.columns))
        
    else:
        st.warning("Please upload data first!")


if choice=="Advanced Profiling":
    st.title("📈 Advanced Data Profiling")
    
    if 'df' in locals():
        # Basic Overview
        st.subheader("📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
        with col4:
            st.metric("Categorical Columns", len(df.select_dtypes(include=['object', 'category']).columns))
        
        # Advanced Visualizations
        st.subheader("🎨 Advanced Visualizations")
        
        # Distribution plots
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            selected_dist_col = st.selectbox("Select Column for Distribution Analysis", numeric_cols)
            
            # Create subplots for multiple views
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Histogram', 'Box Plot', 'Violin Plot', 'Q-Q Plot'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Histogram
            fig.add_trace(
                go.Histogram(x=df[selected_dist_col], name='Histogram'),
                row=1, col=1
            )
            
            # Box Plot
            fig.add_trace(
                go.Box(y=df[selected_dist_col], name='Box Plot'),
                row=1, col=2
            )
            
            # Violin Plot
            fig.add_trace(
                go.Violin(y=df[selected_dist_col], name='Violin Plot'),
                row=2, col=1
            )
            
            # Q-Q Plot (approximation using scatter)
            from scipy import stats
            qq_data = stats.probplot(df[selected_dist_col].dropna(), dist="norm")
            fig.add_trace(
                go.Scatter(x=qq_data[0][0], y=qq_data[0][1], mode='markers', name='Q-Q Plot'),
                row=2, col=2
            )
            
            fig.update_layout(height=800, title_text=f"Distribution Analysis for {selected_dist_col}")
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation Analysis
        if len(numeric_cols) > 1:
            st.subheader("🔗 Correlation Analysis")
            
            # Correlation matrix heatmap
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, 
                           title="Correlation Heatmap",
                           color_continuous_scale="RdBu",
                           aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
            
            # Scatter plot matrix (sample for performance)
            if st.checkbox("Show Scatter Plot Matrix (Sample)"):
                sample_df = df[numeric_cols].sample(min(500, len(df)))
                fig = px.scatter_matrix(sample_df, title="Scatter Plot Matrix (Sample)")
                st.plotly_chart(fig, use_container_width=True)
        
        # Categorical Analysis
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            st.subheader("📋 Categorical Analysis")
            
            selected_cat_col = st.selectbox("Select Categorical Column", categorical_cols)
            
            # Value counts with plot
            cat_counts = df[selected_cat_col].value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(cat_counts)
            with col2:
                fig = px.bar(x=cat_counts.index, y=cat_counts.values, 
                           title=f"Distribution of {selected_cat_col}")
                st.plotly_chart(fig, use_container_width=True)
        
        # Missing Values Pattern
        st.subheader("🔍 Missing Values Pattern")
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Missing %': (missing_data.values / len(df)) * 100
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
            
            fig = px.bar(missing_df, x='Column', y='Missing %', 
                        title="Missing Values Percentage by Column")
            st.plotly_chart(fig, use_container_width=True)
            
            # Missing values heatmap
            fig = px.imshow(df.isnull(), title="Missing Values Pattern")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values found! ✅")
        
        # Statistical Summary
        st.subheader("📈 Statistical Summary")
        
        # Descriptive statistics
        st.write("**Numeric Variables:**")
        st.dataframe(df.describe(include=[np.number]))
        
        if len(categorical_cols) > 0:
            st.write("**Categorical Variables:**")
            cat_stats = []
            for col in categorical_cols:
                stats_dict = {
                    'Column': col,
                    'Unique Values': df[col].nunique(),
                    'Most Frequent': df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A',
                    'Frequency': df[col].value_counts().iloc[0] if not df[col].empty else 0
                }
                cat_stats.append(stats_dict)
            
            cat_stats_df = pd.DataFrame(cat_stats)
            st.dataframe(cat_stats_df)
        
        # Data Quality Report
        st.subheader("📋 Data Quality Report")
        
        quality_issues = []
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            quality_issues.append(f"Duplicate rows: {duplicates}")
        
        # Check for potential outliers
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
            if len(outliers) > 0:
                quality_issues.append(f"Potential outliers in {col}: {len(outliers)}")
        
        # Check for constant columns
        constant_cols = []
        for col in df.columns:
            if df[col].nunique() == 1:
                constant_cols.append(col)
        
        if constant_cols:
            quality_issues.append(f"Constant columns: {', '.join(constant_cols)}")
        
        if quality_issues:
            st.warning("Data Quality Issues Found:")
            for issue in quality_issues:
                st.write(f"⚠️ {issue}")
        else:
            st.success("No major data quality issues found! ✅")
        
        # Export profiling report
        st.subheader("💾 Export Report")
        if st.button("Generate Profiling Report"):
            # Create a comprehensive report
            report_data = {
                'Dataset_Info': {
                    'Rows': df.shape[0],
                    'Columns': df.shape[1],
                    'Numeric_Columns': len(numeric_cols),
                    'Categorical_Columns': len(categorical_cols),
                    'Missing_Values': missing_data.sum(),
                    'Duplicate_Rows': duplicates
                },
                'Numeric_Summary': df.describe(include=[np.number]).to_dict(),
                'Data_Quality_Issues': quality_issues
            }
            
            # Save as JSON
            import json
            with open('profiling_report.json', 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            st.success("Profiling report saved as 'profiling_report.json'")
            
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
        
        # Download engineered data if available
        if os.path.exists('engineered_data.csv'):
            with open('engineered_data.csv', 'r') as f:
                engineered_csv = f.read()
            st.download_button("Download Engineered Data", engineered_csv, "engineered_data.csv", "text/csv")
        
        # Download profiling report if available
        if os.path.exists('profiling_report.json'):
            with open('profiling_report.json', 'r') as f:
                report_json = f.read()
            st.download_button("Download Profiling Report", report_json, "profiling_report.json", "application/json")
    
    # Export session info
    st.subheader("📋 Session Summary")
    if 'df' in locals():
        session_info = {
            'dataset_shape': df.shape,
            'columns': list(df.columns),
            'data_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object', 'category']).columns)
        }
        
        import json
        session_json = json.dumps(session_info, indent=2, default=str)
        st.download_button("Download Session Summary", session_json, "session_summary.json", "application/json")


if choice=="Model Comparison":
    st.title("🔍 Model Comparison")
    
    if 'df' in locals():
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        
        if st.button('Compare Models'): 
            st.info("Training multiple models for comparison...")
            
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
            elif len(np.unique(y)) <= 20:
                is_classification = True
            
            models = []
            results = []
            
            if is_classification:
                st.write("🎯 **Classification Models**")
                models = [
                    ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
                    ('Logistic Regression', LogisticRegression(random_state=42, max_iter=1000)),
                    ('SVM', SVC(random_state=42)),
                    ('Decision Tree', DecisionTreeClassifier(random_state=42)),
                    ('KNN', KNeighborsClassifier())
                ]
                
                for name, model in models:
                    try:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        results.append({'Model': name, 'Accuracy': accuracy, 'Type': 'Classification'})
                    except Exception as e:
                        st.warning(f"Error training {name}: {str(e)}")
            else:
                st.write("📈 **Regression Models**")
                models = [
                    ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=42)),
                    ('Linear Regression', LinearRegression()),
                    ('SVR', SVR()),
                    ('Decision Tree', DecisionTreeRegressor(random_state=42)),
                    ('KNN', KNeighborsRegressor())
                ]
                
                for name, model in models:
                    try:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        results.append({'Model': name, 'MSE': mse, 'R²': r2, 'Type': 'Regression'})
                    except Exception as e:
                        st.warning(f"Error training {name}: {str(e)}")
            
            # Display results
            if results:
                results_df = pd.DataFrame(results)
                st.subheader("Model Comparison Results")
                st.dataframe(results_df)
                
                # Plot results
                if is_classification:
                    fig = px.bar(results_df, x='Model', y='Accuracy', title='Model Accuracy Comparison')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = make_subplots(rows=1, cols=2, subplot_titles=('MSE Comparison', 'R² Comparison'))
                    fig.add_trace(go.Bar(x=results_df['Model'], y=results_df['MSE'], name='MSE'), row=1, col=1)
                    fig.add_trace(go.Bar(x=results_df['Model'], y=results_df['R²'], name='R²'), row=1, col=2)
                    fig.update_layout(title_text="Regression Models Comparison")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show best model
                if is_classification:
                    best_model = results_df.loc[results_df['Accuracy'].idxmax()]
                    st.success(f"🏆 Best Model: {best_model['Model']} with Accuracy: {best_model['Accuracy']:.4f}")
                else:
                    best_model = results_df.loc[results_df['R²'].idxmax()]
                    st.success(f"🏆 Best Model: {best_model['Model']} with R²: {best_model['R²']:.4f}")
            else:
                st.error("No models could be trained successfully!")
    else:
        st.warning("Please upload data first!")


if choice=="Clustering":
    st.title("🧩 Clustering & Segmentation")
    
    if 'df' in locals():
        st.subheader("🎯 Unsupervised Learning")
        
        # Select numeric columns for clustering
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            selected_cols = st.multiselect("Select Columns for Clustering", numeric_cols, default=list(numeric_cols)[:2])
            
            if len(selected_cols) >= 2:
                X = df[selected_cols]
                
                # Handle missing values
                if X.isnull().sum().sum() > 0:
                    X = X.fillna(X.mean())
                    st.info("Missing values filled with column means")
                
                # Scale the data
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Number of clusters
                n_clusters = st.slider("Number of Clusters", 2, 10, 3)
                
                if st.button("Run Clustering"):
                    # K-Means clustering
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    cluster_labels = kmeans.fit_predict(X_scaled)
                    
                    # Add cluster labels to dataframe
                    df_clustered = df.copy()
                    df_clustered['Cluster'] = cluster_labels
                    
                    st.success(f"K-Means clustering completed with {n_clusters} clusters!")
                    
                    # Cluster visualization
                    if len(selected_cols) == 2:
                        fig = px.scatter(df_clustered, x=selected_cols[0], y=selected_cols[1], 
                                       color='Cluster', title="K-Means Clustering Results")
                        st.plotly_chart(fig, use_container_width=True)
                    elif len(selected_cols) >= 3:
                        fig = px.scatter_3d(df_clustered, x=selected_cols[0], y=selected_cols[1], z=selected_cols[2],
                                          color='Cluster', title="3D Clustering Results")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Cluster statistics
                    st.subheader("📊 Cluster Statistics")
                    cluster_stats = df_clustered.groupby('Cluster')[selected_cols].agg(['mean', 'std', 'count'])
                    st.dataframe(cluster_stats)
                    
                    # Cluster sizes
                    cluster_sizes = df_clustered['Cluster'].value_counts().sort_index()
                    fig = px.bar(x=cluster_sizes.index, y=cluster_sizes.values, 
                               title="Cluster Sizes", labels={'x': 'Cluster', 'y': 'Number of Points'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Save clustered data
                    if st.button("Save Clustered Data"):
                        df_clustered.to_csv("clustered_data.csv", index=False)
                        st.success("Clustered data saved as 'clustered_data.csv'")
                        
                        csv = df_clustered.to_csv(index=False)
                        st.download_button("Download Clustered Data", csv, "clustered_data.csv", "text/csv")
            else:
                st.warning("Please select at least 2 columns for clustering")
        else:
            st.warning("Need at least 2 numeric columns for clustering")
    else:
        st.warning("Please upload data first!")


if choice=="Statistical Testing":
    st.title("📉 Statistical Testing & Hypothesis Testing")
    
    if 'df' in locals():
        st.subheader("🔬 Statistical Tests")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            # Select columns for testing
            col1 = st.selectbox("Select First Variable", numeric_cols)
            col2 = st.selectbox("Select Second Variable", numeric_cols)
            
            # Correlation test
            st.write("**Correlation Analysis:**")
            correlation, p_value = stats.pearsonr(df[col1].dropna(), df[col2].dropna())
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Correlation Coefficient", f"{correlation:.4f}")
            with col2:
                st.metric("P-value", f"{p_value:.4f}")
            
            if p_value < 0.05:
                st.success("Significant correlation detected (p < 0.05)")
            else:
                st.info("No significant correlation (p >= 0.05)")
            
            # Normality test
            st.write("**Normality Test (Shapiro-Wilk):**")
            selected_col = st.selectbox("Select Column for Normality Test", numeric_cols)
            
            if st.button("Test Normality"):
                stat, p_val = stats.shapiro(df[selected_col].dropna())
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Test Statistic", f"{stat:.4f}")
                with col2:
                    st.metric("P-value", f"{p_val:.4f}")
                
                if p_val < 0.05:
                    st.warning("Data is not normally distributed (p < 0.05)")
                else:
                    st.success("Data appears normally distributed (p >= 0.05)")
            
            # T-test
            st.write("**Independent T-Test:**")
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            if len(categorical_cols) > 0:
                cat_col = st.selectbox("Select Categorical Variable", categorical_cols)
                num_col = st.selectbox("Select Numeric Variable for T-Test", numeric_cols)
                
                if st.button("Run T-Test"):
                    groups = df.groupby(cat_col)[num_col].apply(list)
                    
                    if len(groups) == 2:
                        group1, group2 = groups
                        stat, p_val = stats.ttest_ind(group1, group2)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("T-Statistic", f"{stat:.4f}")
                        with col2:
                            st.metric("P-value", f"{p_val:.4f}")
                        
                        if p_val < 0.05:
                            st.success("Significant difference between groups (p < 0.05)")
                        else:
                            st.info("No significant difference between groups (p >= 0.05)")
                    else:
                        st.warning("T-Test requires exactly 2 groups")
            
            # ANOVA
            st.write("**ANOVA Test:**")
            if len(categorical_cols) > 0:
                cat_col_anova = st.selectbox("Select Categorical Variable for ANOVA", categorical_cols)
                num_col_anova = st.selectbox("Select Numeric Variable for ANOVA", numeric_cols)
                
                if st.button("Run ANOVA"):
                    groups = df.groupby(cat_col_anova)[num_col_anova].apply(list)
                    
                    if len(groups) >= 3:
                        stat, p_val = stats.f_oneway(*groups)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("F-Statistic", f"{stat:.4f}")
                        with col2:
                            st.metric("P-value", f"{p_val:.4f}")
                        
                        if p_val < 0.05:
                            st.success("Significant difference between groups (p < 0.05)")
                        else:
                            st.info("No significant difference between groups (p >= 0.05)")
                    else:
                        st.warning("ANOVA requires at least 3 groups")
        
        # Descriptive statistics
        st.subheader("📈 Descriptive Statistics")
        if len(numeric_cols) > 0:
            selected_stats_col = st.selectbox("Select Column for Statistics", numeric_cols)
            
            stats_data = df[selected_stats_col].describe()
            st.dataframe(stats_data)
            
            # Additional statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Skewness", f"{stats.skew(df[selected_stats_col].dropna()):.4f}")
            with col2:
                st.metric("Kurtosis", f"{stats.kurtosis(df[selected_stats_col].dropna()):.4f}")
            with col3:
                st.metric("Variance", f"{df[selected_stats_col].var():.4f}")
            with col4:
                st.metric("Range", f"{df[selected_stats_col].max() - df[selected_stats_col].min():.4f}")
    else:
        st.warning("Please upload data first!")


if choice=="Time Series":
    st.title("⏰ Time Series Analysis")
    
    if 'df' in locals():
        st.subheader("📅 Time Series Components")
        
        # Check for datetime columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        
        if len(datetime_cols) > 0:
            date_col = st.selectbox("Select Date Column", datetime_cols)
            value_col = st.selectbox("Select Value Column", df.select_dtypes(include=[np.number]).columns)
            
            if st.button("Analyze Time Series"):
                # Prepare time series data
                ts_data = df[[date_col, value_col]].copy()
                ts_data = ts_data.sort_values(date_col)
                ts_data.set_index(date_col, inplace=True)
                
                # Plot time series
                fig = px.line(ts_data, y=value_col, title=f"Time Series: {value_col}")
                st.plotly_chart(fig, use_container_width=True)
                
                # Moving averages
                window = st.slider("Moving Average Window", 3, 30, 7)
                ts_data[f'MA_{window}'] = ts_data[value_col].rolling(window=window).mean()
                
                fig = px.line(ts_data, y=[value_col, f'MA_{window}'], 
                             title=f"Time Series with {window}-day Moving Average")
                st.plotly_chart(fig, use_container_width=True)
                
                # Basic statistics
                st.subheader("📊 Time Series Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean", f"{ts_data[value_col].mean():.4f}")
                with col2:
                    st.metric("Std Dev", f"{ts_data[value_col].std():.4f}")
                with col3:
                    st.metric("Min", f"{ts_data[value_col].min():.4f}")
                with col4:
                    st.metric("Max", f"{ts_data[value_col].max():.4f}")
        else:
            st.warning("No datetime columns found. Please convert a column to datetime first.")
            
            # Option to convert column to datetime
            if st.checkbox("Convert Column to Datetime"):
                col_to_convert = st.selectbox("Select Column to Convert", df.columns)
                
                if st.button("Convert to Datetime"):
                    try:
                        df[col_to_convert] = pd.to_datetime(df[col_to_convert])
                        st.success(f"Converted {col_to_convert} to datetime")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error converting column: {str(e)}")
    else:
        st.warning("Please upload data first!")


if choice=="Model Explainability":
    st.title("📊 Model Explainability & Interpretability")
    
    if os.path.exists('best_model.pkl'):
        # Load the trained model
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        if 'df' in locals():
            st.subheader("🔍 Feature Importance Analysis")
            
            # Get feature importance if available
            if hasattr(model, 'feature_importances_'):
                # This works for tree-based models
                importance = model.feature_importances_
                
                # Get feature names (assuming we have the same preprocessing as during training)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                feature_names = list(numeric_cols)
                
                if len(importance) == len(feature_names):
                    feature_importance = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importance
                    }).sort_values('Importance', ascending=False)
                    
                    # Plot feature importance
                    fig = px.bar(feature_importance.head(15), 
                               x='Importance', y='Feature',
                               title="Top 15 Feature Importance",
                               orientation='h')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(feature_importance)
                else:
                    st.warning("Feature importance length mismatch. Model may have been trained with different features.")
            else:
                st.info("Feature importance not available for this model type.")
            
            # Model characteristics
            st.subheader("🤖 Model Characteristics")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Model Type:**", type(model).__name__)
                st.write("**Model Parameters:**", len(model.get_params()))
            
            with col2:
                if hasattr(model, 'n_estimators'):
                    st.metric("Number of Estimators", model.n_estimators)
                if hasattr(model, 'max_depth'):
                    st.metric("Max Depth", model.max_depth if model.max_depth else "No Limit")
            
            # SHAP-like explanation (simplified)
            st.subheader("📈 Feature Impact Analysis")
            
            if 'df' in locals():
                target_col = st.selectbox("Select Target Column", df.columns)
                
                if st.button("Analyze Feature Impact"):
                    X = df.drop(columns=[target_col])
                    y = df[target_col]
                    
                    # Use only numeric features
                    X_numeric = X.select_dtypes(include=[np.number])
                    
                    if len(X_numeric.columns) > 0:
                        # Simple correlation-based importance
                        correlations = X_numeric.corrwith(y).abs().sort_values(ascending=False)
                        
                        fig = px.bar(x=correlations.values, y=correlations.index,
                                   title="Feature Correlation with Target",
                                   orientation='h')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.dataframe(correlations)
                    else:
                        st.warning("No numeric features available for analysis.")
    else:
        st.warning("No trained model found. Please run modeling first!")


if choice=="Reports":
    st.title("📋 Comprehensive Reports")
    
    if 'df' in locals():
        st.subheader("📊 Data Science Project Report")
        
        # Project Overview
        st.write("### 🎯 Project Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Dataset Size", f"{df.shape[0]:,} rows")
        with col2:
            st.metric("Features", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Duplicates", df.duplicated().sum())
        
        # Data Quality Report
        st.write("### 📋 Data Quality Assessment")
        
        quality_score = 100
        issues = []
        
        # Check missing values
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        if missing_pct > 5:
            quality_score -= 10
            issues.append(f"High missing values: {missing_pct:.1f}%")
        
        # Check duplicates
        duplicate_pct = (df.duplicated().sum() / df.shape[0]) * 100
        if duplicate_pct > 1:
            quality_score -= 5
            issues.append(f"Duplicates: {duplicate_pct:.1f}%")
        
        # Check constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() == 1]
        if constant_cols:
            quality_score -= len(constant_cols) * 2
            issues.append(f"Constant columns: {len(constant_cols)}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Data Quality Score", f"{quality_score}/100")
        with col2:
            if issues:
                st.write("⚠️ Issues:")
                for issue in issues:
                    st.write(f"- {issue}")
            else:
                st.success("✅ No major issues")
        
        # Feature Analysis
        st.write("### 🔍 Feature Analysis")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Numeric Features", len(numeric_cols))
            st.metric("Categorical Features", len(categorical_cols))
        with col2:
            if len(numeric_cols) > 0:
                st.metric("Avg Correlation", f"{df[numeric_cols].corr().abs().mean().mean():.3f}")
        
        # Statistical Summary
        st.write("### 📈 Statistical Summary")
        
        if len(numeric_cols) > 0:
            st.dataframe(df[numeric_cols].describe())
        
        # Recommendations
        st.write("### 💡 Recommendations")
        
        recommendations = []
        
        if missing_pct > 10:
            recommendations.append("Consider imputation strategies for missing values")
        
        if duplicate_pct > 2:
            recommendations.append("Remove duplicate rows to improve data quality")
        
        if len(constant_cols) > 0:
            recommendations.append(f"Remove constant columns: {', '.join(constant_cols)}")
        
        if len(numeric_cols) < len(categorical_cols):
            recommendations.append("Consider encoding categorical variables for modeling")
        
        if not recommendations:
            recommendations.append("Data appears clean and ready for analysis!")
        
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
        
        # Export Report
        st.write("### 💾 Export Report")
        
        if st.button("Generate Full Report"):
            report_data = {
                'project_overview': {
                    'dataset_shape': df.shape,
                    'missing_values': df.isnull().sum().sum(),
                    'duplicates': df.duplicated().sum(),
                    'quality_score': quality_score
                },
                'feature_analysis': {
                    'numeric_features': len(numeric_cols),
                    'categorical_features': len(categorical_cols),
                    'numeric_columns': list(numeric_cols),
                    'categorical_columns': list(categorical_cols)
                },
                'data_quality_issues': issues,
                'recommendations': recommendations,
                'statistical_summary': df.describe().to_dict() if len(numeric_cols) > 0 else {}
            }
            
            import json
            report_json = json.dumps(report_data, indent=2, default=str)
            
            st.download_button("Download Full Report", report_json, "data_science_report.json", "application/json")
            st.success("Report generated successfully!")
    else:
        st.warning("Please upload data first!")