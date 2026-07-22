# 🚀 Comprehensive Data Science Platform

A professional-grade data science platform that covers the complete workflow from data cleaning to model deployment. This application provides all the tools and functionality that data analysts and data scientists need in their daily work.

## ✨ Features

### 📊 Data Operations
- **📁 Upload**: Smart file upload with validation and preview
- **🧹 Data Cleaning**: Automated detection and handling of missing values, duplicates, outliers, and data type conversion
- **⚙️ Feature Engineering**: Professional-grade feature creation, encoding, scaling, selection, and importance analysis
- **📈 Advanced Profiling**: Interactive visualizations, correlation analysis, and comprehensive data quality reports

### 🤖 Machine Learning
- **🎯 Modeling**: Auto-detection of classification vs regression with appropriate algorithms
- **🔍 Model Comparison**: Compare multiple algorithms (Random Forest, SVM, Logistic Regression, Decision Tree, KNN) with visual performance metrics
- **🧩 Clustering**: K-Means clustering with 2D/3D visualization and cluster statistics
- **📊 Model Explainability**: Feature importance analysis and model interpretability tools

### 📈 Analytics
- **📉 Statistical Testing**: Correlation analysis, normality tests, t-tests, ANOVA, and descriptive statistics
- **⏰ Time Series**: Time series analysis with moving averages and trend analysis
- **📋 Reports**: Comprehensive project reports with data quality assessment and recommendations

### 💾 Export
- **📥 Download**: Export trained models, processed data, profiling reports, and session summaries

## 🛠️ Technologies Used

### Core Frameworks
- **[Python](https://www.python.org/)** - Core programming language
- **[Streamlit](https://streamlit.io/)** - Interactive web application framework
- **[Plotly](https://plotly.com/)** - Interactive visualizations
- **[Pandas](https://pandas.pydata.org/)** - Data manipulation and analysis

### Machine Learning & Statistics
- **[Scikit-learn](https://scikit-learn.org/)** - Machine learning algorithms and tools
- **[SciPy](https://scipy.org/)** - Scientific computing and statistical tests
- **[Seaborn](https://seaborn.pydata.org/)** - Statistical data visualization
- **[Matplotlib](https://matplotlib.org/)** - Plotting library

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/praj2408/Automated-Machine-Learning-App
cd Automated-Machine-Learning-App-main
```

### 2. Create a virtual environment
```bash
python -m venv .venv
```

Activate it:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Windows Command Prompt
.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Run the application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

> If you are starting from a fresh clone, make sure you run the commands from the repository root.

## 📖 Usage Guide

### 1. **Upload Your Data**
- Supported formats: CSV
- Automatic data type detection
- Real-time data preview

### 2. **Data Cleaning**
- Missing value analysis and handling
- Duplicate detection and removal
- Outlier detection with IQR method
- Data type conversion tools

### 3. **Feature Engineering**
- Categorical encoding (Label, One-Hot, Target)
- Feature scaling (Standard, Min-Max, Robust)
- Polynomial and interaction features
- Automated feature selection
- Feature importance analysis

### 4. **Advanced Profiling**
- Interactive distribution analysis
- Correlation heatmaps
- Categorical variable analysis
- Data quality assessment
- Statistical summaries

### 5. **Machine Learning**
- Automatic classification/regression detection
- Multiple algorithm comparison
- Performance metrics visualization
- Model training and evaluation

### 6. **Clustering**
- K-Means clustering
- 2D/3D cluster visualization
- Cluster statistics and analysis
- Export clustered data

### 7. **Statistical Testing**
- Correlation analysis
- Normality tests
- T-tests and ANOVA
- Descriptive statistics

### 8. **Time Series Analysis**
- Time series visualization
- Moving average analysis
- Trend analysis
- Statistical summaries

### 9. **Model Explainability**
- Feature importance plots
- Model characteristics
- Feature impact analysis

### 10. **Reports & Export**
- Comprehensive project reports
- Data quality scoring
- Automated recommendations
- Multiple export formats

## 🎯 Key Capabilities

### Data Analysis
- **Automated EDA**: One-click exploratory data analysis
- **Quality Assessment**: Automatic data quality scoring and issue detection
- **Statistical Analysis**: Comprehensive statistical tests and summaries

### Machine Learning
- **AutoML**: Automatic algorithm selection and hyperparameter tuning
- **Model Comparison**: Side-by-side performance comparison of multiple algorithms
- **Interpretability**: Feature importance and model explainability tools

### Visualization
- **Interactive Charts**: Plotly-based interactive visualizations
- **Statistical Plots**: Distribution, correlation, and statistical plots
- **3D Visualization**: 3D clustering and scatter plots

### Export & Reporting
- **Multiple Formats**: CSV, JSON, pickle files
- **Comprehensive Reports**: Detailed project documentation
- **Session Management**: Save and export complete analysis sessions

## 📊 Supported Algorithms

### Classification
- Random Forest
- Logistic Regression
- Support Vector Machines (SVM)
- Decision Trees
- K-Nearest Neighbors (KNN)

### Regression
- Random Forest Regressor
- Linear Regression
- Support Vector Regression (SVR)
- Decision Tree Regressor
- K-Nearest Neighbors Regressor

### Clustering
- K-Means Clustering
- Customizable cluster numbers
- 2D/3D visualization

## 🔧 Requirements

- Python 3.11+ (3.14 recommended for this project)
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- SciPy
- Matplotlib
- Seaborn
- Jinja2
- OpenPyXL
- PyArrow

Install everything with:

```bash
python -m pip install -r requirements.txt
```

## 📈 Performance Features

- **Memory Optimization**: Efficient handling of large datasets
- **Interactive UI**: Real-time feedback and updates
- **Error Handling**: Comprehensive error management
- **Progress Tracking**: Visual progress indicators
- **Auto-save**: Automatic session saving

## 🎨 User Experience

- **Intuitive Navigation**: Organized sidebar with clear sections
- **Visual Feedback**: Success/warning/info messages
- **Responsive Design**: Works on different screen sizes
- **Professional UI**: Modern, clean interface
- **Helpful Tooltips**: Contextual guidance

## 🔄 Workflow Integration

This platform supports the complete data science workflow:

1. **Data Ingestion** → Upload and validate data
2. **Data Cleaning** → Handle quality issues
3. **Feature Engineering** → Create optimal features
4. **Exploratory Analysis** → Understand data patterns
5. **Model Development** → Train and compare models
6. **Evaluation** → Assess model performance
7. **Interpretation** → Understand model decisions
8. **Deployment** → Export models and reports

## 🛠️ Troubleshooting

If the app does not start on a fresh machine:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

If `streamlit` is missing, install dependencies again from the project root. If a port is already busy, use:

```bash
streamlit run app.py --server.port 8501
```

## 🤝 Contributing

Contributions are welcome! Please follow the standard GitHub workflow for pull requests.

## 📧 Contact

For questions or support, please contact the project team.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🎉 What's New in This Version

- ✅ **Complete Data Science Workflow**: From data cleaning to model deployment
- ✅ **Professional UI**: Modern, intuitive interface with organized sections
- ✅ **Advanced Analytics**: Statistical testing, time series, clustering
- ✅ **Model Comparison**: Compare multiple algorithms side-by-side
- ✅ **Interactive Visualizations**: Plotly-based charts and dashboards
- ✅ **Comprehensive Reports**: Automated data quality assessment and recommendations
- ✅ **Export Capabilities**: Multiple formats for models, data, and reports
- ✅ **Error Handling**: Robust error management and user feedback
- ✅ **Python 3.14 Compatible**: Updated for latest Python version

This is now a **complete data science platform** that covers all aspects of a data analyst's and data scientist's work! 🚀
