import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from scipy import stats
from scipy.stats.mstats import winsorize
from scipy.stats import boxcox
import io
from packaging import version
import missingno as msno
import category_encoders as ce  # For binary and other encodings
import traceback


# Machine Learning imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, davies_bouldin_score
)

# Page configuration
st.set_page_config(page_title="Data Preprocessing & ML Tool", layout="wide")

# Title
st.title("📊 Data Preprocessing & Machine Learning Tool")

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Your Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        # Read the file
        df = pd.read_csv(uploaded_file)

    else:
        st.info("Please upload a CSV file to get started.")
        st.stop()

# Main content

def show_missing_value_percent(df):
    st.subheader("🕳 Missing Values (% per column)")
    # حساب النسبة المئوية للقيم المفقودة
    missing_ratio = df.isnull().sum() / len(df) * 100
    missing_ratio = missing_ratio[missing_ratio > 0].sort_values(ascending=False)

    if missing_ratio.empty:
        st.success("✅ No missing values in the dataset.")
    else:
        st.dataframe(missing_ratio.to_frame(name="Missing %"))
        st.info(f"🔢 Number of columns with missing values: {len(missing_ratio)}")
        
        
    # Data Overview Section
st.header("📊 Data Overview")
# Display original data
st.subheader("Original Data")
st.dataframe(df.head())
# Display  data info
st.subheader("🧾 Data Info")
buffer = io.StringIO()
df.info(buf=buffer)
info_str = buffer.getvalue()
st.code(info_str, language='text')
# display data description
st.subheader(" 🧮 Data Description")
st.dataframe(df.describe())   
    
#Duplicates Section
st.header("Duplicate Values")

try:
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            st.warning(f"⚠ Number of duplicate rows: {duplicate_count}")
            st.dataframe(df[df.duplicated()])
        else:
            st.success("✅ No duplicate values found.")
except Exception as e:
        st.error(f"❌ Error: {e}")


# Handle Duplicate Values

try:
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            df = df.drop_duplicates()
            st.session_state['df'] = df  # تحديث البيانات داخل الجلسة
            st.success(f"🧹 Removed {duplicate_count} duplicate rows successfully!")
        else:
            st.success("✅ No duplicate values found.")
except Exception as e:
        st.error(f"❌ Error while removing duplicates: {e}")
      
show_missing_value_percent(df)

st.subheader("Drop Columns with Missing Values")

columns_to_drop = st.multiselect(
        "Select columns to drop (delete)", 
        options=df.columns,
        help="Select multiple columns to remove from the dataset"
    )


st.button("Drop Selected Columns")
if columns_to_drop:
    df.drop(columns=columns_to_drop, inplace=True, axis=1)
    st.success(f"Successfully dropped columns: {', '.join(columns_to_drop)}")
            
        

st.dataframe(df.head())
            
          


# 2. Visualization for missing values
st.header("Missing Values Visualization")

option = st.radio(
    "Choose way for visualization",
    ("Matrix", "Heatmap", "Bar Chart"),
    horizontal=True
)

if option == "Matrix":
    fig, ax = plt.subplots(figsize=(10, 6))
    msno.matrix(df, ax=ax, color=(0.53, 0.81, 0.92))
    ax.set_title("Missing Values Matrix")
    st.pyplot(fig)

if option == "Heatmap":
    fig, ax = plt.subplots(figsize=(10, 6))
    msno.heatmap(df, ax=ax, cmap='magma')
    ax.set_title("Missing Values Heatmap")
    st.pyplot(fig)

if option == "Bar Chart":
    fig, ax = plt.subplots(figsize=(10, 6))
    msno.bar(df, ax=ax, color='teal')
    ax.set_title("Missing Values Bar Chart")
    st.pyplot(fig)

missing_cols = df.columns[df.isnull().any()].tolist()
df_missing = df[missing_cols]

if len(missing_cols) > 0:
    st.subheader("Distribution of Columns with Missing Values")
    st.write("Histograms to guide imputation strategy:")
    
    for col in missing_cols:
        st.markdown(f"**{col}** (Missing: {df[col].isnull().sum()} values)")
        
        fig, ax = plt.subplots(figsize=(8, 3))
        
        if pd.api.types.is_numeric_dtype(df[col]):
            sns.histplot(df[col], kde=True, color="skyblue", bins=30)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
        else:
            sns.countplot(x=df[col], palette="viridis")
            plt.title(f"Categories in {col}")
            plt.xticks(rotation=45)
        
        st.pyplot(fig)
        st.write("---")
else:
    st.success("No missing values found in the dataset!")


# 3. Missing Values Imputation
st.header("3. Handle Missing Values")
# Get columns with missing values
missing_cols = df.columns[df.isnull().any()].tolist()


    
missing_stats_before = pd.DataFrame({
        'Column': missing_cols,
        'Missing Count': [df[col].isnull().sum() for col in missing_cols],
        'Missing %': [round(df[col].isnull().mean()*100, 2) for col in missing_cols],
        'Data Type': [df[col].dtype for col in missing_cols]
    }).sort_values('Missing Count', ascending=False)
    
st.write("### Missing Values Statistics (Before)")
st.dataframe(missing_stats_before.style.format({'Missing %': '{:.2f}%'}))

    # Let user select which columns to impute
cols_to_impute = st.multiselect(
        "Select columns to impute", 
        options=missing_cols,
        default=missing_cols,
        help="Choose which columns to apply missing value imputation"
    )

    # Initialize imputation_settings dictionary
imputation_settings = {}
    
for col in cols_to_impute:
        st.markdown(f"### Column: {col}")
        st.write(f"- Current missing values: {df[col].isnull().sum()} ({round(df[col].isnull().mean()*100, 2)}%)")
        
        # Initialize variables for this column
        strategy = None
        fill_value = None
        n_neighbors = None
        max_iter = None
        
        # Determine column type
        col_type = "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "categorical"
        
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Select imputation method based on column type
            if col_type == "numeric":
                method = st.selectbox(
                    f"Method for {col}",
                    ["Simple Imputer", "KNN Imputer", "Iterative Imputer", "Fill with value"],
                    key=f"method_{col}"
                )
            else:
                method = st.selectbox(
                    f"Method for {col}",
                    ["Simple Imputer", "Fill with value"],
                    key=f"method_{col}"
                )
        
        with col2:
            # Show method-specific options
            if method == "Simple Imputer":
                if col_type == "numeric":
                    strategy = st.selectbox(
                        f"Strategy for {col}",
                        ["mean", "median", "most_frequent", "constant"],
                        key=f"strategy_{col}"
                    )
                    if strategy == "constant":
                        fill_value = st.number_input(
                            f"Fill value for {col}",
                            value=0,
                            key=f"fill_{col}"
                        )
                else:
                    strategy = st.selectbox(
                        f"Strategy for {col}",
                        ["most_frequent", "constant"],
                        key=f"strategy_{col}"
                    )
                    if strategy == "constant":
                        fill_value = st.text_input(
                            f"Fill value for {col}",
                            value="UNKNOWN",
                            key=f"fill_{col}"
                        )
            
            elif method == "Fill with value":
                if col_type == "numeric":
                    fill_value = st.number_input(
                        f"Fill value for {col}",
                        value=0,
                        key=f"fill_{col}"
                    )
                else:
                    fill_value = st.text_input(
                        f"Fill value for {col}",
                        value="UNKNOWN",
                        key=f"fill_{col}"
                    )
            
            elif method == "KNN Imputer":
                n_neighbors = st.number_input(
                    f"Number of neighbors for {col}",
                    min_value=1,
                    value=5,
                    key=f"neighbors_{col}"
                )
            
            elif method == "Iterative Imputer":
                max_iter = st.number_input(
                    f"Max iterations for {col}",
                    min_value=1,
                    value=10,
                    key=f"iter_{col}"
                )
        
        # Store settings for this column
        imputation_settings[col] = {
            "type": col_type,
            "method": method,
            "strategy": strategy,
            "fill_value": fill_value,
            "n_neighbors": n_neighbors,
            "max_iter": max_iter
        }

    # Apply imputation when button clicked

        
        
        # Group columns by imputation method for efficiency
simple_numeric = []
simple_categorical = []
fill_values = {}
knn_cols = []
iterative_cols = []

for col, settings in imputation_settings.items():
    if settings["method"] == "Simple Imputer":
        if settings["type"] == "numeric":
            simple_numeric.append((col, settings["strategy"], settings["fill_value"]))
        else:
            simple_categorical.append((col, settings["strategy"], settings["fill_value"]))
    elif settings["method"] == "Fill with value":
        fill_values[col] = settings["fill_value"]
    elif settings["method"] == "KNN Imputer":
        knn_cols.append(col)
    elif settings["method"] == "Iterative Imputer":
        iterative_cols.append(col)

# Apply SimpleImputer to numeric columns
for col, strategy, fill_value in simple_numeric:
    if strategy == "constant":
        imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
    else:
        imputer = SimpleImputer(strategy=strategy)
    df[[col]] = imputer.fit_transform(df[[col]])

# Apply SimpleImputer to categorical columns
for col, strategy, fill_value in simple_categorical:
    if strategy == "constant":
        imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
    else:
        imputer = SimpleImputer(strategy=strategy)
    df[[col]] = imputer.fit_transform(df[[col]])

# Apply fill with value
for col, value in fill_values.items():
    df[col] = df[col].fillna(value)

# Apply KNN Imputer (must be done together for all KNN columns)
if knn_cols:
    n_neighbors = imputation_settings[knn_cols[0]]["n_neighbors"]
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    df[knn_cols] = knn_imputer.fit_transform(df[knn_cols])

# Apply Iterative Imputer (must be done together for all iterative columns)
if iterative_cols:
    max_iter = imputation_settings[iterative_cols[0]]["max_iter"]
    iterative_imputer = IterativeImputer(max_iter=max_iter, random_state=42)
    df[iterative_cols] = iterative_imputer.fit_transform(df[iterative_cols])

# Show after imputation results
st.subheader("Imputation Results")

# Calculate missing values after imputation
missing_stats_after = pd.DataFrame({
    'Column': cols_to_impute,
    'Missing Before': [df[col].isnull().sum() for col in cols_to_impute],
    'Missing After': [df[col].isnull().sum() for col in cols_to_impute],
    'Method': [imputation_settings[col]["method"] for col in cols_to_impute],
    'Status': ["✅ Completed" if df[col].isnull().sum() == 0 
            else "⚠ Partial" for col in cols_to_impute]
})

st.write("### Missing Values Count Comparison")
st.dataframe(missing_stats_after)

# Visualize remaining missing values
if df.isnull().any().any():
    st.subheader(" Missing Values After Imputation")
    fig_after, ax_after = plt.subplots(figsize=(12, 6))
    msno.matrix(df, ax=ax_after, color=(0.8, 0.3, 0.3))
    ax_after.set_title("Remaining Missing Values", fontsize=14)
    st.pyplot(fig_after)
    plt.close(fig_after)
else:
    st.success("🎉 All missing values have been successfully imputed!")
    
    # Show quick distribution comparison
    st.subheader(" Missing Values After Imputation")
    fig_after, ax_after = plt.subplots(figsize=(12, 6))
    msno.matrix(df, ax=ax_after, color=(0.8, 0.3, 0.3))
    ax_after.set_title("Remaining Missing Values", fontsize=14)
    st.pyplot(fig_after)
    plt.close(fig_after)
        
        
        
        # Update the dataframe



df_num = df.select_dtypes(include=[np.number])

# Draw Boxplots
st.subheader("📦 show outliers")

if not df_num.empty:
    num_cols = df_num.columns
    
    # Draw boxplots for each numeric column in separate subplot
    fig, axes = plt.subplots(nrows=1, ncols=len(num_cols), figsize=(5 * len(num_cols), 5))
    
    if len(num_cols) == 1:
        axes = [axes]  # Convert to list if single column
                
    for i, col in enumerate(num_cols):
        sns.boxplot(data=df_num, y=col, ax=axes[i], color="#b480b8")
        axes[i].set_title(f'Boxplot of {col}', fontsize=14)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    
    

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

  
    # 5-6. Outlier Detection & Handling
st.header("5-6. Outlier Detection & Handling")

    # Get numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

if not numeric_cols:
    st.warning("No numeric columns found for outlier detection!")
else:
    # Let user select which columns to process
    selected_cols = st.multiselect(
        "Select columns for outlier handling",
        options=numeric_cols,
        default=numeric_cols,
        help="Choose which numeric columns to analyze and process"
    )
    
    if not selected_cols:
        st.warning("Please select at least one column")
        st.stop()
    
    # Outlier detection settings
    out_col1, out_col2 = st.columns(2)
    
    with out_col1:
        outlier_method = st.selectbox("Select outlier detection method", ['Z-Score', 'IQR'])
    
    with out_col2:
        handle_method = st.selectbox("Choose how to handle outliers", 
                                ['Remove outliers', 'Winsorization', 'Clip outliers'])
    
    # Button to detect and handle outliers in selected columns


original_row_count = len(df)
results = []
        
# Create figure for before/after comparison
fig, axes = plt.subplots(len(selected_cols), 2, figsize=(15, 5*len(selected_cols)))

if len(selected_cols) == 1:
    axes = [axes]  # Ensure axes is 2D even for single column

for i, col in enumerate(selected_cols):
    # Before handling - left column
    sns.boxplot(y=df[col], ax=axes[i][0], color='skyblue')
    axes[i][0].set_title(f"{col} (Before)")
    
    # Detect outliers
    if outlier_method == 'Z-Score':
        z_scores = np.abs(stats.zscore(df[col]))
        outliers_mask = z_scores > 3
        lower_bound = None
        upper_bound = None
    else:  # IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
    
    outlier_count = sum(outliers_mask)
    
    # Handle outliers
    if handle_method == 'Remove outliers':
        if outlier_method == 'Z-Score':
            df = df[z_scores <= 3]
        else:
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        action = "removed"
    elif handle_method == 'Winsorization':
        df[col] = winsorize(df[col], limits=[0.05, 0.05])
        action = "winsorized"
    else:  # Clip outliers
        if lower_bound is not None and upper_bound is not None:
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        action = "clipped"
    
    # After handling - right column
    sns.boxplot(y=df[col], ax=axes[i][1], color='lightgreen')
    axes[i][1].set_title(f"{col} (After {action})")
    
    

plt.tight_layout()

# Show before/after comparison
st.subheader("Before vs After Outlier Handling")
st.pyplot(fig)


# Show row count changes if rows were removed
if handle_method == 'Remove outliers':
    new_row_count = len(df)
    st.write(f"📊 Row count before: {original_row_count} | After: {new_row_count}")
    st.write(f"📉 Rows removed: {original_row_count - new_row_count}")
    

# Update the data in session state

st.success(f"Outlier handling completed for {len(selected_cols)} selected columns!")



st.header("📊 Data Visualization")

# Create tabs for different visualization types
viz_tab1, viz_tab2 = st.tabs(["Numerical Data", "Categorical Data"])

with viz_tab1:
    st.subheader("Numerical Data Visualization")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if not numeric_cols:
        st.warning("No numerical columns found for visualization!")
    else:
        # Single variable plots
        st.markdown("### Single Variable Analysis")
        num_col1 = st.selectbox("Select numerical column", numeric_cols, key='num_col1')
        
        plot_type = st.radio("Select plot type", 
                           ["Histogram", "Density Plot", "Box Plot", "Violin Plot"], 
                           key='num_plot_type')
        
        if st.button("Generate Numerical Plot"):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if plot_type == "Histogram":
                sns.histplot(df[num_col1], ax=ax, kde=False, color='skyblue')
                ax.set_title(f"Histogram of {num_col1}")
            elif plot_type == "Density Plot":
                sns.kdeplot(df[num_col1], ax=ax, color='purple', fill=True)
                ax.set_title(f"Density Plot of {num_col1}")
            elif plot_type == "Box Plot":
                sns.boxplot(y=df[num_col1], ax=ax, color='lightgreen')
                ax.set_title(f"Box Plot of {num_col1}")
            elif plot_type == "Violin Plot":
                sns.violinplot(y=df[num_col1], ax=ax, color='orange')
                ax.set_title(f"Violin Plot of {num_col1}")
            
            st.pyplot(fig)
            plt.close(fig)
        
        # Two variable plots
        st.markdown("### Two Variable Analysis")
        col1, col2 = st.columns(2)
        with col1:
            num_col_x = st.selectbox("Select X-axis column", numeric_cols, key='num_col_x')
        with col2:
            num_col_y = st.selectbox("Select Y-axis column", numeric_cols, key='num_col_y')
        
        two_var_plot_type = st.radio("Select plot type", 
                                    ["Scatter Plot", "Line Plot", "Hexbin Plot"], 
                                    key='two_var_plot')
        
        if st.button("Generate Two Variable Plot"):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if two_var_plot_type == "Scatter Plot":
                sns.scatterplot(data=df, x=num_col_x, y=num_col_y, ax=ax, alpha=0.7)
                ax.set_title(f"Scatter Plot: {num_col_x} vs {num_col_y}")
            elif two_var_plot_type == "Line Plot":
                sns.lineplot(data=df, x=num_col_x, y=num_col_y, ax=ax)
                ax.set_title(f"Line Plot: {num_col_x} vs {num_col_y}")
            elif two_var_plot_type == "Hexbin Plot":
                plt.hexbin(df[num_col_x], df[num_col_y], gridsize=25, cmap='Blues')
                plt.colorbar()
                ax.set_title(f"Hexbin Plot: {num_col_x} vs {num_col_y}")
            
            st.pyplot(fig)
            plt.close(fig)

with viz_tab2:
    st.subheader("Categorical Data Visualization")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not cat_cols:
        st.warning("No categorical columns found for visualization!")
    else:
        # Single categorical variable plots
        st.markdown("### Single Variable Analysis")
        cat_col = st.selectbox("Select categorical column", cat_cols, key='cat_col')
        
        cat_plot_type = st.radio("Select plot type", 
                               ["Bar Plot", "Pie Chart", "Lollipop Plot", "Treemap"],
                               key='cat_plot_type')
        
        if st.button("Generate Categorical Plot"):
            fig, ax = plt.subplots(figsize=(10, 6))
            value_counts = df[cat_col].value_counts()
            
            if cat_plot_type == "Bar Plot":
                sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax, palette='viridis')
                ax.set_title(f"Bar Plot of {cat_col}")
                ax.tick_params(axis='x', rotation=45)
            elif cat_plot_type == "Pie Chart":
                ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', 
                      startangle=90, colors=sns.color_palette('pastel'))
                ax.set_title(f"Pie Chart of {cat_col}")
            elif cat_plot_type == "Lollipop Plot":
                ax.stem(value_counts.index, value_counts.values, basefmt=" ")
                ax.set_title(f"Lollipop Plot of {cat_col}")
                ax.tick_params(axis='x', rotation=45)
            elif cat_plot_type == "Treemap":
                import squarify
                squarify.plot(sizes=value_counts.values, 
                            label=value_counts.index, 
                            color=sns.color_palette('Spectral', len(value_counts)))
                ax.set_title(f"Treemap of {cat_col}")
                ax.axis('off')
            
            st.pyplot(fig)
            plt.close(fig)
        
        # Two variable analysis (categorical vs numerical)
        st.markdown("### Categorical vs Numerical Analysis")
        col1, col2 = st.columns(2)
        with col1:
            cat_col_x = st.selectbox("Select categorical column", cat_cols, key='cat_col_x')
        with col2:
            num_col_y = st.selectbox("Select numerical column", numeric_cols, key='num_col_y2')
        
        cat_num_plot_type = st.radio("Select plot type", 
                                   ["Box Plot", "Violin Plot", "Swarm Plot", "Bar Plot"],
                                   key='cat_num_plot')
        
        if st.button("Generate Categorical-Numerical Plot"):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if cat_num_plot_type == "Box Plot":
                sns.boxplot(data=df, x=cat_col_x, y=num_col_y, ax=ax, palette='Set2')
                ax.set_title(f"Box Plot: {cat_col_x} vs {num_col_y}")
            elif cat_num_plot_type == "Violin Plot":
                sns.violinplot(data=df, x=cat_col_x, y=num_col_y, ax=ax, palette='Set2')
                ax.set_title(f"Violin Plot: {cat_col_x} vs {num_col_y}")
            elif cat_num_plot_type == "Swarm Plot":
                sns.swarmplot(data=df, x=cat_col_x, y=num_col_y, ax=ax, palette='Set2')
                ax.set_title(f"Swarm Plot: {cat_col_x} vs {num_col_y}")
            elif cat_num_plot_type == "Bar Plot":
                sns.barplot(data=df, x=cat_col_x, y=num_col_y, ax=ax, palette='Set2', 
                           estimator=np.mean, ci=95)
                ax.set_title(f"Bar Plot (Mean): {cat_col_x} vs {num_col_y}")
            
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
            plt.close(fig)



st.header("🔗 Correlation Analysis")

if not numeric_cols:
    st.warning("No numerical columns found for correlation analysis!")
else:
    st.markdown("### Correlation Matrix")
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title("Correlation Matrix of Numerical Features")
    st.pyplot(fig)
    plt.close(fig)
    
    st.markdown("### Top Correlated Pairs")
    
    # Get top correlated pairs
    corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
    top_pairs = corr_pairs[corr_pairs != 1].head(10)
    
    # Display top pairs
    st.write("Top 10 most correlated feature pairs:")
    st.dataframe(top_pairs.reset_index().rename(columns={'level_0': 'Feature 1', 
                                                      'level_1': 'Feature 2', 
                                                      0: 'Correlation'}))







st.header("4. Categorical Data Encoding")

# تحديد الأعمدة التصنيفية
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

if categorical_cols:
    col1, col2 = st.columns(2)

    # ----------------------------
    # العمود الأول: اختيار طريقة الترميز لكل عمود
    # ----------------------------
    with col1:
        st.subheader("Select Columns and Encoding Method")

        # تخزين الإعدادات مؤقتاً في قاموس
        encoding_settings = {}

        for col in categorical_cols:
            method = st.selectbox(
                f"Encoding method for column *{col}*",
                ['None', 'Label Encoding', 'One-Hot Encoding', 'Binary Encoding', 'Most Frequent Encoding'],
                key=f"encode_{col}"
            )
            encoding_settings[col] = method
    with col2:
        st.subheader("Preview & Apply")


try:
        for col, method in encoding_settings.items():
            if method == 'Label Encoding':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                st.success(f"✅ Label Encoding applied to '{col}'")

            elif method == 'One-Hot Encoding':
                ohe = OneHotEncoder(sparse_output=False)
                encoded = ohe.fit_transform(df[[col]])
                new_cols = ohe.get_feature_names_out([col])
                df_encoded = pd.DataFrame(encoded, columns=new_cols, index=df.index)
                df.drop(columns=[col], inplace=True)
                df = pd.concat([df, df_encoded], axis=1)
                st.success(f"✅ One‑Hot Encoding applied to '{col}'")

            elif method == 'Binary Encoding':
                be = ce.BinaryEncoder(cols=[col])
                df = be.fit_transform(df)
                st.success(f"✅ Binary Encoding applied to '{col}'")

            elif method == 'Most Frequent Encoding':
                ce_count = ce.CountEncoder(cols=[col], normalize=True)
                df[f"{col}_MostFreq"] = ce_count.fit_transform(df[[col]])
                df.drop(columns=[col], inplace=True)
                st.success(f"✅ Most-Frequent Encoding applied to '{col}'")

            # إذا كانت 'None' → لا تفعل شيئاً

        # عرض البيانات بعد التغيير
        st.success("🎉 All selected encodings applied successfully!")

except Exception as e:
        st.error(f"⚠ Error: {e}")


st.dataframe(df.head())



# Correlation Analysis Section



# 7-8. Feature Scaling
# Enhanced Feature Scaling Section
st.header("7-8. Feature Scaling")

scale_col1, scale_col2 = st.columns(2)

with scale_col1:
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    scale_columns = st.multiselect("Select numeric columns to scale", 
                                    numeric_cols if numeric_cols else [])

with scale_col2:
    scale_method = st.selectbox("Select scaling method", 
                                ['Normalization (MinMaxScaler)', 'Standardization (StandardScaler)'])

if len(scale_columns) > 0:
    if scale_method == 'Normalization (MinMaxScaler)':
        scaler = MinMaxScaler()
        df[scale_columns] = scaler.fit_transform(df[scale_columns])
        st.success("Applied MinMax scaling to selected columns")
        st.dataframe(df.head())
    else:
        scaler = StandardScaler()
        df[scale_columns] = scaler.fit_transform(df[scale_columns])
        st.success("Applied Standard scaling to selected columns")
        st.dataframe(df.head())

        
# 9-10. Skewness handling
st.header("9-10. Skewness Analysis & Transformation")

skew_col1, skew_col2 = st.columns(2)

with skew_col1:
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    skew_column = st.selectbox("Select numeric column for skewness analysis", 
                                ['None'] + numeric_cols if numeric_cols else ['None'])

if skew_column != 'None':
    # Calculate skewness
    skewness = df[skew_column].skew()
    st.write(f"Skewness of {skew_column}: {skewness:.2f}")
    
    # Interpret skewness
    if abs(skewness) < 0.5:
        st.success("The distribution is approximately symmetric")
    elif 0.5 <= abs(skewness) < 1:
        st.warning("The distribution is moderately skewed")
    else:
        st.error("The distribution is highly skewed")
    
    # Show distribution plot
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df[skew_column], kde=True, ax=ax,color='#ef8d56')
    plt.title(f'Distribution of {skew_column}')
    st.pyplot(fig)
    
    # Transformation options
    with skew_col2:
        transform_method = st.selectbox("Select transformation method", 
                                        ['Log Transformation', 'Box-Cox Transformation'])
    
    st.button("Apply Transformation")
    if transform_method == 'Log Transformation':
            if (df[skew_column] <= 0).any():
                st.warning("Log transform requires positive values. Adding constant to make values positive.")
                constant = abs(df[skew_column].min()) + 1
                df[skew_column] = np.log(df[skew_column] + constant)
            else:
                df[skew_column] = np.log(df[skew_column])
            st.success("Applied log transformation")
    else:  # Box-Cox
            if (df[skew_column] <= 0).any():
                st.error("Box-Cox requires strictly positive values. Cannot apply transformation.")
            else:
                df[skew_column], _ = boxcox(df[skew_column])
                st.success("Applied Box-Cox transformation")
        
        # Show after transformation
    new_skewness = df[skew_column].skew()
    st.write(f"New skewness: {new_skewness:.2f}")
        
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df[skew_column], kde=True, ax=ax)
    plt.title(f'Distribution after transformation')
    st.pyplot(fig)
        
        



st.header("Machine Learning Models")



# اختيار المشكلة ونوع النموذج
problem_type = st.radio("Select problem type", ["Classification", "Regression", "Clustering"])

if problem_type in ["Classification", "Regression"]:
    # اختيار المتغير الهدف للتصنيف/الانحدار
    target_col = st.selectbox("Select target variable", df.columns)
    
    # تقسيم البيانات
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # معالجة البيانات
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # ترميز المتغير الهدف للتصنيف
    if problem_type == "Classification":
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
    
    # تقسيم البيانات
    test_size = st.slider("Test size ratio", 0.1, 0.5, 0.2, 0.05)
    random_state = st.slider("Random state", 0, 100, 42)
    
    try:
        if problem_type == "Classification":
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, 
                test_size=test_size, 
                random_state=random_state,
                stratify=y_encoded if len(np.unique(y_encoded)) > 1 else None
            )
        else:  # Regression
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=random_state
            )
    except ValueError as e:
        st.error(f"Error in train-test split: {str(e)}")
        st.stop()

# اختيار النموذج
if problem_type == "Classification":
    st.subheader("Classification Models")
    model_name = st.selectbox("Select model", 
                            ["SVM", "Random Forest", "Naive Bayes", 
                            "Logistic Regression", "KNN", "Decision Tree"])
    
    model_params = {}
    if model_name == "SVM":
        model_params['C'] = st.slider("Regularization parameter (C)", 0.01, 10.0, 1.0)
        model_params['kernel'] = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
        model = SVC(**model_params, probability=True)
    
    elif model_name == "Random Forest":
        model_params['n_estimators'] = st.slider("Number of trees", 10, 200, 100)
        model_params['max_depth'] = st.slider("Max depth", 1, 20, 5)
        model = RandomForestClassifier(**model_params, random_state=42)
    
    elif model_name == "Naive Bayes":
        model = GaussianNB()
    
    elif model_name == "Logistic Regression":
        model_params['C'] = st.slider("Inverse of regularization strength (C)", 
                                    0.01, 10.0, 1.0)
        model_params['penalty'] = st.selectbox("Penalty", ["l2", "none"])
        model = LogisticRegression(**model_params, random_state=42, max_iter=1000)
    
    elif model_name == "KNN":
        model_params['n_neighbors'] = st.slider("Number of neighbors", 1, 20, 5)
        model_params['weights'] = st.selectbox("Weights", ["uniform", "distance"])
        model = KNeighborsClassifier(**model_params)
    
    elif model_name == "Decision Tree":
        model_params['max_depth'] = st.slider("Max depth", 1, 20, 5)
        model_params['criterion'] = st.selectbox("Criterion", ["gini", "entropy"])
        model = DecisionTreeClassifier(**model_params, random_state=42)

elif problem_type == "Regression":
    st.subheader("Regression Models")
    model_name = st.selectbox("Select model", 
                            ["Linear Regression", "SVR", "Random Forest", 
                            "KNN", "Decision Tree"])
    
    model_params = {}
    if model_name == "Linear Regression":
        model = LinearRegression()
    
    elif model_name == "SVR":
        model_params['C'] = st.slider("Regularization parameter (C)", 0.01, 10.0, 1.0)
        model_params['kernel'] = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
        model = SVR(**model_params)
    
    elif model_name == "Random Forest":
        model_params['n_estimators'] = st.slider("Number of trees", 10, 200, 100)
        model_params['max_depth'] = st.slider("Max depth", 1, 20, 5)
        model = RandomForestRegressor(**model_params, random_state=42)
    
    elif model_name == "KNN":
        model_params['n_neighbors'] = st.slider("Number of neighbors", 1, 20, 5)
        model_params['weights'] = st.selectbox("Weights", ["uniform", "distance"])
        model = KNeighborsRegressor(**model_params)
    
    elif model_name == "Decision Tree":
        model_params['max_depth'] = st.slider("Max depth", 1, 20, 5)
        model_params['criterion'] = st.selectbox("Criterion", ["squared_error", "friedman_mse"])
        model = DecisionTreeRegressor(**model_params, random_state=42)

elif problem_type == "Clustering":
    st.subheader("Clustering Models")
    model_name = st.selectbox("Select model", 
                            ["K-Means", "DBSCAN", "Agglomerative"])
    
    model_params = {}
    if model_name == "K-Means":
        model_params['n_clusters'] = st.slider("Number of clusters", 2, 10, 3)
        model = KMeans(**model_params, random_state=42)
    
    elif model_name == "DBSCAN":
        model_params['eps'] = st.slider("Epsilon (eps)", 0.1, 2.0, 0.5, 0.1)
        model_params['min_samples'] = st.slider("Minimum samples", 1, 20, 5)
        model = DBSCAN(**model_params)
    
    elif model_name == "Agglomerative":
        model_params['n_clusters'] = st.slider("Number of clusters", 2, 10, 3)
        model_params['linkage'] = st.selectbox("Linkage", ["ward", "complete", "average", "single"])
        model = AgglomerativeClustering(**model_params)

# إنشاء وتدريب النموذج
if problem_type in ["Classification", "Regression"]:
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            model_pipeline.fit(X_train, y_train)
            
            if problem_type == "Classification":
                y_pred = model_pipeline.predict(X_test)
                y_prob = model_pipeline.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
                y_test_original = le.inverse_transform(y_test)
                y_pred_original = le.inverse_transform(y_pred)
                
                # حساب المقاييس
                st.subheader("Classification Metrics")
                metrics_df = pd.DataFrame({
                    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
                    "Value": [
                        accuracy_score(y_test_original, y_pred_original),
                        precision_score(y_test_original, y_pred_original, average='weighted', zero_division=0),
                        recall_score(y_test_original, y_pred_original, average='weighted', zero_division=0),
                        f1_score(y_test_original, y_pred_original, average='weighted', zero_division=0)
                    ]
                })
                st.dataframe(metrics_df.style.format({"Value": "{:.4f}"}))
                
                if y_prob is not None and len(np.unique(y_test)) > 1:
                    try:
                        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
                        st.write(f"ROC AUC: {roc_auc:.4f}")
                    except Exception as e:
                        st.warning(f"Could not calculate ROC AUC: {str(e)}")

                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test_original, y_pred_original)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=le.classes_,
                        yticklabels=le.classes_)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)
                
                # تقرير التصنيف
                st.subheader("Classification Report")
                report = classification_report(y_test_original, y_pred_original, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format("{:.4f}"))
            
            elif problem_type == "Regression":
                y_pred = model_pipeline.predict(X_test)
                
                st.subheader("Regression Metrics")
                metrics_df = pd.DataFrame({
                    "Metric": ["MSE", "RMSE", "MAE", "R²"],
                    "Value": [
                        mean_squared_error(y_test, y_pred),
                        np.sqrt(mean_squared_error(y_test, y_pred)),
                        mean_absolute_error(y_test, y_pred),
                        r2_score(y_test, y_pred)
                    ]
                })
                st.dataframe(metrics_df.style.format({"Value": "{:.4f}"}))
                
                # رسم نتائج الانحدار
                st.subheader("Regression Plot")
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred, alpha=0.5)
                ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
                ax.set_xlabel('Actual')
                ax.set_ylabel('Predicted')
                st.pyplot(fig)

elif problem_type == "Clustering":
    if st.button("Apply Clustering"):
        with st.spinner("Clustering data..."):
            # معالجة البيانات للتجميع
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                ])
            
            X_processed = preprocessor.fit_transform(X)
            
            # تطبيق التجميع
            clusters = model.fit_predict(X_processed)
            
            # إضافة التجميع إلى البيانات الأصلية
            df_clustered = df.copy()
            df_clustered['Cluster'] = clusters
            
            st.subheader("Clustering Results")
            
            # حساب مقاييس التجميع (إذا كان ذلك ممكنًا)
            if model_name != "DBSCAN" or len(np.unique(clusters)) > 1:
                try:
                    silhouette = silhouette_score(X_processed, clusters)
                    davies_bouldin = davies_bouldin_score(X_processed, clusters)
                    
                    metrics_df = pd.DataFrame({
                        "Metric": ["Silhouette Score", "Davies-Bouldin Index"],
                        "Value": [silhouette, davies_bouldin]
                    })
                    st.dataframe(metrics_df.style.format({"Value": "{:.4f}"}))
                except:
                    st.warning("Could not calculate clustering metrics for this configuration")
            
            # عرض توزيع العناقيد
            st.subheader("Cluster Distribution")
            cluster_counts = pd.Series(clusters).value_counts().sort_index()
            st.bar_chart(cluster_counts)
            
            # عرض البيانات مع العناقيد
            st.subheader("Clustered Data Preview")
            st.dataframe(df_clustered.head())
            
            # حفظ البيانات المجمعة
            if st.button("Download Clustered Data"):
                csv = df_clustered.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="clustered_data.csv",
                    mime="text/csv"
                )