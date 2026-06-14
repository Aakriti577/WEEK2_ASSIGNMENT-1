# Import Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# Q1: Load dataset and display first five records
df = pd.read_csv("Dataset 2.csv")
print("First Five Records:")
print(df.head())

# Q2: Number of rows and columns
print("\nShape of Dataset:")
print(df.shape)

# Q3: Display all column names
print("\nColumn Names:")
print(df.columns)

# Q4: Identify numerical and categorical features
print("\nNumerical Features:")
print(df.select_dtypes(include=['int64', 'float64']).columns)
print("\nCategorical Features:")
print(df.select_dtypes(include=['object']).columns)

# Q5: Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Q6: Average age of users
print("\nAverage Age:")
print(df['Age'].mean())

# Q7: Average watch hours per week
print("\nAverage Watch Hours Per Week:")
print(df['WatchHoursPerWeek'].mean())

# Q8: Average monthly spending
print("\nAverage Monthly Spending:")
print(df['MonthlySpend'].mean())

# Q9: Count users in each subscription category
print("\nUsers in Each Subscription Category:")
print(df['SubscriptionType'].value_counts())

# Q10: Percentage of users who renewed subscriptions
renew_percentage = (
    df['SubscriptionRenewed'].value_counts(normalize=True) * 100
)
print("\nSubscription Renewal Percentage:")
print(renew_percentage)

# Q11: Convert categorical features into numerical form
df_encoded = df.copy()
le = LabelEncoder()
for col in df_encoded.select_dtypes(include='object').columns:
    df_encoded[col] = le.fit_transform(df_encoded[col])
print("\nEncoded Dataset:")
print(df_encoded.head())

# Q12: Define feature set (X) and target variable (y)
X = df_encoded.drop(['SubscriptionRenewed', 'UserID'], axis=1)
y = df_encoded['SubscriptionRenewed']
print("\nFeatures (X):")
print(X.head())
print("\nTarget Variable (y):")
print(y.head())

# Q13: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
print("\nTraining Set Shape:", X_train.shape)
print("Testing Set Shape:", X_test.shape)

# Q14: Train Decision Tree Model
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Q15: Evaluate using accuracy
y_pred_dt = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)
print("\nDecision Tree Accuracy:")
print(dt_accuracy)

# Q16: Confusion Matrix
cm = confusion_matrix(y_test, y_pred_dt)
print("\nConfusion Matrix:")
print(cm)

# Q17: Train KNN Classifier (K = 5)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Q18: Compare KNN and Decision Tree Accuracy
y_pred_knn = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
print("\nKNN Accuracy:")
print(knn_accuracy)
print("\nDecision Tree Accuracy:")
print(dt_accuracy)

if dt_accuracy > knn_accuracy:
    print("\nDecision Tree performed better.")
elif knn_accuracy > dt_accuracy:
    print("\nKNN performed better.")
else:
    print("\nBoth models performed equally.")

# Q19: Train Linear Regression Model
X_reg = df_encoded.drop(['MonthlySpend', 'UserID'], axis=1)
y_reg = df_encoded['MonthlySpend']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)
lr = LinearRegression()
lr.fit(X_train_reg, y_train_reg)

# Q20: Predict Monthly Spending for a New User
new_user = [[25, 1, 2, 15, 2, 1, 5, 1]]
predicted_spending = lr.predict(new_user)
print("\nPredicted Monthly Spending:")
print(predicted_spending[0])