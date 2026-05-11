import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import joblib

# Column names for NSL-KDD dataset
columns = [
    "duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot",
    "num_failed_logins","logged_in","num_compromised","root_shell",
    "su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate",
    "dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty"
]

# Load training and testing datasets
train_data = pd.read_csv("data/KDDTrain+.txt", names=columns)
test_data = pd.read_csv("data/KDDTest+.txt", names=columns)

# Remove difficulty column
train_data.drop("difficulty", axis=1, inplace=True)
test_data.drop("difficulty", axis=1, inplace=True)

# Convert labels into binary values
# normal = 0
# attack = 1

attack_mapping = {

    'normal': 'Normal',

    'back': 'DoS',
    'land': 'DoS',
    'neptune': 'DoS',
    'pod': 'DoS',
    'smurf': 'DoS',
    'teardrop': 'DoS',

    'ipsweep': 'Probe',
    'nmap': 'Probe',
    'portsweep': 'Probe',
    'satan': 'Probe',

    'ftp_write': 'R2L',
    'guess_passwd': 'R2L',
    'imap': 'R2L',
    'multihop': 'R2L',

    'buffer_overflow': 'U2R',
    'loadmodule': 'U2R',
    'perl': 'U2R',
    'rootkit': 'U2R'
}

train_data['label'] = train_data['label'].map(
    attack_mapping
)

test_data['label'] = test_data['label'].map(
    attack_mapping
)
# Encode categorical columns
categorical_columns = ['protocol_type', 'service', 'flag']

encoder = LabelEncoder()

for col in categorical_columns:
    train_data[col] = encoder.fit_transform(train_data[col])
    test_data[col] = encoder.transform(test_data[col])

# Features and labels
X_train = train_data.drop('label', axis=1)
y_train = train_data['label']

X_test = test_data.drop('label', axis=1)
y_test = test_data['label']

# Create model
models = {

    "Random Forest":
    RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "Logistic Regression":
    LogisticRegression(max_iter=1000),

    "Decision Tree":
    DecisionTreeClassifier(),

    "SVM":
    SVC(),

    "XGBoost":
    XGBClassifier()
}
# Train model
model.fit(X_train, y_train)
results = []
# Make predictions
best_accuracy = 0
best_model = None

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    results.append([name,accuracy,precision,recall,f1
    ])

    print(f"\n{name}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
results_df = pd.DataFrame(
    results,
    columns=[
        'Model',
        'Accuracy',
        'Precision',
        'Recall',
        'F1'
    ]
)

results_df.plot(
    x='Model',
    y=['Accuracy','Precision','Recall','F1'],
    kind='bar',
    figsize=(10,5)
)

plt.title("Model Comparison")

plt.show()
print(classification_report(y_test, predictions))

# Save model
joblib.dump(best_model, "models/intrusion_model.pkl")

print("Model Saved Successfully")
