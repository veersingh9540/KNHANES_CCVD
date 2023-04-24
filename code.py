df = pd.read_csv('KNHANES.csv')

#Data cleaning, preprocessing and normalization

# Drop missing values
df.dropna(inplace=True)

# Split data into features and target
X = df.drop('CCVD', axis=1)
y = df['CCVD']

# Normalization
X_norm = (X - X.mean()) / X.std()

#Feature selection
# Feature selection using correlation matrix
corr_matrix = df.corr()
corr_matrix['CCVD'].sort_values(ascending=False)
# Select top n correlated features
selected_features = ['age', 'BMI', 'waist', 'SBP', 'DBP', 'cholesterol', 'triglycerides', 'HDL', 'LDL']
X_norm = X_norm[selected_features]

#Splitting the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)

#Logistic Regression 
# Model training
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)

# Model evaluation
lr_pred = lr_model.predict(X_test)
print('Logistic Regression:')
print('Accuracy: ', accuracy_score(y_test, lr_pred))
print('Precision: ', precision_score(y_test, lr_pred))
print('Recall: ', recall_score(y_test, lr_pred))
print('F1 score: ', f1_score(y_test, lr_pred))


#Decision tree

# Model training
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Model evaluation
dt_pred = dt_model.predict(X_test)
print('Decision Trees:')
print('Accuracy: ', accuracy_score(y_test, dt_pred))
print('Precision: ', precision_score(y_test, dt_pred))
print('Recall: ', recall_score(y_test, dt_pred))
print('F1 score: ', f1_score(y_test, dt_pred))


#Random forest
# Model training
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Model evaluation
rf_pred = rf_model.predict(X_test)
print('Random Forest:')
print('Accuracy: ', accuracy_score(y_test, rf_pred))
print('Precision: ', precision_score(y_test, rf_pred))
print('Recall: ', recall_score(y_test, rf_pred))
print('F1 score: ', f1_score(y_test, rf_pred))

#Neural Networks

# Model training
nn_model = MLPClassifier(random_state=42)
nn_model.fit(X_train, y_train)

# Model evaluation
nn_pred = nn_model.predict(X_test)
print('Neural Networks:')
print('Accuracy: ', accuracy_score(y_test, nn_pred))
print('Precision: ', precision_score(y_test, nn_pred))
print('Recall: ', recall_score(y_test, nn_pred))
print('F1 score: ', f







