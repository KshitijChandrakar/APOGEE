
# placed after main.py

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# %%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# %%
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_train, test_size=0.3)

# %%
# Initialize and fit the model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
importances = rf.feature_importances_
feature_names = df.columns

# Create a DataFrame for visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print(importance_df.to_string(index=False))
