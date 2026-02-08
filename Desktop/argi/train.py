import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. LOAD DATA
# For now, we use a dictionary. Later, you can replace this with: pd.read_csv('your_file.csv')
data = {
    'crop': ['Rice', 'Wheat', 'Maize', 'Rice', 'Wheat', 'Maize'],
    'rain': [1000, 800, 1200, 1100, 850, 1300],
    'temp': [30, 25, 28, 29, 24, 27],
    'yield': [50, 40, 55, 52, 42, 58] # This is what we want to predict
}
df = pd.DataFrame(data)

# 2. ENCODE CATEGORICAL DATA (Turn words to numbers)
encoder = LabelEncoder()
df['crop'] = encoder.fit_transform(df['crop']) 

# 3. SPLIT DATA
X = df[['crop', 'rain', 'temp']] # Input features
y = df['yield']                 # Target output

# 4. TRAIN THE MODEL (The "Random Forest" is excellent for agriculture)
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# 5. SAVE THE "BRAIN" AND THE "TRANSLATOR"
# We save the model AND the encoder so the website knows '1' means 'Rice'
joblib.dump(model, 'crop_model.pkl')
joblib.dump(encoder, 'crop_encoder.pkl')

print("Success! Phase 1 Complete. 'crop_model.pkl' is ready.")