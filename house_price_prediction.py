import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import seaborn
#instead of taking a new dataset we can use default datasets from sklearn
from sklearn.datasets import fetch_california_housing
#organizing data into excel sheet
df=pd.DataFrame(fetch_california_housing().data,columns=fetch_california_housing().feature_names)
df['Price']=fetch_california_housing().target
print(df.head())
#converting them into graphs
X=df.drop('Price',axis=1)
Y=df['Price'] 
#training the model
from sklearn.model_selection  import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
#spliting and scaling the data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
print("Data split and scaled successfully")
#using linear regression
from sklearn.linear_model import LinearRegression
model=LinearRegression()
#predicting the accuracy score
model.fit(X_train_scaled,Y_train)
print("Model Training complete")
from sklearn.metrics import mean_squared_error,r2_score
predictions=model.predict(X_test_scaled)
score=r2_score(Y_test,predictions)
error=mean_squared_error(Y_test,predictions)
print(f"Accuracy Score (R2): {score:.2f}")
print(f"Mean Squared Error: {error:.2f}")
import numpy as np

# Let's define a 'fake' house

new_house = np.array([[8.0, 15.0, 6.0, 1.0, 300.0, 3.0, 34.0, -118.0]])

#scaling the data
new_house_scaled = scaler.transform(new_house)

# 2. Predict
predicted_price = model.predict(new_house_scaled)

# The dataset is in units of $100,000, so we multiply by 100,000
final_price = predicted_price[0] * 100000

print(f"The predicted price for this house is: ${final_price:,.2f}")
import tkinter as tk
from tkinter import messagebox

def get_prediction():
    try:
        # 1. Get values from the input boxes
        income = float(ent_income.get())
        age = float(ent_age.get())
        rooms = float(ent_rooms.get())
        pop = float(ent_pop.get())
        
        # 2. Prepare the data (filling in defaults for the 4 features we didn't ask for)
        # Format: [Income, Age, Rooms, Bedrooms, Population, Occupancy, Lat, Long]
        user_data = np.array([[income, age, rooms, 1.0, pop, 3.0, 34.0, -118.0]])
        
        # 3. Scale and Predict
        user_data_scaled = scaler.transform(user_data)
        prediction = model.predict(user_data_scaled)
        
        # 4. Show the result
        final_price = prediction[0] * 100000
        lbl_result.config(text=f"Predicted Price: ${final_price:,.2f}", fg="green")
        
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers!")
import joblib

# Save the trained model to a file
joblib.dump(model, 'house_model.pkl')

# Save the scaler 
joblib.dump(scaler, 'scaler.pkl')

print("Model and Scaler saved successfully as .pkl files!")
model = joblib.load('house_model.pkl')
scaler = joblib.load('scaler.pkl')        

# --- Setup the Window ---
root = tk.Tk()
root.title("House Price Predictor")
root.geometry("300x400")

tk.Label(root, text="House Price Predictor", font=("Arial", 14, "bold")).pack(pady=10)

# Input Fields
tk.Label(root, text="Median Income (tens of thousands):").pack()
ent_income = tk.Entry(root)
ent_income.pack()

tk.Label(root, text="House Age:").pack()
ent_age = tk.Entry(root)
ent_age.pack()

tk.Label(root, text="Average Rooms:").pack()
ent_rooms = tk.Entry(root)
ent_rooms.pack()

tk.Label(root, text="Neighborhood Population:").pack()
ent_pop = tk.Entry(root)
ent_pop.pack()

# Predict Button
btn_predict = tk.Button(root, text="Predict Price", command=get_prediction, bg="blue", fg="white")
btn_predict.pack(pady=20)

# Result Label
lbl_result = tk.Label(root, text="", font=("Arial", 12, "bold"))
lbl_result.pack()

root.mainloop()


