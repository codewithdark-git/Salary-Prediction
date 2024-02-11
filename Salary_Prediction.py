import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Define the training data
fig, ax = plt.subplots()

years_of_experience = np.array([1, 2, 3, 4, 5, 6])
salary = np.array([20000, 40000 , 65000, 95000, 130000, 150000])
ax.scatter(years_of_experience, salary, color='red', label='Actual Data')

# Reshape the data (needed for scikit-learn)
years_of_experience = years_of_experience.reshape(-1, 1)

# Create and train the linear regression model
model = LinearRegression()
model.fit(years_of_experience, salary)

# Input: Years of experience from the user
years = float(input("Enter the years of experience: "))

# Predict the salary using the trained model
predicted_salary = model.predict(np.array([[years]]))

print(f"Predicted Salary: ${predicted_salary[0]:.2f} for the {years} Years of experience : ")

ax.legend()
ax.set_xlabel('Years of Experience')
ax.set_ylabel('Salary')
ax.set_title('Years of Experience vs Salary')
plt.show()
