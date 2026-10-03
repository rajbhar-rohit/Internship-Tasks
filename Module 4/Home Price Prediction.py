import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Sample Dataset

data = { 
    # Area: House area in square feet
    "Area": [800, 1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000],
    # Beadrooms: Number of bedrooms
    "Bedrooms": [2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
    # Age: Age of House in years
    "Age": [10, 9, 8, 7, 5, 6, 4, 3, 2, 1,],
    # Price: House price in Lakh (₹)
    "Price": [25, 32, 40, 50, 58, 68, 75, 85, 95, 105]
}

df = pd.DataFrame(data)

print(df)