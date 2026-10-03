import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sample Dataset

data ={
    "Name": ["Aman", "Rahul", "Param", "Priya"],
    "Marks": [85, 72, 90, 78],
    "Attendance": [92, 85, 95, 88],
    "Age": [19, 20, 19, 19]
}

df = pd.DataFrame(data)

# Exploring Dataset

print("Dataset:")
print(df)

print("\nFirst 4 rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nColumn Names:")
print(df.columns)

# Performing Operations

marks = np.array(df["Marks"])

print("\nMarks:")
print(marks)

print("Average Marks:", np.mean(marks))
print("Highest Marks:", np.max(marks))
print("Lowest Marks:", np.min(marks))
print("Standard Deviation:", np.std(marks))

# Data Cleaning


# Checking missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Removing duplicate rows
df = df.drop_duplicates()



# Simple Data Analysis

print("\nStudents with marks greater than 75:")
print(df[df["Marks"] > 75])

print("\nAverage Attendance:")
print(df["Attendance"].mean())


# Data Visualization

# Bar Chart
plt.figure(figsize=(7,4))
plt.bar(df["Name"], df["Marks"])
plt.title("Student Marks")
plt.xlabel("Students")
plt.ylabel("Marks")
plt.show()


# Line Chart
plt.figure(figsize=(7, 4))
plt.plot(df["Name"], df["Marks"], "o-")
plt.title("Marks Trend")
plt.xlabel("Student")
plt.ylabel("Marks")
plt.show()

# Pie Chart
plt.figure(figsize=(6, 6))
plt.pie(
    df["Marks"],
    labels=df['Name'],
    autopct="%1.1f%%"
)
plt.title("Marks Distribution")
plt.show()