import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("Titanic-Dataset.csv")
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Sex'] = np.where(df['Sex'] == 'male', 0, 1)
predictions = []
for _, row in df.iterrows():
    if row['Sex'] == 1 or row['Pclass'] == 1 or row['Age'] < 50:
        predictions.append(1)  # Survived
    else:
        predictions.append(0)  # Did not survive

df['Predicted'] = predictions
total_passengers = len(df)
correct_predictions = (df['Predicted'] == df['Survived']).sum()
accuracy = correct_predictions / total_passengers

print("Total Passengers:", total_passengers)
print("Correct Predictions:", correct_predictions)
print("Prediction Accuracy:", round(accuracy * 100, 2), "%")
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
df['Survived'].value_counts().plot(kind='bar', color=['red', 'green'])
plt.title('Actual Survival')
plt.xlabel('0 = Did not Survive, 1 = Survived')
plt.ylabel('Count')

plt.subplot(2, 2, 2)
df['Predicted'].value_counts().plot(kind='bar', color=['red', 'green'])
plt.title('Predicted Survival')
plt.xlabel('0 = Did not Survive, 1 = Survived')
plt.ylabel('Count')

plt.subplot(2, 2, 3)
pd.crosstab(df['Sex'], df['Survived']).plot(kind='bar', color=['red', 'green'], rot=0)
plt.title('Actual Survival by Gender')
plt.ylabel('Count')

plt.subplot(2, 2, 4)
pd.crosstab(df['Sex'], df['Predicted']).plot(kind='bar', color=['red', 'green'], rot=0)
plt.title('Predicted Survival by Gender')
plt.ylabel('Count')

plt.tight_layout()
plt.show()
