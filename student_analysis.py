import pandas as pd
import numpy as np


df = pd.read_csv("students.csv")

print("----- STUDENT DATA -----")
print(df)


df["Total"] = df[["Maths", "Science", "English"]].sum(axis=1)
df["Average"] = df["Total"] / 3


def grade(avg):
    if avg >= 85:
        return "A"
    elif avg >= 70:
        return "B"
    elif avg >= 50:
        return "C"
    else:
        return "Fail"

df["Grade"] = df["Average"].apply(grade)

print("\n----- RESULT ANALYSIS -----")
print(df)


top_student = df.loc[df["Average"].idxmax()]
print("\nTop Performer:")
print(top_student[["Name", "Average", "Grade"]])


weak_students = df[df["Grade"] == "Fail"]
print("\nWeak Students:")
print(weak_students[["Name", "Average", "Grade"]])


subjects = ["Maths", "Science", "English"]
subject_avg = np.mean(df[subjects], axis=0)

print("\nSubject-wise Average Marks:")
for sub, avg in zip(subjects, subject_avg):
    print(f"{sub}: {avg:.2f}")
