import pandas as pd
import numpy as np

# 1. Load dataset
df = pd.read_csv("students.csv")

subjects = ["Maths", "Science", "English"]

# 2. Data cleaning
df["Name"] = df["Name"].astype("string").str.strip()

for subject in subjects:
    df[subject] = pd.to_numeric(df[subject], errors="coerce")
    df.loc[~df[subject].between(0, 100), subject] = np.nan

df = df.dropna(subset=["Name"])

print("\nOriginal Data:")
print(df)

# 3. Calculate total and average
df["Total"] = df[subjects].sum(axis=1, min_count=3)
df["Average"] = (df["Total"] / 3).round(2)

# 4. Assign grades
def calculate_grade(avg):
    if pd.isna(avg):
        return "Incomplete"
    elif avg >= 85:
        return "A"
    elif avg >= 70:
        return "B"
    elif avg >= 50:
        return "C"
    else:
        return "D"

df["Grade"] = df["Average"].apply(calculate_grade)

# 5. Pass or Fail
df["Status"] = np.where(
    df[subjects].isna().any(axis=1),
    "Incomplete",
    np.where(df[subjects].ge(35).all(axis=1), "Pass", "Fail")
)

# 6. Calculate ranking
df["Rank"] = df["Average"].rank(
    method="min",
    ascending=False
).astype("Int64")

df = df.sort_values(
    "Average",
    ascending=False,
    na_position="last"
)

print("\nStudent Results:")
print(df)

# 7. Find topper and lowest scorer
valid = df.dropna(subset=["Average"])

if not valid.empty:
    print("\nTopper:")
    print(valid.loc[valid["Average"].idxmax()])

    print("\nLowest Scorer:")
    print(valid.loc[valid["Average"].idxmin()])

# 8. Subject-wise analysis
print("\nSubject-wise Analysis:")

for subject in subjects:
    print("\n", subject)
    print("Average:", df[subject].mean())
    print("Highest Marks:", df[subject].max())
    print("Lowest Marks:", df[subject].min())
    print("Pass Count:", (df[subject] >= 35).sum())
    print("Fail Count:", (df[subject] < 35).sum())

# 9. Class statistics using NumPy
averages = df["Average"].dropna().to_numpy()

if len(averages) > 0:
    print("\nClass Statistics:")
    print("Mean:", np.mean(averages))
    print("Median:", np.median(averages))
    print("Standard Deviation:", np.std(averages))

# 10. Grade distribution
print("\nGrade Distribution:")
print(df["Grade"].value_counts())

# 11. Search student
name = input("\nEnter student name to search: ").strip()

if name:
    result = df[df["Name"].str.contains(name, case=False, na=False)]

    if result.empty:
        print("Student not found.")
    else:
        print(result)

# 12. Filter by grade
grade = input("\nEnter grade to filter (A/B/C/D): ").strip().upper()

if grade in ["A", "B", "C", "D"]:
    print("\nStudents with grade", grade)
    print(df[df["Grade"] == grade])

# 13. Export results
df.to_csv("student_results.csv", index=False)

print("\nResults successfully saved!")
