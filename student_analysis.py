
import pandas as pd
import numpy as np

# =========================================
# 1. LOAD DATA
# =========================================

df = pd.read_csv("students.csv")

SUBJECTS = ["Maths", "Science", "English"]
PASS_MARKS = 35

required_columns = ["Name"] + SUBJECTS

if not all(col in df.columns for col in required_columns):
    raise ValueError(
        f"CSV must contain these columns: {required_columns}"
    )

# Clean names and convert marks to numeric values
df["Name"] = df["Name"].astype("string").str.strip()
df["Name"] = df["Name"].replace("", pd.NA)

for subject in SUBJECTS:
    df[subject] = pd.to_numeric(
        df[subject], errors="coerce"
    )

    df.loc[
        ~df[subject].between(0, 100),
        subject
    ] = np.nan

df = df.dropna(subset=["Name"]).copy()

if df.empty:
    raise ValueError("No valid student records found.")

print("\n========== ORIGINAL DATA ==========")
print(df.to_string(index=False))


# =========================================
# 2. DATA QUALITY ANALYSIS
# =========================================

print("\n========== DATA QUALITY ==========")

print("\nMissing values:")
print(df[SUBJECTS].isna().sum())

print("\nDuplicate student records:")
print(df.duplicated(subset=["Name"] + SUBJECTS).sum())

# Count records with at least one missing mark
incomplete_records = df[SUBJECTS].isna().any(axis=1).sum()
print("Incomplete records:", incomplete_records)

# Count students with any invalid/missing subject mark
print("Students with incomplete marks:", incomplete_records)


# =========================================
# 3. TOTAL AND AVERAGE
# =========================================

# Calculate results only when all subject marks are present
df["Total"] = df[SUBJECTS].sum(axis=1, min_count=len(SUBJECTS))
df["Average"] = df["Total"] / len(SUBJECTS)

# Round average for display
df["Average"] = df["Average"].round(2)


# =========================================
# 4. GRADE AND PERFORMANCE CATEGORY
# =========================================

df["Grade"] = np.select(
    [
        df["Average"].ge(85),
        df["Average"].ge(70),
        df["Average"].ge(50),
        df["Average"].notna()
    ],
    ["A", "B", "C", "D"],
    default="Incomplete"
)

df["Performance"] = np.select(
    [
        df["Average"].ge(85),
        df["Average"].ge(70) & df["Average"].lt(85),
        df["Average"].ge(50) & df["Average"].lt(70),
        df["Average"].notna()
    ],
    [
        "Excellent",
        "Good",
        "Average",
        "Needs Improvement"
    ],
    default="Incomplete"
)


# =========================================
# 5. PASS / FAIL ANALYSIS
# =========================================

# A student passes only if every subject has >= 35
df["Status"] = np.select(
    [
        df[SUBJECTS].isna().any(axis=1),
        df[SUBJECTS].ge(PASS_MARKS).all(axis=1)
    ],
    ["Incomplete", "Pass"],
    default="Fail"
)

# Number of subjects passed
df["Subjects Passed"] = (
    df[SUBJECTS].ge(PASS_MARKS).sum(axis=1)
)

df["Subjects Failed"] = (
    df[SUBJECTS].lt(PASS_MARKS).sum(axis=1)
)

# Incomplete marks should not be interpreted as failed subjects
df.loc[
    df[SUBJECTS].isna().any(axis=1),
    ["Subjects Passed", "Subjects Failed"]
] = pd.NA


# =========================================
# 6. RANKING
# =========================================

df["Rank"] = (
    df["Average"]
    .rank(method="min", ascending=False)
    .astype("Int64")
)

df = df.sort_values(
    "Average",
    ascending=False,
    na_position="last"
)

print("\n========== COMPLETE RESULTS ==========")
print(df.to_string(index=False))


# =========================================
# 7. TOPPER AND LOWEST SCORER
# =========================================

valid = df.dropna(subset=["Average"])

if not valid.empty:
    topper = valid.loc[valid["Average"].idxmax()]
    lowest = valid.loc[valid["Average"].idxmin()]

    print("\n========== TOPPER ==========")
    print(topper[["Name", "Total", "Average", "Grade"]])

    print("\n========== LOWEST AVERAGE ==========")
    print(lowest[["Name", "Total", "Average", "Grade"]])


# =========================================
# 8. SUBJECT-WISE ANALYSIS
# =========================================

print("\n========== SUBJECT ANALYSIS ==========")

for subject in SUBJECTS:
    marks = df[subject].dropna()

    if marks.empty:
        print(f"\n{subject}: No valid marks")
        continue

    topper_index = df[subject].idxmax()
    lowest_index = df[subject].idxmin()

    print(f"\nSubject: {subject}")
    print(f"Average: {marks.mean():.2f}")
    print(f"Median: {marks.median():.2f}")
    print(f"Maximum: {marks.max()}")
    print(f"Minimum: {marks.min()}")
    print(f"Standard Deviation: {marks.std():.2f}")
    print(f"Topper: {df.loc[topper_index, 'Name']}")
    print(f"Lowest Scorer: {df.loc[lowest_index, 'Name']}")
    print(f"Pass Count: {(marks >= PASS_MARKS).sum()}")
    print(f"Fail Count: {(marks < PASS_MARKS).sum()}")


# =========================================
# 9. CLASS STATISTICS USING NUMPY
# =========================================

averages = df["Average"].dropna().to_numpy()

print("\n========== CLASS STATISTICS ==========")

if len(averages) > 0:
    print(f"Mean: {np.mean(averages):.2f}")
    print(f"Median: {np.median(averages):.2f}")
    print(f"Variance: {np.var(averages):.2f}")
    print(f"Standard Deviation: {np.std(averages):.2f}")

    print(f"25th Percentile: {np.percentile(averages, 25):.2f}")
    print(f"50th Percentile: {np.percentile(averages, 50):.2f}")
    print(f"75th Percentile: {np.percentile(averages, 75):.2f}")

    print(f"Above Class Average: {(averages > np.mean(averages)).sum()}")
    print(f"Below Class Average: {(averages < np.mean(averages)).sum()}")


# =========================================
# 10. GRADE DISTRIBUTION
# =========================================

print("\n========== GRADE DISTRIBUTION ==========")
print(df["Grade"].value_counts())

print("\n========== STATUS DISTRIBUTION ==========")
print(df["Status"].value_counts())


# =========================================
# 11. SUBJECT CORRELATION
# =========================================

print("\n========== SUBJECT CORRELATION ==========")

correlation = df[SUBJECTS].corr()
print(correlation.round(2))



# =========================================
# 12. ABOVE / BELOW SUBJECT AVERAGE
# =========================================

print("\n========== SUBJECT-WISE COMPARISON ==========")

subject_means = df[SUBJECTS].mean()

print("\nSubject averages:")
print(subject_means.round(2))

for subject in SUBJECTS:
    mean_marks = subject_means[subject]

    if pd.isna(mean_marks):
        continue

    above_average = df.loc[
        df[subject] > mean_marks, "Name"
    ].tolist()

    below_average = df.loc[
        df[subject] < mean_marks, "Name"
    ].tolist()

    print(f"\n{subject}")
    print("Above average:", above_average)
    print("Below average:", below_average)


# =========================================
# 13. CONSISTENCY ANALYSIS
# =========================================


df["Marks Range"] = (
    df[SUBJECTS].max(axis=1) -
    df[SUBJECTS].min(axis=1)
)


df["Subject Std Dev"] = df[SUBJECTS].std(axis=1)

print("\n========== STUDENT CONSISTENCY ==========")
print(
    df[["Name", "Marks Range", "Subject Std Dev"]]
    .to_string(index=False)
)



# =========================================
# 14. OUTLIER DETECTION
# =========================================

print("\n========== OUTLIER ANALYSIS ==========")

for subject in SUBJECTS:
    marks = df[subject]
    mean_marks = marks.mean()
    std_marks = marks.std()

    if pd.isna(std_marks) or std_marks == 0:
        print(f"{subject}: Not enough variation to detect outliers")
        continue

    lower = mean_marks - 2 * std_marks
    upper = mean_marks + 2 * std_marks

    outliers = df.loc[
        marks.notna() & ((marks < lower) | (marks > upper)),
        ["Name", subject]
    ]

    print(f"\n{subject} outliers:")
    if outliers.empty:
        print("No outliers found")
    else:
        print(outliers.to_string(index=False))


# =========================================
# 15. IDENTIFY STUDENTS NEEDING SUPPORT
# =========================================

weak_students = df[
    (df["Status"] == "Fail") |
    (df["Average"] < 50)
]

print("\n========== STUDENTS NEEDING SUPPORT ==========")

if weak_students.empty:
    print("No students identified by these criteria.")
else:
    print(
        weak_students[
            ["Name", "Average", "Grade", "Status"]
        ].to_string(index=False)
    )


# =========================================
# 16. SEARCH FOR A STUDENT
# =========================================

search_name = input(
    "\nEnter student name to search (Enter to skip): "
).strip()

if search_name:
    result = df[
        df["Name"].str.contains(
            search_name,
            case=False,
            na=False,
            regex=False
        )
    ]

    if result.empty:
        print("Student not found.")
    else:
        print("\n========== SEARCH RESULT ==========")
        print(result.to_string(index=False))


# =========================================
# 17. FILTER STUDENTS BY GRADE
# =========================================

grade_filter = input(
    "\nEnter grade to filter (A/B/C/D or Enter to skip): "
).strip().upper()

if grade_filter in ["A", "B", "C", "D"]:
    filtered = df[df["Grade"] == grade_filter]

    print(f"\n========== GRADE {grade_filter} STUDENTS ==========")
    print(filtered.to_string(index=False))


# =========================================
# 18. EXPORT RESULTS
# =========================================

df.to_csv("student_results.csv", index=False)

print("\nResults saved to student_results.csv")
print("\n========== PROJECT COMPLETED ==========")
