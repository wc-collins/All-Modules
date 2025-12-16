import csv
import warnings
import matplotlib.pyplot as plt
import numpy as np
from patient import Patient

Patient.instantiate_from_csv('UpdatedLuminex.csv', 'UpdatedMetaData.csv')


# GRAPH 1: Distribution of Education Levels

# Build dictionary grouped by education
Patient.sort_ed()
Patient.subsort_thal()

# Count patients in each education level
education_counts = {lvl: len(pats) for lvl, pats in Patient.education_lvl.items()}

# Plot a bar chart of education levels
plt.bar(education_counts.keys(), education_counts.values())
plt.xticks(rotation=45, ha="right")
plt.xlabel("Education Level")
plt.ylabel("Number of Patients")
plt.title("Distribution of Education Levels in Patient Dataset")
plt.tight_layout()
plt.show()

# GRAPH 2: Mean Thal Scores by Education Level with Error Bars

# Prepare labels
labels = list(Patient.education_lvl.keys())

# Compute mean and standard deviation of Thal scores for each education level
mean_thal_scores = []
std_devs = []

for lvl in labels:
    thal_scores = [p.thal_score for p in Patient.education_lvl[lvl] if p.thal_score is not None]
    if thal_scores:
        mean_thal_scores.append(np.mean(thal_scores))
        std_devs.append(np.std(thal_scores))
    else:
        mean_thal_scores.append(0)
        std_devs.append(0)

mean_thal_scores = np.array(mean_thal_scores)
std_devs = np.array(std_devs)

# Error bars: only upward
lower_errors = np.zeros_like(std_devs)
upper_errors = std_devs

# Plot mean Thal scores with error bars
plt.bar(labels, mean_thal_scores, yerr=[lower_errors, upper_errors], capsize=5, color='lightblue')
plt.xticks(rotation=45, ha="right")
plt.xlabel("Education Level")
plt.ylabel("Mean Thal Score")
plt.title("Mean Thal Scores by Education Level with Standard Deviation")
plt.tight_layout()
plt.show()

# ========================
# ANOVA: Compare Thal scores between education levels
# ========================

# Regression Analysis
from scipy import stats

# Prepare groups of Thal scores by education level
thal_groups = []
for lvl in labels:
    thal_scores = [p.thal_score for p in Patient.education_lvl[lvl] if p.thal_score is not None]
    if thal_scores:
        thal_groups.append(thal_scores)

# Perform one-way ANOVA
anova_result = stats.f_oneway(*thal_groups)

# Print results
print("ANOVA Results:")
print(f"F-statistic: {anova_result.statistic:.3f}")
print(f"P-value: {anova_result.pvalue:.3e}")
if anova_result.pvalue < 0.05:
    print("Result: Significant differences in Thal scores between education levels (reject null).")
else:
    print("Result: No significant differences in Thal scores between education levels (fail to reject null).")

    import matplotlib.pyplot as plt

# Initialize lists
onset_age_list = []
ab_ratio = []

# Process patient data
for patient in Patient.all_patients:
    onset_age_list.append(patient.age_symp_on)
    
    if patient.ABeta40:
        ratio = patient.ABeta42 / patient.ABeta40
        ab_ratio.append(ratio if ratio <= 500 else None)  # Filter out unrealistic ratios
        '''
        one data point had a AB40/AB42ratio of around 2000, which is is around 20x the highest of the other points, so we 
        decided to filter it out for the sake of 
        '''
    else:
        ab_ratio.append(None)  # Use None for missing or invalid data

# Filter out None values for plotting
filtered_data = [(age, ratio) for age, ratio in zip(onset_age_list, ab_ratio) if ratio is not None]
ages, ratios = zip(*filtered_data)

# Scatter plot
plt.scatter(ages, ratios, color='blue')
plt.xlabel("Age of Onset of Cognitive Symptoms")
plt.ylabel("Aβ42/Aβ40 Ratio")
plt.title("Aβ42/Aβ40 Ratio vs Age of Onset of Cognitive Symptoms")
plt.grid(True)
plt.show()
# ========================
# Export Death Age and Aβ42 to CSV
import pandas as pd
print(ab_ratio)
print(onset_age_list)
# Create a DataFrame
df = pd.DataFrame({
'Age of Onset': onset_age_list,
'ABeta42/Abeta40 Ratio': ab_ratio
})
# Write to CSV
df.to_csv('patient_data.csv', index=False)
print("CSV file 'patient_data.csv' has been created.")