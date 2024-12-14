import pandas as pd
from collections import defaultdict

# Read the CSV file
df = pd.read_csv(r'C:\Users\Asus\Desktop\SEMESTER_7\data\final_normalized_disease_symptoms.csv')

# Create a dictionary to store symptom-disease mappings
symptom_disease_map = defaultdict(set)

# Process each row
for _, row in df.iterrows():
    disease = row['Disease']
    # Split the combined symptoms and strip whitespace
    symptoms = [symptom.strip() for symptom in row['Combined_Symptoms'].split(',')]
    
    # Add each symptom-disease mapping
    for symptom in symptoms:
        symptom_disease_map[symptom].add(disease)

# Create a list to store the data for the new CSV
csv_data = []

# Process each symptom and its associated diseases
for symptom, diseases in symptom_disease_map.items():
    csv_data.append({
        'Symptom': symptom,
        'Number_of_Diseases': len(diseases),
        'Associated_Diseases': ', '.join(sorted(diseases))
    })

# Convert to DataFrame and sort by number of diseases (descending)
output_df = pd.DataFrame(csv_data)
output_df = output_df.sort_values(by='Number_of_Diseases', ascending=False)

# Save to a new CSV file
output_path = r'C:\Users\Asus\Desktop\SEMESTER_7\data\grouped_symptoms_diseases.csv'
output_df.to_csv(output_path, index=False)

# Print summary
print(f"Analysis complete! New CSV file created at: {output_path}")
print(f"Total number of symptoms analyzed: {len(symptom_disease_map)}")
print(f"Number of unique symptoms: {len(csv_data)}")

# Display first few rows of the new CSV
print("\nFirst few rows of the new CSV file:")
print(output_df.head().to_string())
