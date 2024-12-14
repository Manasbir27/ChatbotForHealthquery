import os
import csv
import networkx as nx
from collections import defaultdict
from transformers import AutoTokenizer
import requests
from fuzzywuzzy import process
import re

# Set Hugging Face token
os.environ['HUGGINGFACE_TOKEN'] = "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"
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
headers = {"Authorization": f"Bearer {os.environ['HUGGINGFACE_TOKEN']}"}

def validate_email(email):
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_phone(phone):
    """Validate phone number format."""
    pattern = r'^\+?[0-9]{10,12}$'
    return re.match(pattern, phone) is not None

def validate_date(date):
    """Validate date format (DD/MM/YYYY)."""
    pattern = r'^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/\d{4}$'
    return re.match(pattern, date) is not None

def get_valid_input(prompt, validator=None, error_message=None):
    """Get user input with validation."""
    while True:
        value = input(prompt).strip()
        if not value:
            print("This field is mandatory and cannot be empty.")
            continue
        if validator:
            if validator(value):
                return value
            print(error_message)
        else:
            return value

def query_model(payload):
    """Query the Hugging Face model with the provided payload."""
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error querying the model: {e}")
        return {}

def extract_symptoms_from_response(response_text):
    """Extract symptoms from the model's response."""
    lines = response_text.split('\n')
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("symptoms:"):
            return '\n'.join(lines[i:]).strip()
    return response_text.strip()

def analyze_symptoms(user_input):
    """Analyze user input and return a list of detected symptoms."""
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
    messages = [
        {"role": "system", "content": "You are a medical assistant. Analyze the given text for symptoms. Respond with a list of detected symptoms only, starting with 'Symptoms:'."},
        {"role": "user", "content": user_input},
    ]
    
    try:
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
        input_text = tokenizer.decode(inputs[0])
        
        payload = {
            "inputs": input_text,
            "parameters": {
                "max_new_tokens": 150,
                "temperature": 0.3,
                "top_p": 0.95,
                "do_sample": True
            }
        }
        
        response = query_model(payload)
        if isinstance(response, list) and response and 'generated_text' in response[0]:
            assistant_response = response[0]['generated_text'].split("[/INST]")[-1].strip()
            symptoms = extract_symptoms_from_response(assistant_response)
            return [s.strip().lower() for s in symptoms.replace("Symptoms:", "").strip().split(",")]
        else:
            return []
    except Exception as e:
        print(f"Error: An unexpected error occurred. Details: {str(e)}")
        return []

def create_knowledge_graph(csv_file):
    """Create a knowledge graph from the CSV file of diseases and symptoms."""
    G = nx.Graph()
    symptoms_dict = defaultdict(set)
    
    with open(csv_file, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            symptom = row['Symptom'].strip().lower()
            disease = row['Disease'].strip().lower()
            relation = row['Relation'].strip().lower()

            G.add_node(disease, node_type='disease')
            G.add_node(symptom, node_type='symptom')
            
            if relation == 'has_symptom':
                G.add_edge(disease, symptom)
                symptoms_dict[symptom].add(disease)

    print(f"Knowledge graph created with {len(G.nodes())} nodes and {len(G.edges())} edges.")
    return G, symptoms_dict

def match_symptoms_with_graph(G, detected_symptoms):
    """Match detected symptoms with the knowledge graph using fuzzy matching."""
    potential_diseases = defaultdict(set)
    all_symptoms_in_graph = [n for n, d in G.nodes(data=True) if d['node_type'] == 'symptom']
    
    matched_symptoms = set()
    for symptom in detected_symptoms:
        matches = process.extract(symptom, all_symptoms_in_graph, limit=4)
        for match, score in matches:
            if score >= 60:  # Increased threshold for more precise matching
                matched_symptoms.add(match)
    
    if not matched_symptoms:
        print("No matched symptoms found.")
        return potential_diseases

    all_diseases = set(n for n, d in G.nodes(data=True) if d['node_type'] == 'disease')
    
    for disease in all_diseases:
        disease_symptoms = set(G.neighbors(disease))
        matching_symptoms = matched_symptoms.intersection(disease_symptoms)
        
        if matching_symptoms:
            potential_diseases[disease] = matching_symptoms
    
    return dict(sorted(potential_diseases.items(), key=lambda x: len(x[1]), reverse=True))

def ask_initial_symptoms(G, potential_diseases, confirmed_symptoms):
    """Ask questions for each of the top potential diseases."""
    max_questions_per_disease = 5
    max_diseases = 3
    disease_scores = defaultdict(int)
    
    sorted_diseases = sorted(potential_diseases.items(), key=lambda x: len(x[1]), reverse=True)[:max_diseases]
    
    print("\nAsking specific questions about potential conditions...")
    
    for disease, initial_symptoms in sorted_diseases:
        print(f"\nChecking symptoms for: {disease.capitalize()}")
        disease_symptoms = set(G.neighbors(disease)) - initial_symptoms - confirmed_symptoms
        questions_asked = 0
        
        for symptom in disease_symptoms:
            if questions_asked >= max_questions_per_disease:
                break
                
            response = get_valid_input(f"Do you experience {symptom}? (yes/no/unsure) or 'quit': ", 
                                       lambda x: x.lower() in ['yes', 'no', 'unsure', 'quit'],
                                       "Invalid input. Please enter 'yes', 'no', 'unsure', or 'quit'.")
            
            if response == 'quit':
                return confirmed_symptoms, True, disease_scores
            
            questions_asked += 1
            
            if response == 'yes':
                confirmed_symptoms.add(symptom)
                disease_scores[disease] += 1
    
    return confirmed_symptoms, False, disease_scores

def diagnose_diseases(G, potential_diseases, confirmed_symptoms, disease_scores):
    """Diagnose diseases based on matching symptoms and confirmed symptoms."""
    diagnosed_diseases = []
    
    for disease, initial_symptoms in potential_diseases.items():
        all_disease_symptoms = set(G.neighbors(disease))
        matched_symptoms = initial_symptoms.union(confirmed_symptoms.intersection(all_disease_symptoms))
        total_matched = len(matched_symptoms)
        total_symptoms = len(all_disease_symptoms)
        
        # Calculate confidence based on the proportion of matched symptoms
        confidence = (total_matched / total_symptoms) * 100
        
        diagnosed_diseases.append((disease, confidence, matched_symptoms))
    
    # Sort by number of matched symptoms (descending) and then by confidence (descending)
    diagnosed_diseases.sort(key=lambda x: (len(x[2]), x[1]), reverse=True)
    return diagnosed_diseases[:3]

def patient_registration():
    """Collect patient registration details with validation."""
    print("\n=== Patient Registration Form ===")
    registration_details = {
        "Name": get_valid_input("Full Name: "),
        "Address": get_valid_input("Complete Address: "),
        "Phone Number": get_valid_input(
            "Phone Number (10-12 digits): ",
            validate_phone,
            "Invalid phone number format. Please enter 10-12 digits."
        ),
        "DOB": get_valid_input(
            "Date of Birth (DD/MM/YYYY): ",
            validate_date,
            "Invalid date format. Please use DD/MM/YYYY format."
        ),
        "Age": get_valid_input("Age: ", lambda x: x.isdigit() and 0 < int(x) < 150, "Please enter a valid age."),
        "Sex": get_valid_input(
            "Sex (M/F/O): ",
            lambda x: x.upper() in ['M', 'F', 'O'],
            "Please enter M, F, or O."
        ),
        "Email": get_valid_input(
            "Email: ",
            validate_email,
            "Invalid email format."
        ),
        "Blood Group": get_valid_input(
            "Blood Group (A+/A-/B+/B-/O+/O-/AB+/AB-): ",
            lambda x: x.upper() in ['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-'],
            "Invalid blood group. Please enter a valid blood group."
        ),
        "Symptoms": [],
        "Diagnosed Disease": None,
        "Disease Confidence": None,
        "Disease Symptoms": None
    }
    return registration_details

def print_final_report(patient_info):
    """Print formatted final report with all details."""
    print("\n" + "="*50)
    print("PATIENT DIAGNOSIS REPORT")
    print("="*50)
    
    print("\nPATIENT INFORMATION:")
    print("-"*20)
    for key in ["Name", "Age", "Sex", "DOB", "Phone Number", "Email", "Blood Group", "Address"]:
        print(f"{key:15}: {patient_info[key]}")
    
    print("\nDIAGNOSIS DETAILS:")
    print("-"*20)
    if patient_info["Diagnosed Disease"]:
        print(f"Diagnosed Disease  : {patient_info['Diagnosed Disease'].capitalize()}")
        print(f"Confidence Level  : {patient_info['Disease Confidence']:.2f}%")
        print("\nIdentified Symptoms:")
        for symptom in patient_info["Disease Symptoms"]:
            print(f"- {symptom}")
    else:
        print("No definitive diagnosis made. Please consult a healthcare professional.")
    
    print("\n" + "="*50)

def save_report_to_file(patient_info):
    """Save the patient report to a file."""
    try:
        filename = f"medical_report_{patient_info['Name'].replace(' ', '_')}_{patient_info['DOB'].replace('/', '-')}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*50 + "\n")
            f.write("PATIENT DIAGNOSIS REPORT\n")
            f.write("="*50 + "\n\n")
            
            f.write("PATIENT INFORMATION:\n")
            f.write("-"*20 + "\n")
            for key in ["Name", "Age", "Sex", "DOB", "Phone Number", "Email", "Blood Group", "Address"]:
                f.write(f"{key:15}: {patient_info[key]}\n")
            
            f.write("\nDIAGNOSIS DETAILS:\n")
            f.write("-"*20 + "\n")
            if patient_info["Diagnosed Disease"]:
                f.write(f"Diagnosed Disease  : {patient_info['Diagnosed Disease'].capitalize()}\n")
                f.write(f"Confidence Level  : {patient_info['Disease Confidence']:.2f}%\n")
                f.write("\nIdentified Symptoms:\n")
                for symptom in patient_info["Disease Symptoms"]:
                    f.write(f"- {symptom}\n")
            else:
                f.write("No definitive diagnosis made. Please consult a healthcare professional.\n")
            
            f.write("\n" + "="*50 + "\n")
        print(f"\nReport saved successfully as: {filename}")
    except Exception as e:
        print(f"\nError saving report: {str(e)}")

def main():
    print("Welcome to the Disease Diagnosis Chatbot.")
    
    patient_info = patient_registration()
    
    csv_file = r'C:\Users\Asus\Desktop\SEMESTER_7\data\new_symptom_disease_relations.csv'
    G, symptoms_dict = create_knowledge_graph(csv_file)
    all_symptoms = set()

    while True:
        user_input = input("Describe your symptoms (or type 'add'/'list'/'quit'): ").lower()
        if user_input == 'quit':
            break
        elif user_input == 'add':
            continue
        elif user_input == 'list':
            print(f"Current symptoms: {', '.join(all_symptoms) if all_symptoms else 'No symptoms recorded yet.'}")
            continue

        detected_symptoms = set(analyze_symptoms(user_input))
        if detected_symptoms:
            all_symptoms.update(detected_symptoms)
            print(f"Detected Symptoms: {', '.join(all_symptoms)}")
            patient_info["Symptoms"] = list(all_symptoms)
        else:
            print("No symptoms detected. Try again.")
            continue

        potential_diseases = match_symptoms_with_graph(G, all_symptoms)
        if not potential_diseases:
            print("No potential diseases found. Please consult a healthcare professional.")
            continue

        print("\nInitial potential diseases:")
        for disease, matched_symptoms in list(potential_diseases.items())[:3]:
            num_matched_symptoms = len(matched_symptoms)
            print(f"- {disease.capitalize()} (Matched symptoms: {num_matched_symptoms} - {', '.join(matched_symptoms)})")

        confirmed_symptoms, quit_flag, disease_scores = ask_initial_symptoms(G, potential_diseases, set())
        if quit_flag:
            break

        diagnosed_diseases = diagnose_diseases(G, potential_diseases, confirmed_symptoms, disease_scores)
        if diagnosed_diseases:
            top_disease = diagnosed_diseases[0]
            patient_info["Diagnosed Disease"] = top_disease[0]
            patient_info["Disease Confidence"] = top_disease[1]
            patient_info["Disease Symptoms"] = top_disease[2]
            
            print("\nTop 3 potential diagnoses:")
            for disease, confidence, symptoms in diagnosed_diseases:
                print(f"- {disease.capitalize()} (Confidence: {confidence:.2f}%, Matched symptoms: {len(symptoms)})")
                print(f"  Symptoms: {', '.join(symptoms)}")
                print()
        else:
            print("No diagnosis could be made based on the provided symptoms.")
            print("Please consult a healthcare professional for proper medical evaluation.")

    print_final_report(patient_info)
    
    save_report = input("\nWould you like to save this report to a file? (yes/no): ").lower()
    if save_report == 'yes':
        save_report_to_file(patient_info)

if __name__ == "__main__":
    main()