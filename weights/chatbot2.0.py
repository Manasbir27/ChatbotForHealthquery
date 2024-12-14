import os
import csv
import requests
from transformers import AutoTokenizer
from fuzzywuzzy import process
from collections import defaultdict
from math import log

# API Configuration
os.environ['HUGGINGFACE_TOKEN'] = "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"
MIXTRAL_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
headers = {"Authorization": f"Bearer {os.environ['HUGGINGFACE_TOKEN']}"}

# Weight configurations
WEIGHTS = {
    'symptom_match': 1.0,         
    'yes_response': 1.0,          
    'no_response': -1.0,          
    'unsure_response': 0.0,       
    'disease_frequency': 0.3,     
    'symptom_specificity': 0.4,   
    'symptom_count_penalty': 0.1  
}

def load_symptom_disease_data(csv_file):
    """Load and process symptom-disease relationships with frequency analysis."""
    disease_symptoms = defaultdict(set)
    symptom_diseases = defaultdict(set)
    disease_frequency = defaultdict(int)
    symptom_frequency = defaultdict(int)
    all_symptoms = set()
    
    with open(csv_file, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            symptom = row['Symptom'].strip().lower()
            disease = row['Disease'].strip().lower()
            
            all_symptoms.add(symptom)
            disease_symptoms[disease].add(symptom)
            symptom_diseases[symptom].add(disease)
            disease_frequency[disease] += 1
            symptom_frequency[symptom] += 1
    
    return disease_symptoms, symptom_diseases, all_symptoms

def calculate_disease_frequencies(symptom_diseases):
    """Calculate how common each disease is based on symptom associations."""
    disease_freq = defaultdict(int)
    total_associations = 0
    
    for symptom, diseases in symptom_diseases.items():
        for disease in diseases:
            disease_freq[disease] += 1
            total_associations += 1
    
    disease_weights = {}
    for disease, freq in disease_freq.items():
        relative_freq = freq / total_associations
        disease_weights[disease] = (1 - relative_freq) * WEIGHTS['disease_frequency']
    
    return disease_weights

def calculate_symptom_specificity(symptom_diseases):
    """Calculate how specific each symptom is."""
    total_diseases = len({disease for diseases in symptom_diseases.values() for disease in diseases})
    
    specificity_scores = {}
    for symptom, diseases in symptom_diseases.items():
        specificity = 1 - (len(diseases) / total_diseases)
        specificity_scores[symptom] = specificity * WEIGHTS['symptom_specificity']
    
    return specificity_scores

def query_mistral(text):
    """Extract symptoms using Mistral AI model."""
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
    
    prompt = f"""Extract only the medical symptoms from the following text. Reply with just a comma-separated list of symptoms, nothing else.
    
Text: {text}
Symptoms:"""
    
    messages = [
        {"role": "system", "content": "You are a medical symptom detector. Extract and list only the symptoms mentioned."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
        input_text = tokenizer.decode(inputs[0])
        
        payload = {
            "inputs": input_text,
            "parameters": {
                "max_new_tokens": 100,
                "temperature": 0.1,
                "top_p": 0.9,
                "do_sample": True
            }
        }
        
        response = requests.post(MIXTRAL_URL, headers=headers, json=payload)
        response.raise_for_status()
        response_data = response.json()
        
        if isinstance(response_data, list) and response_data:
            symptoms = response_data[0]['generated_text'].split("[/INST]")[-1].strip()
            return [s.strip().lower() for s in symptoms.split(',') if s.strip()]
            
    except Exception as e:
        print(f"Error in Mistral analysis: {str(e)}")
    return []

def extract_symptoms_from_text(text, all_symptoms):
    """Extract symptoms using fuzzy matching as fallback."""
    text = text.lower()
    detected_symptoms = set()
    
    words = text.replace(',', ' ').split()
    phrases = [
        ' '.join(words[i:i+n]) 
        for n in range(1, 5)
        for i in range(len(words) - n + 1)
    ]
    
    for phrase in phrases:
        matches = process.extract(phrase, all_symptoms, limit=2)
        for match, score in matches:
            if score >= 80:
                detected_symptoms.add(match)
    
    return list(detected_symptoms)

def match_symptoms(detected_symptoms, all_symptoms, disease_symptoms, symptom_diseases):
    """Match detected symptoms with weighted scoring."""
    matched_symptoms = set()
    potential_diseases = defaultdict(lambda: {
        'matched_symptoms': set(),
        'score': 0,
        'match_percentage': 0
    })
    
    disease_weights = calculate_disease_frequencies(symptom_diseases)
    symptom_weights = calculate_symptom_specificity(symptom_diseases)
    
    for symptom in detected_symptoms:
        matches = process.extract(symptom, all_symptoms, limit=1)
        for match, score in matches:
            if score >= 60:
                matched_symptoms.add(match)
                
                for disease in symptom_diseases[match]:
                    potential_diseases[disease]['matched_symptoms'].add(match)
                    
                    base_score = WEIGHTS['symptom_match']
                    specificity_bonus = symptom_weights[match]
                    disease_rarity_bonus = disease_weights[disease]
                    
                    total_score = base_score + specificity_bonus + disease_rarity_bonus
                    potential_diseases[disease]['score'] += total_score
    
    final_diseases = {}
    for disease, info in potential_diseases.items():
        total_possible_symptoms = len(disease_symptoms[disease])
        matched_count = len(info['matched_symptoms'])
        
        unmatched_penalty = (total_possible_symptoms - matched_count) * WEIGHTS['symptom_count_penalty']
        final_score = info['score'] - unmatched_penalty
        
        match_percentage = (matched_count / len(detected_symptoms)) * 100
        
        if match_percentage >= 60:
            final_diseases[disease] = {
                'matched_symptoms': info['matched_symptoms'],
                'match_percentage': match_percentage,
                'total_symptoms': total_possible_symptoms,
                'weighted_score': final_score
            }
    
    return matched_symptoms, dict(sorted(
        final_diseases.items(),
        key=lambda x: x[1]['weighted_score'],
        reverse=True
    )[:3])

def ask_additional_symptoms(disease_symptoms, disease_info, confirmed_symptoms, symptom_diseases):
    """Ask follow-up questions with weighted scoring."""
    disease_scores = defaultdict(lambda: {
        'score': 0,
        'total_questions': 0,
        'symptoms': set(),
        'yes_count': 0,
        'no_count': 0,
        'unsure_count': 0
    })
    
    print("\nAsking follow-up questions for better diagnosis...")
    
    disease_weights = calculate_disease_frequencies(symptom_diseases)
    symptom_weights = calculate_symptom_specificity(symptom_diseases)
    
    for disease, info in disease_info.items():
        print(f"\nChecking symptoms for: {disease.title()}")
        unconfirmed_symptoms = disease_symptoms[disease] - confirmed_symptoms
        questions_asked = 0
        
        disease_scores[disease]['symptoms'].update(confirmed_symptoms)
        base_score = len(confirmed_symptoms) * WEIGHTS['symptom_match']
        specificity_bonus = sum(symptom_weights[s] for s in confirmed_symptoms)
        disease_scores[disease]['score'] = base_score + specificity_bonus + disease_weights[disease]
        
        for symptom in unconfirmed_symptoms:
            if questions_asked >= 5:
                break
                
            response = input(f"Do you experience {symptom}? (yes/no/unsure): ").lower().strip()
            if response == 'quit':
                return None
            
            if response == 'yes':
                disease_scores[disease]['symptoms'].add(symptom)
                score_change = WEIGHTS['yes_response'] + symptom_weights[symptom]
                disease_scores[disease]['yes_count'] += 1
            elif response == 'no':
                score_change = WEIGHTS['no_response']
                disease_scores[disease]['no_count'] += 1
            else:  # unsure
                score_change = WEIGHTS['unsure_response']
                disease_scores[disease]['unsure_count'] += 1
            
            disease_scores[disease]['score'] += score_change
            disease_scores[disease]['total_questions'] += 1
            questions_asked += 1
    
    final_scores = {}
    for disease, scores in disease_scores.items():
        total_responses = scores['total_questions'] + len(info['matched_symptoms'])
        if total_responses > 0:
            weighted_score = scores['score'] * (1 + disease_weights[disease])
            confidence = (weighted_score / total_responses) * 100
            
            final_scores[disease] = {
                'confidence': min(confidence, 100),
                'symptoms': scores['symptoms'],
                'yes_count': scores['yes_count'],
                'no_count': scores['no_count'],
                'unsure_count': scores['unsure_count'],
                'rarity_score': disease_weights[disease]
            }
    
    return final_scores

def analyze_with_ai(symptoms):
    """Analyze symptoms using Mixtral AI for advanced medical analysis."""
    symptoms_text = ", ".join(sorted(symptoms))
    
    prompt = f"""As a medical expert, analyze these symptoms and provide potential diagnoses:

Symptoms: {symptoms_text}

Based on these symptoms, provide your analysis in exactly this format:

CONDITION 1: [most likely condition]
CONFIDENCE 1: [percentage]
EXPLANATION 1: [brief explanation why this matches]

CONDITION 2: [second most likely condition]
CONFIDENCE 2: [percentage]
EXPLANATION 2: [brief explanation why this matches]

CONDITION 3: [third most likely condition]
CONFIDENCE 3: [percentage]
EXPLANATION 3: [brief explanation why this matches]"""

    try:
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
        messages = [
            {"role": "system", "content": "You are a medical expert. Analyze symptoms and suggest potential conditions with explanations."},
            {"role": "user", "content": prompt}
        ]
        
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
        input_text = tokenizer.decode(inputs[0])
        
        payload = {
            "inputs": input_text,
            "parameters": {
                "max_new_tokens": 500,
                "temperature": 0.3,
                "top_p": 0.95,
                "do_sample": True
            }
        }
        
        response = requests.post(MIXTRAL_URL, headers=headers, json=payload)
        response.raise_for_status()
        response_data = response.json()
        
        if isinstance(response_data, list) and response_data:
            analysis = response_data[0]['generated_text'].split("[/INST]")[-1].strip()
            return parse_ai_response(analysis, symptoms)
            
    except Exception as e:
        print(f"\nNote: Advanced AI analysis unavailable: {str(e)}")
    return None

def parse_ai_response(text, symptoms):
    """Parse AI response into structured format."""
    results = []
    current_condition = {}
    
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        if line.lower().startswith('condition'):
            if current_condition:
                results.append(current_condition)
            current_condition = {'symptoms': symptoms}
            current_condition['disease'] = line.split(':', 1)[1].strip()
        elif line.lower().startswith('confidence'):
            try:
                conf_str = line.split(':', 1)[1].strip().rstrip('%')
                current_condition['confidence'] = float(conf_str)
            except (ValueError, IndexError):
                current_condition['confidence'] = 0
        elif line.lower().startswith('explanation'):
            current_condition['explanation'] = line.split(':', 1)[1].strip()
    
    if current_condition:
        results.append(current_condition)
    
    return results

def print_results(csv_results, ai_results, all_symptoms):
    """Print comprehensive results from both analyses."""
    print("\n=== Comprehensive Diagnosis Results ===")
    
    if all_symptoms:
        print("\nIdentified Symptoms:", ", ".join(sorted(all_symptoms)))
    
    if csv_results:
        print("\nBased on weighted database analysis:")
        for disease, info in list(csv_results.items())[:3]:
            print(f"\n{disease.title()}:")
            print(f"Confidence: {info['confidence']:.1f}%")
            print(f"Response Summary: {info['yes_count']} yes, {info['no_count']} no, {info['unsure_count']} unsure")
            print(f"Disease Rarity Score: {info['rarity_score']:.2f}")
            print("Matched Symptoms:", ", ".join(info['symptoms']))
    
    if ai_results:
        print("\nBased on AI medical analysis:")
        for result in ai_results:
            print(f"\n{result['disease'].title()}:")
            print(f"AI Confidence: {result['confidence']:.1f}%")
            if 'explanation' in result and result['explanation']:
                print("Explanation:", result['explanation'])
    
    print("\nIMPORTANT: This is not a medical diagnosis.")
    print("Please consult a healthcare professional for proper medical evaluation.")

def main():
    print("Welcome to the Enhanced Medical Diagnosis System")
    
    try:
        csv_file = 'new_symptom_disease_relations.csv'
        disease_symptoms, symptom_diseases, all_symptoms = load_symptom_disease_data(csv_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    while True:
        user_input = input("\nDescribe your symptoms (or type 'quit'): ").strip()
        
        if user_input.lower() == 'quit':
            break
            
        # Try Mistral first, fallback to text extraction
        detected_symptoms = query_mistral(user_input)
        if not detected_symptoms:
            print("Falling back to text-based symptom detection...")
            detected_symptoms = extract_symptoms_from_text(user_input, all_symptoms)
            
        if not detected_symptoms:
            print("No symptoms detected. Please try describing them differently.")
            continue
            
        print("\nDetected Symptoms:")
        for symptom in detected_symptoms:
            print(f"- {symptom}")
        
        # Database matching with weighted scoring
        matched_symptoms, potential_diseases = match_symptoms(
            detected_symptoms, all_symptoms, disease_symptoms, symptom_diseases)
        
        all_confirmed_symptoms = set(detected_symptoms)
        
        if potential_diseases:
            print("\nAnalyzing potential conditions...")
            csv_results = ask_additional_symptoms(
                disease_symptoms, potential_diseases, matched_symptoms, symptom_diseases)
            
            if csv_results is None:  # User quit during questions
                continue
            
            # Update all confirmed symptoms
            for info in csv_results.values():
                all_confirmed_symptoms.update(info['symptoms'])
            
            # Get AI analysis
            print("\nPerforming advanced medical analysis...")
            ai_results = analyze_with_ai(all_confirmed_symptoms)
            
            # Print comprehensive results
            print_results(csv_results, ai_results, all_confirmed_symptoms)
        else:
            # No database matches, try AI analysis
            print("\nNo strong matches found in database. Performing AI analysis...")
            ai_results = analyze_with_ai(all_confirmed_symptoms)
            print_results(None, ai_results, all_confirmed_symptoms)
        
        # Ask if user wants to continue
        continue_choice = input("\nWould you like to describe more symptoms? (yes/no): ").lower()
        if continue_choice != 'yes':
            break

    print("\nThank you for using the Enhanced Medical Diagnosis System.")
    print("Remember to consult healthcare professionals for proper medical advice.")

if __name__ == "__main__":
    main()