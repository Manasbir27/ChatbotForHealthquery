from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from flask_cors import CORS
import os
import csv
import requests
from transformers import AutoTokenizer
from fuzzywuzzy import process
from collections import defaultdict
from datetime import datetime
from werkzeug.utils import secure_filename
import json
from patient_registration import PatientRegistrationSystem

app = Flask(__name__, static_folder='static')
CORS(app)

# API Configuration
os.environ['HUGGINGFACE_TOKEN'] = "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"
MIXTRAL_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
headers = {"Authorization": f"Bearer {os.environ['HUGGINGFACE_TOKEN']}"}

# File upload configuration
app.config['UPLOAD_FOLDER'] = 'temp_audio'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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

# Initialize systems
registration_system = PatientRegistrationSystem()

class SymptomSession:
    def __init__(self):
        self.current_disease = None
        self.current_symptoms = set()
        self.asked_questions = set()
        self.disease_scores = defaultdict(lambda: {
            'score': 0,
            'total_questions': 0,
            'symptoms': set(),
            'yes_count': 0,
            'no_count': 0,
            'unsure_count': 0
        })
        self.potential_diseases = {}
        self.disease_weights = {}
        self.symptom_weights = {}
        self.remaining_questions = []

session = SymptomSession()

# Load symptom-disease data
def load_symptom_disease_data():
    disease_symptoms = defaultdict(set)
    symptom_diseases = defaultdict(set)
    all_symptoms = set()
    
    with open('new_symptom_disease_relations.csv', 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            symptom = row['Symptom'].strip().lower()
            disease = row['Disease'].strip().lower()
            all_symptoms.add(symptom)
            disease_symptoms[disease].add(symptom)
            symptom_diseases[symptom].add(disease)
    
    return disease_symptoms, symptom_diseases, all_symptoms

disease_symptoms, symptom_diseases, all_symptoms = load_symptom_disease_data()

# Registration routes
@app.route('/')
def index():
    return render_template('page1.html')

@app.route('/page2')
def page2():
    return render_template('page2.html')

@app.route('/page3')
def page3():
    return render_template('page3.html')

@app.route('/page4')
def page4():
    return render_template('page4.html')

@app.route('/page5')
def page5():
    return render_template('page5.html')

@app.route('/diagnosis')
def diagnosis():
    reason = request.args.get('reason', '')
    return render_template('index.html', initial_reason=reason)

# Form submission routes
@app.route('/submit_form', methods=['POST'])
def submit_form():
    try:
        form_data = request.json
        return jsonify({
            'success': True,
            'redirect': url_for('page5'),
            'form_data': form_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/start_diagnosis', methods=['POST'])
def start_diagnosis():
    try:
        reason = request.json.get('reason', '')
        return jsonify({
            'success': True,
            'redirect': f'/diagnosis?reason={reason}'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Audio processing route
@app.route('/api/process_audio', methods=['POST'])
def process_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': 'No audio file provided'})
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'success': False, 'error': 'Empty filename'})
        
        filename = secure_filename(f"temp_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(filepath)
        
        text = registration_system.speech_processor.transcribe_audio(filepath)
        if not text:
            return jsonify({'success': False, 'error': 'Failed to transcribe audio'})
            
        extracted_info = registration_system.form_processor.process_text(text)
        if 'dob' in extracted_info:
            age = registration_system.form_processor.calculate_age(extracted_info['dob'])
            if age:
                extracted_info['age'] = age
                
        return jsonify({'success': True, 'data': extracted_info})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    finally:
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)

# Diagnosis system functions
def calculate_disease_frequencies():
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

def calculate_symptom_specificity():
    total_diseases = len({disease for diseases in symptom_diseases.values() for disease in diseases})
    
    specificity_scores = {}
    for symptom, diseases in symptom_diseases.items():
        specificity = 1 - (len(diseases) / total_diseases)
        specificity_scores[symptom] = specificity * WEIGHTS['symptom_specificity']
    
    return specificity_scores

def query_mistral(text):
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
    
    prompt = f"""Extract ONLY the medical symptoms explicitly mentioned in the text. List ONLY symptoms that are directly stated, do not infer or add additional symptoms. Reply with just a comma-separated list of symptoms, nothing else.
    
Text: {text}
Symptoms:"""
    
    messages = [
        {"role": "system", "content": "You are a medical symptom detector. Extract and list only explicitly mentioned symptoms."},
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

def match_symptoms(detected_symptoms):
    matched_symptoms = set()
    potential_diseases = defaultdict(lambda: {
        'matched_symptoms': set(),
        'score': 0,
        'match_percentage': 0
    })
    
    session.disease_weights = calculate_disease_frequencies()
    session.symptom_weights = calculate_symptom_specificity()
    
    # Exact matches
    for symptom in detected_symptoms:
        if symptom in all_symptoms:
            matched_symptoms.add(symptom)
            for disease in symptom_diseases[symptom]:
                potential_diseases[disease]['matched_symptoms'].add(symptom)
                score = (WEIGHTS['symptom_match'] + 
                        session.symptom_weights[symptom] + 
                        session.disease_weights[disease])
                potential_diseases[disease]['score'] += score
    
    # Fuzzy matching
    unmatched = set(detected_symptoms) - matched_symptoms
    for symptom in unmatched:
        matches = process.extract(symptom, all_symptoms, limit=1)
        for match, score in matches:
            if score >= 85:
                matched_symptoms.add(match)
                for disease in symptom_diseases[match]:
                    potential_diseases[disease]['matched_symptoms'].add(match)
                    weighted_score = (WEIGHTS['symptom_match'] * (score/100) +
                                    session.symptom_weights[match] +
                                    session.disease_weights[disease])
                    potential_diseases[disease]['score'] += weighted_score
    
    # Calculate final scores
    final_diseases = {}
    for disease, info in potential_diseases.items():
        total_symptoms = len(disease_symptoms[disease])
        matched_count = len(info['matched_symptoms'])
        
        unmatched_penalty = (total_symptoms - matched_count) * WEIGHTS['symptom_count_penalty']
        final_score = info['score'] - unmatched_penalty
        
        match_percentage = (matched_count / len(detected_symptoms)) * 100
        coverage_percentage = (matched_count / total_symptoms) * 100
        confidence = (match_percentage + coverage_percentage) / 2
        
        if confidence >= 30:
            final_diseases[disease] = {
                'matched_symptoms': info['matched_symptoms'],
                'match_percentage': confidence,
                'score': final_score,
                'total_symptoms': total_symptoms
            }
    
    return matched_symptoms, dict(sorted(
        final_diseases.items(),
        key=lambda x: x[1]['score'],
        reverse=True
    )[:3])

def prepare_follow_up_questions(potential_diseases, confirmed_symptoms):
    questions = []
    max_questions_per_disease = 5
    
    sorted_diseases = sorted(
        potential_diseases.items(),
        key=lambda x: x[1]['score'],
        reverse=True
    )
    
    for disease, info in sorted_diseases:
        disease_questions = 0
        unconfirmed_symptoms = disease_symptoms[disease] - confirmed_symptoms
        
        symptom_priorities = [
            (symptom, session.symptom_weights.get(symptom, 0))
            for symptom in unconfirmed_symptoms
            if symptom not in session.asked_questions
        ]
        symptom_priorities.sort(key=lambda x: x[1], reverse=True)
        
        for symptom, _ in symptom_priorities:
            if disease_questions < max_questions_per_disease:
                questions.append({
                    'disease': disease,
                    'symptom': symptom,
                    'specificity': session.symptom_weights.get(symptom, 0)
                })
                disease_questions += 1
    
    questions.sort(key=lambda x: x['specificity'], reverse=True)
    return questions[:10]

def analyze_with_ai(symptoms):
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

# Diagnosis routes
@app.route('/analyze_symptoms', methods=['POST'])
def analyze_symptoms():
    user_input = request.json.get('message', '')
    
    session.current_symptoms = set()
    session.asked_questions = set()
    session.disease_scores = defaultdict(lambda: {
        'score': 0,
        'total_questions': 0,
        'symptoms': set(),
        'yes_count': 0,
        'no_count': 0,
        'unsure_count': 0
    })
    
    detected_symptoms = query_mistral(user_input)
    if not detected_symptoms:
        return jsonify({
            'status': 'error',
            'message': 'No symptoms detected. Please try describing them differently.'
        })
    
    matched_symptoms, potential_diseases = match_symptoms(detected_symptoms)
    session.current_symptoms = matched_symptoms
    session.potential_diseases = potential_diseases
    
    if potential_diseases:
        session.remaining_questions = prepare_follow_up_questions(
            potential_diseases, 
            matched_symptoms
        )
        
        if session.remaining_questions:
            return jsonify({
                'status': 'success',
                'detected_symptoms': list(detected_symptoms),
                'matched_symptoms': list(matched_symptoms),
                'has_questions': True,
                'next_question': session.remaining_questions[0]
            })
    
    ai_results = analyze_with_ai(list(detected_symptoms))
    return jsonify({
        'status': 'success',
        'detected_symptoms': list(detected_symptoms),
        'matched_symptoms': list(matched_symptoms),
        'ai_results': ai_results,
        'has_questions': False
    })

@app.route('/answer_question', methods=['POST'])
def answer_question():
    answer = request.json.get('answer', '').lower()
    
    if not session.remaining_questions:
        return process_final_results()
    
    current_question = session.remaining_questions[0]
    disease = current_question['disease']
    symptom = current_question['symptom']
    
    session.asked_questions.add(symptom)
    
    if answer == 'yes':
        session.current_symptoms.add(symptom)
        session.disease_scores[disease]['symptoms'].add(symptom)
        score_change = (WEIGHTS['yes_response'] + 
                       session.symptom_weights.get(symptom, 0))
        session.disease_scores[disease]['yes_count'] += 1
    elif answer == 'no':
        score_change = WEIGHTS['no_response']
        session.disease_scores[disease]['no_count'] += 1
    else:
        score_change = WEIGHTS['unsure_response']
        session.disease_scores[disease]['unsure_count'] += 1
    
    session.disease_scores[disease]['score'] += score_change
    session.disease_scores[disease]['total_questions'] += 1
    session.remaining_questions = session.remaining_questions[1:]
    
    if session.remaining_questions:
        return jsonify({
            'status': 'continue',
            'next_question': session.remaining_questions[0]
        })
    else:
        return process_final_results()

def process_final_results():
    results = {
        'status': 'complete',
        'all_symptoms': list(session.current_symptoms),
        'csv_results': []
    }
    
    for disease, scores in session.disease_scores.items():
        if scores['total_questions'] > 0:
            base_confidence = (scores['score'] / scores['total_questions']) * 100
            
            symptom_bonus = sum(session.symptom_weights.get(s, 0) 
                              for s in scores['symptoms'])
            disease_bonus = session.disease_weights.get(disease, 0)
            
            final_confidence = min(100, base_confidence + 
                                 (symptom_bonus * 10) + 
                                 (disease_bonus * 10))
            
            results['csv_results'].append({
                'disease': disease,
                'confidence': final_confidence,
                'symptoms': list(scores['symptoms']),
                'yes_count': scores['yes_count'],
                'no_count': scores['no_count'],
                'unsure_count': scores['unsure_count'],
                'rarity_score': session.disease_weights.get(disease, 0)
            })
    
    results['csv_results'].sort(key=lambda x: x['confidence'], reverse=True)
    
    ai_results = analyze_with_ai(list(session.current_symptoms))
    if ai_results:
        results['ai_results'] = ai_results
    
    return jsonify(results)

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/api/save_registration', methods=['POST'])
def save_registration():
    try:
        data = request.json
        required_fields = ['name', 'dob', 'bloodGroup', 'address', 'phone', 'email', 'reason']
        missing_fields = [field for field in required_fields if not data.get(field)]
        
        if missing_fields:
            return jsonify({
                'success': False, 
                'error': f'Missing fields: {", ".join(missing_fields)}'
            })
            
        return jsonify({
            'success': True,
            'redirect': url_for('page2')
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)