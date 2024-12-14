import os
import sounddevice as sd
import wavio
import numpy as np
import torch
from transformers import pipeline
import speech_recognition as sr
from datetime import datetime, date
import json
import re
import time

# Updated FFmpeg paths setup
os.environ["FFMPEG_BINARY"] = "C:\\Users\\Asus\\Desktop\\ffmpeg-master-latest-win64-gpl\\bin\\ffmpeg.exe"
os.environ["PATH"] += os.pathsep + "C:\\Users\\Asus\\Desktop\\ffmpeg-master-latest-win64-gpl\\bin"

class AudioRecorder:
    def __init__(self, sample_rate=44100, channels=2):
        self.sample_rate = sample_rate
        self.channels = channels
        self.recording = None
        
    def record_audio(self, duration):
        """Record audio for specified duration with countdown"""
        print(f"\nPreparing to record for {duration} seconds...")
        for i in [3, 2, 1]:
            print(f"{i}...")
            time.sleep(1)
        print("Start speaking now!")
        print("Recording in progress...")
        
        try:
            recording = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels
            )
            # Add progress indicator
            for i in range(duration):
                time.sleep(1)
                print(".", end="", flush=True)
            sd.wait()
            print("\nRecording completed!")
            self.recording = recording
            return recording
        except Exception as e:
            print(f"\nError during recording: {e}")
            return None
    
    def save_audio(self, filename="temp_recording.wav"):
        """Save the recorded audio to a WAV file"""
        if self.recording is not None:
            try:
                wavio.write(filename, self.recording, self.sample_rate, sampwidth=2)
                return filename
            except Exception as e:
                print(f"Error saving audio file: {e}")
        return None

class SpeechProcessor:
    def __init__(self):
        print("Initializing speech recognition systems...")
        try:
            self.transcriber = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-base",
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.8
            print("Speech recognition systems initialized successfully.")
        except Exception as e:
            print(f"Error initializing speech recognition: {e}")
            raise

    def clean_transcription(self, text):
        """Enhanced text cleaning with context awareness"""
        # First, protect words that shouldn't be modified
        protected_words = ['fatigue', 'attitude', 'atmosphere', 'eating', 'rating']
        
        # Convert to lowercase and protect certain words
        protected_text = text.lower()
        for word in protected_words:
            protected_text = protected_text.replace(word, f"PROTECTED_{word}_PROTECTED")
        
        # Handle email-specific patterns
        email_pattern = r'([a-zA-Z0-9._%+-]+)\s+(?:at|@)\s+([a-zA-Z]+)(?:\s+dot\s+|\.)com'
        
        def email_replacer(match):
            email_part = match.group(1).replace(" ", "")
            domain = match.group(2)
            return f"{email_part}@{domain}.com"
        
        protected_text = re.sub(email_pattern, email_replacer, protected_text)
        
        # Clean up specific patterns
        replacements = {
            'at the rate': '@',
            'at gmail dot com': '@gmail.com',
            'at yahoo dot com': '@yahoo.com',
            'at hotmail dot com': '@hotmail.com',
            'at outlook dot com': '@outlook.com'
        }
        
        for old, new in replacements.items():
            protected_text = protected_text.replace(old, new)
        
        # Restore protected words
        for word in protected_words:
            protected_text = protected_text.replace(f"PROTECTED_{word}_PROTECTED", word)
        
        # Clean up extra spaces and punctuation
        protected_text = re.sub(r'\s+', ' ', protected_text)
        return protected_text.strip()

    def transcribe_audio(self, audio_file):
        """Enhanced transcription with multiple attempts"""
        text = None
        
        try:
            # Try Whisper first
            result = self.transcriber(audio_file)
            text = result["text"]
            print(f"\nRaw transcription: {text}")
        except Exception as e:
            print(f"Whisper transcription failed: {e}")
        
        # If Whisper fails or text is too short, try Google Speech Recognition
        if not text or len(text) < 5:
            try:
                with sr.AudioFile(audio_file) as source:
                    audio = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio)
                print(f"Google Speech Recognition: {text}")
            except Exception as e:
                print(f"Speech recognition failed: {e}")
        
        if text:
            cleaned_text = self.clean_transcription(text)
            print(f"Cleaned transcription: {cleaned_text}")
            return cleaned_text
        return None

class FormProcessor:
    def __init__(self):
        # Enhanced patterns for better recognition
        self.patterns = {
            'name': [
                r'(?:my name is|i am called|i am|this is|name is)\s+([A-Za-z\s]+?)(?=\s*(?:my|$|\.|live|phone|email))',
                r'([A-Za-z\s]+?)(?=\s*(?:is my name|here|\.))'
            ],
            'dob': [
                r'(?:i was born on|birth date is|born on|dob is)\s+(?:the\s+)?([\d]{1,2})(?:th|st|nd|rd)?\s+(?:of\s+)?(january|february|march|april|may|june|july|august|september|october|november|december),?\s+(\d{4})',
                r'(?:born|birth).+?([\d]{1,2})(?:th|st|nd|rd)?\s+(january|february|march|april|may|june|july|august|september|october|november|december),?\s+(\d{4})'
            ],
            'phone': [
                r'(?:my phone|contact|number|phone number|phone is|mobile|contact number)\s*(?:is|:)?\s*((?:\+?\d{1,3}[-\s]?)?\d{3}[-\s]?\d{3}[-\s]?\d{4})',
                r'(\d{3}[-\s]?\d{3}[-\s]?\d{4})'
            ],
            'email': [
                r'(?:my email|mail|email address|email id|email is|mail id).?\s([a-zA-Z0-9][a-zA-Z0-9._%+-]@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
                r'([a-zA-Z0-9][a-zA-Z0-9._%+-]*)\s+(?:at)\s+([a-zA-Z]+)(?:\.|\s+dot\s+)com',
                r'([a-zA-Z0-9][a-zA-Z0-9._%+-]*)@([a-zA-Z0-9.-]+)\.([a-zA-Z]{2,})'
            ],
            'address': [
                r'(?:i live|stay|reside|my address is|address is|live at|stay at)\s+(?:at|in|on)\s+([^.]+?)(?=\s*(?:my phone|my email|$|\.|my reason|i am here|phone|email))',
                r'(?:address|location|residence).+?is\s+([^.]+?)(?=\s*(?:my|$|\.))',
                r'live\s+(?:at|in|on)\s+([^.]+?)(?=\s*(?:my|$|\.))'
            ],
            'reason': [
                r'(?:reason for visit|visiting because|here because|suffering from|problem is|having|came here for)\s+([^.]+?)(?=\s*(?:my phone|my email|$|\.))',
                r'(?:i am here|came here|visiting)\s+(?:because|for|as)\s+([^.]+?)(?=\s*(?:my phone|my email|$|\.))',
                r'(?:i have|having|suffering from)\s+([^.]+?)(?=\s*(?:my phone|my email|$|\.))'
            ]
        }
        
        self.month_to_num = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }

    def extract_field_info(self, text, field):
        """Enhanced field information extraction"""
        if field in self.patterns:
            patterns = self.patterns[field] if isinstance(self.patterns[field], list) else [self.patterns[field]]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    if field == 'dob':
                        day, month, year = match.groups()
                        month_num = self.month_to_num[month.lower()]
                        return f"{day} {month_num:02d} {year}"
                    elif field == 'email':
                        if len(match.groups()) > 1:  # Pattern with 'at' and 'dot'
                            username, domain = match.groups()[:2]
                            return f"{username}@{domain}.com"
                        else:  # Full email pattern
                            email = match.group(1)
                            if '@' in email and '.' in email.split('@')[1]:
                                return email.lower()
                    elif field == 'name':
                        name = match.group(1).strip()
                        name_parts = name.split()
                        if len(name_parts) >= 2:  # Ensure at least first and last name
                            return ' '.join(name_parts)
                    else:
                        return match.group(1).strip()
        return None

    def calculate_age(self, dob_str):
        """Calculate age from DOB string"""
        try:
            day, month_str, year = dob_str.split()
            month = int(month_str)
            dob = date(int(year), month, int(day))
            today = date.today()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            return age
        except Exception as e:
            print(f"Error calculating age: {e}")
            return None

    def process_text(self, text):
        """Enhanced text processing with improved extraction"""
        extracted_info = {}
        text = text.lower()
        
        # Process name first with special handling
        name_value = None
        for pattern in self.patterns['name']:
            match = re.search(pattern, text)
            if match:
                name_value = match.group(1).strip()
                if len(name_value.split()) >= 2:
                    extracted_info['name'] = name_value
                    break
        
        # Process other fields
        for field in self.patterns.keys():
            if field != 'name':  # Skip name as it's already processed
                value = self.extract_field_info(text, field)
                if value:
                    extracted_info[field] = value
                    if field == 'dob':
                        age = self.calculate_age(value)
                        if age:
                            extracted_info['age'] = age

        # Special handling for reason if not found
        if 'reason' not in extracted_info:
            sentences = text.split('.')
            for sentence in reversed(sentences):
                if any(word in sentence for word in ['because', 'having', 'suffering', 'problem', 'pain', 'feeling']):
                    extracted_info['reason'] = sentence.strip()
                    break
                    
        return extracted_info

class HeadacheAssessment:
    def __init__(self):
        self.frequency_options = ['None', '1-4', '5-10', '11-14', 'More than 15', 'Daily']
        self.type_options = ['Pounding', 'Crushing', 'Dull', 'Pressure']
        self.time_options = ['Morning', 'Afternoon', 'Evening', 'Middle of the night', 'Menstrual', 'Constant']
        self.severity_options = ['Mild', 'Moderate', 'Severe']
        
    def get_assessment(self):
        assessment = {}
        
        print("\nHeadache Assessment")
        print("="*50)
        
        # Frequency
        print("\nHow many days do you experience headaches per month?")
        for i, option in enumerate(self.frequency_options, 1):
            print(f"{i}. {option}")
        while True:
            try:
                choice = int(input("Enter number (1-6): "))
                if 1 <= choice <= len(self.frequency_options):
                    assessment['frequency'] = self.frequency_options[choice-1]
                    break
            except ValueError:
                print("Please enter a valid number.")
        
        # Type
        print("\nWhat type of headache do you experience?")
        for i, option in enumerate(self.type_options, 1):
            print(f"{i}. {option}")
        while True:
            try:
                choice = int(input("Enter number (1-4): "))
                if 1 <= choice <= len(self.type_options):
                    assessment['type'] = self.type_options[choice-1]
                    break
            except ValueError:
                print("Please enter a valid number.")
        
        # Time
        print("\nWhen do your headaches generally occur?")
        for i, option in enumerate(self.time_options, 1):
            print(f"{i}. {option}")
        while True:
            try:
                choice = int(input("Enter number (1-6): "))
                if 1 <= choice <= len(self.time_options):
                    assessment['time'] = self.time_options[choice-1]
                    break
            except ValueError:
                print("Please enter a valid number.")
        
        # Severity
        print("\nHow severe are your headaches?")
        for i, option in enumerate(self.severity_options, 1):
            print(f"{i}. {option}")
        while True:
            try:
                choice = int(input("Enter number (1-3): "))
                if 1 <= choice <= len(self.severity_options):
                    assessment['severity'] = self.severity_options[choice-1]
                    break
            except ValueError:
                print("Please enter a valid number.")
        
        return assessment

class MedicalHistory:
    def __init__(self):
        self.conditions = [
            'Neurologic (seizure, stroke, etc)',
            'Heart Disease',
            'Lung Problems',
            'Diabetes',
            'High Blood Pressure',
            'Cancer',
            'Kidney or urinary problems',
            'Sexual',
            'Psychological',
            'Nervous breakdown',
            'Ears, nose, or throat problems',
            'Abdominal, stomach/intestinal'
        ]
    
    def get_medical_history(self):
        history = {'conditions': {}, 'allergies': [], 'medications': []}
        
        print("\nMedical History")
        print("="*50)
        
        # Medical conditions
        print("\nHave you had any of the following problems?")
        for condition in self.conditions:
            while True:
                response = input(f"{condition} (Y/N): ").upper()
                if response in ['Y', 'N']:
                    if response == 'Y':
                        explanation = input("Please explain: ")
                        history['conditions'][condition] = explanation
                    else:
                        history['conditions'][condition] = "No"
                    break
                print("Please enter Y or N.")
        
        # Allergies
        print("\nAllergies to Medications:")
        while True:
            allergy = input("Enter medication allergy and reaction (or press Enter to finish): ")
            if not allergy:
                break
            history['allergies'].append(allergy)
        
        # Current medications
        print("\nCurrent Medications:")
        while True:
            medication = input("Enter medication with dosage and frequency (or press Enter to finish): ")
            if not medication:
                break
            history['medications'].append(medication)
        
        return history

class SocialHistory:
    def get_social_history(self):
        social = {}
        
        print("\nSocial History")
        print("="*50)
        
        # Smoking
        social['smoking'] = {}
        social['smoking']['regular'] = input("Do you smoke regularly? (Y/N): ").upper()
        if social['smoking']['regular'] == 'Y':
            social['smoking']['duration'] = input("How long? ")
            social['smoking']['type'] = input("What do you smoke? (Cigarettes/Pipe/Cigars): ")
            social['smoking']['per_day'] = input("How many per day? ")
        
        # Alcohol
        social['alcohol'] = {}
        social['alcohol']['drinks'] = input("Do you drink alcohol? (Y/N): ").upper()
        if social['alcohol']['drinks'] == 'Y':
            social['alcohol']['regular'] = input("Do you drink regularly? (Y/N): ").upper()
            social['alcohol']['type'] = input("What do you drink? (Beer/Wine/Hard liquor): ")
            social['alcohol']['per_day'] = input("How much per day? ")
            social['alcohol']['per_week'] = input("How much per week? ")
            social['alcohol']['duration'] = input("How long? ")
        
        # Other information
        social['drugs'] = input("Do you use any street drugs? (Y/N): ").upper()
        social['addiction'] = input("Are you or have you been addicted to any drugs or alcohol? (Y/N): ").upper()
        social['transfusions'] = input("Any blood transfusions? (Y/N): ").upper()
        social['tattoos'] = input("Any tattoos? (Y/N): ").upper()
        social['risky_sexual'] = input("Any risky sexual activity for STDs? (Y/N): ").upper()
        social['marital_status'] = input("Are you single, married, divorced, or widowed? ")
        social['occupation'] = input("What is your job? ")
        
        return social

class FamilyHistory:
    def __init__(self):
        self.conditions = [
            'Similar type of illness that you have now',
            'Heart Disease',
            'Stroke',
            'High Blood Pressure',
            "Alzheimer's or dementia",
            'Diabetes',
            'Migraines',
            'Cancer',
            'Seizure disorder or epilepsy',
            'Blood clotting disorder',
            'Muscle disease',
            'Nerve disease or neuropathy',
            'Tremor',
            "Parkinson's Disease"
        ]
    
    def get_family_history(self):
        history = {'conditions': {}}
        
        print("\nFamily History")
        print("="*50)
        
        print("\nFor each condition, enter the relative who had it (or N for none)")
        for condition in self.conditions:
            relative = input(f"{condition}: ")
            if relative.upper() != 'N':
                history['conditions'][condition] = relative
        
        history['adopted'] = input("Are you adopted? (Y/N): ").upper()
        
        return history

class PatientRegistrationSystem:
    def __init__(self):
        print("Initializing Patient Registration System...")
        self.audio_recorder = AudioRecorder()
        self.speech_processor = SpeechProcessor()
        self.form_processor = FormProcessor()
        self.headache_assessment = HeadacheAssessment()
        self.medical_history = MedicalHistory()
        self.social_history = SocialHistory()
        self.family_history = FamilyHistory()
        self.form_data = {}

    def show_speaking_format(self):
        """Display the format for speaking information"""
        print("\n" + "="*50)
        print("SPEAKING FORMAT GUIDE")
        print("="*50)
        print("\nOption 1: Speak all details at once (30 seconds):")
        print("---")
        print("Example format:")
        print("My name is John Smith.")
        print("I was born on 27th September 2003.")
        print("I live at 123 Main Street, Apartment 4B, New York.")
        print("My phone number is 123 456 7890.")
        print("My email id is john.smith@email.com")
        print("I'm here because I have been experiencing severe headaches.")
        print("---")
        print("\nTips for better recognition:")
        print("• Speak clearly and at a moderate pace")
        print("• For email, say: 'My email ID is ___ at gmail dot com'")
        print("• For phone, say each digit clearly: 'nine two zero five...'")
        print("\nOption 2: Answer questions one by one (10 seconds each)")
        print("="*50)

    def verify_and_correct_info(self):
        """Allow user to verify and correct information"""
        while True:
            print("\nCurrent Information:")
            print("="*50)
            fields = ['name', 'dob', 'age', 'address', 'phone', 'email', 'reason']
            for idx, field in enumerate(fields, 1):
                if field in self.form_data:
                    print(f"{idx}. {field.title()}: {self.form_data[field]}")
            
            print("\nWould you like to correct any information?")
            print("Enter the number of the field to correct (1-7), or 0 to finish")
            
            try:
                choice = input("Choice (0-7): ")
                if choice == '0':
                    break
                elif choice.isdigit() and 1 <= int(choice) <= len(fields):
                    field = fields[int(choice)-1]
                    print(f"\nCurrent {field}: {self.form_data.get(field, 'Not set')}")
                    while True:
                        correction_method = input("Would you like to (1) Speak or (2) Type the correction? ")
                        if correction_method == '1':
                            self.record_field(field)
                            break
                        elif correction_method == '2':
                            self.manual_input(field)
                            break
                        else:
                            print("Invalid choice. Please select 1 or 2.")
                    
                    if field == 'dob' and 'dob' in self.form_data:
                        age = self.form_processor.calculate_age(self.form_data['dob'])
                        if age:
                            self.form_data['age'] = age
                else:
                    print("Invalid choice. Please enter a number between 0 and 7.")
            except ValueError:
                print("Please enter a valid number.")

    def record_field(self, field_name, is_full_details=False):
        """Record audio for a specific field"""
        duration = 30 if is_full_details else 10
        
        if is_full_details:
            print("\nPlease provide your complete details (30 seconds) including:")
            print("✓ Your full name")
            print("✓ Date of birth (e.g., 'I was born on 27th September 2003')")
            print("✓ Complete address")
            print("✓ Phone number (speak digits clearly)")
            print("✓ Email ID (say: 'My email ID is ___ at gmail dot com')")
            print("✓ Reason for visit")
        else:
            print(f"\nPlease tell me your {field_name} (10 seconds)")
            if field_name == 'email':
                print("Format: My email ID is example at gmail dot com")
            elif field_name == 'dob':
                print("Format: I was born on 27th September 2003")
            elif field_name == 'phone':
                print("Format: My phone number is nine two zero five...")
            elif field_name == 'name':
                print("Format: My name is [First] [Middle] [Last]")
        
        input("\nPress Enter when you're ready to start recording...")
        
        self.audio_recorder.record_audio(duration)
        audio_file = self.audio_recorder.save_audio(f"temp_{field_name}.wav")
        
        if audio_file:
            text = self.speech_processor.transcribe_audio(audio_file)
            if text:
                print(f"\nTranscribed text: {text}")
                extracted_info = self.form_processor.process_text(text)
                
                if extracted_info:
                    print("\nExtracted information:")
                    for key, value in extracted_info.items():
                        print(f"{key.title()}: {value}")
                        if value and (field_name == 'full_details' or key == field_name):
                            self.form_data[key] = value
                
                os.remove(audio_file)
                return bool(extracted_info)
        
        return False

    def manual_input(self, field_name):
        """Enhanced manual input with validation"""
        if field_name == 'dob':
            print("\nEnter date of birth in format: DD MM YYYY (e.g., 27 09 2003)")
        elif field_name == 'email':
            print("\nEnter email in format: example@domain.com")
        elif field_name == 'phone':
            print("\nEnter phone number (10 digits): XXXXXXXXXX")
        
        while True:
            value = input(f"Please enter {field_name}: ").strip()
            
            if field_name == 'dob':
                try:
                    age = self.form_processor.calculate_age(value)
                    if age:
                        self.form_data['dob'] = value
                        self.form_data['age'] = age
                        break
                    else:
                        print("Invalid date format. Please try again.")
                except:
                    print("Invalid date format. Please try again.")
            
            elif field_name == 'email':
                if '@' in value and '.' in value.split('@')[1]:
                    self.form_data[field_name] = value.lower()
                    break
                else:
                    print("Invalid email format. Please try again.")
            
            elif field_name == 'phone':
                phone_digits = ''.join(filter(str.isdigit, value))
                if len(phone_digits) == 10:
                    self.form_data[field_name] = phone_digits
                    break
                else:
                    print("Invalid phone number. Please enter 10 digits.")
            
            else:
                self.form_data[field_name] = value
                break

    def fill_form(self):
        """Fill the complete patient registration form"""
        print("\nWelcome to the Patient Registration System")
        self.show_speaking_format()
        
        while True:
            choice = input("\nWould you like to:\n1. Speak all details at once (30 seconds)\n2. Answer questions one by one (10 seconds each)\nEnter your choice (1 or 2): ")
            if choice in ['1', '2']:
                break
            print("Invalid choice. Please select 1 or 2.")
        
        if choice == '1':
            self.record_field('full_details', is_full_details=True)
            
            # Check for missing required fields
            required_fields = ['name', 'dob', 'address', 'phone', 'email', 'reason']
            missing_fields = [field for field in required_fields if field not in self.form_data]
            
            if missing_fields:
                print("\nI need some additional information:")
                for field in missing_fields:
                    print(f"\nPlease provide your {field}:")
                    while True:
                        input_choice = input(f"Would you like to (1) Speak or (2) Type? ")
                        if input_choice == '1':
                            if self.record_field(field):
                                break
                        elif input_choice == '2':
                            self.manual_input(field)
                            break
                        else:
                            print("Invalid choice. Please select 1 or 2.")
        else:
            fields = ['name', 'dob', 'address', 'phone', 'email', 'reason']
            for field in fields:
                if field not in self.form_data:
                    while True:
                        input_choice = input(f"\nFor {field}, would you like to (1) Speak or (2) Type? ")
                        if input_choice == '1':
                            if self.record_field(field):
                                break
                        elif input_choice == '2':
                            self.manual_input(field)
                            break
                        else:
                            print("Invalid choice. Please select 1 or 2.")
        
        # Add additional assessments
        self.form_data['headache_assessment'] = self.headache_assessment.get_assessment()
        self.form_data['medical_history'] = self.medical_history.get_medical_history()
        self.form_data['social_history'] = self.social_history.get_social_history()
        self.form_data['family_history'] = self.family_history.get_family_history()
        
        # Verify and correct information
        self.verify_and_correct_info()
        return self.form_data

    def display_form(self):
        """Display all collected information"""
        print("\nPatient Registration Details:")
        print("="*50)
        
        # Basic Information
        display_order = ['name', 'age', 'dob', 'address', 'phone', 'email', 'reason']
        for field in display_order:
            if field in self.form_data:
                formatted_value = self.form_data[field]
                if field == 'age':
                    formatted_value = f"{formatted_value} years"
                elif field == 'phone':
                    if len(formatted_value) == 10:
                        formatted_value = f"{formatted_value[:3]}-{formatted_value[3:6]}-{formatted_value[6:]}"
                print(f"{field.title()}: {formatted_value}")
        
        # Headache Assessment
        print("\nHeadache Assessment:")
        print("-"*30)
        for key, value in self.form_data['headache_assessment'].items():
            print(f"{key.title()}: {value}")
        
        # Medical History
        print("\nMedical History:")
        print("-"*30)
        print("\nMedical Conditions:")
        for condition, status in self.form_data['medical_history']['conditions'].items():
            print(f"- {condition}: {status}")
        
        print("\nAllergies to Medications:")
        if self.form_data['medical_history']['allergies']:
            for allergy in self.form_data['medical_history']['allergies']:
                print(f"- {allergy}")
        else:
            print("- None reported")
        
        print("\nCurrent Medications:")
        if self.form_data['medical_history']['medications']:
            for medication in self.form_data['medical_history']['medications']:
                print(f"- {medication}")
        else:
            print("- None reported")
        
        # Social History
        print("\nSocial History:")
        print("-"*30)
        
        # Smoking
        print("\nSmoking History:")
        if self.form_data['social_history']['smoking']['regular'] == 'Y':
            print(f"- Smokes {self.form_data['social_history']['smoking']['type']}")
            print(f"- {self.form_data['social_history']['smoking']['per_day']} per day")
            print(f"- Duration: {self.form_data['social_history']['smoking']['duration']}")
        else:
            print("- Non-smoker")
        
        # Alcohol
        print("\nAlcohol Consumption:")
        if self.form_data['social_history']['alcohol']['drinks'] == 'Y':
            print(f"- Drinks {self.form_data['social_history']['alcohol']['type']}")
            print(f"- {self.form_data['social_history']['alcohol']['per_day']} per day")
            print(f"- {self.form_data['social_history']['alcohol']['per_week']} per week")
            print(f"- Duration: {self.form_data['social_history']['alcohol']['duration']}")
        else:
            print("- No alcohol consumption")
        
        # Other Social History
        print("\nOther Social Information:")
        print(f"- Street Drugs: {self.form_data['social_history']['drugs']}")
        print(f"- History of Addiction: {self.form_data['social_history']['addiction']}")
        print(f"- Blood Transfusions: {self.form_data['social_history']['transfusions']}")
        print(f"- Tattoos: {self.form_data['social_history']['tattoos']}")
        print(f"- Risk for STDs: {self.form_data['social_history']['risky_sexual']}")
        print(f"- Marital Status: {self.form_data['social_history']['marital_status']}")
        print(f"- Occupation: {self.form_data['social_history']['occupation']}")
        
        # Family History
        print("\nFamily History:")
        print("-"*30)
        for condition, relative in self.form_data['family_history']['conditions'].items():
            print(f"- {condition}: {relative}")
        print(f"Adopted: {self.form_data['family_history']['adopted']}")

    def save_form(self, filename=None):
        """Save the form data to a JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"patient_registration_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.form_data, f, indent=4)
            print(f"\nForm data successfully saved to {filename}")
        except Exception as e:
            print(f"Error saving form data: {e}")

def main():
    try:
        print("Starting Patient Registration System...")
        registration_system = PatientRegistrationSystem()
        registration_system.fill_form()
        registration_system.display_form()
        registration_system.save_form()
        print("\nRegistration process completed successfully!")
    except Exception as e:
        print(f"An error occurred during registration: {e}")
        print("Please try again or contact system administrator.")

if __name__ == "__main__":
    main()