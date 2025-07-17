import os
import base64
import re
from email.header import decode_header
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import datetime
import time
import json
import matplotlib.pyplot as plt
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.utils import make_msgid, parsedate_to_datetime
from dotenv import load_dotenv
import asyncio # New import for async operations
import google.generativeai as genai # New import for Gemini API

# You'll need to install these (if you haven't yet):
# pip install google-api-python-client google-auth-oauthlib google-auth-httplib2
# pip install nltk spacy matplotlib python-dotenv google-generativeai
# python -m spacy download en_core_web_sm
import nltk
# nltk.download('punkt') # Uncomment and run if you don't have it
# nltk.download('stopwords') # Uncomment and run if you don't have it
# nltk.download('punkt_tab') # Uncomment and run if you don't have it
from nltk.corpus import stopwords
from collections import Counter, defaultdict
import spacy

# --- Load environment variables from .env file ---
load_dotenv()

# --- Configuration ---
# IMPORTANT: If modifying these scopes, delete the file token.json and re-run to re-authenticate.
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.readonly"
]

# The file token.json stores the user's access and refresh tokens, and is
# created automatically when the authorization flow completes for the first
# time.
TOKEN_FILE = "token.json"
CREDENTIALS_FILE = "credentials.json"

# --- Email Sending Configuration ---
# Now loaded from .env file
SENDER_EMAIL_ADDRESS = os.getenv("SENDER_EMAIL_ADDRESS")
SENDER_EMAIL_PASSWORD = os.getenv("SENDER_EMAIL_PASSWORD")
RECIPIENT_EMAIL_ADDRESS = os.getenv("RECIPIENT_EMAIL_ADDRESS")
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465 # For SSL

# Path to your ManagerFM logo image
MANAGER_FM_LOGO_PATH = "/Users/andrewledet/Desktop/gmailapi/managerFMlogo.png"

# Path to your Gemini prompt file
GEMINI_PROMPT_FILE_PATH = "/Users/andrewledet/Desktop/gmailapi/prompt_Contextual_Morning_Briefing_Theme.txt"

# Dynamically generate the output JSON filename with a timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# --- NLP Setup ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    exit()

# Combine NLTK stopwords with custom stopwords for better filtering
more_stopwords = set([
    "br", "https", "know", "wrote", "jul", "us", "pm", "good", "like", "play", "week",
    "matthew", "india", "let", "hi", "today", "pat", "get", "team", "andrew", "time",
    "email", "day", "new", "will", "see", "can", "just", "one", "also", "from", "to",
    "for", "with", "and", "or", "but", "a", "an", "the", "in", "on", "at", "of", "is",
    "are", "was", "were", "be", "been", "has", "have", "had", "do", "does", "did",
    "would", "could", "should", "i", "you", "he", "she", "it", "we", "they", "me",
    "him", "her", "us", "them", "my", "your", "his", "her", "its", "our", "their",
    "this", "that", "these", "those", "what", "where", "when", "why", "how", "which",
    "who", "whom", "whose", "here", "there", "when", "where", "why", "how", "all",
    "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will",
    "just", "don", "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren",
    "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn",
    "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn", "etc", "etcetera",
    "regards", "best", "thanks", "thank", "please", "find", "attached", "see", "look",
    "forward", "best", "regards", "sincerely", "kindly", "hope", "know", "think",
    "feel", "make", "take", "get", "go", "come", "want", "need", "like", "play", "check",
    "out", "about", "this", "that", "these", "those", "here", "there", "when", "where",
    "why", "how", "what", "which", "who", "whom", "whose", "etc", "etcetera", "regards",
    "best", "thanks", "thank", "please", "find", "attached", "see", "look", "forward",
    "best", "regards", "sincerely", "kindly", "hope", "know", "think", "feel", "make",
    "take", "get", "go", "come", "want", "need", "play", "check", "out", "about", "also",
    "just", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "first", "second", "third", "fourth", "fifth", "last", "next", "previous", "current",
    "new", "old", "good", "great", "bad", "big", "small", "long", "short", "high", "low",
    "early", "late", "up", "down", "in", "out", "on", "off", "over", "under", "through",
    "around", "at", "by", "for", "from", "into", "like", "of", "off", "on", "out", "over",
    "through", "to", "under", "up", "with", "as", "at", "but", "by", "for", "from", "if",
    "in", "into", "like", "more", "most", "much", "my", "no", "not", "of", "off", "on",
    "or", "other", "out", "over", "per", "so", "some", "such", "than", "that", "the",
    "their", "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "you", "your"
])
all_stopwords = stopwords.words('english')
all_stopwords.extend(more_stopwords)
all_stopwords = set(all_stopwords) # Convert to set for faster lookup

# --- Configure Gemini API ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("WARNING: GEMINI_API_KEY not found in .env. Gemini API features will not be available.")

# --- Helper Functions ---

def clean_text(text):
    """Removes common email/text artifacts and cleans text for NLP."""
    text = re.sub(r'Subject:.*', '', text, flags=re.DOTALL)
    text = re.sub(r'From:.*', '', text, flags=re.DOTALL)
    text = re.sub(r'To:.*', '', text, flags=re.DOTALL)
    text = re.sub(r'Date:.*', '', text, flags=re.DOTALL)
    text = re.sub(r'Content-Type:.*', '', text, flags=re.DOTALL)
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def decode_email_header(header):
    """Decodes email headers which can be encoded in various ways."""
    decoded_parts = decode_header(header)
    decoded_string = ""
    for part, charset in decoded_parts:
        if isinstance(part, bytes):
            try:
                decoded_string += part.decode(charset if charset else 'utf-8')
            except (UnicodeDecodeError, LookupError):
                decoded_string += part.decode('latin-1', errors='ignore')
        else:
            decoded_string += part
    return decoded_string

def get_email_body_from_gmail_api_payload(payload):
    """Extracts the plain text body from a Gmail API message payload."""
    if 'parts' in payload:
        for part in payload['parts']:
            mime_type = part.get('mimeType')
            if mime_type == 'text/plain':
                if 'body' in part and 'data' in part['body']:
                    return base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
            if 'parts' in part:
                body = get_email_body_from_gmail_api_payload(part)
                if body: return body
    elif 'body' in payload and 'data' in payload['body']:
        return base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='ignore')
    return ""

def is_marketing_email(email_body, subject, sender_email):
    """Checks if an email is likely a marketing email."""
    body_lower = email_body.lower()
    subject_lower = subject.lower()
    sender_email_lower = sender_email.lower()

    if "unsubscribe" in body_lower: return True
    marketing_keywords = ["promo", "newsletter", "discount", "offer", "sale", "webinar", "event", "free trial", "coupon", "exclusive"]
    if any(keyword in subject_lower for keyword in marketing_keywords): return True
    if any(keyword in body_lower for keyword in marketing_keywords): return True
    marketing_sender_patterns = ["noreply", "info@", "support@", "marketing@", "updates@", "notifications@"]
    if any(pattern in sender_email_lower for pattern in marketing_sender_patterns): return True
    return False

def extract_entities(text):
    """Extracts named entities (people, organizations) using spaCy."""
    doc = nlp(text)
    people = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    return people, orgs

def extract_keywords_for_themes(text, num_keywords=10):
    """
    Extracts common keywords (potential themes) from text,
    filtering by stopwords and Part-of-Speech.
    """
    doc = nlp(text.lower())
    filtered_words = []
    for token in doc:
        # Filter out stopwords, punctuation, numbers, and non-alphabetic tokens
        if token.is_alpha and not token.is_stop and token.text not in all_stopwords:
            # Prioritize nouns and adjectives for thematic keywords
            if token.pos_ in ["NOUN", "PROPN", "ADJ"]: # NOUN: noun, PROPN: proper noun, ADJ: adjective
                filtered_words.append(token.text)
    
    word_counts = Counter(filtered_words)
    return [word for word, count in word_counts.most_common(num_keywords)]

def get_simplified_thread_topic(subject, thread_emails_bodies):
    """
    Attempts to extract a simplified, thematic topic from an email thread subject and bodies.
    Prioritizes nouns/adjectives from the subject.
    """
    # Try to get a topic from the subject first
    doc = nlp(subject.lower())
    subject_keywords = []
    for token in doc:
        if token.is_alpha and not token.is_stop and token.text not in all_stopwords:
            if token.pos_ in ["NOUN", "PROPN", "ADJ"]:
                subject_keywords.append(token.text)
    
    if subject_keywords:
        # Return the most common non-stopword noun/adjective from the subject
        return Counter(subject_keywords).most_common(1)[0][0].capitalize()

    # If subject doesn't yield good keywords, try the first email body
    if thread_emails_bodies:
        first_body_doc = nlp(thread_emails_bodies[0].lower())
        body_keywords = []
        for token in first_body_doc:
            if token.is_alpha and not token.is_stop and token.text not in all_stopwords:
                if token.pos_ in ["NOUN", "PROPN", "ADJ"]:
                    body_keywords.append(token.text)
        if body_keywords:
            return Counter(body_keywords).most_common(1)[0][0].capitalize()

    # Fallback to truncated subject if no meaningful topic found
    return (subject[:50] + '...') if len(subject) > 50 else subject


def parse_email_date(date_string):
    """Parses an email date string into a datetime object."""
    try:
        return parsedate_to_datetime(date_string)
    except Exception as e:
        print(f"Warning: Could not parse date string '{date_string}': {e}")
        return None

# --- Authentication Function ---
def authenticate_gmail():
    """Shows basic usage of the Gmail API.
    Loads credentials from token.json or initiates OAuth flow.
    """
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
    return creds

# --- Fetch and Process Calendar Events ---
def fetch_and_process_calendar_events(service, time_min, time_max):
    """
    Fetches calendar events within a specified time range and processes them.
    """
    processed_events = []
    page_token = None
    total_events_fetched = 0

    time_min_str = time_min.isoformat(timespec='seconds').replace('+00:00', 'Z')
    time_max_str = time_max.isoformat(timespec='seconds').replace('+00:00', 'Z')

    print(f"\nDEBUG: Calendar API timeMin: {time_min_str}")
    print(f"DEBUG: Calendar API timeMax: {time_max_str}")
    print(f"\nStarting calendar event fetch from {time_min_str} to {time_max_str}...")

    while True:
        try:
            events_result = service.events().list(
                calendarId='primary',
                timeMin=time_min_str,
                timeMax=time_max_str,
                maxResults=100,
                singleEvents=True,
                orderBy='startTime',
                pageToken=page_token
            ).execute()
        except HttpError as e:
            print(f"Error fetching calendar events: {e}. Skipping calendar data collection.")
            return []

        events = events_result.get('items', [])
        if not events:
            print("No more calendar events found.")
            break

        print(f"Found {len(events)} events on this page. Processing...")
        for event in events:
            total_events_fetched += 1
            event_data = {
                'id': event.get('id'),
                'summary': event.get('summary', 'No Title'),
                'description': clean_text(event.get('description', '')),
                'start_time': event['start'].get('dateTime', event['start'].get('date')),
                'end_time': event['end'].get('dateTime', event['end'].get('date')),
                'location': event.get('location', ''),
                'organizer_email': event['organizer'].get('email', ''),
                'organizer_name': event['organizer'].get('displayName', event['organizer'].get('email', '')),
                'attendees': [
                    {'email': att.get('email'), 'name': att.get('displayName', att.get('email'))}
                    for att in event.get('attendees', []) if att.get('email')
                ],
                'status': event.get('status')
            }
            processed_events.append(event_data)
            time.sleep(0.05)

        page_token = events_result.get('nextPageToken')
        if not page_token:
            break

    print(f"Finished fetching {total_events_fetched} calendar events.")
    return processed_events

# --- Generate Visualizations ---
def generate_visualizations(emails_data, calendar_events_data, timestamp_str):
    """
    Generates and saves bar charts for top senders and most active threads.
    Returns a dictionary of chart filenames with their Content-IDs for embedding,
    and the top_threads data for executive summary.
    """
    print("\nGenerating visualizations...")
    chart_files = {}

    # --- 1. Top Senders (from emails) ---
    sender_counts = Counter()
    for email in emails_data:
        if email['from_name'] and email['from_email'] and not any(pattern in email['from_email'].lower() for pattern in ["noreply", "info@", "support@", "marketing@"]):
            sender_counts[email['from_name']] += 1
    
    top_senders = sender_counts.most_common(10)
    if top_senders:
        senders = [s[0] for s in top_senders]
        counts = [s[1] for s in top_senders]

        plt.figure(figsize=(10, 6))
        plt.barh(senders, counts, color='#3498db') # Consistent color
        plt.xlabel('Number of Emails')
        plt.ylabel('Sender')
        plt.title('Top 10 Email Senders (Excluding Marketing)', color='#2c3e50')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        sender_chart_filename = f"top_senders_chart_{timestamp_str}.png"
        plt.savefig(sender_chart_filename)
        plt.close()
        chart_files['top_senders'] = {'path': sender_chart_filename, 'cid': make_msgid()[1:-1]}
        print(f"Saved Top Senders chart to {sender_chart_filename}")
    else:
        print("No top senders data to visualize.")

    # --- 2. Most Active Threads (from emails) ---
    thread_counts = Counter()
    thread_subjects = {} # This will be returned
    thread_bodies = defaultdict(list) # To store bodies for topic extraction
    thread_id_to_simplified_topic_map = {} # New map for simplified topics

    for email in emails_data:
        thread_id = email['threadId']
        thread_counts[thread_id] += 1
        if thread_id not in thread_subjects:
            thread_subjects[thread_id] = email['subject']
        thread_bodies[thread_id].append(email['body']) # Store body

    top_threads = thread_counts.most_common(10)
    if top_threads:
        thread_labels = []
        thread_message_counts = []
        for thread_id, count in top_threads:
            subject = thread_subjects.get(thread_id, "Unknown Subject")
            bodies = thread_bodies.get(thread_id, [])
            # Get a simplified topic for the thread
            simplified_topic = get_simplified_thread_topic(subject, bodies)
            thread_id_to_simplified_topic_map[thread_id] = simplified_topic # Store simplified topic
            thread_labels.append(f"{simplified_topic} ({count} messages)") # Updated label
            thread_message_counts.append(count)

        plt.figure(figsize=(10, 6))
        plt.barh(thread_labels, thread_message_counts, color='#e74c3c') # Consistent color
        plt.xlabel('Number of Messages')
        plt.ylabel('Email Thread Topic') # Updated label
        plt.title('Top 10 Most Active Email Threads', color='#2c3e50')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        thread_chart_filename = f"top_threads_chart_{timestamp_str}.png"
        plt.savefig(thread_chart_filename)
        plt.close()
        chart_files['top_threads'] = {'path': thread_chart_filename, 'cid': make_msgid()[1:-1]}
        print(f"Saved Most Active Threads chart to {thread_chart_filename}")
    else:
        print("No active thread data to visualize.")
    
    return chart_files, top_threads, thread_subjects, thread_id_to_simplified_topic_map # Return new map

# --- Analyze Email Interactions (for prioritization) ---
def analyze_email_interactions(emails_data, user_email):
    """
    Analyzes email data for top interacted contacts, average response times,
    and identifies emails awaiting a response.
    """
    interacted_counts = defaultdict(int)
    response_times_per_sender = defaultdict(list)
    emails_awaiting_response = []
    
    threads = defaultdict(list)
    for email in emails_data:
        threads[email['threadId']].append(email)

    # New: Map names to their most frequent email addresses
    name_email_associations = defaultdict(Counter) # Stores {name: {email: count}}

    for thread_id, email_list in threads.items():
        sorted_emails = sorted(email_list, key=lambda x: parse_email_date(x['date']) if parse_email_date(x['date']) else datetime.datetime.min)

        last_incoming_email_info = None # (sender_email, date_obj)
        user_replied_in_thread = False

        for email in sorted_emails:
            email_date_obj = parse_email_date(email['date'])
            if not email_date_obj:
                continue

            normalized_from_email = email['from_email'].lower()
            normalized_to_recipients = [r.lower() for r in email['to_recipients']]
            normalized_cc_recipients = [r.lower() for r in email['cc_recipients']]

            # Populate name_email_associations
            if email['from_name'] and email['from_email']:
                name_email_associations[email['from_name']][email['from_email'].lower()] += 1
            
            for recipient_full in email['to_recipients'] + email['cc_recipients']:
                # Extract name and email from "Name <email>" format
                match = re.match(r'^(.*?)\s*<([^>]+)>', recipient_full)
                if match:
                    name, email_addr = match.group(1).strip(), match.group(2)
                    if name: name_email_associations[name][email_addr.lower()] += 1
                elif '@' in recipient_full: # If it's just an email, use email as name for association
                    name_email_associations[recipient_full][recipient_full.lower()] += 1

            is_sent_by_user = (normalized_from_email == user_email.lower())
            is_received_by_user = (user_email.lower() in normalized_to_recipients or user_email.lower() in normalized_cc_recipients)

            if is_sent_by_user:
                for recipient in normalized_to_recipients + normalized_cc_recipients:
                    if recipient != user_email.lower():
                        interacted_counts[recipient] += 1
                user_replied_in_thread = True # User sent a message in this thread
            elif is_received_by_user:
                interacted_counts[normalized_from_email] += 1
            
            # Logic for response time calculation
            if not is_sent_by_user and is_received_by_user:
                last_incoming_email_info = (email['from_email'], email_date_obj, email['subject'])
            elif is_sent_by_user and last_incoming_email_info:
                original_sender_email, incoming_date_obj, _ = last_incoming_email_info
                
                # Ensure the response is to the last incoming email in the thread
                # and that the response time is positive
                if email_date_obj > incoming_date_obj:
                    response_time = email_date_obj - incoming_date_obj
                    response_times_per_sender[original_sender_email].append(response_time)
                
                last_incoming_email_info = None # Reset after a response is found
        
        # After processing all emails in a thread, check for awaiting response
        if last_incoming_email_info and not user_replied_in_thread:
            # Only add if the incoming email is recent enough but not too recent (e.g., older than 1 hour, younger than 14 days)
            time_since_incoming = datetime.datetime.now(datetime.timezone.utc) - last_incoming_email_info[1]
            if datetime.timedelta(hours=1) < time_since_incoming < datetime.timedelta(days=14):
                emails_awaiting_response.append({
                    'subject': last_incoming_email_info[2],
                    'sender': last_incoming_email_info[0],
                    'date': last_incoming_email_info[1].strftime('%Y-%m-%d %H:%M')
                })


    # Finalize name_to_email_map by picking the most common email for each name
    final_name_to_email_map = {}
    for name, email_counts in name_email_associations.items():
        if email_counts:
            final_name_to_email_map[name] = email_counts.most_common(1)[0][0]
        else:
            final_name_to_email_map[name] = name # Fallback to name itself if no email found

    avg_response_times = {}
    for sender, times in response_times_per_sender.items():
        if times:
            total_seconds = sum(t.total_seconds() for t in times)
            avg_seconds = total_seconds / len(times)
            
            if avg_seconds < 60:
                avg_response_times[sender] = f"{int(avg_seconds)} seconds"
            elif avg_seconds < 3600:
                avg_response_times[sender] = f"{int(avg_seconds / 60)} minutes"
            elif avg_seconds < 86400:
                avg_response_times[sender] = f"{int(avg_seconds / 3600)} hours"
            else:
                avg_response_times[sender] = f"{int(avg_seconds / 86400)} days"
    
    top_interacted_contacts = Counter(interacted_counts).most_common(10)

    key_organizations = Counter()
    combined_text_for_themes = ""
    for email in emails_data:
        combined_text_for_themes += email['body'] + " "
        people_in_body, orgs_in_body = extract_entities(email['body'])
        for org in orgs_in_body: key_organizations[org] += 1

    return top_interacted_contacts, avg_response_times, key_organizations, combined_text_for_themes, final_name_to_email_map, emails_awaiting_response

def get_consolidated_contacts_summary(key_people_combined, top_interacted_contacts, avg_response_times, name_to_email_map, user_email):
    """
    Consolidates key people/mentions with email interaction counts and average response times.
    Returns a list of dictionaries, sorted by total interactions/mentions, limited to top 10,
    and excluding the user's own email.
    """
    consolidated_data = defaultdict(lambda: {
        'display_name': 'Unknown',
        'interactions': 0, # Mentions/interactions from calendar/email bodies
        'emails_exchanged': 0, # Emails sent/received
        'avg_response_time': 'N/A',
        'primary_email': None # To store the email if available, for filtering
    })

    user_email_lower = user_email.lower()

    # Step 1: Populate with data from email interactions (emails_exchanged, avg_response_time)
    # Use email as the primary key for initial consolidation
    for email_address, emails_count in top_interacted_contacts:
        normalized_email = email_address.lower()
        if normalized_email == user_email_lower: # Exclude user's own email early
            continue

        consolidated_data[normalized_email]['emails_exchanged'] = emails_count
        consolidated_data[normalized_email]['primary_email'] = normalized_email
        if normalized_email in avg_response_times:
            consolidated_data[normalized_email]['avg_response_time'] = avg_response_times[normalized_email]
        
        # Try to find a display name for this email using the name_to_email_map
        found_name = None
        for name, mapped_email in name_to_email_map.items():
            if mapped_email and mapped_email.lower() == normalized_email:
                found_name = name
                break
        if found_name:
            consolidated_data[normalized_email]['display_name'] = found_name
        else:
            consolidated_data[normalized_email]['display_name'] = normalized_email # Default to email if no name found

    # Step 2: Add interactions/mentions from key_people_combined
    # Iterate through key_people_combined (which has names as keys)
    for person_name, interactions_count in key_people_combined.items():
        # Check if this person_name or its mapped email is the user's email
        if person_name.lower() == user_email_lower or (name_to_email_map.get(person_name, '').lower() == user_email_lower):
            continue # Exclude user's own interactions

        # Try to find the email for this person_name using the name_to_email_map
        person_email = name_to_email_map.get(person_name)
        
        if person_email:
            normalized_email = person_email.lower()
            # If this email is already in our consolidated data, update its interactions
            if normalized_email in consolidated_data:
                consolidated_data[normalized_email]['interactions'] += interactions_count
                # Ensure display name is set if it was previously just the email
                if consolidated_data[normalized_email]['display_name'] == normalized_email:
                    consolidated_data[normalized_email]['display_name'] = person_name
            else:
                # If this email is new (person mentioned but no direct email interaction yet)
                consolidated_data[normalized_email]['display_name'] = person_name
                consolidated_data[normalized_email]['interactions'] = interactions_count
                consolidated_data[normalized_email]['primary_email'] = normalized_email
                # avg_response_time and emails_exchanged remain 0/N/A
        else:
            # Case: person_name is a name, but no email mapping found (e.g., mentioned in calendar only)
            # Or person_name is an email not in top_interacted_contacts
            # We need to add it as a separate entry if it's not already covered by an email key
            if person_name not in consolidated_data: # Check if this name is already a key (e.g., if it was a raw email)
                consolidated_data[person_name]['display_name'] = person_name
                consolidated_data[person_name]['interactions'] += interactions_count
                # primary_email remains None, emails_exchanged 0, avg_response_time N/A

    # Step 3: Convert to list and sort
    final_list = []
    for key, data in consolidated_data.items():
        final_list.append({
            'contact': data['display_name'],
            'interactions': data['interactions'],
            'emails_exchanged': data['emails_exchanged'],
            'avg_response_time': data['avg_response_time']
        })
    
    # Sort by interactions in descending order and limit to top 10
    final_list.sort(key=lambda x: x['interactions'], reverse=True)
    return final_list[:10]


# --- Load Prompt from File ---
def load_prompt_from_file(file_path):
    """Loads the prompt text from a specified file."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"ERROR: Prompt file not found at {file_path}. Please check the path.")
        return None
    except Exception as e:
        print(f"ERROR: Could not read prompt file {file_path}: {e}")
        return None

# --- Generate Digest with Gemini API ---
async def generate_digest_with_gemini(combined_data_json, prompt_text):
    """
    Calls the Gemini API to generate a digest from the combined data.
    """
    if not GEMINI_API_KEY:
        print("Skipping Gemini API call: GEMINI_API_KEY not set.")
        return "Gemini API not configured."

    if not prompt_text:
        print("Skipping Gemini API call: Prompt text is empty.")
        return "No prompt provided for Gemini API."

    try:
        model = genai.GenerativeModel('gemini-1.5-flash') # Using gemini-1.5-flash for efficiency
        
        # Convert combined_data_json to a string for the prompt
        data_string = json.dumps(combined_data_json, indent=2)

        full_prompt = f"{prompt_text}\n\nHere is the data:\n{data_string}"
        
        print("\nDEBUG: Sending request to Gemini API...")
        response = await model.generate_content_async(full_prompt)
        
        if response.candidates:
            digest = response.candidates[0].content.parts[0].text
            print("\nDEBUG: Gemini API response received.")
            return digest
        else:
            print("DEBUG: Gemini API response had no candidates or content.")
            return "Could not generate digest."
    except Exception as e:
        print(f"ERROR: Failed to call Gemini API: {e}")
        return f"Error generating digest: {e}"


# --- Format Brief as HTML ---
def format_brief_as_html(key_people, key_organizations, themes, chart_files, consolidated_contacts_summary, time_window_days, start_date, end_date, all_processed_emails_data, logo_cid, thread_id_to_simplified_topic_map, llm_digest, emails_awaiting_response, upcoming_meetings):
    """
    Formats the analysis results into an HTML string for email.
    Args:
        key_people (Counter): Top people (from simple aggregation).
        key_organizations (Counter): Top organizations/projects.
        themes (list): Top themes/keywords.
        chart_files (dict): Dictionary of chart filenames and their CIDs.
        consolidated_contacts_summary (list): List of dicts for consolidated contact info.
        time_window_days (int): Number of days for the brief's time window.
        start_date (datetime): Start date of the brief's period.
        end_date (datetime): End date of the brief's period.
        all_processed_emails_data (list): All processed email data to look up thread subjects.
        logo_cid (str): Content-ID for the embedded logo image.
        thread_id_to_simplified_topic_map (dict): Dictionary of thread IDs to their simplified topics.
        llm_digest (str): The digest generated by the LLM.
        emails_awaiting_response (list): List of emails awaiting user response.
        upcoming_meetings (list): List of important upcoming meetings.
    Returns:
        str: HTML content for the email body.
    """
    # Format date range for header
    date_range_str = f"{start_date.strftime('%B %d')} - {end_date.strftime('%B %d, %Y')}"

    # --- Executive Summary / Highlights ---
    highlights = []
    if key_people:
        most_engaged_person = key_people.most_common(1)[0][0]
        highlights.append(f"<strong>{most_engaged_person}</strong> remains your most engaged contact this week.")
    
    # Get top threads for executive summary
    thread_counts_for_summary = Counter()
    for email in all_processed_emails_data:
        thread_id = email['threadId']
        thread_counts_for_summary[thread_id] += 1
    
    top_threads_for_summary_list = thread_counts_for_summary.most_common(1)
    if top_threads_for_summary_list:
        most_active_thread_id, _ = top_threads_for_summary_list[0]
        # Use the passed thread_id_to_simplified_topic_map for the highlight
        most_active_thread_topic = thread_id_to_simplified_topic_map.get(most_active_thread_id, "an active discussion")
        highlights.append(f"The <strong>'{most_active_thread_topic}'</strong> thread saw the most activity.")
    
    if consolidated_contacts_summary:
        # Find the fastest response from the consolidated summary
        fastest_response_contact = None
        min_seconds = float('inf')

        for contact_data in consolidated_contacts_summary:
            time_str = contact_data['avg_response_time']
            if time_str != 'N/A':
                seconds = 0
                if 'seconds' in time_str: seconds = int(time_str.split(' ')[0])
                elif 'minutes' in time_str: seconds = int(time_str.split(' ')[0]) * 60
                elif 'hours' in time_str: seconds = int(time_str.split(' ')[0]) * 3600
                elif 'days' in time_str: seconds = int(time_str.split(' ')[0]) * 86400
                
                if seconds < min_seconds:
                    min_seconds = seconds
                    fastest_response_contact = contact_data['contact']
        
        if fastest_response_contact:
            # Find the actual avg_response_time string for the fastest contact
            for contact_data in consolidated_contacts_summary:
                if contact_data['contact'] == fastest_response_contact:
                    highlights.append(f"Your average response time to <strong>{fastest_response_contact}</strong> was {contact_data['avg_response_time']}.")
                    break


    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ManagerFM Weekly Brief</title>
        <style>
            body {{ font-family: 'Inter', Arial, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 0; background-color: #f4f4f4; }}
            .container {{ max-width: 600px; margin: 20px auto; padding: 20px; border: 1px solid #ddd; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; padding-bottom: 20px; border-bottom: 1px solid #eee; }}
            .header img {{ max-width: 150px; height: auto; }}
            .header h1 {{ font-size: 24px; color: #333; margin: 10px 0; }}
            .header p {{ color: #555; font-size: 14px; }}
            .section {{ margin-bottom: 25px; padding-bottom: 15px; border-bottom: 1px solid #eee; }}
            .section:last-child {{ border-bottom: none; margin-bottom: 0; padding-bottom: 0; }}
            .section h2 {{ font-size: 20px; color: #2c3e50; margin-bottom: 10px; padding-bottom: 5px; display: flex; align-items: center; }}
            .section h2 .icon {{ margin-right: 8px; font-size: 1.2em; color: #3498db; }} /* Icon styling */
            ul {{ list-style-type: none; padding: 0; margin: 0; }}
            li {{ margin-bottom: 8px; padding-left: 15px; position: relative; }}
            li::before {{ content: '•'; color: #3498db; position: absolute; left: 0; }} /* Custom bullet */
            .list-item strong {{ color: #34495e; }}
            .chart-container {{ text-align: center; margin-top: 20px; }}
            .chart-img {{ max-width: 100%; height: auto; display: block; margin: 0 auto; border: 1px solid #ddd; border-radius: 4px; }} /* Added border-radius */
            .chart-caption {{ font-size: 12px; color: #777; margin-top: 5px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
            th {{ background-color: #f2f2f2; color: #333; }}
            .footer {{ text-align: center; margin-top: 30px; font-size: 12px; color: #777; border-top: 1px solid #eee; padding-top: 20px; }}
            .actionable-insight {{ font-style: italic; color: #555; margin-top: 10px; font-size: 0.95em; }}

            /* Mobile Responsiveness */
            @media only screen and (max-width: 600px) {{
                .container {{ width: 95%; border-radius: 0; box-shadow: none; padding: 15px; }}
                .header h1 {{ font-size: 20px; }}
                .section h2 {{ font-size: 18px; }}
                th, td {{ padding: 8px; font-size: 14px; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <img src="cid:{logo_cid}" alt="ManagerFM Logo">
                <h1>ManagerFM Weekly Brief</h1>
                <p>Your consolidated insights for {date_range_str}</p>
            </div>

            <div class="section">
                <h2><span class="icon">✨</span> This Week's Highlights</h2>
                <ul>
    """
    if highlights:
        for highlight in highlights:
            html_content += f"<li class='list-item'>{highlight}</li>"
    else:
        html_content += "<li class='list-item'>No significant highlights identified this week.</li>"
    html_content += """
                </ul>
            </div>

            <div class="section">
                <h2><span class="icon">📝</span> AI-Generated Digest</h2>
                <p>Here's a summary of your week's communications:</p>
                <div style="background-color: #f9f9f9; border: 1px solid #eee; padding: 15px; border-radius: 5px;">
                    <p style="white-space: pre-wrap; margin: 0; font-size: 14px;">{llm_digest}</p>
                </div>
            </div>

            <div class="section">
                <h2><span class="icon">👥</span> Key Contacts & Response Times</h2>
                <p>Your most engaged contacts and how quickly you respond:</p>
                <table>
                    <thead>
                        <tr>
                            <th>Contact</th>
                            <th>Interactions/Mentions</th>
                            <th>Emails Exchanged</th>
                            <th>Avg. Response Time</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    if consolidated_contacts_summary:
        for contact_data in consolidated_contacts_summary:
            html_content += f"""
                        <tr>
                            <td>{contact_data['contact']}</td>
                            <td>{contact_data['interactions']}</td>
                            <td>{contact_data['emails_exchanged']}</td>
                            <td>{contact_data['avg_response_time']}</td>
                        </tr>
            """
    else:
        html_content += "<tr><td colspan='4'>No key contacts or response time data available.</td></tr>"
    html_content += """
                    </tbody>
                </table>
                <p class="actionable-insight"><em>💡 Actionable Insight: Prioritize responses to key contacts and those with longer average response times.</em></p>
            </div>
    """

    # Add charts if available, reordered
    if 'top_senders' in chart_files:
        html_content += f"""
            <div class="section">
                <h2><span class="icon">📈</span> Your Top Email Senders</h2>
                <p>Visualizing your most frequent email senders (excluding marketing emails):</p>
                <div class="chart-container">
                    <img src="cid:{chart_files['top_senders']['cid']}" class="chart-img" alt="Top Email Senders Chart">
                    <p class="chart-caption"><em>Data from your Google Mail.</em></p>
                </div>
            </div>
        """
    if 'top_threads' in chart_files:
        html_content += f"""
            <div class="section">
                <h2><span class="icon">💬</span> Most Active Email Threads</h2>
                <p>The conversations that generated the most messages:</p>
                <div class="chart-container">
                    <img src="cid:{chart_files['top_threads']['cid']}" class="chart-img" alt="Most Active Email Threads Chart">
                    <p class="chart-caption"><em>Data from your Google Mail.</em></p>
                </div>
            </div>
        """

    # New section for Emails Awaiting Response
    html_content += """
            <div class="section">
                <h2><span class="icon">📧</span> Emails Awaiting Your Response</h2>
                <p>These emails might be waiting for your reply:</p>
                <ul>
    """
    if emails_awaiting_response:
        for email in emails_awaiting_response:
            html_content += f"""
                    <li><strong>Subject:</strong> {email['subject']}<br>
                        <strong>From:</strong> {email['sender']}<br>
                        <strong>Date:</strong> {email['date']}</li>
            """
    else:
        html_content += "<li class='list-item'>No emails currently awaiting your response.</li>"
    html_content += """
                </ul>
            </div>
    """

    # New section for Upcoming Meetings
    html_content += """
            <div class="section">
                <h2><span class="icon">📅</span> Upcoming Meetings</h2>
                <p>Important meetings on your calendar for the next 7 days:</p>
                <ul>
    """
    if upcoming_meetings:
        for meeting in upcoming_meetings:
            attendees_str = ", ".join([att['name'] for att in meeting['attendees']])
            html_content += f"""
                    <li><strong>{meeting['summary']}</strong><br>
                        <strong>Time:</strong> {meeting['start_time']}<br>
                        <strong>Location:</strong> {meeting['location'] if meeting['location'] else 'N/A'}<br>
                        <strong>Attendees:</strong> {attendees_str if attendees_str else 'N/A'}</li>
            """
    else:
        html_content += "<li class='list-item'>No important upcoming meetings found.</li>"
    html_content += """
                </ul>
            </div>
    """


    html_content += """
            <div class="section">
                <h2><span class="icon">🏢</span> Key Organizations/Projects</h2>
                <p>Mentions across your communications:</p>
                <ul>
    """
    if key_organizations:
        for org, count in key_organizations.most_common(10):
            html_content += f"<li class='list-item'><strong>{org}</strong> ({count} mentions)</li>"
    else:
        html_content += "<li class='list-item'>No key organizations/projects identified.</li>"
    html_content += """
                </ul>
            </div>

            <div class="section">
                <h2><span class="icon">🏷️</span> Top Themes/Keywords</h2>
                <p>Recurring topics from your emails and calendar events:</p>
                <p style="font-size: 14px; line-height: 1.8;">
    """
    if themes:
        html_content += ", ".join(themes)
    else:
        html_content += "No top themes/keywords identified."
    html_content += """
                </p>
            </div>

            <div class="footer">
                <p>Generated by ManagerFM. Data from your Google Mail and Calendar.</p>
                <p>&copy; 2025 ManagerFM. All rights reserved.</p>
                <p><a href="#">Unsubscribe</a> | <a href="#">Privacy Policy</a></p>
            </div>
        </div>
    </body>
    </html>
    """
    print(f"DEBUG: Length of generated HTML content: {len(html_content)} characters.") # Debug print
    return html_content

# --- New Function: Send Email Brief ---
def send_email_brief(sender_email, sender_password, recipient_email, subject, html_content, chart_files, logo_path, logo_cid):
    """
    Sends an HTML email with embedded charts and logo.
    """
    msg = MIMEMultipart('mixed')
    msg['From'] = f"ManagerFM <{sender_email}>"
    msg['To'] = recipient_email
    msg['Subject'] = subject

    # Create a related part for HTML and inline images
    msg_related = MIMEMultipart('related')

    # Attach HTML content to the related part
    msg_related.attach(MIMEText(html_content, 'html')) # Attach HTML first within 'related'

    # Attach logo image using the provided logo_cid
    try:
        with open(logo_path, 'rb') as img_file:
            img = MIMEImage(img_file.read())
            img.add_header('Content-ID', f"<{logo_cid}>")
            msg_related.attach(img)
        print(f"Attached logo file: {logo_path}")
    except FileNotFoundError:
        print(f"Warning: Logo file {logo_path} not found, cannot embed. Logo will not appear.")
    except Exception as e:
        print(f"Error attaching logo {logo_path}: {e}. Logo will not appear.")

    # Attach chart images to the related part
    for chart_name, chart_info in chart_files.items():
        try:
            with open(chart_info['path'], 'rb') as img_file:
                img = MIMEImage(img_file.read())
                img.add_header('Content-ID', f"<{chart_info['cid']}>")
                msg_related.attach(img)
            os.remove(chart_info['path'])
            print(f"Cleaned up chart file: {chart_info['path']}")
        except FileNotFoundError:
            print(f"Warning: Chart file {chart_info['path']} not found, cannot embed.")
        except Exception as e:
            print(f"Error attaching chart {chart_info['path']}: {e}")
    
    # Create an alternative part for plain text and the related HTML part
    msg_alternative = MIMEMultipart('alternative')
    
    # Create plain text version
    plain_text_content = re.sub(r'<[^>]+>', '', html_content) # Strip HTML tags
    msg_alternative.attach(MIMEText(plain_text_content, 'plain'))
    
    # Attach the related part (which contains HTML and images) to the alternative part
    msg_alternative.attach(msg_related)

    # Attach the alternative part to the main message
    msg.attach(msg_alternative)

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
        print(f"\nEmail brief successfully sent to {recipient_email}!")
    except smtplib.SMTPAuthenticationError:
        print("\nERROR: SMTP Authentication Failed. Please check your SENDER_EMAIL_ADDRESS and SENDER_EMAIL_PASSWORD (ensure it's a Gmail App Password if using Gmail).")
    except Exception as e:
        print(f"\nERROR: Failed to send email: {e}")
    
    return logo_cid 

# --- Main Orchestration Function ---

async def generate_manager_briefing_data():
    """
    Orchestrates fetching and processing of email and calendar data for ManagerFM.
    Generates visualizations and sends an email brief.
    """
    # Check if essential environment variables are set
    if not all([SENDER_EMAIL_ADDRESS, SENDER_EMAIL_PASSWORD, RECIPIENT_EMAIL_ADDRESS]):
        print("ERROR: Email configuration missing. Please set SENDER_EMAIL_ADDRESS, SENDER_EMAIL_PASSWORD, and RECIPIENT_EMAIL_ADDRESS in your .env file.")
        return

    creds = authenticate_gmail()
    if not creds:
        print("Authentication failed. Exiting.")
        return

    try:
        gmail_service = build("gmail", "v1", credentials=creds)
        calendar_service = build("calendar", "v3", credentials=creds)
        if not calendar_service:
            print("\nDEBUG: Failed to build Google Calendar service. Exiting.")
            return

        all_processed_emails_data = []
        all_processed_calendar_data = []

        time_window_days = 14 # Set to 14 days for broader context
        end_time = datetime.datetime.now(datetime.timezone.utc)
        start_time = end_time - datetime.timedelta(days=time_window_days)

        date_filter_gmail = start_time.strftime("%Y/%m/%d")
        query_parts_gmail = [
            f"after:{date_filter_gmail}",
            '-"unsubscribe"', '-category:promotions', '-category:social',
            '-category:updates', '-category:forums',
            '-"promo"', '-"newsletter"', '-"discount"', '-"offer"',
            '-"sale"', '-"webinar"', '-"event"', '-"free trial"',
            '-"coupon"', '-"exclusive"',
            '-from:noreply@*', '-from:info@*', '-from:marketing@*',
            '-from:updates@*', '-from:notifications@*'
        ]
        gmail_api_query = " ".join(query_parts_gmail)
        print(f"Using Gmail API query for filtering: '{gmail_api_query}'")
        print(f"Fetching emails from {time_window_days} days ago.")

        next_page_token_email = None
        total_emails_processed = 0
        email_page_number = 0

        print("\nStarting email fetch and analysis...")
        while True:
            email_page_number += 1
            print(f"\nFetching page {email_page_number} of email IDs...")
            results_email = gmail_service.users().messages().list(
                userId="me", maxResults=50, pageToken=next_page_token_email, q=gmail_api_query
            ).execute()
            messages = results_email.get("messages", [])

            if not messages:
                print("No more email messages found or no messages matching the filter criteria.")
                break

            print(f"Found {len(messages)} email IDs on this page. Fetching full content...")
            for i, message in enumerate(messages):
                total_emails_processed += 1
                if total_emails_processed % 25 == 0:
                    print(f"Processed {total_emails_processed} emails in total...")

                msg_id = message["id"]
                thread_id = message["threadId"]
                
                try:
                    msg = gmail_service.users().messages().get(userId="me", id=msg_id, format="full").execute()
                    time.sleep(0.1)
                except HttpError as e:
                    print(f"Error fetching email ID {msg_id}: {e}. Skipping this message.")
                    continue
                except Exception as e:
                    print(f"An unexpected error occurred while fetching email ID {msg_id}: {e}. Skipping this message.")
                    continue

                headers = msg["payload"]["headers"]
                subject = ""
                sender_email = ""
                sender_name = ""
                to_recipients = []
                cc_recipients = []
                date_sent = ""
                has_attachments = 'attachmentId' in str(msg["payload"])

                for header in headers:
                    if header["name"] == "Subject":
                        subject = decode_email_header(header["value"])
                    elif header["name"] == "From":
                        sender_raw = header["value"]
                        sender_name_match = re.match(r'^(.*?)\s*<([^>]+)>', sender_raw)
                        if sender_name_match:
                            sender_name = decode_email_header(sender_name_match.group(1).strip())
                            sender_email = sender_name_match.group(2)
                        else:
                            sender_email = decode_email_header(sender_raw)
                            sender_name = sender_email
                    elif header["name"] == "To":
                        to_recipients = [decode_email_header(r.strip()) for r in header["value"].split(',')]
                    elif header["name"] == "Cc":
                        cc_recipients = [decode_email_header(r.strip()) for r in header["value"].split(',')]
                    elif header["name"] == "Date":
                        date_sent = decode_email_header(header["value"])

                body = get_email_body_from_gmail_api_payload(msg["payload"])
                cleaned_body = clean_text(body)

                if is_marketing_email(cleaned_body, subject, sender_email):
                    continue

                email_data = {
                    'id': msg_id,
                    'threadId': thread_id,
                    'subject': subject,
                    'from_name': sender_name,
                    'from_email': sender_email,
                    'to_recipients': to_recipients,
                    'cc_recipients': cc_recipients,
                    'date': date_sent,
                    'body': cleaned_body,
                    'has_attachments': has_attachments
                }
                all_processed_emails_data.append(email_data)

            next_page_token_email = results_email.get("nextPageToken")
            if not next_page_token_email:
                break

        print(f"\n--- Finished processing {total_emails_processed} total emails ---")
        print(f"Collected data for {len(all_processed_emails_data)} non-marketing emails.")

        all_processed_calendar_data = fetch_and_process_calendar_events(calendar_service, start_time, end_time)

        # --- Combine all data into a single structure (for optional saving or direct LLM input) ---
        combined_briefing_data = {
            "emails": all_processed_emails_data,
            "calendar_events": all_processed_calendar_data
        }

        # --- Perform detailed email interaction analysis ---
        top_interacted_contacts, avg_response_times, key_organizations_from_emails, combined_email_text, name_to_email_map, emails_awaiting_response = \
            analyze_email_interactions(all_processed_emails_data, SENDER_EMAIL_ADDRESS)

        # --- Filter upcoming meetings ---
        upcoming_meetings = []
        now = datetime.datetime.now(datetime.timezone.utc)
        # Look ahead for the next 7 days
        look_ahead_end_time = now + datetime.timedelta(days=7)

        for event in all_processed_calendar_data:
            try:
                event_start_str = event['start_time']
                # Handle both full datetime and date-only formats from calendar API
                if 'T' in event_start_str:
                    event_start_dt = datetime.datetime.fromisoformat(event_start_str)
                else:
                    event_start_dt = datetime.datetime.strptime(event_start_str, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc)
                
                # Ensure timezone awareness for comparison
                if event_start_dt.tzinfo is None:
                    event_start_dt = event_start_dt.replace(tzinfo=datetime.timezone.utc)

                # Check if event is in the future and within the look-ahead window
                if now < event_start_dt < look_ahead_end_time:
                    is_important = False
                    # Check if user is organizer
                    if event['organizer_email'].lower() == SENDER_EMAIL_ADDRESS.lower():
                        is_important = True
                    # Check if user is a confirmed attendee
                    for attendee in event['attendees']:
                        if attendee['email'].lower() == SENDER_EMAIL_ADDRESS.lower() and attendee.get('responseStatus') == 'accepted':
                            is_important = True
                            break
                    
                    if is_important:
                        upcoming_meetings.append({
                            'summary': event['summary'],
                            'start_time': event_start_dt.strftime('%Y-%m-%d %H:%M'),
                            'location': event['location'],
                            'attendees': event['attendees']
                        })
            except Exception as e:
                print(f"Warning: Could not process calendar event '{event.get('summary', 'N/A')}': {e}")
        
        # Sort upcoming meetings by start time
        upcoming_meetings.sort(key=lambda x: datetime.datetime.strptime(x['start_time'], '%Y-%m-%d %H:%M'))


        # --- Generate Visualizations (temporarily saves files for embedding) ---
        chart_files, top_threads_for_summary, thread_subjects_map, thread_id_to_simplified_topic_map = \
            generate_visualizations(all_processed_emails_data, all_processed_calendar_data, timestamp)

        # --- Aggregate key_people and key_organizations from both emails and calendar events ---
        key_people_combined = Counter()
        key_organizations_combined = Counter(key_organizations_from_emails) # Start with orgs from emails
        
        for email in all_processed_emails_data:
            if email['from_name'] and email['from_email'] and not any(pattern in email['from_email'].lower() for pattern in ["noreply", "info@", "support@", "marketing@"]):
                key_people_combined[email['from_name']] += 1
            people_in_body, orgs_in_body = extract_entities(email['body'])
            for person in people_in_body: key_people_combined[person] += 1
            # Organizations already added from email analysis

        combined_text_for_themes = combined_email_text # Start with email text
        for event in all_processed_calendar_data:
            if event['organizer_name'] and event['organizer_email']:
                 key_people_combined[event['organizer_name']] += 1
            for attendee in event['attendees']:
                if attendee['name']: key_people_combined[attendee['name']] += 1
            
            event_text = event['summary'] + " " + event['description']
            people_in_event, orgs_in_event = extract_entities(clean_text(event_text))
            for person in people_in_event: key_people_combined[person] += 1
            for org in orgs_in_event: key_organizations_combined[org] += 1
            combined_text_for_themes += " " + clean_text(event_text)

        themes = extract_keywords_for_themes(combined_text_for_themes, num_keywords=20)

        # --- Get Consolidated Contacts Summary ---
        consolidated_contacts_summary = get_consolidated_contacts_summary(key_people_combined, top_interacted_contacts, avg_response_times, name_to_email_map, SENDER_EMAIL_ADDRESS)

        # --- Generate LLM Digest ---
        llm_prompt = load_prompt_from_file(GEMINI_PROMPT_FILE_PATH)
        llm_digest = "Digest could not be generated."
        if llm_prompt:
            llm_digest = await generate_digest_with_gemini(combined_briefing_data, llm_prompt)
        
        # --- Attach logo and get its CID before formatting HTML ---
        logo_cid = make_msgid()[1:-1]
        try:
            with open(MANAGER_FM_LOGO_PATH, 'rb') as img_file:
                logo_img = MIMEImage(img_file.read())
                logo_img.add_header('Content-ID', f"<{logo_cid}>")
                # The logo image itself is attached in send_email_brief
            print(f"Prepared logo file for embedding: {MANAGER_FM_LOGO_PATH}")
        except FileNotFoundError:
            print(f"Warning: Logo file {MANAGER_FM_LOGO_PATH} not found, cannot embed. Using empty CID.")
            logo_cid = ""
        except Exception as e:
            print(f"Error preparing logo {MANAGER_FM_LOGO_PATH}: {e}. Using empty CID.")
            logo_cid = ""

        # --- Format and Send Email Brief ---
        email_subject = f"ManagerFM Weekly Brief - {datetime.date.today().strftime('%Y-%m-%d')}"
        html_email_body = format_brief_as_html(
            key_people_combined, key_organizations_combined, themes, chart_files,
            consolidated_contacts_summary, # Pass consolidated summary here
            time_window_days, start_time, end_time, all_processed_emails_data, logo_cid,
            thread_id_to_simplified_topic_map, # Pass thread_id_to_simplified_topic_map here
            llm_digest, # Pass the generated digest
            emails_awaiting_response, # Pass emails awaiting response
            upcoming_meetings # Pass upcoming meetings
        )
        send_email_brief(SENDER_EMAIL_ADDRESS, SENDER_EMAIL_PASSWORD, RECIPIENT_EMAIL_ADDRESS, email_subject, html_email_body, chart_files, logo_path=MANAGER_FM_LOGO_PATH, logo_cid=logo_cid)

        # --- Console Output (for debugging/verification) ---
        print("\n--- Consolidated Analysis Results (Console Preview) ---")
        print("\nAI-Generated Digest:")
        print(llm_digest)
        print("\nKey Contacts & Response Times:")
        for contact_data in consolidated_contacts_summary:
            print(f"- Contact: {contact_data['contact']}, Interactions: {contact_data['interactions']}, Emails Exchanged: {contact_data['emails_exchanged']}, Avg. Response Time: {contact_data['avg_response_time']}")
        
        print("\nEmails Awaiting Your Response:")
        if emails_awaiting_response:
            for email in emails_awaiting_response:
                print(f"- Subject: {email['subject']}, From: {email['sender']}, Date: {email['date']}")
        else:
            print("- None")

        print("\nUpcoming Meetings:")
        if upcoming_meetings:
            for meeting in upcoming_meetings:
                attendees_str = ", ".join([att['name'] for att in meeting['attendees']])
                print(f"- Summary: {meeting['summary']}, Time: {meeting['start_time']}, Location: {meeting['location']}, Attendees: {attendees_str}")
        else:
            print("- None")

        print("\nTop 10 Key Organizations/Projects (from Email & Calendar):")
        for org, count in key_organizations_combined.most_common(10):
            print(f"- {org} ({count} mentions)")
        print("\nTop 20 Themes/Keywords (from Email & Calendar):")
        for theme in themes:
            print(f"- {theme}")

    except HttpError as error:
        print(f"An API error occurred: {error}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(generate_manager_briefing_data())
