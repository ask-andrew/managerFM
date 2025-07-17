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
import matplotlib.pyplot as plt # Import for plotting

# You'll need to install these (if you haven't yet):
# pip install google-api-python-client google-auth-oauthlib google-auth-httplib2
# pip install nltk spacy matplotlib
# python -m spacy download en_core_web_sm
import nltk
# nltk.download('punkt') # Uncomment and run if you don't have it
# nltk.download('stopwords') # Uncomment and run if you don't have it
# nltk.download('punkt_tab') # Uncomment and run if you don't have it
from nltk.corpus import stopwords
from collections import Counter
import spacy

# --- Configuration ---
# IMPORTANT: If modifying these scopes, delete the file token.json and re-run to re-authenticate.
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.readonly" # Added Calendar scope
]

# The file token.json stores the user's access and refresh tokens, and is
# created automatically when the authorization flow completes for the first
# time.
TOKEN_FILE = "token.json"
# CREDENTIALS_FILE is downloaded from Google Cloud Console
CREDENTIALS_FILE = "credentials.json"

# Dynamically generate the output JSON filename with a timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_JSON_FILE = f"manager_briefing_data_{timestamp}.json" # Consolidated output file for all data

# --- NLP Setup ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    exit()

stop_words = set(stopwords.words('english'))

# --- Helper Functions (retained from previous versions) ---

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
            if 'parts' in part: # Recurse into nested parts
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
    """Extracts common keywords (potential themes) from text."""
    words = nltk.word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    word_counts = Counter(filtered_words)
    return [word for word, count in word_counts.most_common(num_keywords)]

# --- Authentication Function (Must be defined before generate_manager_briefing_data) ---
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
    Args:
        service: Authenticated Google Calendar API service object.
        time_min (datetime): Start datetime for fetching events.
        time_max (datetime): End datetime for fetching events.
    Returns:
        list: A list of dictionaries, each representing a structured calendar event.
    """
    processed_events = []
    page_token = None
    total_events_fetched = 0

    # Format timestamps for Google Calendar API to RFC 3339 format (YYYY-MM-DDTHH:MM:SSZ)
    # The datetime object from datetime.datetime.now(datetime.timezone.utc) already includes +00:00 offset.
    # We need to replace this with 'Z' for the API.
    time_min_str = time_min.isoformat(timespec='seconds').replace('+00:00', 'Z')
    time_max_str = time_max.isoformat(timespec='seconds').replace('+00:00', 'Z')

    # DEBUG: Print the exact time range being sent to the Calendar API
    print(f"\nDEBUG: Calendar API timeMin: {time_min_str}")
    print(f"DEBUG: Calendar API timeMax: {time_max_str}")

    print(f"\nStarting calendar event fetch from {time_min_str} to {time_max_str}...")

    while True:
        try:
            events_result = service.events().list(
                calendarId='primary', # 'primary' refers to the user's default calendar
                timeMin=time_min_str,
                timeMax=time_max_str,
                maxResults=100, # Fetch up to 100 events per page
                singleEvents=True, # Expand recurring events into individual instances
                orderBy='startTime',
                pageToken=page_token
            ).execute()
        except HttpError as e:
            print(f"Error fetching calendar events: {e}. Skipping calendar data collection.")
            return [] # Return empty list on error

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
                'description': clean_text(event.get('description', '')), # Clean description for NLP
                'start_time': event['start'].get('dateTime', event['start'].get('date')),
                'end_time': event['end'].get('dateTime', event['end'].get('date')),
                'location': event.get('location', ''),
                'organizer_email': event['organizer'].get('email', ''),
                'organizer_name': event['organizer'].get('displayName', event['organizer'].get('email', '')),
                'attendees': [
                    {'email': att.get('email'), 'name': att.get('displayName', att.get('email'))}
                    for att in event.get('attendees', []) if att.get('email')
                ],
                'status': event.get('status') # e.g., 'confirmed', 'cancelled'
            }
            processed_events.append(event_data)
            time.sleep(0.05) # Small delay for calendar API calls too

        page_token = events_result.get('nextPageToken')
        if not page_token:
            break

    print(f"Finished fetching {total_events_fetched} calendar events.")
    return processed_events

# --- New Function: Generate Visualizations ---

def generate_visualizations(emails_data, calendar_events_data, timestamp_str):
    """
    Generates and saves bar charts for top senders and most active threads.
    Args:
        emails_data (list): List of structured email data.
        calendar_events_data (list): List of structured calendar event data.
        timestamp_str (str): Timestamp string for naming output files.
    """
    print("\nGenerating visualizations...")

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
        plt.barh(senders, counts, color='skyblue')
        plt.xlabel('Number of Emails')
        plt.ylabel('Sender')
        plt.title('Top 10 Email Senders (Excluding Marketing)')
        plt.gca().invert_yaxis() # Put highest count at the top
        plt.tight_layout()
        sender_chart_filename = f"top_senders_chart_{timestamp_str}.png"
        plt.savefig(sender_chart_filename)
        plt.close() # Close the plot to free memory
        print(f"Saved Top Senders chart to {sender_chart_filename}")
    else:
        print("No top senders data to visualize.")

    # --- 2. Most Active Threads (from emails) ---
    thread_counts = Counter()
    thread_subjects = {} # To map threadId to a representative subject

    for email in emails_data:
        thread_id = email['threadId']
        thread_counts[thread_id] += 1
        if thread_id not in thread_subjects:
            thread_subjects[thread_id] = email['subject'] # Use the subject of the first email encountered

    top_threads = thread_counts.most_common(10)
    if top_threads:
        thread_labels = []
        thread_message_counts = []
        for thread_id, count in top_threads:
            subject = thread_subjects.get(thread_id, "Unknown Subject")
            # Truncate long subjects for better display
            display_subject = (subject[:40] + '...') if len(subject) > 40 else subject
            thread_labels.append(f"{display_subject} ({thread_id[:6]}...)") # Show truncated ID
            thread_message_counts.append(count)

        plt.figure(figsize=(10, 6))
        plt.barh(thread_labels, thread_message_counts, color='lightcoral')
        plt.xlabel('Number of Messages')
        plt.ylabel('Email Thread')
        plt.title('Top 10 Most Active Email Threads')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        thread_chart_filename = f"top_threads_chart_{timestamp_str}.png"
        plt.savefig(thread_chart_filename)
        plt.close()
        print(f"Saved Most Active Threads chart to {thread_chart_filename}")
    else:
        print("No active thread data to visualize.")

# --- Main Orchestration Function ---

def generate_manager_briefing_data():
    """
    Orchestrates fetching and processing of email and calendar data for ManagerFM.
    Saves combined data to a JSON file and generates visualizations.
    """
    creds = authenticate_gmail() # This now handles both Gmail and Calendar scopes
    if not creds:
        print("Authentication failed. Exiting.")
        return

    try:
        gmail_service = build("gmail", "v1", credentials=creds)
        
        # DEBUG: Confirm calendar service is built
        calendar_service = build("calendar", "v3", credentials=creds)
        if calendar_service:
            print("\nDEBUG: Google Calendar service successfully built.")
        else:
            print("\nDEBUG: Failed to build Google Calendar service.")
            return # Exit if calendar service not built

        all_processed_emails_data = []
        all_processed_calendar_data = []

        # --- Define Time Window (e.g., last 30 days for both emails and calendar) ---
        time_window_days = 30
        end_time = datetime.datetime.now(datetime.timezone.utc) # Current time in UTC
        start_time = end_time - datetime.timedelta(days=time_window_days)

        # --- Prepare Gmail API Query ---
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

        # --- Fetch and Process Emails ---
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
                    time.sleep(0.1) # Delay for Gmail API calls
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

        # --- Fetch and Process Calendar Events ---
        all_processed_calendar_data = fetch_and_process_calendar_events(calendar_service, start_time, end_time)

        # --- Combine all data into a single structure ---
        combined_briefing_data = {
            "emails": all_processed_emails_data,
            "calendar_events": all_processed_calendar_data
        }

        # --- Save combined data to JSON file ---
        try:
            with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
                json.dump(combined_briefing_data, f, ensure_ascii=False, indent=4)
            print(f"Successfully saved combined briefing data to {OUTPUT_JSON_FILE}")
        except Exception as e:
            print(f"Error saving data to JSON file: {e}")

        # --- Generate and Save Visualizations ---
        generate_visualizations(all_processed_emails_data, all_processed_calendar_data, timestamp)

        # --- Preliminary Analysis Results (for console output) ---
        key_people = Counter()
        key_organizations = Counter()
        
        # Aggregate from emails
        for email in all_processed_emails_data:
            if email['from_name'] and email['from_email'] and not any(pattern in email['from_email'].lower() for pattern in ["noreply", "info@", "support@", "marketing@"]):
                key_people[email['from_name']] += 1
            people_in_body, orgs_in_body = extract_entities(email['body'])
            for person in people_in_body: key_people[person] += 1
            for org in orgs_in_body: key_organizations[org] += 1

        # Aggregate from calendar events (attendees, organizer, summary/description entities)
        for event in all_processed_calendar_data:
            if event['organizer_name'] and event['organizer_email']:
                 key_people[event['organizer_name']] += 1
            for attendee in event['attendees']:
                if attendee['name']: key_people[attendee['name']] += 1
            
            event_text = event['summary'] + " " + event['description']
            people_in_event, orgs_in_event = extract_entities(clean_text(event_text))
            for person in people_in_event: key_people[person] += 1
            for org in orgs_in_event: key_organizations[org] += 1


        combined_text_for_themes = " ".join([e['body'] for e in all_processed_emails_data])
        combined_text_for_themes += " ".join([clean_text(e['summary'] + " " + e['description']) for e in all_processed_calendar_data])


        print("\n--- Consolidated Analysis Results ---")

        print("\nTop 10 Key People (from Email & Calendar):")
        for person, count in key_people.most_common(10):
            print(f"- {person} ({count} interactions/mentions)")

        print("\nTop 10 Key Organizations/Projects (from Email & Calendar):")
        for org, count in key_organizations.most_common(10):
            print(f"- {org} ({count} mentions)")

        print("\nTop 20 Themes/Keywords (from Email & Calendar):")
        themes = extract_keywords_for_themes(combined_text_for_themes, num_keywords=20)
        for theme in themes:
            print(f"- {theme}")

    except HttpError as error:
        print(f"An API error occurred: {error}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    generate_manager_briefing_data()
