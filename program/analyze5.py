# analyze4.py (Complete, Refactored & Ready to Run)

import os
import re
import json
import base64
import datetime
import logging
import asyncio
from collections import Counter, defaultdict
from email.utils import make_msgid, parsedate_to_datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import smtplib

# --- Third-party Libraries ---
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import nltk
import spacy
import google.generativeai as genai

# --- Google API Libraries ---
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# --- Load environment variables ---
load_dotenv()

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# --- Constants & Config ---
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.readonly" # Added calendar scope as per previous discussions
]

REQUIRED_ENV_VARS = [
    "SENDER_EMAIL_ADDRESS",
    "SENDER_EMAIL_PASSWORD",
    "RECIPIENT_EMAIL_ADDRESS",
    "MANAGER_FM_LOGO_PATH",
    "GEMINI_API_KEY",
    "GEMINI_PROMPT_FILE_PATH"
]

# --- Environment Validation ---
def validate_env_vars():
    missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

# --- NLP Initialization ---
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        logging.error("SpaCy model not found. Please run: python -m spacy download en_core_web_sm")
        exit(1)

# --- Gmail Authentication ---
def authenticate_gmail():
    creds = None
    token_path = "token.json"
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as token_file:
            token_file.write(creds.to_json())
    return creds # Corrected: Return credentials object, not the service build

# --- Fetch & Parse Emails ---
async def fetch_recent_messages(service, days=14, max_results=50): # Changed to 14 days
    date_after = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days))
    # Refined query to exclude more marketing/social emails
    query = f"after:{int(date_after.timestamp())} is:inbox -category:promotions -category:social -category:updates -category:forums -\"unsubscribe\" -\"promo\" -\"newsletter\" -\"discount\" -\"offer\" -\"sale\" -\"webinar\" -\"event\" -\"free trial\" -\"coupon\" -\"exclusive\" -from:noreply@* -from:info@* -from:marketing@* -from:updates@* -from:notifications@*"
    
    messages = []
    next_page_token = None
    while True:
        results = service.users().messages().list(userId='me', q=query, maxResults=max_results, pageToken=next_page_token).execute()
        messages.extend(results.get('messages', []))
        next_page_token = results.get('nextPageToken')
        if not next_page_token:
            break
    return messages

async def get_message_body(service, msg_id):
    msg = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
    
    # Function to recursively get plain text body from payload parts
    def get_plain_text_from_payload(payload):
        if 'parts' in payload:
            for part in payload['parts']:
                if part.get('mimeType') == 'text/plain':
                    data = part.get('body', {}).get('data')
                    if data:
                        return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
                elif 'parts' in part:
                    # Recursively check nested parts
                    nested_text = get_plain_text_from_payload(part)
                    if nested_text:
                        return nested_text
        elif payload.get('mimeType') == 'text/plain':
            data = payload.get('body', {}).get('data')
            if data:
                return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
        return None

    return get_plain_text_from_payload(msg.get('payload', {}))

# --- Fetch and Process Calendar Events ---
async def fetch_and_process_calendar_events(service, time_min, time_max):
    processed_events = []
    time_min_str = time_min.isoformat(timespec='seconds').replace('+00:00', 'Z')
    time_max_str = time_max.isoformat(timespec='seconds').replace('+00:00', 'Z')

    try:
        events_result = service.events().list(
            calendarId='primary',
            timeMin=time_min_str,
            timeMax=time_max_str,
            maxResults=100,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        events = events_result.get('items', [])
        for event in events:
            event_data = {
                'id': event.get('id'),
                'summary': event.get('summary', 'No Title'),
                'description': event.get('description', ''),
                'start_time': event['start'].get('dateTime', event['start'].get('date')),
                'end_time': event['end'].get('dateTime', event['end'].get('date')),
                'location': event.get('location', ''),
                'organizer_email': event['organizer'].get('email', ''),
                'organizer_name': event['organizer'].get('displayName', event['organizer'].get('email', '')),
                'attendees': [
                    {'email': att.get('email'), 'name': att.get('displayName', att.get('email')), 'responseStatus': att.get('responseStatus')}
                    for att in event.get('attendees', []) if att.get('email')
                ],
                'status': event.get('status')
            }
            processed_events.append(event_data)
    except HttpError as e:
        logging.error(f"Error fetching calendar events: {e}")
    return processed_events

# --- Analysis & Summary Generation ---
def analyze_emails_for_topics(texts, nlp):
    # Simple topic extraction: most common nouns
    all_tokens = []
    for text in texts:
        doc = nlp(text)
        all_tokens += [token.lemma_.lower() for token in doc if token.pos_ in ("NOUN", "PROPN")]
    common = Counter(all_tokens).most_common(10)
    return common

def parse_email_date_obj(date_string):
    """Parses an email date string into a datetime object."""
    try:
        # Try parsing with timezone first
        return parsedate_to_datetime(date_string)
    except Exception:
        # Fallback for simpler date strings if needed, though parsedate_to_datetime is robust
        try:
            return datetime.datetime.strptime(date_string, '%a, %d %b %Y %H:%M:%S %z')
        except ValueError:
            try:
                return datetime.datetime.strptime(date_string, '%a, %d %b %Y %H:%M:%S %Z')
            except ValueError:
                return None # Indicate parsing failure

def analyze_email_interactions(emails_data, user_email):
    email_exchange_counts = defaultdict(int)
    response_times_per_sender = defaultdict(list)
    emails_awaiting_response = []
    
    threads = defaultdict(list)
    for email_entry in emails_data:
        threads[email_entry['threadId']].append(email_entry)

    name_email_associations = defaultdict(Counter)

    for thread_id, email_list in threads.items():
        sorted_emails = sorted(email_list, key=lambda x: parse_email_date_obj(x['date']) if parse_email_date_obj(x['date']) else datetime.datetime.min)

        last_incoming_email_info = None # (sender_email, date_obj, subject)
        
        for email_entry in sorted_emails:
            email_date_obj = parse_email_date_obj(email_entry['date'])
            if not email_date_obj:
                continue

            normalized_from_email = email_entry['from_email'].lower()
            normalized_to_recipients = [r.lower() for r in email_entry['to_recipients']]
            normalized_cc_recipients = [r.lower() for r in email_entry['cc_recipients']]

            # Populate name_email_associations
            if email_entry['from_name'] and email_entry['from_email']:
                name_email_associations[email_entry['from_name']][email_entry['from_email'].lower()] += 1
            
            for recipient_full in email_entry['to_recipients'] + email_entry['cc_recipients']:
                match = re.match(r'^(.*?)\s*<([^>]+)>', recipient_full)
                if match:
                    name, email_addr = match.group(1).strip(), match.group(2)
                    if name: name_email_associations[name][email_addr.lower()] += 1
                elif '@' in recipient_full:
                    name_email_associations[recipient_full][recipient_full.lower()] += 1

            is_sent_by_user = (normalized_from_email == user_email.lower())
            is_received_by_user = (user_email.lower() in normalized_to_recipients or user_email.lower() in normalized_cc_recipients)

            if is_sent_by_user:
                for recipient in normalized_to_recipients + normalized_cc_recipients:
                    if recipient != user_email.lower():
                        email_exchange_counts[recipient] += 1
            elif is_received_by_user:
                email_exchange_counts[normalized_from_email] += 1
                
            if not is_sent_by_user and is_received_by_user:
                last_incoming_email_info = (email_entry['from_email'], email_date_obj, email_entry['subject'])
            elif is_sent_by_user and last_incoming_email_info:
                original_sender_email, incoming_date_obj, _ = last_incoming_email_info
                if email_date_obj > incoming_date_obj:
                    response_time = email_date_obj - incoming_date_obj
                    response_times_per_sender[original_sender_email].append(response_time)
                last_incoming_email_info = None
        
        # Check for emails awaiting response at the end of each thread
        sorted_emails_desc = sorted(email_list, key=lambda x: parse_email_date_obj(x['date']) if parse_email_date_obj(x['date']) else datetime.datetime.min, reverse=True)
        
        if sorted_emails_desc:
            latest_email_in_thread = sorted_emails_desc[0]
            latest_email_date_obj = parse_email_date_obj(latest_email_in_thread['date'])

            if latest_email_date_obj:
                normalized_latest_from_email = latest_email_in_thread['from_email'].lower()
                normalized_latest_to_recipients = [r.lower() for r in latest_email_in_thread['to_recipients']]
                normalized_latest_cc_recipients = [r.lower() for r in latest_email_in_thread['cc_recipients']]

                is_latest_sent_by_user = (normalized_latest_from_email == user_email.lower())
                is_latest_received_by_user = (user_email.lower() in normalized_latest_to_recipients or user_email.lower() in normalized_latest_cc_recipients)

                # Check if user has replied *anywhere* in the thread after this specific incoming email
                user_replied_after_this_incoming = False
                for email_in_thread in sorted_emails:
                    email_in_thread_date_obj = parse_email_date_obj(email_in_thread['date'])
                    if email_in_thread_date_obj and email_in_thread_date_obj > latest_email_date_obj:
                        if email_in_thread['from_email'].lower() == user_email.lower():
                            user_replied_after_this_incoming = True
                            break
                
                if is_latest_received_by_user and not is_latest_sent_by_user and not user_replied_after_this_incoming:
                    time_since_incoming = datetime.datetime.now(datetime.timezone.utc) - latest_email_date_obj
                    if datetime.timedelta(hours=1) < time_since_incoming < datetime.timedelta(days=14):
                        emails_awaiting_response.append({
                            'subject': latest_email_in_thread['subject'],
                            'sender': latest_email_in_thread['from_email'],
                            'date': latest_email_date_obj.strftime('%Y-%m-%d %H:%M')
                        })

    final_name_to_email_map = {}
    for name, email_counts in name_email_associations.items():
        if email_counts:
            final_name_to_email_map[name] = email_counts.most_common(1)[0][0]
        else:
            final_name_to_email_map[name] = name

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
    
    top_email_exchange_contacts = Counter(email_exchange_counts).most_common(10)

    key_organizations = Counter()
    combined_text_for_themes = ""
    for email_entry in emails_data:
        combined_text_for_themes += email_entry['body'] + " "
        people_in_body, orgs_in_body = extract_entities(email_entry['body'])
        for org in orgs_in_body: key_organizations[org] += 1

    return top_email_exchange_contacts, avg_response_times, key_organizations, combined_text_for_themes, final_name_to_email_map, emails_awaiting_response

def get_consolidated_contacts_summary(key_people_combined, top_email_exchange_contacts, avg_response_times, name_to_email_map, user_email):
    consolidated_data = defaultdict(lambda: {
        'display_name': 'Unknown',
        'interactions': 0,
        'emails_exchanged': 0,
        'avg_response_time': 'N/A',
        'primary_email': None
    })

    user_email_lower = user_email.lower()

    for email_address, emails_count in top_email_exchange_contacts:
        normalized_email = email_address.lower()
        if normalized_email == user_email_lower:
            continue

        consolidated_data[normalized_email]['emails_exchanged'] = emails_count
        consolidated_data[normalized_email]['primary_email'] = normalized_email
        if normalized_email in avg_response_times:
            consolidated_data[normalized_email]['avg_response_time'] = avg_response_times[normalized_email]
        
        found_name = None
        for name, mapped_email in name_to_email_map.items():
            if mapped_email and mapped_email.lower() == normalized_email:
                found_name = name
                break
        if found_name:
            consolidated_data[normalized_email]['display_name'] = found_name
        else:
            consolidated_data[normalized_email]['display_name'] = normalized_email

    for person_name, interactions_count in key_people_combined.items():
        if person_name.lower() == user_email_lower or (name_to_email_map.get(person_name, '').lower() == user_email_map.get(person_name, '').lower()): # Fixed comparison
            continue

        person_email = name_to_email_map.get(person_name)
        
        if person_email:
            normalized_email = person_email.lower()
            if normalized_email in consolidated_data:
                consolidated_data[normalized_email]['interactions'] += interactions_count
                if consolidated_data[normalized_email]['display_name'] == normalized_email:
                    consolidated_data[normalized_email]['display_name'] = person_name
            else:
                consolidated_data[normalized_email]['display_name'] = person_name
                consolidated_data[normalized_email]['interactions'] = interactions_count
                consolidated_data[normalized_email]['primary_email'] = normalized_email
        else:
            if person_name not in consolidated_data:
                consolidated_data[person_name]['display_name'] = person_name
                consolidated_data[person_name]['interactions'] += interactions_count

    final_list = []
    for key, data in consolidated_data.items():
        final_list.append({
            'contact': data['display_name'],
            'interactions': data['interactions'],
            'emails_exchanged': data['emails_exchanged'],
            'avg_response_time': data['avg_response_time']
        })
    
    final_list.sort(key=lambda x: (x['interactions'], x['emails_exchanged']), reverse=True)
    return final_list[:10]


# --- Load Prompt from File ---
def load_prompt_from_file(file_path):
    """Loads the prompt text from a specified file."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        logging.error(f"Prompt file not found at {file_path}. Please check the path.")
        return None
    except Exception as e:
        logging.error(f"Could not read prompt file {file_path}: {e}")
        return None

# --- Generate Digest with Gemini API ---
async def generate_gemini_summary(combined_data_json, prompt_text):
    """
    Calls the Gemini API to generate a digest from the combined data.
    """
    if not os.getenv("GEMINI_API_KEY"):
        logging.warning("Skipping Gemini API call: GEMINI_API_KEY not set.")
        return "<!-- Gemini API not configured. -->"

    if not prompt_text:
        logging.warning("Skipping Gemini API call: Prompt text is empty.")
        return "<!-- No prompt provided for Gemini API. -->"

    try:
        # Corrected model name
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        
        data_string = json.dumps(combined_data_json, indent=2)
        full_prompt = f"{prompt_text}\n\nHere is the data:\n{data_string}"
        
        logging.info("Sending request to Gemini API...")
        response = await model.generate_content_async(full_prompt)
        
        if response.candidates:
            digest = response.candidates[0].content.parts[0].text
            logging.info("Gemini API response received.")
            return digest
        else:
            logging.warning("Gemini API response had no candidates or content.")
            return "<!-- Could not generate digest. -->"
    except Exception as e:
        logging.error(f"Failed to call Gemini API: {e}")
        return f"<!-- Error generating digest: {e} -->"


# --- Chart Generation ---
def generate_topic_chart(common_topics, output_path="topic_chart.png"):
    if not common_topics:
        logging.warning("No common topics to generate chart.")
        return None

    labels, counts = zip(*common_topics)
    plt.figure(figsize=(10, 6)) # Added figure size for better readability
    plt.bar(labels, counts, color='#3498db') # Consistent color
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Topic')
    plt.ylabel('Frequency')
    plt.title('Top Topics', color='#2c3e50')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Generated topic chart at {output_path}")
    return output_path

# --- Email Sending ---
def send_email(sender_email, sender_password, recipient_email, subject, html_content, chart_files, logo_path, logo_cid):
    msg = MIMEMultipart('mixed')
    msg['From'] = f"ManagerFM <{sender_email}>"
    msg['To'] = recipient_email
    msg['Subject'] = subject

    msg_related = MIMEMultipart('related')
    msg_related.attach(MIMEText(html_content, 'html'))

    # Attach logo image
    if logo_path and os.path.exists(logo_path):
        try:
            with open(logo_path, 'rb') as img_file:
                img = MIMEImage(img_file.read())
                img.add_header('Content-ID', f"<{logo_cid}>")
                msg_related.attach(img)
            logging.info(f"Attached logo file: {logo_path}")
        except Exception as e:
            logging.error(f"Error attaching logo {logo_path}: {e}. Logo will not appear.")
    else:
        logging.warning(f"Logo file {logo_path} not found, cannot embed.")

    # Attach chart images
    for chart_name, chart_info in chart_files.items():
        try:
            if os.path.exists(chart_info['path']):
                with open(chart_info['path'], 'rb') as img_file:
                    img = MIMEImage(img_file.read())
                    img.add_header('Content-ID', f"<{chart_info['cid']}>")
                    msg_related.attach(img)
                os.remove(chart_info['path'])
                logging.info(f"Cleaned up chart file: {chart_info['path']}")
            else:
                logging.warning(f"Chart file {chart_info['path']} not found, cannot embed.")
        except Exception as e:
            logging.error(f"Error attaching chart {chart_info['path']}: {e}")
    
    msg_alternative = MIMEMultipart('alternative')
    plain_text_content = re.sub(r'<[^>]+>', '', html_content)
    msg_alternative.attach(MIMEText(plain_text_content, 'plain'))
    msg_alternative.attach(msg_related)
    msg.attach(msg_alternative)

    try:
        with smtplib.SMTP_SSL(os.getenv('SMTP_SERVER', 'smtp.gmail.com'), int(os.getenv('SMTP_PORT', 465))) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        logging.info('Summary email dispatched.')
    except smtplib.SMTPAuthenticationError:
        logging.error("SMTP Authentication Failed. Please check your SENDER_EMAIL_ADDRESS and SENDER_EMAIL_PASSWORD (ensure it's a Gmail App Password if using Gmail).")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")

# --- Main Workflow ---
async def main():
    validate_env_vars()
    nlp = load_spacy_model()
    
    # Authenticate Gmail and Calendar services
    creds = authenticate_gmail() # This now returns the credentials object
    gmail_service = build("gmail", "v1", credentials=creds) # Use creds here
    calendar_service = build("calendar", "v3", credentials=creds) # Use creds here
    logging.info('Authenticated with Google services.')

    time_window_days = 14
    end_time = datetime.datetime.now(datetime.timezone.utc)
    start_time = end_time - datetime.timedelta(days=time_window_days)

    # Fetch emails
    msgs = await fetch_recent_messages(gmail_service, days=time_window_days)
    if not msgs:
        logging.info('No recent emails found. Exiting.')
        return

    email_details = []
    for m in msgs:
        body = await get_message_body(gmail_service, m['id'])
        if body:
            # Extract headers for full email data
            full_msg = gmail_service.users().messages().get(userId='me', id=m['id'], format='full').execute()
            headers = full_msg.get('payload', {}).get('headers', [])
            
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            from_header = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown Sender')
            to_header = next((h['value'] for h in headers if h['name'] == 'To'), '')
            cc_header = next((h['value'] for h in headers if h['name'] == 'Cc'), '')
            date_header = next((h['value'] for h in headers if h['name'] == 'Date'), '')

            sender_name_match = re.match(r'^(.*?)\s*<([^>]+)>', from_header)
            sender_name = sender_name_match.group(1).strip() if sender_name_match else from_header
            sender_email = sender_name_match.group(2) if sender_name_match else from_header

            email_details.append({
                'id': m['id'],
                'threadId': m['threadId'],
                'subject': subject,
                'from_name': sender_name,
                'from_email': sender_email,
                'to_recipients': [re.sub(r'.*<([^>]+)>.*', r'\1', r).strip() for r in to_header.split(',')] if to_header else [],
                'cc_recipients': [re.sub(r'.*<([^>]+)>.*', r'\1', r).strip() for r in cc_header.split(',')] if cc_header else [],
                'date': date_header,
                'body': body
            })
    logging.info(f'Fetched and decoded {len(email_details)} email bodies.')

    # Fetch calendar events
    calendar_events = await fetch_and_process_calendar_events(calendar_service, start_time, end_time)
    logging.info(f'Fetched {len(calendar_events)} calendar events.')

    # Perform analysis
    top_email_exchange_contacts, avg_response_times, key_organizations_from_emails, combined_email_text, name_to_email_map, emails_awaiting_response = \
        analyze_email_interactions(email_details, os.getenv("SENDER_EMAIL_ADDRESS"))

    # Filter upcoming meetings
    upcoming_meetings = []
    now = datetime.datetime.now(datetime.timezone.utc)
    look_ahead_end_time = now + datetime.timedelta(days=7) # Next 7 days
    for event in calendar_events:
        try:
            event_start_str = event['start_time']
            if 'T' in event_start_str:
                event_start_dt = datetime.datetime.fromisoformat(event_start_str)
            else:
                event_start_dt = datetime.datetime.strptime(event_start_str, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc)
            
            if event_start_dt.tzinfo is None:
                event_start_dt = event_start_dt.replace(tzinfo=datetime.timezone.utc)

            if now < event_start_dt < look_ahead_end_time:
                is_important = False
                if event['organizer_email'].lower() == os.getenv("SENDER_EMAIL_ADDRESS").lower():
                    is_important = True
                for attendee in event['attendees']:
                    if attendee['email'].lower() == os.getenv("SENDER_EMAIL_ADDRESS").lower() and attendee.get('responseStatus') == 'accepted':
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
            logging.warning(f"Could not process calendar event '{event.get('summary', 'N/A')}': {e}")
    upcoming_meetings.sort(key=lambda x: datetime.datetime.strptime(x['start_time'], '%Y-%m-%d %H:%M'))


    # Aggregate key_people and key_organizations from both emails and calendar events
    key_people_combined = Counter()
    key_organizations_combined = Counter(key_organizations_from_emails)
    
    for email_entry in email_details:
        if email_entry['from_name'] and email_entry['from_email'] and not any(pattern in email_entry['from_email'].lower() for pattern in ["noreply", "info@", "support@", "marketing@"]):
            key_people_combined[email_entry['from_name']] += 1
        people_in_body, orgs_in_body = extract_entities(email_entry['body'])
        for person in people_in_body: key_people_combined[person] += 1

    combined_text_for_themes = combined_email_text
    for event in calendar_events:
        if event['organizer_name'] and event['organizer_email']:
             key_people_combined[event['organizer_name']] += 1
        for attendee in event['attendees']:
            if attendee['name']: key_people_combined[attendee['name']] += 1
        
        event_text = event['summary'] + " " + event['description']
        people_in_event, orgs_in_event = extract_entities(event_text)
        for person in people_in_event: key_people_combined[person] += 1
        for org in orgs_in_event: key_organizations_combined[org] += 1
        combined_text_for_themes += " " + event_text

    themes = analyze_emails_for_topics([combined_text_for_themes], nlp) # Re-using analyze_emails_for_topics for combined text

    consolidated_contacts_summary = get_consolidated_contacts_summary(key_people_combined, top_email_exchange_contacts, avg_response_times, name_to_email_map, os.getenv("SENDER_EMAIL_ADDRESS"))

    # Generate visualizations
    chart_files, _, _, thread_id_to_simplified_topic_map = generate_visualizations(email_details, calendar_events, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Prepare data for LLM
    llm_input_data = {
        "emails": email_details,
        "calendar_events": calendar_events,
        "consolidated_contacts_summary": consolidated_contacts_summary,
        "emails_awaiting_response": emails_awaiting_response,
        "upcoming_meetings": upcoming_meetings,
        "key_organizations": key_organizations_combined.most_common(10),
        "top_themes_keywords": [item[0] for item in themes] # Pass only keywords
    }

    # Generate LLM Digest
    llm_prompt = load_prompt_from_file(os.getenv("GEMINI_PROMPT_FILE_PATH"))
    llm_digest = "<!-- Error: Digest could not be generated. -->"
    if llm_prompt:
        llm_digest = await generate_gemini_summary(llm_input_data, llm_prompt)
    
    # Attach logo and get its CID
    logo_cid = make_msgid()[1:-1]
    
    # Format and Send Email Brief
    email_subject = f"ManagerFM Weekly Brief - {datetime.date.today().strftime('%Y-%m-%d')}"
    html_email_body = format_brief_as_html(
        key_people_combined, key_organizations_combined, [item[0] for item in themes], chart_files, # Pass themes as list of strings
        consolidated_contacts_summary, time_window_days, start_time, end_time, 
        email_details, logo_cid, thread_id_to_simplified_topic_map, llm_digest
    )
    send_email(os.getenv("SENDER_EMAIL_ADDRESS"), os.getenv("SENDER_EMAIL_PASSWORD"), os.getenv("RECIPIENT_EMAIL_ADDRESS"), email_subject, html_email_body, chart_files, os.getenv("MANAGER_FM_LOGO_PATH"), logo_cid)

    logging.info("ManagerFM briefing process completed.")

if __name__ == '__main__':
    asyncio.run(main())
