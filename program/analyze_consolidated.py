import os
import base64
import re
import json
import datetime
import time
import logging
import asyncio
from collections import Counter, defaultdict
from email.header import decode_header
from email.utils import make_msgid, parsedate_to_datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import smtplib

# Third-party Libraries
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import nltk
import spacy
import google.generativeai as genai

# Google API Libraries
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Load environment variables
load_dotenv()

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Constants & Config
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.readonly"
]

TOKEN_FILE = "token.json"
CREDENTIALS_FILE = "credentials.json"

# Required environment variables
REQUIRED_ENV_VARS = [
    "SENDER_EMAIL_ADDRESS",
    "SENDER_EMAIL_PASSWORD", 
    "RECIPIENT_EMAIL_ADDRESS",
    "GEMINI_API_KEY",
    "GEMINI_PROMPT_FILE_PATH"
]

# Environment Validation
def validate_env_vars():
    missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

# NLP Initialization
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        logging.error("SpaCy model not found. Please run: python -m spacy download en_core_web_sm")
        exit(1)

# Initialize NLP components
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Helper Functions
def clean_text(text):
    """Removes common email artifacts and cleans text for NLP."""
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
                if body:
                    return body
    elif 'body' in payload and 'data' in payload['body']:
        return base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='ignore')
    return ""

def is_marketing_email(email_body, subject, sender_email):
    """Performs a secondary check if an email is likely a marketing email."""
    body_lower = email_body.lower()
    subject_lower = subject.lower()
    sender_email_lower = sender_email.lower()

    if "unsubscribe" in body_lower:
        return True
    
    marketing_keywords = ["promo", "newsletter", "discount", "offer", "sale", "webinar", "event", "free trial", "coupon", "exclusive"]
    if any(keyword in subject_lower for keyword in marketing_keywords):
        return True
    if any(keyword in body_lower for keyword in marketing_keywords):
        return True
    
    marketing_sender_patterns = ["noreply", "info@", "support@", "marketing@", "updates@", "notifications@"]
    if any(pattern in sender_email_lower for pattern in marketing_sender_patterns):
        return True
    
    return False

def extract_entities(text):
    """Extracts named entities (people, organizations) using spaCy."""
    nlp = load_spacy_model()
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

# Gmail Authentication
def authenticate_gmail():
    """Authenticates with Gmail API and returns credentials."""
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
    return creds

# Fetch & Parse Emails
async def fetch_recent_messages(service, days=14, max_results=50):
    """Fetches recent email messages with filtering."""
    date_after = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days))
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

async def get_message_details(service, msg_id):
    """Gets full message details including headers and body."""
    msg = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
    
    headers = msg["payload"]["headers"]
    subject = ""
    sender_email = ""
    sender_name = ""
    to_recipients = []
    cc_recipients = []
    date_sent = ""
    
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
    
    return {
        'id': msg_id,
        'threadId': msg['threadId'],
        'subject': subject,
        'from_name': sender_name,
        'from_email': sender_email,
        'to_recipients': to_recipients,
        'cc_recipients': cc_recipients,
        'date': date_sent,
        'body': cleaned_body,
        'has_attachments': 'attachmentId' in str(msg["payload"])
    }

# Fetch and Process Calendar Events
async def fetch_calendar_events(service, time_min, time_max):
    """Fetches calendar events within a specified time range."""
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

# Analysis Functions
def analyze_email_interactions(emails_data, user_email):
    """Analyzes email interactions and response patterns."""
    email_exchange_counts = defaultdict(int)
    response_times_per_sender = defaultdict(list)
    emails_awaiting_response = []
    
    threads = defaultdict(list)
    for email_entry in emails_data:
        threads[email_entry['threadId']].append(email_entry)

    name_email_associations = defaultdict(Counter)

    for thread_id, email_list in threads.items():
        sorted_emails = sorted(email_list, key=lambda x: parsedate_to_datetime(x['date']) if parsedate_to_datetime(x['date']) else datetime.datetime.min)

        last_incoming_email_info = None
        
        for email_entry in sorted_emails:
            email_date_obj = parsedate_to_datetime(email_entry['date'])
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
                    if name: 
                        name_email_associations[name][email_addr.lower()] += 1

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
        
        # Check for emails awaiting response
        sorted_emails_desc = sorted(email_list, key=lambda x: parsedate_to_datetime(x['date']) if parsedate_to_datetime(x['date']) else datetime.datetime.min, reverse=True)
        
        if sorted_emails_desc:
            latest_email_in_thread = sorted_emails_desc[0]
            latest_email_date_obj = parsedate_to_datetime(latest_email_in_thread['date'])

            if latest_email_date_obj:
                normalized_latest_from_email = latest_email_in_thread['from_email'].lower()
                normalized_latest_to_recipients = [r.lower() for r in latest_email_in_thread['to_recipients']]
                normalized_latest_cc_recipients = [r.lower() for r in latest_email_in_thread['cc_recipients']]

                is_latest_sent_by_user = (normalized_latest_from_email == user_email.lower())
                is_latest_received_by_user = (user_email.lower() in normalized_latest_to_recipients or user_email.lower() in normalized_latest_cc_recipients)

                user_replied_after_this_incoming = False
                for email_in_thread in sorted_emails:
                    email_in_thread_date_obj = parsedate_to_datetime(email_in_thread['date'])
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
    
    return top_email_exchange_contacts, avg_response_times, final_name_to_email_map, emails_awaiting_response

def get_upcoming_meetings(calendar_events, user_email):
    """Filters for upcoming important meetings."""
    upcoming_meetings = []
    now = datetime.datetime.now(datetime.timezone.utc)
    look_ahead_end_time = now + datetime.timedelta(days=7)
    
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
                if event['organizer_email'].lower() == user_email.lower():
                    is_important = True
                for attendee in event['attendees']:
                    if attendee['email'].lower() == user_email.lower() and attendee.get('responseStatus') == 'accepted':
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
    return upcoming_meetings

# Load Prompt from File
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

# Generate Digest with Gemini API
async def generate_gemini_summary(combined_data_json, prompt_text):
    """Calls the Gemini API to generate a digest from the combined data."""
    if not os.getenv("GEMINI_API_KEY"):
        logging.warning("Skipping Gemini API call: GEMINI_API_KEY not set.")
        return "<!-- Gemini API not configured. -->"

    if not prompt_text:
        logging.warning("Skipping Gemini API call: Prompt text is empty.")
        return "<!-- No prompt provided for Gemini API. -->"

    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
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

# Chart Generation
def generate_topic_chart(common_topics, output_path="topic_chart.png"):
    """Generates a bar chart for common topics."""
    if not common_topics:
        logging.warning("No common topics to generate chart.")
        return None

    labels, counts = zip(*common_topics)
    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color='#3498db')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Topic')
    plt.ylabel('Frequency')
    plt.title('Top Topics')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Generated topic chart at {output_path}")
    return output_path

# Email Sending
def send_email(sender_email, sender_password, recipient_email, subject, html_content, chart_files=None):
    """Sends an HTML email with optional chart attachments."""
    msg = MIMEMultipart('mixed')
    msg['From'] = f"ManagerFM <{sender_email}>"
    msg['To'] = recipient_email
    msg['Subject'] = subject

    msg_related = MIMEMultipart('related')
    msg_related.attach(MIMEText(html_content, 'html'))

    # Attach chart images if provided
    if chart_files:
        for chart_name, chart_path in chart_files.items():
            try:
                if os.path.exists(chart_path):
                    with open(chart_path, 'rb') as img_file:
                        img = MIMEImage(img_file.read())
                        img.add_header('Content-ID', f"<{chart_name}>")
                        msg_related.attach(img)
                    os.remove(chart_path)
                    logging.info(f"Attached and cleaned up chart file: {chart_path}")
            except Exception as e:
                logging.error(f"Error attaching chart {chart_path}: {e}")
    
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
        logging.error("SMTP Authentication Failed. Please check your credentials.")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")

def format_brief_as_html(digest_content, key_people, key_organizations, themes, chart_files=None):
    """Formats the brief as HTML email content."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>ManagerFM Daily Brief</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .header {{ background-color: #0ea5e9; color: white; padding: 20px; text-align: center; }}
            .content {{ padding: 20px; }}
            .section {{ margin-bottom: 30px; }}
            .section h2 {{ color: #0ea5e9; border-bottom: 2px solid #0ea5e9; padding-bottom: 5px; }}
            ul {{ padding-left: 20px; }}
            li {{ margin-bottom: 5px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>ManagerFM Daily Brief</h1>
            <p>{datetime.date.today().strftime('%B %d, %Y')}</p>
        </div>
        
        <div class="content">
            <div class="section">
                <h2>AI-Generated Summary</h2>
                {digest_content}
            </div>
            
            <div class="section">
                <h2>Key People</h2>
                <ul>
                    {''.join([f'<li>{person} ({count} interactions)</li>' for person, count in key_people.most_common(10)])}
                </ul>
            </div>
            
            <div class="section">
                <h2>Key Organizations</h2>
                <ul>
                    {''.join([f'<li>{org} ({count} mentions)</li>' for org, count in key_organizations.most_common(10)])}
                </ul>
            </div>
            
            <div class="section">
                <h2>Top Themes</h2>
                <ul>
                    {''.join([f'<li>{theme}</li>' for theme in themes[:10]])}
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

# Main Workflow
async def main():
    """Main execution function."""
    validate_env_vars()
    
    # Authenticate with Google services
    creds = authenticate_gmail()
    gmail_service = build("gmail", "v1", credentials=creds)
    calendar_service = build("calendar", "v3", credentials=creds)
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
        try:
            email_data = await get_message_details(gmail_service, m['id'])
            if email_data and not is_marketing_email(email_data['body'], email_data['subject'], email_data['from_email']):
                email_details.append(email_data)
            time.sleep(0.1)  # Rate limiting
        except Exception as e:
            logging.error(f"Error processing email {m['id']}: {e}")
            continue

    logging.info(f'Processed {len(email_details)} emails.')

    # Fetch calendar events
    calendar_events = await fetch_calendar_events(calendar_service, start_time, end_time)
    logging.info(f'Fetched {len(calendar_events)} calendar events.')

    # Perform analysis
    user_email = os.getenv("SENDER_EMAIL_ADDRESS")
    top_email_exchange_contacts, avg_response_times, name_to_email_map, emails_awaiting_response = \
        analyze_email_interactions(email_details, user_email)

    upcoming_meetings = get_upcoming_meetings(calendar_events, user_email)

    # Aggregate key people and organizations
    key_people_combined = Counter()
    key_organizations_combined = Counter()
    
    for email_entry in email_details:
        if email_entry['from_name'] and email_entry['from_email'] and not any(pattern in email_entry['from_email'].lower() for pattern in ["noreply", "info@", "support@", "marketing@"]):
            key_people_combined[email_entry['from_name']] += 1
        people_in_body, orgs_in_body = extract_entities(email_entry['body'])
        for person in people_in_body: 
            key_people_combined[person] += 1
        for org in orgs_in_body: 
            key_organizations_combined[org] += 1

    for event in calendar_events:
        if event['organizer_name'] and event['organizer_email']:
             key_people_combined[event['organizer_name']] += 1
        for attendee in event['attendees']:
            if attendee['name']: 
                key_people_combined[attendee['name']] += 1
        
        event_text = event['summary'] + " " + event['description']
        people_in_event, orgs_in_event = extract_entities(event_text)
        for person in people_in_event: 
            key_people_combined[person] += 1
        for org in orgs_in_event: 
            key_organizations_combined[org] += 1

    # Extract themes
    combined_text_for_themes = " ".join([e['body'] for e in email_details])
    combined_text_for_themes += " ".join([clean_text(e['summary'] + " " + e['description']) for e in calendar_events])
    themes = extract_keywords_for_themes(combined_text_for_themes, num_keywords=20)

    # Generate topic chart
    topic_counts = Counter()
    for theme in themes:
        topic_counts[theme] = combined_text_for_themes.lower().count(theme.lower())
    
    chart_files = {}
    chart_path = generate_topic_chart(topic_counts.most_common(10))
    if chart_path:
        chart_files['topic_chart'] = chart_path

    # Prepare data for LLM
    llm_input_data = {
        "emails": email_details,
        "calendar_events": calendar_events,
        "top_email_contacts": [{"contact": contact, "count": count} for contact, count in top_email_exchange_contacts],
        "emails_awaiting_response": emails_awaiting_response,
        "upcoming_meetings": upcoming_meetings,
        "key_organizations": [{"org": org, "count": count} for org, count in key_organizations_combined.most_common(10)],
        "top_themes_keywords": themes
    }

    # Generate LLM Digest
    llm_prompt = load_prompt_from_file(os.getenv("GEMINI_PROMPT_FILE_PATH"))
    llm_digest = "<!-- Error: Digest could not be generated. -->"
    if llm_prompt:
        llm_digest = await generate_gemini_summary(llm_input_data, llm_prompt)
    
    # Save data to JSON for debugging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"manager_briefing_data_{timestamp}.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(llm_input_data, f, ensure_ascii=False, indent=4)
        logging.info(f"Saved briefing data to {output_file}")
    except Exception as e:
        logging.error(f"Error saving data to JSON file: {e}")
    
    # Format and Send Email Brief
    email_subject = f"ManagerFM Weekly Brief - {datetime.date.today().strftime('%Y-%m-%d')}"
    html_email_body = format_brief_as_html(
        llm_digest, key_people_combined, key_organizations_combined, themes, chart_files
    )
    
    send_email(
        os.getenv("SENDER_EMAIL_ADDRESS"), 
        os.getenv("SENDER_EMAIL_PASSWORD"), 
        os.getenv("RECIPIENT_EMAIL_ADDRESS"), 
        email_subject, 
        html_email_body, 
        chart_files
    )

    logging.info("ManagerFM briefing process completed.")

if __name__ == '__main__':
    asyncio.run(main())