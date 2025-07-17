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
import json # Import for JSON operations

# You'll need to install these (which you've already done!):
# pip install google-api-python-client google-auth-oauthlib google-auth-httplib2
# And these for NLP (if you haven't yet):
# pip install nltk spacy
# python -m spacy download en_core_web_sm
import nltk
# nltk.download('punkt') # Uncomment and run if you don't have it
# nltk.download('stopwords') # Uncomment and run if you don't have it
# nltk.download('punkt_tab') # Uncomment and run if you don't have it
from nltk.corpus import stopwords
from collections import Counter
import spacy

# --- Configuration ---
# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# The file token.json stores the user's access and refresh tokens, and is
# created automatically when the authorization flow completes for the first
# time.
TOKEN_FILE = "token.json"
CREDENTIALS_FILE = "credentials.json" # Download this from Google Cloud Console
OUTPUT_JSON_FILE = "processed_emails_data.json" # New: File to save structured email data

# --- NLP Setup ---
# Load spaCy's English model for Named Entity Recognition (NER)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    exit()

# Set of common English stopwords to filter out
stop_words = set(stopwords.words('english'))

# --- Helper Functions ---

def clean_text(text):
    """Removes common email artifacts and cleans text for NLP."""
    text = re.sub(r'Subject:.*', '', text, flags=re.DOTALL) # Remove subject line if present
    text = re.sub(r'From:.*', '', text, flags=re.DOTALL) # Remove From line
    text = re.sub(r'To:.*', '', text, flags=re.DOTALL) # Remove To line
    text = re.sub(r'Date:.*', '', text, flags=re.DOTALL) # Remove Date line
    text = re.sub(r'Content-Type:.*', '', text, flags=re.DOTALL) # Remove Content-Type
    text = re.sub(r'[\r\n]+', ' ', text) # Replace newlines with spaces
    text = re.sub(r'\s+', ' ', text).strip() # Remove multiple spaces
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
                decoded_string += part.decode('latin-1', errors='ignore') # Fallback
        else:
            decoded_string += part
    return decoded_string

def get_email_body_from_gmail_api_payload(payload):
    """
    Extracts the plain text body from a Gmail API message payload.
    Handles different parts (text/plain, text/html, attachments).
    """
    if 'parts' in payload:
        for part in payload['parts']:
            mime_type = part.get('mimeType')
            if mime_type == 'text/plain':
                if 'body' in part and 'data' in part['body']:
                    return base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
            elif mime_type == 'text/html':
                # You might want to process HTML if plain text isn't available
                # For now, we prioritize plain text.
                pass
            # Recursively check nested parts
            if 'parts' in part:
                body = get_email_body_from_gmail_api_payload(part)
                if body:
                    return body
    elif 'body' in payload and 'data' in payload['body']:
        return base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='ignore')
    return ""

def is_marketing_email(email_body, subject, sender_email):
    """
    Performs a secondary, local check if an email is likely a marketing email.
    This is a fallback for emails that might slip past the API filter.
    """
    body_lower = email_body.lower()
    subject_lower = subject.lower()
    sender_email_lower = sender_email.lower()

    # Rule 1: Check for 'unsubscribe' link/text (even if API filters, good to double check)
    if "unsubscribe" in body_lower:
        return True

    # Rule 2: Common marketing phrases in subject or body (can be expanded)
    marketing_keywords = ["promo", "newsletter", "discount", "offer", "sale", "webinar", "event", "free trial", "coupon", "exclusive"]
    if any(keyword in subject_lower for keyword in marketing_keywords):
        return True
    if any(keyword in body_lower for keyword in marketing_keywords):
        return True

    # Rule 3: Common marketing sender patterns (can be improved with regex)
    marketing_sender_patterns = ["noreply", "info@", "support@", "marketing@", "updates@", "notifications@"]
    if any(pattern in sender_email_lower for pattern in marketing_sender_patterns):
        return True

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
    # Filter out stopwords and non-alphabetic words
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    # Count word frequencies
    word_counts = Counter(filtered_words)
    return [word for word, count in word_counts.most_common(num_keywords)]

# --- Main Script ---

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

def analyze_gmail_messages():
    """Connects to Gmail API, fetches, filters, and analyzes emails."""
    creds = authenticate_gmail()
    if not creds:
        print("Authentication failed. Exiting.")
        return

    try:
        service = build("gmail", "v1", credentials=creds)

        key_people = Counter()
        key_organizations = Counter()
        all_processed_emails_data = [] # Stores structured data for each email

        next_page_token = None
        total_messages_processed = 0
        page_number = 0

        # --- Calculate date 30 days ago for API query ---
        thirty_days_ago = datetime.datetime.now() - datetime.timedelta(days=30)
        date_filter = thirty_days_ago.strftime("%Y/%m/%d")
        print(f"Filtering emails from after: {date_filter} (last 30 days)")

        # --- Construct the API-level filter query ---
        query_parts = [
            f"after:{date_filter}", # Filter emails from the last 30 days
            '-"unsubscribe"', # Exclude emails with "unsubscribe" in the body
            '-category:promotions', # Exclude emails in the Promotions category
            '-category:social',     # Exclude emails in the Social category
            '-category:updates',    # Exclude emails in the Updates category
            '-category:forums',     # Exclude emails in the Forums category
            '-"promo"',
            '-"newsletter"',
            '-"discount"',
            '-"offer"',
            '-"sale"',
            '-"webinar"',
            '-"event"',
            '-"free trial"',
            '-"coupon"',
            '-"exclusive"',
            '-from:noreply@*',
            '-from:info@*',
            '-from:marketing@*',
            '-from:updates@*',
            '-from:notifications@*'
        ]
        gmail_api_query = " ".join(query_parts)
        print(f"Using Gmail API query for filtering: '{gmail_api_query}'")


        print("Starting email fetch and analysis...")

        while True:
            page_number += 1
            print(f"\nFetching page {page_number} of message IDs...")
            # Fetch message IDs in smaller batches with the time filter
            results = service.users().messages().list(
                userId="me",
                maxResults=50, # Keep at 50 messages per page for IDs
                pageToken=next_page_token,
                q=gmail_api_query # Apply the combined filter here!
            ).execute()

            messages = results.get("messages", [])

            if not messages:
                print("No more messages found or no messages matching the filter criteria.")
                break

            print(f"Found {len(messages)} message IDs on this page. Fetching full content...")

            for i, message in enumerate(messages):
                total_messages_processed += 1
                if total_messages_processed % 25 == 0: # Adjusted progress indicator
                    print(f"Processed {total_messages_processed} messages in total...")

                msg_id = message["id"]
                thread_id = message["threadId"]
                
                try:
                    # Fetch the full message content
                    msg = service.users().messages().get(userId="me", id=msg_id, format="full").execute()
                    time.sleep(0.2) # Add a small delay after each message fetch
                except HttpError as e:
                    print(f"Error fetching message ID {msg_id}: {e}. Skipping this message.")
                    continue
                except Exception as e:
                        print(f"An unexpected error occurred while fetching message ID {msg_id}: {e}. Skipping this message.")
                        continue


                headers = msg["payload"]["headers"]
                subject = ""
                sender_email = ""
                sender_name = ""
                to_recipients = []
                cc_recipients = []
                date_sent = ""
                has_attachments = 'attachmentId' in str(msg["payload"]) # Simple heuristic for attachments

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
                            sender_name = sender_email # Fallback if no name found
                    elif header["name"] == "To":
                        to_recipients = [decode_email_header(r.strip()) for r in header["value"].split(',')]
                    elif header["name"] == "Cc":
                        cc_recipients = [decode_email_header(r.strip()) for r in header["value"].split(',')]
                    elif header["name"] == "Date":
                        date_sent = decode_email_header(header["value"])


                body = get_email_body_from_gmail_api_payload(msg["payload"])
                cleaned_body = clean_text(body)

                # Secondary local filter (in case API filter missed something)
                if is_marketing_email(cleaned_body, subject, sender_email):
                    # print(f"Secondary filter skipped marketing email: {subject}") # Uncomment for verbose skipping
                    continue

                # Store all extracted email data
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

                # Update key people and organizations based on this email
                if sender_name and sender_email:
                    if not any(pattern in sender_email.lower() for pattern in ["noreply", "info@", "support@", "marketing@"]):
                        key_people[sender_name] += 1
                elif sender_email:
                    if not any(pattern in sender_email.lower() for pattern in ["noreply", "info@", "support@", "marketing@"]):
                        key_people[sender_email] += 1

                people_in_body, orgs_in_body = extract_entities(cleaned_body)
                for person in people_in_body:
                    key_people[person] += 1
                for org in orgs_in_body:
                    key_organizations[org] += 1

            next_page_token = results.get("nextPageToken")
            if not next_page_token:
                break # No more pages

        print(f"\n--- Finished processing {total_messages_processed} total messages ---")
        print(f"Collected data for {len(all_processed_emails_data)} non-marketing emails.")

        # --- Save collected data to JSON file ---
        try:
            with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
                json.dump(all_processed_emails_data, f, ensure_ascii=False, indent=4)
            print(f"Successfully saved processed email data to {OUTPUT_JSON_FILE}")
        except Exception as e:
            print(f"Error saving data to JSON file: {e}")

        # --- Final Analysis Results (using collected data) ---
        combined_text_for_themes = " ".join([e['body'] for e in all_processed_emails_data])

        print("\n--- Analysis Results ---")

        print("\nTop 10 Key People (based on sender and mentions):")
        for person, count in key_people.most_common(10):
            print(f"- {person} ({count} interactions/mentions)")

        print("\nTop 10 Key Organizations/Projects (based on mentions):")
        for org, count in key_organizations.most_common(10):
            print(f"- {org} ({count} mentions)")

        print("\nTop 20 Themes/Keywords (from non-marketing emails):")
        themes = extract_keywords_for_themes(combined_text_for_themes, num_keywords=20)
        for theme in themes:
            print(f"- {theme}")

    except HttpError as error:
        print(f"An API error occurred: {error}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    analyze_gmail_messages()
