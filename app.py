from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import imaplib
import email
from email.header import decode_header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import google.generativeai as genai
from dotenv import load_dotenv
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    print("✓ Gemini AI configured")
else:
    model = None
    print("✗ Warning: GEMINI_API_KEY not found in .env")

# Load knowledge base with RAG
KNOWLEDGE_BASE = {}
RAG_INDEX = None

def load_knowledge_base():
    """Load knowledge.txt and create RAG index"""
    global KNOWLEDGE_BASE, RAG_INDEX
    knowledge_file = 'knowledge.txt'
    
    if os.path.exists(knowledge_file):
        with open(knowledge_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Split into chunks for RAG
            chunks = []
            current_section = ""
            section_title = ""
            
            for line in content.split('\n'):
                if line.startswith('====='):
                    if current_section:
                        chunks.append({
                            'title': section_title,
                            'content': current_section.strip(),
                            'combined': f"{section_title}\n{current_section.strip()}"
                        })
                    section_title = line.strip('= ')
                    current_section = ""
                else:
                    current_section += line + "\n"
            
            # Add last section
            if current_section:
                chunks.append({
                    'title': section_title,
                    'content': current_section.strip(),
                    'combined': f"{section_title}\n{current_section.strip()}"
                })
            
            KNOWLEDGE_BASE = {
                'content': content,
                'chunks': chunks,
                'loaded': True
            }
            
            # Create TF-IDF vectorizer for RAG retrieval
            if chunks:
                chunk_texts = [chunk['combined'] for chunk in chunks]
                RAG_INDEX = {
                    'vectorizer': TfidfVectorizer(stop_words='english'),
                    'texts': chunk_texts,
                    'chunks': chunks
                }
                RAG_INDEX['vectors'] = RAG_INDEX['vectorizer'].fit_transform(chunk_texts)
            
            print(f"✓ Knowledge base loaded: {len(content)} characters, {len(chunks)} chunks")
            print(f"✓ RAG index created with {len(chunks)} document chunks")
    else:
        KNOWLEDGE_BASE = {
            'content': '',
            'chunks': [],
            'loaded': False
        }
        print("✗ Warning: knowledge.txt not found")
    return KNOWLEDGE_BASE

# Load knowledge base on startup
load_knowledge_base()

def connect_to_email():
    """Connect to email server using IMAP"""
    try:
        email_server = os.getenv('EMAIL_SERVER', 'imap.gmail.com')
        email_address = os.getenv('EMAIL_ADDRESS')
        email_password = os.getenv('EMAIL_PASSWORD')
        
        if not email_address or not email_password:
            print("✗ Email credentials not configured")
            return None
            
        mail = imaplib.IMAP4_SSL(email_server)
        mail.login(email_address, email_password)
        return mail
    except Exception as e:
        print(f"✗ Email connection error: {e}")
        return None

def fetch_emails(limit=50):
    """Fetch emails from inbox"""
    mail = connect_to_email()
    if not mail:
        return []
    
    try:
        mail.select('inbox')
        status, messages = mail.search(None, 'ALL')
        
        if status != 'OK':
            print("✗ Failed to search emails")
            return []
        
        email_ids = messages[0].split()
        emails = []
        
        # Get latest emails first
        for email_id in reversed(email_ids[-limit:]):
            try:
                status, msg_data = mail.fetch(email_id, '(RFC822)')
                
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        
                        # Decode subject
                        subject_header = msg.get('Subject', '')
                        subject = ''
                        if subject_header:
                            decoded = decode_header(subject_header)
                            subject = decoded[0][0]
                            if isinstance(subject, bytes):
                                subject = subject.decode(errors='ignore')
                        
                        # Get sender
                        from_email = msg.get('From', '')
                        
                        # Get body
                        body = ""
                        if msg.is_multipart():
                            for part in msg.walk():
                                content_type = part.get_content_type()
                                if content_type == "text/plain":
                                    try:
                                        payload = part.get_payload(decode=True)
                                        if payload:
                                            body = payload.decode(errors='ignore')
                                            break
                                    except:
                                        pass
                        else:
                            try:
                                payload = msg.get_payload(decode=True)
                                if payload:
                                    body = payload.decode(errors='ignore')
                            except:
                                body = str(msg.get_payload())
                        
                        emails.append({
                            'email_id': email_id.decode(),
                            'from': from_email,
                            'subject': subject or 'No Subject',
                            'body': body.strip() or 'No content',
                            'received_time': msg.get('Date', 'Unknown'),
                            'attachments': []
                        })
            except Exception as e:
                print(f"Error processing email {email_id}: {e}")
                continue
        
        mail.close()
        mail.logout()
        print(f"✓ Fetched {len(emails)} emails")
        return emails
        
    except Exception as e:
        print(f"✗ Fetch error: {e}")
        return []

def send_email_reply(to_email, subject, body):
    """Send email reply via SMTP"""
    try:
        smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = int(os.getenv('SMTP_PORT', 587))
        email_address = os.getenv('EMAIL_ADDRESS')
        email_password = os.getenv('EMAIL_PASSWORD')
        
        if not email_address or not email_password:
            return {'success': False, 'error': 'Email credentials not configured'}
        
        msg = MIMEMultipart()
        msg['From'] = email_address
        msg['To'] = to_email
        msg['Subject'] = f"Re: {subject}" if not subject.startswith('Re:') else subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(email_address, email_password)
        server.send_message(msg)
        server.quit()
        
        print(f"✓ Email sent to {to_email}")
        return {'success': True, 'message': 'Email sent successfully'}
        
    except Exception as e:
        print(f"✗ Send email error: {e}")
        return {'success': False, 'error': str(e)}

def analyze_email(email_data):
    """Analyze email content to determine intent, urgency, and sender type"""
    subject = email_data.get('subject', '').lower()
    body = email_data.get('body', '').lower()
    combined_text = f"{subject} {body}"
    
    intent = "General"
    if any(word in combined_text for word in ['price', 'pricing', 'cost', 'quote', 'payment']):
        intent = "Pricing"
    elif any(word in combined_text for word in ['meeting', 'schedule', 'call', 'appointment']):
        intent = "Meeting"
    elif any(word in combined_text for word in ['help', 'issue', 'problem', 'support']):
        intent = "Support"
    elif any(word in combined_text for word in ['buy', 'purchase', 'interested', 'sales']):
        intent = "Sales"
    
    urgency = "Medium"
    if any(word in combined_text for word in ['urgent', 'asap', 'immediately', 'emergency']):
        urgency = "High"
    elif any(word in combined_text for word in ['whenever', 'no rush', 'flexible']):
        urgency = "Low"
    
    sender_type = "Lead"
    
    return {
        'intent': intent,
        'urgency': urgency,
        'sender_type': sender_type
    }

def extract_sender_name(from_email):
    """Extract sender's first name from email"""
    match = re.match(r'^([^<]+)', from_email)
    if match:
        name = match.group(1).strip()
        first_name = name.split()[0] if name else "there"
        first_name = first_name.replace('"', '').replace("'", '')
        return first_name
    return "there"

def is_automated_email(email_data):
    """Detect if email is automated/notification"""
    subject = email_data.get('subject', '').lower()
    body = email_data.get('body', '').lower()
    from_email = email_data.get('from', '').lower()
    
    automated_indicators = [
        'job alert', 'jobs@linkedin', 'linkedin.com', 'new jobs match',
        'noreply', 'no-reply', 'donotreply', 'automated', 'notification',
        'unsubscribe', 'newsletter', 'manage your settings'
    ]
    
    combined_text = f"{subject} {body} {from_email}"
    
    for indicator in automated_indicators:
        if indicator in combined_text:
            return True
    
    return False

def rag_retrieve(query, top_k=3):
    """RAG: Retrieve most relevant chunks from knowledge base"""
    if not RAG_INDEX or not KNOWLEDGE_BASE.get('loaded'):
        return []
    
    try:
        # Transform query to vector
        query_vector = RAG_INDEX['vectorizer'].transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, RAG_INDEX['vectors'])[0]
        
        # Get top-k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        retrieved_chunks = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Threshold for relevance
                chunk = RAG_INDEX['chunks'][idx].copy()
                chunk['similarity'] = float(similarities[idx])
                retrieved_chunks.append(chunk)
        
        print(f"✓ RAG Retrieved {len(retrieved_chunks)} relevant chunks")
        for i, chunk in enumerate(retrieved_chunks):
            print(f"  [{i+1}] {chunk['title']} (similarity: {chunk['similarity']:.3f})")
        
        return retrieved_chunks
    
    except Exception as e:
        print(f"✗ RAG retrieval error: {e}")
        return []

def generate_draft_with_rag(email_data, analysis):
    """Generate draft using RAG (Retrieval-Augmented Generation)"""
    
    # Check for automated emails first
    if is_automated_email(email_data):
        return {
            'required': False,
            'content': """[AUTOMATED EMAIL DETECTED - NO REPLY NEEDED]

This is an automated notification (job alert, newsletter, etc.).
Recommendation: Do not reply to this email."""
        }
    
    if not KNOWLEDGE_BASE.get('loaded'):
        return {
            'required': True,
            'content': """Thank you for your email.

I'm currently unable to access our information database.

Best regards,
Support Team"""
        }
    
    sender_name = extract_sender_name(email_data.get('from', ''))
    query = f"{email_data.get('subject', '')} {email_data.get('body', '')}"
    
    # RAG STEP 1: RETRIEVE relevant chunks from knowledge base
    retrieved_chunks = rag_retrieve(query, top_k=3)
    
    if not retrieved_chunks:
        # No relevant info found
        return {
            'required': True,
            'content': f"""Hi {sender_name},

Thank you for reaching out to TechStart Solutions!

I'd be happy to help you. To provide the most accurate information, could you please clarify what specific information you're looking for?

You can reach us at:
• Email: hello@techstart.solutions
• Phone: (555) 123-4567

Best regards,
TechStart Solutions Team"""
        }
    
    # RAG STEP 2: Build context from retrieved chunks
    context = ""
    for i, chunk in enumerate(retrieved_chunks):
        context += f"\n--- {chunk['title']} ---\n{chunk['content']}\n"
    
    # RAG STEP 3: GENERATE response using retrieved context
    if model and GEMINI_API_KEY:
        try:
            prompt = f"""You are an email assistant for TechStart Solutions.

RETRIEVED RELEVANT INFORMATION FROM KNOWLEDGE BASE:
{context}

CUSTOMER EMAIL:
Subject: {email_data.get('subject')}
Body: {email_data.get('body')}

INSTRUCTIONS:
1. Read the customer's question carefully
2. Use ONLY the information provided above to answer
3. Be specific with details (prices, durations, technologies, addresses)
4. If the information exists above, answer directly - do NOT ask for clarification
5. Start with "Hi {sender_name},"
6. End with "Best regards, TechStart Solutions Team"

Write a professional email response now:"""

            response = model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.7,
                    'max_output_tokens': 1024,
                }
            )
            
            draft_content = response.text.strip()
            
            if draft_content and len(draft_content) > 50:
                print(f"✓ RAG-based draft generated: {len(draft_content)} characters")
                return {
                    'required': True,
                    'content': draft_content,
                    'rag_chunks_used': len(retrieved_chunks)
                }
        
        except Exception as e:
            print(f"✗ Gemini generation error: {e}")
    
    # Fallback: Use retrieved chunks directly
    fallback_response = f"""Hi {sender_name},

Thank you for reaching out to TechStart Solutions!

Based on your inquiry, here's the relevant information:

"""
    
    for chunk in retrieved_chunks:
        if 'SERVICES' in chunk['title']:
            fallback_response += f"**Our Services:**\n{chunk['content'][:300]}...\n\n"
        elif 'PRICING' in chunk['title']:
            fallback_response += f"**Pricing Information:**\n{chunk['content'][:300]}...\n\n"
        elif 'CONTACT' in chunk['title']:
            fallback_response += f"**Contact Us:**\n{chunk['content'][:200]}...\n\n"
    
    fallback_response += """Would you like more details about any specific aspect?

Best regards,
TechStart Solutions Team"""
    
    return {
        'required': True,
        'content': fallback_response,
        'rag_chunks_used': len(retrieved_chunks)
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/emails', methods=['GET'])
def get_emails():
    try:
        limit = int(request.args.get('limit', 50))
        emails = fetch_emails(limit=limit)
        return jsonify({
            'success': True,
            'emails': emails,
            'count': len(emails)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_email_endpoint():
    try:
        email_data = request.json
        
        print(f"\n{'='*60}")
        print(f"📧 Analyzing email: {email_data.get('subject')}")
        print(f"{'='*60}")
        
        analysis = analyze_email(email_data)
        draft_reply = generate_draft_with_rag(email_data, analysis)
        
        print(f"✓ Draft generated using RAG")
        if 'rag_chunks_used' in draft_reply:
            print(f"✓ Used {draft_reply['rag_chunks_used']} knowledge chunks\n")
        
        return jsonify({
            'email_id': email_data.get('email_id'),
            'email_analysis': analysis,
            'draft_reply': draft_reply
        })
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/send', methods=['POST'])
def send_email_endpoint():
    try:
        data = request.json
        result = send_email_reply(data.get('to'), data.get('subject'), data.get('body'))
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/knowledge', methods=['GET'])
def get_knowledge_status():
    return jsonify({
        'loaded': KNOWLEDGE_BASE.get('loaded', False),
        'chunks': len(KNOWLEDGE_BASE.get('chunks', [])),
        'rag_enabled': RAG_INDEX is not None
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 RAG-BASED EMAIL ASSISTANT STARTING...")
    print("="*60)
    print(f"✓ Flask Server: http://0.0.0.0:5000")
    print(f"✓ Knowledge Base: {len(KNOWLEDGE_BASE.get('chunks', []))} chunks")
    print(f"✓ RAG Index: {'Enabled' if RAG_INDEX else 'Disabled'}")
    print(f"✓ Gemini API: {'Configured' if GEMINI_API_KEY else 'NOT SET'}")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
    
    
    
    
'''from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import imaplib
import email
from email.header import decode_header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import google.generativeai as genai
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    print("✓ Gemini AI configured")
else:
    model = None
    print("✗ Warning: GEMINI_API_KEY not found in .env")

# Load knowledge base
KNOWLEDGE_BASE = {}

def load_knowledge_base():
    """Load knowledge.txt into memory"""
    global KNOWLEDGE_BASE
    knowledge_file = 'knowledge.txt'
    
    if os.path.exists(knowledge_file):
        with open(knowledge_file, 'r', encoding='utf-8') as f:
            content = f.read()
            KNOWLEDGE_BASE = {
                'content': content,
                'loaded': True
            }
            print(f"✓ Knowledge base loaded: {len(content)} characters")
    else:
        KNOWLEDGE_BASE = {
            'content': '',
            'loaded': False
        }
        print("✗ Warning: knowledge.txt not found")
    return KNOWLEDGE_BASE

# Load knowledge base on startup
load_knowledge_base()

def connect_to_email():
    """Connect to email server using IMAP"""
    try:
        email_server = os.getenv('EMAIL_SERVER', 'imap.gmail.com')
        email_address = os.getenv('EMAIL_ADDRESS')
        email_password = os.getenv('EMAIL_PASSWORD')
        
        if not email_address or not email_password:
            print("✗ Email credentials not configured")
            return None
            
        mail = imaplib.IMAP4_SSL(email_server)
        mail.login(email_address, email_password)
        return mail
    except Exception as e:
        print(f"✗ Email connection error: {e}")
        return None

def fetch_emails(limit=50):
    """Fetch emails from inbox"""
    mail = connect_to_email()
    if not mail:
        return []
    
    try:
        mail.select('inbox')
        status, messages = mail.search(None, 'ALL')
        
        if status != 'OK':
            print("✗ Failed to search emails")
            return []
        
        email_ids = messages[0].split()
        emails = []
        
        # Get latest emails first
        for email_id in reversed(email_ids[-limit:]):
            try:
                status, msg_data = mail.fetch(email_id, '(RFC822)')
                
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        
                        # Decode subject
                        subject_header = msg.get('Subject', '')
                        subject = ''
                        if subject_header:
                            decoded = decode_header(subject_header)
                            subject = decoded[0][0]
                            if isinstance(subject, bytes):
                                subject = subject.decode(errors='ignore')
                        
                        # Get sender
                        from_email = msg.get('From', '')
                        
                        # Get body
                        body = ""
                        if msg.is_multipart():
                            for part in msg.walk():
                                content_type = part.get_content_type()
                                if content_type == "text/plain":
                                    try:
                                        payload = part.get_payload(decode=True)
                                        if payload:
                                            body = payload.decode(errors='ignore')
                                            break
                                    except:
                                        pass
                        else:
                            try:
                                payload = msg.get_payload(decode=True)
                                if payload:
                                    body = payload.decode(errors='ignore')
                            except:
                                body = str(msg.get_payload())
                        
                        emails.append({
                            'email_id': email_id.decode(),
                            'from': from_email,
                            'subject': subject or 'No Subject',
                            'body': body.strip() or 'No content',
                            'received_time': msg.get('Date', 'Unknown'),
                            'attachments': []
                        })
            except Exception as e:
                print(f"Error processing email {email_id}: {e}")
                continue
        
        mail.close()
        mail.logout()
        print(f"✓ Fetched {len(emails)} emails")
        return emails
        
    except Exception as e:
        print(f"✗ Fetch error: {e}")
        return []

def send_email_reply(to_email, subject, body):
    """Send email reply via SMTP"""
    try:
        # Get email configuration
        smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = int(os.getenv('SMTP_PORT', 587))
        email_address = os.getenv('EMAIL_ADDRESS')
        email_password = os.getenv('EMAIL_PASSWORD')
        
        if not email_address or not email_password:
            return {'success': False, 'error': 'Email credentials not configured'}
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = email_address
        msg['To'] = to_email
        msg['Subject'] = f"Re: {subject}" if not subject.startswith('Re:') else subject
        
        # Add body
        msg.attach(MIMEText(body, 'plain'))
        
        # Connect and send
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(email_address, email_password)
        server.send_message(msg)
        server.quit()
        
        print(f"✓ Email sent to {to_email}")
        return {'success': True, 'message': 'Email sent successfully'}
        
    except Exception as e:
        print(f"✗ Send email error: {e}")
        return {'success': False, 'error': str(e)}

def analyze_email(email_data):
    """Analyze email content to determine intent, urgency, and sender type"""
    subject = email_data.get('subject', '').lower()
    body = email_data.get('body', '').lower()
    combined_text = f"{subject} {body}"
    
    # Determine intent
    intent = "General"
    if any(word in combined_text for word in ['price', 'pricing', 'cost', 'quote', 'payment', 'plan', 'package', 'fee', 'charge']):
        intent = "Pricing"
    elif any(word in combined_text for word in ['meeting', 'schedule', 'call', 'appointment', 'demo', 'discuss', 'talk']):
        intent = "Meeting"
    elif any(word in combined_text for word in ['help', 'issue', 'problem', 'error', 'bug', 'support', 'not working', 'broken', 'fix']):
        intent = "Support"
    elif any(word in combined_text for word in ['buy', 'purchase', 'interested', 'sales', 'service', 'solution', 'product']):
        intent = "Sales"
    
    # Determine urgency
    urgency = "Medium"
    if any(word in combined_text for word in ['urgent', 'asap', 'immediately', 'emergency', 'critical', 'important', 'now']):
        urgency = "High"
    elif any(word in combined_text for word in ['whenever', 'no rush', 'at your convenience', 'no hurry', 'flexible']):
        urgency = "Low"
    
    # Determine sender type
    sender_email = email_data.get('from', '').lower()
    sender_type = "Lead"
    
    if any(word in body for word in ['customer', 'account', 'subscription', 'existing', 'current']):
        sender_type = "Customer"
    elif any(domain in sender_email for domain in ['@company.com', '@startup.com', '@yourcompany.com']):
        sender_type = "Employee"
    elif any(word in body for word in ['vendor', 'supplier', 'partner', 'partnership']):
        sender_type = "Vendor"
    
    return {
        'intent': intent,
        'urgency': urgency,
        'sender_type': sender_type
    }

def extract_sender_name(from_email):
    """Extract sender's first name from email"""
    # Try to extract name from "Name <email@domain.com>" format
    match = re.match(r'^([^<]+)', from_email)
    if match:
        name = match.group(1).strip()
        # Get first name
        first_name = name.split()[0] if name else "there"
        # Remove quotes if present
        first_name = first_name.replace('"', '').replace("'", '')
        return first_name
    return "there"

def search_knowledge_base(query):
    """Search knowledge base for relevant information"""
    if not KNOWLEDGE_BASE.get('loaded'):
        return []
    
    kb_content = KNOWLEDGE_BASE.get('content', '')
    query_lower = query.lower()
    
    # Split knowledge base into sections
    sections = []
    current_section = ""
    current_title = ""
    
    for line in kb_content.split('\n'):
        if line.startswith('====='):
            if current_section:
                sections.append({
                    'title': current_title,
                    'content': current_section.strip()
                })
            current_title = line.strip('= ')
            current_section = ""
        else:
            current_section += line + "\n"
    
    # Add last section
    if current_section:
        sections.append({
            'title': current_title,
            'content': current_section.strip()
        })
    
    # Find relevant sections based on query keywords
    relevant_sections = []
    
    # Keyword mapping for better matching
    keyword_map = {
        'duration': ['SERVICES'],
        'time': ['SERVICES', 'POLICIES'],
        'timeline': ['SERVICES', 'PRICING'],
        'week': ['SERVICES', 'PRICING'],
        'month': ['SERVICES', 'PRICING'],
        'web development': ['SERVICES'],
        'mobile': ['SERVICES'],
        'app': ['SERVICES'],
        'cloud': ['SERVICES'],
        'price': ['PRICING'],
        'cost': ['PRICING'],
        'pricing': ['PRICING'],
        'payment': ['POLICIES'],
        'refund': ['POLICIES'],
        'support': ['POLICIES'],
        'technology': ['SERVICES', 'FAQS'],
        'tech': ['SERVICES', 'FAQS'],
        'learn': ['SERVICES'],
        'future': ['SERVICES'],
        'hour': ['POLICIES', 'CONTACT'],
        'address': ['CONTACT'],
        'contact': ['CONTACT'],
        'email': ['CONTACT'],
        'phone': ['CONTACT']
    }
    
    # Check which sections are relevant
    sections_to_include = set()
    for keyword, section_names in keyword_map.items():
        if keyword in query_lower:
            sections_to_include.update(section_names)
    
    # If no specific keywords matched, include all sections
    if not sections_to_include:
        sections_to_include = {'SERVICES', 'PRICING', 'POLICIES', 'FAQS', 'CONTACT'}
    
    # Get relevant sections
    for section in sections:
        if section['title'] in sections_to_include:
            relevant_sections.append(section)
    
    return relevant_sections

def is_automated_email(email_data):
    """Detect if email is automated/notification and should not be replied to"""
    subject = email_data.get('subject', '').lower()
    body = email_data.get('body', '').lower()
    from_email = email_data.get('from', '').lower()
    
    # List of automated email indicators
    automated_indicators = [
        # LinkedIn
        'job alert', 'jobs@linkedin', 'linkedin.com', 'new jobs match', 
        'your job alert', 'job search smarter',
        
        # General notifications
        'noreply', 'no-reply', 'donotreply', 'do-not-reply',
        'automated', 'notification', 'alert',
        
        # Social media
        'facebook', 'twitter', 'instagram', 'reddit',
        
        # Newsletter/Marketing
        'unsubscribe', 'newsletter', 'update your preferences',
        'manage your settings', 'click here to unsubscribe',
        
        # System emails
        'password reset', 'verify your email', 'confirmation',
        'receipt', 'invoice', 'order confirmation',
        
        # Specific patterns
        'this email was intended for',
        'you are receiving this email because',
        'manage your job alerts',
    ]
    
    # Check if any indicator is present
    combined_text = f"{subject} {body} {from_email}"
    
    for indicator in automated_indicators:
        if indicator in combined_text:
            return True
    
    # Check for URLs with unsubscribe patterns
    if 'unsubscribe' in body or 'opt-out' in body or 'opt out' in body:
        return True
    
    return False

def extract_info_from_kb(keywords):
    """Directly extract information from knowledge base based on keywords"""
    kb_content = KNOWLEDGE_BASE.get('content', '')
    extracted_info = {}
    
    # Extract specific information
    lines = kb_content.split('\n')
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        
        # Extract web development info
        if 'web development' in line_lower and 'duration:' in lines[i+1].lower() if i+1 < len(lines) else False:
            extracted_info['web_duration'] = lines[i+1].strip()
        if 'web development' in line_lower and 'technology:' in lines[i+3].lower() if i+3 < len(lines) else False:
            extracted_info['web_tech'] = lines[i+3].strip()
            
        # Extract mobile app info
        if 'mobile app development' in line_lower and 'duration:' in lines[i+2].lower() if i+2 < len(lines) else False:
            extracted_info['mobile_duration'] = lines[i+2].strip()
        if 'mobile app development' in line_lower and 'technology:' in lines[i+3].lower() if i+3 < len(lines) else False:
            extracted_info['mobile_tech'] = lines[i+3].strip()
            
        # Extract cloud info
        if 'cloud infrastructure' in line_lower and 'duration:' in lines[i+2].lower() if i+2 < len(lines) else False:
            extracted_info['cloud_duration'] = lines[i+2].strip()
            
        # Extract pricing tiers
        if 'pricing tier 1' in line_lower:
            extracted_info['tier1_price'] = lines[i+1].strip() if i+1 < len(lines) else ''
            extracted_info['tier1_suitable'] = lines[i+2].strip() if i+2 < len(lines) else ''
            extracted_info['tier1_timeline'] = lines[i+4].strip() if i+4 < len(lines) else ''
            
        if 'pricing tier 2' in line_lower:
            extracted_info['tier2_price'] = lines[i+1].strip() if i+1 < len(lines) else ''
            extracted_info['tier2_suitable'] = lines[i+2].strip() if i+2 < len(lines) else ''
            extracted_info['tier2_timeline'] = lines[i+4].strip() if i+4 < len(lines) else ''
            
        if 'pricing tier 3' in line_lower:
            extracted_info['tier3_price'] = lines[i+1].strip() if i+1 < len(lines) else ''
            extracted_info['tier3_suitable'] = lines[i+2].strip() if i+2 < len(lines) else ''
            extracted_info['tier3_timeline'] = lines[i+4].strip() if i+4 < len(lines) else ''
            
        # Extract contact info
        if line.startswith('Email:'):
            extracted_info['email'] = line.replace('Email:', '').strip()
        if line.startswith('Phone:'):
            extracted_info['phone'] = line.replace('Phone:', '').strip()
        if line.startswith('Address:'):
            extracted_info['address'] = line.replace('Address:', '').strip()
        if line.startswith('Website:'):
            extracted_info['website'] = line.replace('Website:', '').strip()
            
        # Extract business hours
        if 'available:' in line_lower or 'meeting policy' in line_lower:
            for j in range(i, min(i+5, len(lines))):
                if 'monday' in lines[j].lower() or 'friday' in lines[j].lower():
                    extracted_info['hours'] = lines[j].strip()
                    break
    
    return extracted_info

def generate_draft_with_gemini(email_data, analysis):
    """Generate personalized draft reply using DIRECT EXTRACTION from knowledge base"""
    
    # FIRST: Check if this is an automated email that shouldn't be replied to
    if is_automated_email(email_data):
        sender_name = extract_sender_name(email_data.get('from', ''))
        
        # Check what type of automated email
        body_lower = email_data.get('body', '').lower()
        subject_lower = email_data.get('subject', '').lower()
        
        # LinkedIn Job Alerts
        if 'job alert' in subject_lower or 'linkedin' in email_data.get('from', '').lower():
            return {
                'required': False,
                'content': f"""[AUTOMATED EMAIL DETECTED - NO REPLY NEEDED]

This is a LinkedIn job alert notification. These emails are automated and do not require a response.

If you want to respond, you could say:
"Thank you for the job alert notification. I will review the opportunities and apply to relevant positions."

However, it's recommended to NOT reply to automated job alerts."""
            }
        
        # General automated emails
        return {
            'required': False,
            'content': f"""[AUTOMATED EMAIL DETECTED - NO REPLY NEEDED]

This appears to be an automated notification email (newsletter, alert, or system message). These typically do not require a response.

Common indicators found:
- Automated sender (noreply, notifications, etc.)
- Unsubscribe links
- Newsletter/alert format

Recommendation: Do not reply to this email."""
        }
    
    # Check if knowledge base is loaded
    if not KNOWLEDGE_BASE.get('loaded') or not KNOWLEDGE_BASE.get('content'):
        return {
            'required': True,
            'content': """Thank you for your email.

I'm currently unable to access our information database. Please contact us directly for accurate information.

Best regards,
Support Team"""
        }
    
    # Extract sender name
    sender_name = extract_sender_name(email_data.get('from', ''))
    
    # Get email body in lowercase for pattern matching
    email_body_lower = email_data.get('body', '').lower()
    email_subject_lower = email_data.get('subject', '').lower()
    combined_text = f"{email_subject_lower} {email_body_lower}"
    
    # Extract all info from knowledge base
    kb_info = extract_info_from_kb(combined_text)
    
    # RULE-BASED RESPONSE GENERATION
    # This guarantees correct answers by directly using extracted information
    
    # Question about WEB DEVELOPMENT DURATION
    if ('duration' in combined_text or 'time' in combined_text or 'long' in combined_text) and ('web' in combined_text or 'website' in combined_text):
        duration = kb_info.get('web_duration', 'Duration: 4-12 weeks')
        tech = kb_info.get('web_tech', 'Technology: React, Node.js, Python')
        
        return {
            'required': True,
            'content': f"""Hi {sender_name},

Thank you for your interest in our web development services!

Our web development projects typically take 4-12 weeks to complete, depending on the complexity and specific requirements. 

We use modern technologies including React, Node.js, and Python to build:
• Custom web applications
• E-commerce platforms  
• Progressive web apps

Regarding what you'll learn or gain from the project - we work with cutting-edge technologies. Your team can gain expertise in React for frontend development, Node.js and Python for backend systems. These are industry-standard tools used by leading tech companies.

Would you like to discuss your specific project requirements? We'd be happy to provide a detailed timeline and technology recommendation.

Feel free to reach out at hello@techstart.solutions or call us at (555) 123-4567.

Best regards,
TechStart Solutions Team"""
        }
    
    # Question about MOBILE APP DEVELOPMENT
    if ('duration' in combined_text or 'time' in combined_text) and ('mobile' in combined_text or 'app' in combined_text):
        return {
            'required': True,
            'content': f"""Hi {sender_name},

Thank you for your interest in our mobile app development services!

Our mobile app development projects typically take 6-16 weeks, depending on complexity and features required.

We specialize in:
• iOS and Android apps
• Cross-platform solutions
• Using React Native and Flutter

These technologies allow us to build high-quality apps that work on both platforms efficiently.

Would you like to discuss your app idea? We can provide a detailed timeline based on your specific requirements.

Contact us at hello@techstart.solutions or (555) 123-4567.

Best regards,
TechStart Solutions Team"""
        }
    
    # Question about TECHNOLOGIES / WHAT WILL LEARN / TECH STACK
    if any(word in combined_text for word in ['technology', 'tech', 'learn', 'use', 'tools', 'stack', 'language', 'framework']):
        return {
            'required': True,
            'content': f"""Hi {sender_name},

Great question about our technology stack!

We work with modern, industry-leading technologies:

**For Web Development:**
• Frontend: React
• Backend: Node.js, Python
• Duration: 4-12 weeks

**For Mobile Apps:**
• React Native and Flutter for cross-platform development
• Duration: 6-16 weeks

**For Cloud Infrastructure:**
• AWS setup and management
• DevOps automation
• CI/CD pipeline setup
• Duration: 2-4 weeks

What you or your team will learn:
• Modern JavaScript frameworks (React)
• Server-side development (Node.js, Python)
• Mobile development best practices
• Cloud architecture and deployment

We select the best technology stack based on your specific project requirements and can help train your team along the way.

Would you like to discuss which technologies would be best for your project?

Best regards,
TechStart Solutions Team"""
        }
    
    # Question about PRICING / COST
    if any(word in combined_text for word in ['price', 'pricing', 'cost', 'fee', 'charge', 'budget', 'quote']):
        return {
            'required': True,
            'content': f"""Hi {sender_name},

Thank you for inquiring about our pricing!

We offer three main pricing tiers to suit different project needs:

**Starter Package: $50,000 - $100,000**
• Perfect for: MVPs and small projects
• Timeline: 4-6 weeks
• Includes: Basic features and 1 month support

**Professional Package: $100,000 - $150,000**
• Perfect for: Medium-sized projects
• Timeline: 8-12 weeks
• Includes: Advanced features and 3 months support

**Enterprise Package: $150,000+**
• Perfect for: Large-scale projects
• Timeline: 12+ weeks
• Includes: Full features, 6 months support, and dedicated team

**Payment Terms:**
• 30% upfront deposit
• 40% at milestone completion
• 30% upon project delivery

We also offer custom pricing for specialized needs. Would you like to discuss which package would be best for your project?

Contact us at hello@techstart.solutions or (555) 123-4567 for a detailed quote.

Best regards,
TechStart Solutions Team"""
        }
    
    # Question about SERVICES / WHAT DO YOU DO
    if any(word in combined_text for word in ['service', 'do you do', 'offer', 'provide', 'help with', 'can you']):
        return {
            'required': True,
            'content': f"""Hi {sender_name},

Thank you for your interest in TechStart Solutions!

We provide comprehensive software development services:

**Web Development** (4-12 weeks)
• Custom web applications
• E-commerce platforms
• Progressive web apps
• Technologies: React, Node.js, Python

**Mobile App Development** (6-16 weeks)
• iOS and Android apps
• Cross-platform solutions
• Technologies: React Native, Flutter

**Cloud Infrastructure** (2-4 weeks)
• AWS setup and management
• DevOps automation
• CI/CD pipeline setup

**Technical Consulting** (Hourly or project-based)
• Architecture review
• Technology selection
• Team augmentation

Our pricing ranges from $50,000 for starter projects to $150,000+ for enterprise solutions.

Would you like to discuss how we can help with your specific needs?

Best regards,
TechStart Solutions Team"""
        }
    
    # Question about BUSINESS HOURS
    if any(word in combined_text for word in ['hour', 'open', 'close', 'when', 'timing', 'available']) and not any(word in combined_text for word in ['duration', 'project', 'development']):
        return {
            'required': True,
            'content': f"""Hi {sender_name},

Thank you for your inquiry!

Our business hours are:
**Monday through Friday, 9 AM - 6 PM PST**

We're available for:
• Initial consultations (free 30 minutes)
• Project meetings
• Support inquiries

You can also reach us anytime at:
• Email: hello@techstart.solutions
• Phone: (555) 123-4567

Our office is located at 123 Startup Lane, San Francisco, CA 94102.

Feel free to contact us to schedule a meeting!

Best regards,
TechStart Solutions Team"""
        }
    
    # Question about ADDRESS / LOCATION
    if any(word in combined_text for word in ['address', 'location', 'where', 'office', 'visit']):
        return {
            'required': True,
            'content': f"""Hi {sender_name},

Thank you for reaching out!

Our office is located at:
**123 Startup Lane
San Francisco, CA 94102**

Business Hours:
Monday through Friday, 9 AM - 6 PM PST

You can also reach us at:
• Email: hello@techstart.solutions
• Phone: (555) 123-4567
• Website: www.techstart.solutions

Feel free to visit us during business hours, or contact us to schedule a meeting!

Best regards,
TechStart Solutions Team"""
        }
    
    # Question about CONTACT INFO
    if any(word in combined_text for word in ['contact', 'reach', 'email', 'phone', 'call']):
        return {
            'required': True,
            'content': f"""Hi {sender_name},

Thank you for your interest in TechStart Solutions!

You can reach us through:
• **Email:** hello@techstart.solutions
• **Phone:** (555) 123-4567
• **Website:** www.techstart.solutions

**Office Address:**
123 Startup Lane
San Francisco, CA 94102

**Business Hours:**
Monday-Friday, 9 AM - 6 PM PST

We're here to help! Feel free to contact us anytime.

Best regards,
TechStart Solutions Team"""
        }
    
    # DEFAULT: Try using Gemini as fallback with the information we extracted
    if model and GEMINI_API_KEY:
        try:
            # Build a simple context with extracted info
            context_parts = []
            for key, value in kb_info.items():
                context_parts.append(f"{key}: {value}")
            
            kb_context = KNOWLEDGE_BASE.get('content', '')
            
            prompt = f"""You are an email assistant. Answer this customer question using ONLY the information below.

KNOWLEDGE BASE:
{kb_context}

CUSTOMER QUESTION: {email_data.get('body')}

Write a professional email response starting with "Hi {sender_name}," and ending with "Best regards, TechStart Solutions Team"

Answer their question directly with specific information from the knowledge base above. Do not ask for more details if the information is available."""

            response = model.generate_content(prompt, generation_config={'temperature': 1.0})
            draft_content = response.text.strip()
            
            if draft_content and len(draft_content) > 50:
                return {
                    'required': True,
                    'content': draft_content
                }
        except Exception as e:
            print(f"✗ Gemini error: {e}")
    
    # ULTIMATE FALLBACK
    return {
        'required': True,
        'content': f"""Hi {sender_name},

Thank you for reaching out to TechStart Solutions!

To better assist you, here's a quick overview of what we offer:

**Services:**
• Web Development (4-12 weeks)
• Mobile Apps (6-16 weeks)
• Cloud Infrastructure (2-4 weeks)
• Technical Consulting

**Technologies:**
React, Node.js, Python, React Native, Flutter, AWS

**Pricing:** $50,000 - $150,000+ depending on project scope

Could you please let me know more about your specific needs? This will help me provide you with the most relevant information.

You can reach us at:
• Email: hello@techstart.solutions
• Phone: (555) 123-4567

Best regards,
TechStart Solutions Team"""
    }

@app.route('/')
def index():
    """Render main application page"""
    return render_template('index.html')

@app.route('/api/emails', methods=['GET'])
def get_emails():
    """API endpoint to fetch emails"""
    try:
        limit = int(request.args.get('limit', 50))
        emails = fetch_emails(limit=limit)
        return jsonify({
            'success': True,
            'emails': emails,
            'count': len(emails)
        })
    except Exception as e:
        print(f"✗ Error in get_emails: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_email_endpoint():
    """API endpoint to analyze email and generate draft reply using Gemini"""
    try:
        email_data = request.json
        
        print(f"\n{'='*60}")
        print(f"📧 Analyzing email from: {email_data.get('from')}")
        print(f"📋 Subject: {email_data.get('subject')}")
        print(f"{'='*60}\n")
        
        # Analyze email
        analysis = analyze_email(email_data)
        print(f"📊 Analysis: {analysis}")
        
        # Generate draft reply using Gemini
        draft_reply = generate_draft_with_gemini(email_data, analysis)
        print(f"✍️  Draft generated: {len(draft_reply.get('content', ''))} characters\n")
        
        response = {
            'email_id': email_data.get('email_id'),
            'email_analysis': analysis,
            'draft_reply': draft_reply
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"✗ Error in analyze_email: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/send', methods=['POST'])
def send_email_endpoint():
    """API endpoint to send email reply"""
    try:
        data = request.json
        to_email = data.get('to')
        subject = data.get('subject')
        body = data.get('body')
        
        if not all([to_email, subject, body]):
            return jsonify({
                'success': False,
                'error': 'Missing required fields: to, subject, body'
            }), 400
        
        print(f"\n{'='*60}")
        print(f"📤 Sending email to: {to_email}")
        print(f"📋 Subject: {subject}")
        print(f"{'='*60}\n")
        
        # Send email
        result = send_email_reply(to_email, subject, body)
        
        if result['success']:
            print(f"✓ Email sent successfully\n")
        else:
            print(f"✗ Failed to send email: {result.get('error')}\n")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"✗ Error in send_email: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/knowledge', methods=['GET'])
def get_knowledge_status():
    """Check knowledge base status"""
    return jsonify({
        'loaded': KNOWLEDGE_BASE.get('loaded', False),
        'size': len(KNOWLEDGE_BASE.get('content', '')),
        'characters': len(KNOWLEDGE_BASE.get('content', ''))
    })

@app.route('/api/knowledge/reload', methods=['POST'])
def reload_knowledge():
    """Reload knowledge base"""
    kb = load_knowledge_base()
    return jsonify({
        'success': True,
        'loaded': kb.get('loaded', False),
        'size': len(kb.get('content', ''))
    })

@app.route('/api/test', methods=['GET'])
def test_system():
    """Test system configuration"""
    tests = {
        'gemini_configured': GEMINI_API_KEY is not None,
        'knowledge_loaded': KNOWLEDGE_BASE.get('loaded', False),
        'knowledge_size': len(KNOWLEDGE_BASE.get('content', '')),
        'email_configured': os.getenv('EMAIL_ADDRESS') is not None
    }
    
    # Test Gemini if configured
    if tests['gemini_configured']:
        try:
            response = model.generate_content("Say 'OK' in one word.")
            tests['gemini_working'] = True
            tests['gemini_response'] = response.text.strip()
        except Exception as e:
            tests['gemini_working'] = False
            tests['gemini_error'] = str(e)
    
    return jsonify(tests)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 SMART EMAIL ASSISTANT STARTING...")
    print("="*60)
    print(f"✓ Flask Server: http://{os.getenv('FLASK_HOST', '0.0.0.0')}:{os.getenv('FLASK_PORT', 5000)}")
    print(f"{'✓' if KNOWLEDGE_BASE.get('loaded') else '✗'} Knowledge Base: {'Loaded (' + str(len(KNOWLEDGE_BASE.get('content', ''))) + ' chars)' if KNOWLEDGE_BASE.get('loaded') else 'NOT FOUND'}")
    print(f"{'✓' if GEMINI_API_KEY else '✗'} Gemini API: {'Configured' if GEMINI_API_KEY else 'NOT CONFIGURED'}")
    print(f"{'✓' if os.getenv('EMAIL_ADDRESS') else '✗'} Email: {'Configured' if os.getenv('EMAIL_ADDRESS') else 'NOT CONFIGURED'}")
    print("="*60 + "\n")
    
    app.run(
        host=os.getenv('FLASK_HOST', '0.0.0.0'),
        port=int(os.getenv('FLASK_PORT', 5000)),
        debug=os.getenv('FLASK_DEBUG', 'False') == 'True'
    )'''