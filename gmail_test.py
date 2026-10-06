#!/usr/bin/env python3
"""
Gmail Authentication Test Script
Run this to verify your email credentials before running the main stock research system
"""

import os
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv

def test_gmail_connection():
    """Test Gmail SMTP connection and authentication"""
    
    # Load environment variables
    load_dotenv()
    
    # Get credentials
    sender_email = os.getenv('SENDER_EMAIL')
    sender_password = os.getenv('SENDER_PASSWORD')
    recipient_email = os.getenv('RECIPIENT_EMAIL')
    
    print("🔧 Gmail Authentication Test")
    print("=" * 50)
    
    # Check if credentials exist
    if not sender_email:
        print("❌ SENDER_EMAIL not found in .env file")
        return False
    
    if not sender_password:
        print("❌ SENDER_PASSWORD not found in .env file")
        return False
        
    if not recipient_email:
        print("❌ RECIPIENT_EMAIL not found in .env file")
        return False
    
    print(f"📧 Sender: {sender_email}")
    print(f"📧 Recipient: {recipient_email}")
    print(f"🔑 Password length: {len(sender_password)} characters")
    
    # Validate password format
    if len(sender_password) != 16:
        print("⚠️  Gmail App Password should be exactly 16 characters")
        print("   Example: abcdefghijklmnop")
        print("   Generate new one at: https://myaccount.google.com/security")
    
    print("\n🔄 Testing SMTP connection...")
    
    try:
        # Test SMTP connection
        print("   Step 1: Connecting to smtp.gmail.com:587...")
        server = smtplib.SMTP('smtp.gmail.com', 587)
        
        print("   Step 2: Starting TLS encryption...")
        server.starttls()
        
        print("   Step 3: Attempting login...")
        server.login(sender_email, sender_password)
        
        print("   Step 4: Preparing test email...")
        test_message = f"""
🧪 Gmail Authentication Test

This is a test email from your stock research system.

Credentials working correctly!
- Sender: {sender_email}  
- Recipient: {recipient_email}
- Timestamp: {os.popen('date').read().strip()}

You can now run the stock research workflow with confidence!
        """.strip()
        
        msg = MIMEText(test_message, 'plain', 'utf-8')
        msg['Subject'] = 'Gmail Authentication Test - SUCCESS'
        msg['From'] = sender_email
        msg['To'] = recipient_email
        
        print("   Step 5: Sending test email...")
        server.send_message(msg)
        server.quit()
        
        print("\n✅ SUCCESS! Gmail authentication working perfectly!")
        print(f"📬 Test email sent to {recipient_email}")
        print("🚀 You can now run the stock research workflow")
        
        return True
        
    except smtplib.SMTPAuthenticationError as e:
        print(f"\n❌ AUTHENTICATION FAILED!")
        print(f"   Error: {e}")
        print("\n🔧 How to fix:")
        print("   1. Make sure 2-Factor Authentication is enabled on Gmail")
        print("   2. Generate new App Password at: https://myaccount.google.com/security")
        print("   3. Use the 16-character app password (not your regular password)")
        print("   4. Update SENDER_PASSWORD in your .env file")
        return False
        
    except smtplib.SMTPConnectError:
        print(f"\n❌ CONNECTION FAILED!")
        print("   Cannot connect to Gmail SMTP server")
        print("🔧 How to fix:")
        print("   1. Check your internet connection")
        print("   2. Try from different network (some block SMTP)")
        print("   3. Temporarily disable VPN/firewall")
        return False
        
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        print("\n🔧 Possible fixes:")
        print("   1. Check email addresses are valid")
        print("   2. Verify .env file format")
        print("   3. Try regenerating Gmail App Password")
        return False

def show_env_template():
    """Show the correct .env file format"""
    print("\n📝 Your .env file should look like this:")
    print("-" * 40)
    print("""
OPENAI_API_KEY=sk-your-openai-key-here
SENDER_EMAIL=your-email@gmail.com
SENDER_PASSWORD=abcdefghijklmnop
RECIPIENT_EMAIL=recipient@email.com
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
    """.strip())
    print("-" * 40)

if __name__ == "__main__":
    success = test_gmail_connection()
    
    if not success:
        show_env_template()
        print("\n🔗 Helpful links:")
        print("   Gmail App Passwords: https://myaccount.google.com/security")
        print("   2-Factor Auth: https://myaccount.google.com/signinoptions/two-step-verification")