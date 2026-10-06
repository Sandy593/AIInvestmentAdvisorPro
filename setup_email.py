#!/usr/bin/env python3
"""
Email Setup Script for Stock Research System
This script helps you configure email credentials and test the setup.
"""

import os
import getpass
from main import NotificationTool

def setup_email_credentials():
    """Interactive setup for email credentials"""
    print("📧 Email Configuration Setup")
    print("=" * 50)
    print()
    
    print("This script will help you set up email notifications.")
    print("You'll need:")
    print("1. A Gmail account with 2-factor authentication enabled")
    print("2. An App Password (not your regular password)")
    print()
    
    # Get email credentials
    sender_email = input("Enter your Gmail address: ").strip()
    
    print("\n⚠️  IMPORTANT: Use an App Password, not your regular Gmail password!")
    print("To get an App Password:")
    print("1. Go to: https://myaccount.google.com/apppasswords")
    print("2. Select 'Mail' and click 'Generate'")
    print("3. Copy the 16-character password")
    print()
    
    sender_password = getpass.getpass("Enter your Gmail App Password: ").strip()
    recipient_email = input("Enter recipient email address: ").strip()
    
    # Validate inputs
    if not all([sender_email, sender_password, recipient_email]):
        print("❌ All fields are required!")
        return False
    
    if '@gmail.com' not in sender_email.lower():
        print("⚠️  Warning: This script is configured for Gmail. Other providers may need different settings.")
    
    # Set environment variables
    os.environ['SENDER_EMAIL'] = sender_email
    os.environ['SENDER_PASSWORD'] = sender_password
    os.environ['RECIPIENT_EMAIL'] = recipient_email
    
    print(f"\n✅ Email credentials set:")
    print(f"   Sender: {sender_email}")
    print(f"   Recipient: {recipient_email}")
    print(f"   Password: {'*' * len(sender_password)}")
    
    return True

def test_email_configuration():
    """Test the email configuration"""
    print("\n🧪 Testing Email Configuration")
    print("=" * 40)
    
    # Check if credentials are set
    sender_email = os.getenv('SENDER_EMAIL')
    sender_password = os.getenv('SENDER_PASSWORD')
    recipient_email = os.getenv('RECIPIENT_EMAIL')
    
    if not all([sender_email, sender_password, recipient_email]):
        print("❌ Email credentials not configured!")
        return False
    
    # Create notification tool and test
    notification_tool = NotificationTool()
    
    test_message = """This is a test email from the Stock Research System.

If you receive this email, your configuration is working correctly!

Test Details:
- Sender: {sender}
- Recipient: {recipient}
- Timestamp: {timestamp}

You can now receive stock research reports via email.
""".format(
        sender=sender_email,
        recipient=recipient_email,
        timestamp=os.popen('date').read().strip()
    )
    
    print("📧 Sending test email...")
    result = notification_tool._run(
        message=test_message,
        method="email",
        subject="Test Email - Stock Research System"
    )
    
    print(f"\n📤 Result: {result}")
    
    if "✅ EMAIL SUCCESS" in result:
        print("\n🎉 Email configuration is working!")
        print("You can now run the stock research system and receive email notifications.")
        return True
    else:
        print("\n❌ Email configuration failed!")
        print("Please check:")
        print("1. Gmail 2-factor authentication is enabled")
        print("2. You're using an App Password (not regular password)")
        print("3. Email addresses are correct")
        return False

def save_to_env_file():
    """Save credentials to .env file for persistence"""
    sender_email = os.getenv('SENDER_EMAIL')
    sender_password = os.getenv('SENDER_PASSWORD')
    recipient_email = os.getenv('RECIPIENT_EMAIL')
    
    if not all([sender_email, sender_password, recipient_email]):
        print("❌ Cannot save to .env file - credentials not set")
        return False
    
    env_content = f"""# Email Configuration
SENDER_EMAIL={sender_email}
SENDER_PASSWORD={sender_password}
RECIPIENT_EMAIL={recipient_email}

# Optional SMTP Settings (Gmail defaults)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ Credentials saved to .env file")
    print("⚠️  Remember to add .env to your .gitignore file for security!")
    return True

def main():
    """Main setup function"""
    print("🚀 Stock Research Email Setup")
    print("=" * 50)
    
    # Setup credentials
    if not setup_email_credentials():
        return
    
    # Test configuration
    if test_email_configuration():
        # Save to .env file
        save_to_env_file()
        
        print("\n✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run: python main.py")
        print("2. Select option 1 to run stock research")
        print("3. You'll receive email notifications with the results")
    else:
        print("\n❌ Setup failed. Please check your credentials and try again.")

if __name__ == "__main__":
    main() 