#!/usr/bin/env python3
"""
Test script to verify email functionality and check email status
"""

import os
import sys
from main import NotificationTool, check_email_status, init_database

def test_email_sending():
    """Test email sending functionality"""
    print("🧪 Testing Email Functionality")
    print("=" * 40)
    
    # Initialize database
    init_database()
    
    # Create notification tool
    notification_tool = NotificationTool()
    
    # Test message
    test_message = "This is a test email from the Stock Research System. If you receive this, email notifications are working correctly!"
    test_subject = "Test Email - Stock Research System"
    
    print(f"📧 Sending test email...")
    print(f"   Subject: {test_subject}")
    print(f"   Message: {test_message[:50]}...")
    
    # Send test email
    result = notification_tool._run(
        message=test_message,
        method="email",
        subject=test_subject
    )
    
    print(f"\n📤 Result: {result}")
    
    # Check email status
    print("\n📊 Checking email status...")
    check_email_status()

def check_environment():
    """Check if email environment variables are configured"""
    print("🔧 Environment Check")
    print("=" * 40)
    
    required_vars = ['SENDER_EMAIL', 'SENDER_PASSWORD', 'RECIPIENT_EMAIL']
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Mask password for security
            if 'PASSWORD' in var:
                masked_value = '*' * len(value) if len(value) > 4 else '****'
                print(f"✅ {var}: {masked_value}")
            else:
                print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: Not set")
    
    print()
    
    # Check if all required vars are set
    all_set = all(os.getenv(var) for var in required_vars)
    if all_set:
        print("✅ All email environment variables are configured!")
        return True
    else:
        print("❌ Some email environment variables are missing.")
        print("\nTo configure email notifications, set these environment variables:")
        print("export SENDER_EMAIL='your-email@gmail.com'")
        print("export SENDER_PASSWORD='your-app-password'")
        print("export RECIPIENT_EMAIL='recipient@example.com'")
        return False

if __name__ == "__main__":
    print("🚀 Email Testing and Verification Tool")
    print("=" * 50)
    
    # Check environment first
    if not check_environment():
        print("\n⚠️  Please configure email environment variables before testing.")
        sys.exit(1)
    
    # Test email sending
    test_email_sending()
    
    print("\n✅ Email testing completed!")
    print("\nTo check email status later, run:")
    print("python main.py check-emails")
    print("or")
    print("python test_email.py") 