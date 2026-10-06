# Email Verification Guide

This guide explains how to verify if emails were sent successfully from the Stock Research System.

## Ways to Verify Email Sending

### 1. **Real-time Console Output**
When emails are sent, you'll see immediate feedback in the console:
```
📧 Email sent successfully to recipient@example.com
```
or
```
❌ Error sending email: [error details]
```

### 2. **Check Email Status in Database**
The system tracks all email attempts in a database. You can check the status using:

```bash
# Method 1: Command line argument
python main.py check-emails

# Method 2: Interactive menu option
python main.py
# Then select option 3: "check email status"

# Method 3: Test script
python test_email.py
```

This will show you a table like:
```
📧 Recent Email Notifications:
--------------------------------------------------------------------------------
✅ 2024-01-15 14:30:25 | recipient@example.com | Stock Research Report - January 2024
❌ 2024-01-15 14:25:10 | recipient@example.com | Test Email | Error: Authentication failed
--------------------------------------------------------------------------------
```

### 3. **Check Your Email Inbox**
The most direct verification method is to check the recipient's email inbox (including spam folder).

### 4. **Test Email Functionality**
Run the test script to verify email configuration:
```bash
python test_email.py
```

This will:
- Check if environment variables are configured
- Send a test email
- Show the result
- Display recent email status

## Email Configuration

### Required Environment Variables
```bash
export SENDER_EMAIL='your-email@gmail.com'
export SENDER_PASSWORD='your-app-password'  # Gmail App Password
export RECIPIENT_EMAIL='recipient@example.com'
```

### Optional Environment Variables
```bash
export SMTP_SERVER='smtp.gmail.com'  # Default: smtp.gmail.com
export SMTP_PORT='587'               # Default: 587
```

## Troubleshooting

### Common Issues

1. **"Email credentials not configured"**
   - Set the required environment variables
   - Use Gmail App Password, not regular password

2. **"Authentication failed"**
   - Check if SENDER_PASSWORD is correct
   - Ensure you're using an App Password for Gmail
   - Verify 2-factor authentication is enabled

3. **"Connection timeout"**
   - Check internet connection
   - Verify SMTP server and port settings

4. **Email not received**
   - Check spam/junk folder
   - Verify RECIPIENT_EMAIL is correct
   - Check email provider's security settings

### Gmail Setup
1. Enable 2-factor authentication
2. Generate an App Password:
   - Go to Google Account settings
   - Security → 2-Step Verification → App passwords
   - Generate password for "Mail"
3. Use the generated password as SENDER_PASSWORD

## Database Schema

The system tracks emails in the `email_notifications` table:
```sql
CREATE TABLE email_notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sent_date DATETIME,
    recipient_email TEXT,
    subject TEXT,
    message_preview TEXT,
    status TEXT,           -- 'success' or 'failed'
    error_message TEXT
);
```

## Logging

The system logs email activities to:
- Console output (with emojis for visibility)
- Python logging system
- SQLite database

Check logs for detailed error information if emails fail to send. 