import email
import getpass
import imaplib

detach_dir = '.' # directory where to save attachments (default current)
#user = raw_input('Enter youe Gmail username: ')
user = '670information'
#pwd = getpass.getpass('Enter your password: ')
pwd = 'bingorgoogle'

# connecting to the gmail imap server
m = imaplib.IMAP4_SSL('imap.gmail.com')
m.login(user, pwd)

# here you a can choose a mail box like INBOX instead
#m.select('[Gmail]/All Mail')
m.select('inbox') 
# use m.list() to get all the mailboxes

''' you could filter using the IMAP rules here
(check http://www.example-code.com/csharp/imap-search-critera.asp)
'''
resp, items = m.search(None, 'ALL') 
items = items[0].split() # getting the mails id

mails = []

emails = []
for emailid in items:
    ''' fetching the mail,`(RFC822)` means get the whole stuff,
    but you can ask for headers only, etc
    '''
    resp, data = m.fetch(emailid, '(RFC822)') 
    email_body = data[0][1] # getting the mail content
    # parsing the mail content to get a mail object
    mail = email.message_from_string(email_body)
    
    curr_mail = {}
    curr_mail['From'] = mail['From']
    curr_mail['Date'] = mail['Date']
    curr_mail['Subject'] = mail['Subject']
    curr_mail['Content'] = mail.get_payload()

    emails.append(curr_mail)
