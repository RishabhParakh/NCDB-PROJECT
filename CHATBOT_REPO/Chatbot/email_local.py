import smtplib
def local_run_email_script(test, sender_email, sender_password, recipient_email, subject, message):
  """Sends an email using the SMTP protocol.

  Args:
    sender_email: The sender's email address.
    sender_password: The sender's email password.
    recipient_email: The recipient's email address.
    subject: The email subject.
    message: The email message body.
  """

  smtp_server = smtplib.SMTP('smtp.gmail.com', 587)
  smtp_server.ehlo()
  smtp_server.starttls()
  smtp_server.login(sender_email, sender_password)

  email_message = 'Subject: {}\n\n{}'.format(subject, message)

  smtp_server.sendmail(sender_email, recipient_email, email_message)
  smtp_server.quit()

if __name__ == '__main__':
  sender_email = 'orliroot@gmail.com'
  sender_password = 'eieowacdfvhgeaag'
#   recipient_email = 'ho2103@nyu.com'
  recipient_email = '178159819@qq.com'
  subject = 'This is a test email!'
  message = 'This is the body of the test email.'

  local_run_email_script(False, sender_email, sender_password, recipient_email, subject, message)



