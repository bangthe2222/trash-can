import smtplib, ssl

port = 465  # For SSL
smtp_server = "smtp.gmail.com"
sender_email = "decive.rasppi4@gmail.com"  # Enter your address
receiver_email = "milayduoc@gmail.com"  # Enter receiver address
password = "raspberrypi4"
message = """\
Subject: Hi there\n"""+"\ntest"+"\nno"
context = ssl.create_default_context()
with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
    server.login(sender_email, password)
    server.sendmail(sender_email, receiver_email, message)