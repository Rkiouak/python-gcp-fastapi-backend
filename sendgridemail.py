import sendgrid
import os
from sendgrid.helpers.mail import *
from google.cloud import firestore
from google.cloud.firestore_v1 import FieldFilter
import secretmanager
from domain.user import SignUpUser


def get_to_emails():
    db = firestore.Client()
    users_ref = db.collection("users")
    queried_users = users_ref.where(filter=FieldFilter("username", "==", username)).get()
    return [user.email for user in queried_users.to_dict()]

def send_email(user: SignUpUser, challenge: str):
    message = Mail(
        from_email='matt@rkiouak.com',
        to_emails=user.email,
        subject='Complete your musings-mr.net sign up',
        html_content=f'<strong>Hello, {user.given_name},</strong>\n To complete your musings-mr.net signup, please visit https://musings-mr.net/users/validate?challenge={challenge}'
    )
    try:
        sg = sendgrid.SendGridAPIClient(secretmanager.get_secret("projects/4042672389/secrets/sendgrid-api-key/versions/latest"))
        response = sg.send(message)
        print(f"Status Code: {response.status_code}")
        print(f"Body: {response.body}")
        print(f"Headers: {response.headers}")
    except Exception as e:
        print(f"Error: {e}")