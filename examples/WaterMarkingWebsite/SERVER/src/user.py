import datetime

from passlib.apps import custom_app_context as pwd_context
import jwt
from django.conf import settings


class UserMongo:

    def __init__(self):

        self.db = settings.USER

    def __getitem__(self, email):

        user = self.db.find_one({'email': email})
        if user is None:
            return {'error': 'Not Found'}

        return user

    def getByEmail(self, email):

        user = self.db.find_one({'email': email})
        if user is None:
            return {'error': 'Not Found'}

        return user

    def putUser(self, user):

        return self.db.insert_one(user)

    def delUser(self, email):

        self.db.delete_one({'email': email})

    def removeSensitiveInfo(self, email):

        user = self.db.find_one({'email': email})
        if user is None:
            return {'error': 'Not Found'}

        del user['passwordHash']
        del user['tokenInitTime']
        del user['_id']

        return user

    def checkAdmin(self, email):
        user = self.__getitem__(email)
        return user['role'] == 'admin'


userDB = UserMongo()


def hash_password(password):
    return pwd_context.encrypt(password)


def verify_password(password, email):
    user = userDB[email]
    if 'error' in user:
        return None
    password_hash = user['passwordHash']
    return pwd_context.verify(password, password_hash)


def verify_token(token, email):
    user = userDB[email]

    if user is None:
        return None

    check = verify_auth_token(token)

    if check is None:
        return None

    if email == check['email']:
        return True

    return None


def generate_auth_token(email):
    return jwt.encode({'email': email, 'exp': datetime.datetime.utcnow() + datetime.timedelta(seconds=100 * 24 * 60 * 60)},
                      settings.SECRET_KEY,
                      algorithm="HS256")


def verify_auth_token(token):
    try:
        data = jwt.decode(token, settings.SECRET_KEY, algorithms='HS256')
        user = userDB[data['email']]
        if 'error' in user:
            return None
    except jwt.ExpiredSignatureError:
        print('exp')
        return None
    except jwt.InvalidTokenError:
        print('inv')
        return None
    except:
        return None

    return {'email': data['email']}


def get_user_by_token(token):
    try:
        data = jwt.decode(token, settings.SECRET_KEY, algorithms='HS256')
        db_user = userDB[data['email']]
        if 'error' in db_user:
            print('email does not exist', data['email'])
            return None
    except jwt.ExpiredSignatureError:
        print('exp')
        return None
    except jwt.InvalidTokenError:
        print('inv')
        return None

    return db_user
