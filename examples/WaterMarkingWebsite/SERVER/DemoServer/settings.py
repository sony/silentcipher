"""
Django settings for DemoServer project.

Generated by 'django-admin startproject' using Django 3.1.4.

For more information on this file, see
https://docs.djangoproject.com/en/3.1/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/3.1/ref/settings/
"""

import json
from pathlib import Path
import datetime
from corsheaders.defaults import default_headers
# Pymongo Collections
from pymongo import MongoClient

with open('../config.json', 'r') as f:
    config = json.load(f)

client = MongoClient()

name = config['name']
db = client[name]

USER = db['USER']
PROJECT = db['PROJECT']
DECODE = db['DECODE']
MANIPULATE = db['MANIPULATE']
FILE_UPLOAD_DIR = config['FILE_UPLOAD_DIR']
DECODE_UPLOAD_DIR = config['DECODE_UPLOAD_DIR']
FILE_UPLOAD_DIR_REL_MODEL_SERVER = config['FILE_UPLOAD_DIR_REL_MODEL_SERVER']
DECODE_UPLOAD_DIR_REL_MODEL_SERVER = config['DECODE_UPLOAD_DIR_REL_MODEL_SERVER']

ALLOWED_EXTENSIONS = [
    'wav', 'mp3', 'aac', 'ogg', 'flac', 'alac', 'aiff', 'dsd', 'pcm',
    'MP4', 'MOV', 'WMV', 'AVI', 'AVCHD', 'FLV', 'F4V', 'SWF', 'MKV', 'WEBM'
]
ENCODE_URL = 'http://127.0.0.1:8001/encode'
DECODE_URL = 'http://127.0.0.1:8001/decode'

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

def addUser(email, password=None, name=None):
	
	existingUser = USER.find_one({'email': email})
	if existingUser:
		return None, False
	
	date = datetime.datetime.now().strftime("%d-%m-%y-%H-%M-%S")
	
	if name is not None:
		name = name.title()
	
	req_user = {
		'email': email,
		'password': password,
		'name': name,
		'date': date,
		'projects': []
	}
	USER.insert_one(req_user)
	
	return req_user, True

for admin_user in config['admin_list']:
    addUser(admin_user['email'], admin_user['password'], admin_user['name'])

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/3.1/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'qwerasdfqwerasdfqwerasdfasdfasdf'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS=['localhost', '127.0.01', config['host']['ip']]

# Application definition

DEFAULT_AUTO_FIELD = 'django.db.models.AutoField'


INSTALLED_APPS = [
    'corsheaders',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'DemoServer.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'DemoServer.wsgi.application'


# Database
# https://docs.djangoproject.com/en/3.1/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}


# Password validation
# https://docs.djangoproject.com/en/3.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/3.1/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/3.1/howto/static-files/

STATIC_URL = '/files/'
STATICFILES_DIRS = [
    BASE_DIR / 'files'
]

# MOS
MOS = {
    'numSamples': 8
}

# SPKIDEN
SPKIDEN = {
    'numSamples': 10
}

# ABX
ABX = {
    'numSamples': 8
}
# qualityComp
QUALITYCOMP = {
    'numSamples': 10
}


CORS_ORIGIN_ALLOW_ALL = True
CORS_ALLOW_HEADERS = list(default_headers) + [
    'email', 'token'
]