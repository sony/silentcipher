from django.http import HttpResponse
from functools import wraps
from django.conf import settings
import json
import random
import os
from src import user

def authenticate(to_authenticate_fn):
	
	@wraps(to_authenticate_fn)
	def inner(request, *args, **kwargs):
		
		email = request.headers.get('email', None)
		token = request.headers.get('token', None)

		if token is not None:
			
			if user.verify_token(token, email):
				return to_authenticate_fn(request, *args, **kwargs, email=email, token=token)
			else:
				print('user verification failed', token)
				return HttpResponse(status=401)
		else:
			return HttpResponse(status=401)
	
	return inner


def create_user(email):
	# This function is called when creating a new user. The combinations and random seed are fixed over here.

	tasks = 'files' + '/tasks.json'
	with open(tasks, 'r') as f:
		tasks = json.load(f)
		userSeed = random.randint(0, 9)
		settings.USER.insert_one(
			{
				'email': email,
				'tasks': {taskI['name']: get_empty_task(taskI['name'], userSeed) for taskI in tasks},
				'seed': userSeed
			}
		)

	return True
