from django.shortcuts import render
from django.http import JsonResponse, FileResponse
from django.conf import settings
from bson import ObjectId
import requests
import datetime
from pathlib import Path

import json
import os

from src import utils
from src import user
from src.utils import authenticate

import librosa
import soundfile as sf

def sanitize_data(data):
    if 'password' in data:
        del data['password']
    if '_id' in data:
        data['_id'] = str(data['_id'])

    return data


def create_user(request):

    if request.method == 'POST':
        req_user = json.loads(request.body)
        email = req_user.get('email')
        password = req_user.get('password')
        name = req_user.get('name')
        if req_user.get('admin_key') != 'asdfzxcvqwer':
            return JsonResponse({"status": False, "error": "Stop trying to create new users!"})
        if not email or not password or not name:
            return JsonResponse({"status": False, "error": "Please provide email, password & name"})
        
        req_user, status = utils.addUser(email, password, name, None, False, settings.USER)
        if req_user is None:
            return JsonResponse({"status": False, "error": "User already exists!"})
        
        return JsonResponse(
            {"status": True, "token": user.generate_auth_token(req_user.get('email'))}
        )


@authenticate
def get_user_data(request, **kwargs):

    if request.method == 'GET':

        user = settings.USER.find_one({'email': kwargs['email']})

        if not user:
            return JsonResponse({'status': False})

        return JsonResponse({'status': True, 'user_data': sanitize_data(user)})


@authenticate
def new_project(request, **kwargs):

    if request.method == 'POST':

        user = settings.USER.find_one({'email': kwargs['email']})
        if not user:
            return JsonResponse({'status': False})
        
        extension = request.FILES['file'].name.split('.')[-1]

        if extension not in settings.ALLOWED_EXTENSIONS:
            return JsonResponse({'status': False})
        
        _id = settings.PROJECT.insert_one(
			{
				'email': kwargs['email'],
				'name': request.POST.get('projectName'),
                'file': request.FILES['file'].name,
                'extension': extension,
                'type': request.POST.get('type'),
                'selected': 0,
                'message': ['']*5,
                'error': [False]*5,
			}
		)
        print(str(_id.inserted_id))
        _id = str(_id.inserted_id)
        
        with open(settings.FILE_UPLOAD_DIR + '/' + str(_id) + '.' + extension, 'wb') as f:
            f.write(request.FILES['file'].read())


        projects = user['projects']
        projects.append({'name': request.POST.get('projectName'), '_id': str(_id)})
        
        settings.USER.update_one({'email': kwargs['email']}, {'$set': {'projects': projects}})

        user = settings.USER.find_one({'email': kwargs['email']})
        
        return JsonResponse({'status': True, 'id': str(_id), 'user_data': sanitize_data(user)})

@authenticate
def get_project_data(request, **kwargs):

    if request.method == 'POST':
        
        data = json.loads(request.body)
        project = settings.PROJECT.find_one({'email': kwargs['email'], '_id': ObjectId(data.get('projectid'))})
        if project:
            return JsonResponse({'status': True, 'project': sanitize_data(project)})
        else:
            return JsonResponse({'status': False})
        
def files(request, **kwargs):

    if request.method == 'GET':

        test_path = (Path(settings.FILE_UPLOAD_DIR) / kwargs['path']).resolve()
        if test_path.parent != Path(settings.FILE_UPLOAD_DIR).resolve():
            # raise Exception(f"Filename {test_path} is not in {Path(settings.FILE_UPLOAD_DIR)} directory")
            return JsonResponse({'status': False})

        return FileResponse(open(settings.FILE_UPLOAD_DIR + '/' + kwargs['path'], 'rb'))
    
@authenticate
def encode_project(request, **kwargs):

    if request.method == 'POST':
        data = json.loads(request.body)
        print(data['project'])
        project_id = ObjectId(data['project']['_id'])
        data['project']['message'] = [int(_) for _ in data['project']['message']]
        del data['project']['_id']
        # try:
        result = json.loads(requests.post(settings.ENCODE_URL, json={
            'model_type': data['model_type'],
            'in_path': settings.FILE_UPLOAD_DIR_REL_MODEL_SERVER + '/' + str(project_id) + '.' + data['project']['extension'],
            'out_path': settings.FILE_UPLOAD_DIR_REL_MODEL_SERVER + '/' + str(project_id) + '_encoded.' + data['project']['extension'],
            'message': data['project']['message'],
            'message_sdr': data['message_sdr']
        }).text)
        # except:
        #     return JsonResponse({'status': False})
        
        print(result)
        data['project']['encoded'] = result['status']
        data['project']['sdr'] = result['sdr']
        if not result['status']:
            return JsonResponse({'status': False})

        settings.PROJECT.update_one({'_id': project_id}, {'$set': data['project']})
        data['project']['_id'] = str(project_id)
        return JsonResponse({'status': True, 'project': data['project']})

@authenticate
def decode(request, **kwargs):

    if request.method == 'POST':

        extension = request.FILES['file'].name.split('.')[-1]
        timestamp = datetime.datetime.now()

        if extension not in settings.ALLOWED_EXTENSIONS:
            return JsonResponse({'status': False})

        _id = settings.DECODE.insert_one({'time': timestamp, 'extension': extension, 'email': kwargs['email']}).inserted_id

        with open(settings.DECODE_UPLOAD_DIR + '/' + str(_id) + '.' + extension, 'wb') as f:
            f.write(request.FILES['file'].read())

        try:
            result = json.loads(requests.post(settings.DECODE_URL, json={
                'model_type': request.POST.get('model_type'),
                'path': settings.DECODE_UPLOAD_DIR_REL_MODEL_SERVER + '/' + str(_id) + '.' + extension,
                'phase_shift_decoding': request.POST.get('phase_shift_decoding'),
            }).text)
        except:
            print('Some error in model server')
            return JsonResponse({'status': False})

        if not result['status']:
            return JsonResponse({'status': False})
        settings.DECODE.update_one({'_id': ObjectId(_id)}, {'$set': {'messages': result['messages'], 'confidences': result['confidences']}})        

        return JsonResponse({'status': True, 'decode': sanitize_data(settings.DECODE.find_one({'_id': ObjectId(_id)}))})
    

@authenticate
def decode_file_location(request, **kwargs):

    if request.method == 'POST':
        
        data = json.loads(request.body)
        path = data['path']
        phase_shift_decoding = data['phase_shift_decoding']
        model_type = data['model_type']
        encoded_file_path = settings.FILE_UPLOAD_DIR_REL_MODEL_SERVER + '/' + path

        try:
            result = json.loads(requests.post(settings.DECODE_URL, json={
                'model_type': model_type,
                'path': encoded_file_path,
                'phase_shift_decoding': phase_shift_decoding
            }).text)
        except:
            print('Some error in model server')
            return JsonResponse({'status': False})

        if not result['status']:
            return JsonResponse({'status': False})

        return JsonResponse({'status': True, 'decode': {'messages': result['messages']}})
    

@authenticate
def apply_distortion(request, **kwargs):

    if request.method == 'POST':

        user = settings.USER.find_one({'email': kwargs['email']})
        if not user:
            return JsonResponse({'status': False})
        
        extension = request.FILES['file'].name.split('.')[-1]

        if extension not in settings.ALLOWED_EXTENSIONS:
            return JsonResponse({'status': False})
        
        if request.POST.get('distorted_path') == 'null':
            _id = settings.MANIPULATE.insert_one(
                {
                    'email': kwargs['email'],
                    'distorted_path': request.POST.get('distorted_path'),
                    'file': request.FILES['file'].name,
                    'extension': extension,
                }
            )
            print(str(_id.inserted_id))
            _id = str(_id.inserted_id)
        else:
            data = settings.MANIPULATE.find_one({'distorted_path': request.POST.get('distorted_path')})
            if data is None:
                return JsonResponse({'status': False})
            _id = str(data['_id'])
        
        with open(settings.FILE_UPLOAD_DIR + '/' + str(_id) + '.' + extension, 'wb') as f:
            f.write(request.FILES['file'].read())

        distorted_path = str(_id) + '_distorted.wav'
        
        audio, sr = librosa.load(settings.FILE_UPLOAD_DIR + '/' + str(_id) + '.' + extension, sr=None)

        print(request.POST.get('processList'))
        for process in json.loads(request.POST.get('processList')):
            print(process)
            audio, sr = distort(audio, sr, process, settings.FILE_UPLOAD_DIR + '/' + str(_id) + '_temp_distorted.wav')

        sf.write(settings.FILE_UPLOAD_DIR + '/' + distorted_path, audio, sr)
        settings.MANIPULATE.update_one({'_id': ObjectId(_id)}, {'$set': {'distorted_path': distorted_path}})        

        return JsonResponse({'status': True, 'distorted_path': distorted_path})


def login(request):

    if request.method == 'POST':

        req_user = json.loads(request.body)
        dbuser = settings.USER.find_one({'email': req_user['email'], 'password': req_user['password']})
        print(dbuser)
        if dbuser:
            return JsonResponse(
                {
                    "status": True, "name": dbuser.get('name', None),
                    "token": user.generate_auth_token(req_user.get('email'))  # .decode('utf-8')
                }
            )

        return JsonResponse({"status": False})


def distort(audio, sr, process, temp_path_audio):

    if process['name'] == 'compression':
        sf.write(temp_path_audio, audio, sr)
        os.system(f'ffmpeg -y -i {temp_path_audio} -vn -b:a {process["bit_rate"]} {temp_path_audio}.{process["algorithm"]} > /dev/null 2>&1')
        os.system(f'ffmpeg -y -i {temp_path_audio}.{process["algorithm"]} {temp_path_audio} > /dev/null 2>&1')  #  >/dev/null 2>&1
        audio, sr = sf.read(temp_path_audio)
    elif process['name'] == 'amp':
        audio = audio * float(process['scale'])
    elif process['name'] == 'crop':
        start = int(sr*float(process['startTime']))
        end = int(sr*float(process['endTime']))
        assert start >=0 and end < len(audio)
        audio = audio[start:end]
    elif process['name'] == 'resample':
        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=int(process['sampling_rate']))
        sr = int(process['sampling_rate'])
    else:
        print('Unknown distortion')
    return audio, sr