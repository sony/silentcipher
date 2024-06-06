from django.conf.urls import url
from django.urls import path

from . import views

urlpatterns = [
    path('create_user', views.create_user, name='create_user'),
    path('login', views.login, name='login'),
    path('get_user_data', views.get_user_data, name='get_user_data'),
    path('new_project', views.new_project, name='new_project'),
    path('get_project_data', views.get_project_data, name='get_project_data'),
    url(r'files/(?P<path>.+)', views.files, name='files'),
    path('encode_project', views.encode_project, name='encode_project'),
    path('decode', views.decode, name='decode'),
    path('decode_file_location', views.decode_file_location, name='decode_file_location'),
    path('apply_distortion', views.apply_distortion, name='apply_distortion')
]