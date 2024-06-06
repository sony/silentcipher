import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { environment } from '../../../environments/environment';
import { HttpClient } from '@angular/common/http';
import { NotificationService } from '../..//services/notification.service';
import { LoginService } from '../../services/login.service';

@Component({
  selector: 'app-audio',
  templateUrl: './audio.page.html',
  styleUrls: ['./audio.page.scss'],
})
export class AudioPage implements OnInit {

  fileName = '';
  formData = null;
  message = null;
  confidence = null;
  no_message_detected = false;

  project = {
    message: [],
    error: [],
    extension: '',
    encoded: false,
    _id: '',
    file: '',
    name: '',
    selected: 0,
    sdr: -1
  };

  model_options = [
    {
      'desc': 'high quality encoding',
      'message_limit': [0, 255],
      'message_len': 5
    }
  ]

  model_type = '44k';

  constructor(private notificationService: NotificationService, private router: Router, private http: HttpClient, private loginService: LoginService) {}

  gen_len(num){
    return Array<number>(num);
  }

  onFileSelected(event) {

    this.formData = new FormData();

    const file:File = event.target.files[0];

    if (file) {

        this.fileName = file.name;
        this.formData.append("file", file);
    }
  }

  submit(){
    if (!this.fileName) {
      this.notificationService.presentToastError('Please provide file to be uploaded');
      return;
    }
    this.formData.append("type", 'audio');
    this.formData.append("phase_shift_decoding", environment.phase_shift_decoding);
    this.formData.append("model_type", this.model_type);
    this.http.post(environment.SERVER_URL + 'api/decode', this.formData, {params: {loading: 'true'}}).subscribe((res: any) => {
      if (res.status){
        this.message = res.decode.messages[0];
        this.confidence = res.decode.confidences[0];
        this.notificationService.presentToastSuccess('Message Decoded Successfully.');
        this.no_message_detected = false;
      }
      else{
        this.notificationService.presentToastError('No message detected in the audio');
        this.no_message_detected = true;
      }
    })

  }

  ngOnInit() {
    this.formData = new FormData();
    this.model_type = '44k';
  }

}
