import { HttpClient } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { NotificationService } from '../../services/notification.service';
import { environment } from '../../../environments/environment';

@Component({
  selector: 'app-audio',
  templateUrl: './audio.page.html',
  styleUrls: ['./audio.page.scss'],
})
export class AudioPage implements OnInit {


  constructor(private notificationService: NotificationService, private http: HttpClient) {}

  ngOnInit() {
    this.formData = new FormData();
  }
  fileName = '';
  formData = null;
  projectName = null;
  processList = [];
  distorted_path = null;
  decoded = [];
  error=false;

  onFileSelected(event) {

    this.formData = new FormData();
    const file:File = event.target.files[0];

    if (file) {
        this.fileName = file.name;
        this.formData.append("file", file);
        if (this.projectName == '' || this.projectName == null){
          this.projectName = this.fileName.split('.')[0]
        }
    }
  }

  addProcess(process_info){
    this.processList.push(process_info);
  }
  visualize(data){
    return JSON.stringify(data);
  }
  removeProcess(index){
    this.processList.splice(index,1);
  }

  applyDistortion(){
    
    this.error=false;
    this.formData.append("processList", JSON.stringify(this.processList));
    this.formData.append("distorted_path", this.distorted_path);

    this.http.post(environment.SERVER_URL + 'api/apply_distortion', this.formData, {params: {loading: 'true'}}).subscribe((res: any) => {
      if (res.status){
        this.distorted_path = res.distorted_path;
        console.log(this.distorted_path)
        this.decoded = [];
      }
      else{
        this.notificationService.presentToastError('Error when applying the distortion!');
      }
    })

  }

  decodeDistortedAudio(){
    this.notificationService.presentToastSuccess('Decoding using the 44k model. To use the 16k model, please use the python package or submit a PR to the repo.');
    this.http.post(environment.SERVER_URL + 'api/decode_file_location', {model_type: '44k', path: this.distorted_path, phase_shift_decoding: environment.phase_shift_decoding}, {params: {loading: 'true'}}).subscribe((res: any) => {
      if (res.status){
        this.decoded = res.decode.messages[0]
        this.error=false;
      }
      else{
        this.error=true;
        this.notificationService.presentToastError('Error when decoding the message!');
      }
    });
  }
  getEnv(){
    return environment;
  }
  gen_len(num){
    return Array<number>(num);
  }


}
