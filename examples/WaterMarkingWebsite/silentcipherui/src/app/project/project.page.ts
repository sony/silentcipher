import { Component, OnInit , ViewChild, ElementRef, HostListener, ChangeDetectorRef } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { LoginService } from '../services/login.service';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../environments/environment';
import { NotificationService } from '../services/notification.service';

declare var WaveSurfer;

@Component({
  selector: 'app-project',
  templateUrl: './project.page.html',
  styleUrls: ['./project.page.scss'],
})
export class ProjectPage implements OnInit {
  id = null;
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
  paused = true;
  encoded_paused = true;
  wr = null;
  wr_encoded = null;
  decoded = [];
  user_provided_SDR=-1;
  model_type='44k';

  // model_options = [
  //   {
  //     'desc': 'high quality encoding',
  //     'message_dim': 5,
  //     'message_len': 21
  //   },
  //   {
  //     'desc': 'high message encoding accuracy',
  //     'message_dim': 17,
  //     'message_len': 11
  //   }
  // ]
  model_options = [
    {
      'desc': 'high quality encoding',
      'message_limit': [0, 255],
      'message_len': 5
    }
  ]

  @ViewChild('container', {read: ElementRef}) container: ElementRef;
  @ViewChild('content', {read: ElementRef}) content: ElementRef;
  @ViewChild('header', {read: ElementRef}) header: ElementRef;
  @ViewChild('waveform', {read: ElementRef}) waveform: ElementRef;
  @ViewChild('encoded_waveform', {read: ElementRef}) encoded_waveform: ElementRef;

  select(i) {
    this.project.selected = i;
    this.project.message = Array<string>(this.model_options[this.project.selected].message_len).fill('');
    this.project.error = Array<boolean>(this.model_options[this.project.selected].message_len).fill(false);

  }

  onResize(event?) {
    this.cd.detectChanges();
    this.container.nativeElement.style.height = window.innerHeight - this.header.nativeElement.offsetHeight + 'px';
    this.waveform.nativeElement.style.width = this.content.nativeElement.offsetWidth - 250 + 'px';
    this.setWR();
  }

  onResizeEncoded(event?) {
    this.cd.detectChanges();
    this.container.nativeElement.style.height = window.innerHeight - this.header.nativeElement.offsetHeight + 'px';
    if (this.encoded_waveform){
      this.encoded_waveform.nativeElement.style.width = this.content.nativeElement.offsetWidth - 250 + 'px';
      this.setWREncoded();
    }
  }

  @HostListener('window:resize', ['$event'])
  resizeEvent(event?) {
    this.onResize(event);
    this.onResizeEncoded(event);
  }

  constructor(
    private route: ActivatedRoute, private loginService: LoginService, private http: HttpClient, private notification: NotificationService,
    private cd: ChangeDetectorRef
  ) {}

  ngOnInit() {
    this.id = this.route.snapshot.params['id'];
    console.log(this.id)
    this.wr = null
    this.wr_encoded = null;
    
    this.project = {
      message: Array<string>(this.model_options[this.project.selected].message_len).fill(''),
      error: Array<boolean>(this.model_options[this.project.selected].message_len).fill(false),
      extension: '',
      encoded: false,
      selected: 0,
      _id: '',
      file: '',
      name: '',
      sdr: -1
    };
    this.user_provided_SDR = -1;
    this.model_type = '44k';
    this.paused = true;
    this.encoded_paused = true;
    this.decoded = Array<number>(this.model_options[this.project.selected].message_len).fill(null);
    this.getProjectInfo();
  }

  getEnv(){
    return environment;
  }

  setWR(){
    if (!this.project._id){
      return;
    }
    this.waveform.nativeElement.innerHTML = '';
    this.wr = WaveSurfer.create({
      // container: '#waveform',
      container: this.waveform.nativeElement,
      waveColor: '#0275d8',
      progressColor: 'lightblue',
      scrollParent: true
    });
    console.log('WR', this.getEnv().SERVER_URL + 'api/files/' + this.project._id + '.' + this.project.extension);
    this.wr.load(this.getEnv().SERVER_URL + 'api/files/' + this.project._id + '.' + this.project.extension)
    var self = this;
    this.wr.on('play', function() {
      self.paused = false;
      console.log('play', self.paused);
      self.cd.detectChanges();
    });

    this.wr.on('pause', function() {
      self.paused = true;
      console.log('pause', self.paused);
      self.cd.detectChanges();
    });

    this.wr.on('finish', function() {
      self.paused = true;
      self.wr.seekTo(0);
      console.log('finish', self.paused);
      self.cd.detectChanges();
    });
  }

  setWREncoded(){
    if (!this.project._id){
      return;
    }
    this.encoded_waveform.nativeElement.innerHTML = '';
    this.wr_encoded = WaveSurfer.create({
      container: this.encoded_waveform.nativeElement,
      waveColor: '#0275d8',
      progressColor: 'lightblue',
      scrollParent: true
    });
    console.log('WR_encoded', this.getEnv().SERVER_URL + 'api/files/' + this.project._id + '_encoded.' + this.project.extension);
    this.wr_encoded.load(this.getEnv().SERVER_URL + 'api/files/' + this.project._id + '_encoded.' + this.project.extension)
    var self = this;
    this.wr_encoded.on('play', function() {
      self.encoded_paused = false;
      console.log('play', self.encoded_paused);
      self.cd.detectChanges();
    });

    this.wr_encoded.on('pause', function() {
      self.encoded_paused = true;
      console.log('pause', self.encoded_paused);
      self.cd.detectChanges();
    });

    this.wr_encoded.on('finish', function() {
      self.encoded_paused = true;
      self.wr_encoded.seekTo(0);
      console.log('finish', self.encoded_paused);
      self.cd.detectChanges();
    });
  }

  getProjectInfo() {
    this.paused = true;
    this.encoded_paused = true;
    this.http.post(environment.SERVER_URL + 'api/get_project_data', {email: this.loginService.email, projectid: this.id}).subscribe((res: any) => {
      if (res.status){
        this.project = res.project;
        this.onResize();
        if (this.project.encoded){
          this.onResizeEncoded();
        }
      }
      else{
        this.notification.presentToastError('Error when getting project!');
      }
    });
  }

  check_message(){
    let error = false;
    for(let i=0; i < this.project.message.length; ++i){
      if (this.project.message[i] == '' || this.project.message[i]==null){
        this.notification.presentToastError('None of the fields should be blank in the message');
        error = true;
        break;
      }
      if (!/^\d+$/.test(this.project.message[i])){
        this.notification.presentToastError('Message should only be integers between 0-255 (0 and 255 are included)');
        error = true;
        break;
      }
      else if (Number(this.project.message[i]) < this.model_options[this.project.selected].message_limit[0] || Number(this.project.message[i]) > this.model_options[this.project.selected].message_limit[1]){
        this.notification.presentToastError('Message should have values less than or equal to ' + this.model_options[this.project.selected].message_limit[1] + ' and greater than or equal to ' + this.model_options[this.project.selected].message_limit[0]);
        error = true;
        break;
      }
    }
    return !error;
  }

  check_message_i(i){
    if (this.project.message[i] == '' || this.project.message[i] == null){
      return
    }
    if (!/^\d+$/.test(this.project.message[i])){
      this.project.error[i] = true;
      this.notification.presentToastError('Message should only be integers between 0-255 (0 and 255 are included)');
    }
    else if (Number(this.project.message[i]) < this.model_options[this.project.selected].message_limit[0] || Number(this.project.message[i]) > this.model_options[this.project.selected].message_limit[1]){
      this.project.error[i] = true;
      this.notification.presentToastError('Message should have values less than or equal to ' + this.model_options[this.project.selected].message_limit[1] + ' and greater than or equal to ' + this.model_options[this.project.selected].message_limit[0]);
    }
    else{
      this.project.error[i] = false;
    }
  }
  encode() {
    this.check_message();
    this.project.encoded = false;
    this.decoded = Array<number>(this.model_options[this.project.selected].message_len).fill(null);
    this.http.post(environment.SERVER_URL + 'api/encode_project', {
      email: this.loginService.email, 
      project: this.project, 
      message_sdr: this.user_provided_SDR == -1? null: this.user_provided_SDR,
      model_type: this.model_type
    }, {params: {loading: 'true'}}).subscribe((res: any) => {
      if (res.status){
        this.project = res.project;
        this.cd.detectChanges();
        if (this.project.encoded){
          this.onResizeEncoded();
        }
      }
      else{
        this.notification.presentToastError('Error when encoding the message!');
      }
    });
  }

  decode(){
    if (!this.project.encoded){
      this.notification.presentToastError('Please Encode the Audio First');
      return;
    }
    console.log('Decoding')
    this.http.post(environment.SERVER_URL + 'api/decode_file_location', {
      path: this.project._id + '_encoded.' + this.project.extension, 
      phase_shift_decoding: environment.phase_shift_decoding,
      model_type: this.model_type
    }, {params: {loading: 'true'}}).subscribe((res: any) => {
      if (res.status){
        this.decoded = res.decode.messages[0]
      }
      else{
        this.notification.presentToastError('Error when decoding the message!');
      }
    });
  }

  check_match(){
    if (this.decoded.length == 0){
      return false;
    }
    if (this.decoded[0] === null){
      return false;
    }
    let result = true;
    for(let i =0;i<this.decoded.length;++i){
      if (this.decoded[i] != this.project.message[i]){
        result = false;
        break
      }
    }

    return result;
  }

  gen_len(num){
    return Array<number>(num);
  }

}
