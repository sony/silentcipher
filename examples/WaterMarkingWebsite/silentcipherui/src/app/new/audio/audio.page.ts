import { Component, OnInit , ViewChild, ElementRef, HostListener, AfterViewChecked  } from '@angular/core';
import { Router } from '@angular/router';
import { environment } from '../../../environments/environment';
import { HttpClient } from '@angular/common/http';
import { NotificationService } from '../../services/notification.service';
import { LoginService } from '../../services/login.service';

@Component({
  selector: 'app-audio',
  templateUrl: './audio.page.html',
  styleUrls: ['./audio.page.scss'],
})
export class AudioPage implements OnInit {

  @ViewChild('container', {read: ElementRef}) container: ElementRef;
  @ViewChild('header', {read: ElementRef}) header: ElementRef;

  @HostListener('window:resize', ['$event'])
  onResize(event?) {
    this.container.nativeElement.style.height = window.innerHeight - this.header.nativeElement.offsetHeight + 'px';
  }

  fileName = '';
  formData = null;
  projectName = null;

  constructor(private notificationService: NotificationService, private router: Router, private http: HttpClient, private loginService: LoginService) {}


  ngOnInit() {
    this.formData = new FormData();
  }

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

  submit(){
    console.log('submit')
    if (!this.fileName) {
      this.notificationService.presentToastError('Please provide file to be uploaded');
      return;
    }
    if (!this.projectName) {
      this.notificationService.presentToastError('Please provide a project name for unique identification');
      return;
    }
    console.log('submit1')
    this.formData.append("projectName", this.projectName);
    this.formData.append("type", 'audio');
    this.http.post(environment.SERVER_URL + 'api/new_project', this.formData, {params: {loading: 'true'}}).subscribe((res: any) => {
      if (res.status){
        this.loginService.user_data = res.user_data;
        this.router.navigateByUrl('/project/' + res.id);
        this.notificationService.presentToastSuccess('Project Creation Success.');
      }
      else{
        this.notificationService.presentToastError('Project Creation Failed. Please contact admin');
      }
    })

  }

}
