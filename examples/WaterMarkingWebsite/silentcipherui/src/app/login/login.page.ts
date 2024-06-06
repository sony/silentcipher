import { HttpClient } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { environment } from '../../environments/environment';
import { LoginService } from '../services/login.service';
import { NotificationService } from '../services/notification.service';

@Component({
  selector: 'app-login',
  templateUrl: './login.page.html',
  styleUrls: ['./login.page.scss'],
})
export class LoginPage implements OnInit {

  email = "";
  password = "";

  constructor(
    public loginService: LoginService) {
  }

  ngOnInit() {
  }

  ionViewDidEnter(){
    if (this.loginService.email){
      this.loginService.afterLogin();
    }
  }

}
