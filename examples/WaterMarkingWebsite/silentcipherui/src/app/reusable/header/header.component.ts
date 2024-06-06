import { HttpClient } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { LoginService } from '../../services/login.service';
import { environment } from '../../../environments/environment';
import { saveAs } from 'file-saver';
import { NotificationService } from '../../services/notification.service';

@Component({
  selector: 'app-header',
  templateUrl: './header.component.html',
  styleUrls: ['./header.component.scss'],
})
export class HeaderComponent implements OnInit {

  constructor(public loginService: LoginService, private http: HttpClient, private router: Router, private notification: NotificationService) { }

  ngOnInit() {
  }

}
