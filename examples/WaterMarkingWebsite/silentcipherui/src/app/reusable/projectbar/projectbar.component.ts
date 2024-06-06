import { Component, OnInit, Input } from '@angular/core';
import {LoginService} from '../../services/login.service'

@Component({
  selector: 'app-projectbar',
  templateUrl: './projectbar.component.html',
  styleUrls: ['./projectbar.component.scss'],
})
export class ProjectbarComponent implements OnInit {

  @Input() name: string;

  constructor(public loginService: LoginService) { }

  ngOnInit() {
    this.loginService.updateUserData()
  }

  getWindow(){
    return window;
  }

}
