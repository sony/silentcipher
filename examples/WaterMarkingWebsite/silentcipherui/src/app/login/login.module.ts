import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

import { LoginPageRoutingModule } from './login-routing.module';

import { LoginPage } from './login.page';

import { HeaderComponentModule } from '../reusable/header/header.module';
import { FontAwesomeModule } from '@fortawesome/angular-fontawesome';

@NgModule({
  imports: [
    CommonModule,
    FormsModule,
    LoginPageRoutingModule,
    HeaderComponentModule,
    FontAwesomeModule
  ],
  declarations: [LoginPage]
})
export class LoginPageModule {}
