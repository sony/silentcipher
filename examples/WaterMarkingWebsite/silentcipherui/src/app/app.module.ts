

import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppComponent } from './app.component';
import { AppRoutingModule } from './app-routing.module';

import { HttpClientModule, HTTP_INTERCEPTORS } from '@angular/common/http';
import { MyHttpLogInterceptor } from './services/http.interceptor';

import * as $ from 'jquery';
import { ToastrModule } from 'ngx-toastr';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';

@NgModule({
  declarations: [AppComponent],
  imports: [BrowserModule, AppRoutingModule, HttpClientModule, BrowserAnimationsModule,
    ToastrModule.forRoot({
      positionClass :'toast-bottom-right'
    })],
  providers: [
    { provide: HTTP_INTERCEPTORS, useClass: MyHttpLogInterceptor, multi: true },
  ],
  bootstrap: [AppComponent]
})
export class AppModule { }