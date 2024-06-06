import { HttpClient } from '@angular/common/http';
import { Injectable, Inject } from '@angular/core';
import { Router } from '@angular/router';
import { environment } from '../../environments/environment';
import { NotificationService } from './notification.service';
import { DOCUMENT } from '@angular/common';

@Injectable({
  providedIn: 'root'
})
export class LoginService {

  email = null;
  name = null;

  // Set admin email IDs. Only admin email IDs can see the analysis!

  user_data = null;
  admin = false;
  localStorage;

  constructor(private notification: NotificationService, private router: Router, private http: HttpClient, @Inject(DOCUMENT) private document: Document) {
    // If user has already logged into the browser earlier, they won't need to login again
    this.localStorage = document.defaultView?.localStorage
    const email = this.localStorage.getItem('email');
    if (email !== null && email !== 'null')
    {
      this.email = email;
      this.name = name;
    }
  }

  login(email, password){
    // Login into the server. If the user is looging in for the first time, songs are randomly sampled and initialized in the database
    
    // Ensuring that the emails should end with @sony.com (Not necessary can be removed)

    // Making the email id lower case for consistency
    email = email.toLowerCase();

    // This API call will intiialize the Database
    console.log('Trying to login')
    this.http.post(environment.SERVER_URL + 'api/login', {email, password}).subscribe((res: any) => {
      console.log('Got the response')
      if (res.status){
        this.email = email;
        this.name = res.name;
        this.localStorage.setItem('email', email);
        this.localStorage.setItem('name', res.name);
        this.localStorage.setItem('token', res.token);
        console.log(res);
        this.afterLogin();
      }
      else{
        // If the login fails this notification toast is presented
        this.notification.presentToastError('Login Failed! Please contact the admin or provide correct password!');
      }
    })
    
  }

  updateUserData(){
    this.http.get(environment.SERVER_URL + 'api/get_user_data').subscribe((res: any) => {
      if (res.status){
        this.user_data = res.user_data;
      }
      else{
        this.notification.presentToastError('Error when getting tasks!');
        this.logout();
      }
    });
  }

  afterLogin(){

    // This function is used to get the current status of the tasks.

    this.http.get(environment.SERVER_URL + 'api/get_user_data').subscribe((res: any) => {
      if (res.status){
        this.user_data = res.user_data;
        this.router.navigateByUrl('/main');
      }
      else{
        this.notification.presentToastError('Error when getting tasks!');
        this.logout();
      }
    });
  }
  logout(){

    // Function to logout the user and redirect to login page

    this.email = null;
    this.name = null;
    this.localStorage.removeItem('email');
    this.localStorage.removeItem('token');
    this.router.navigate(['/login'])
  }

  isLoggedIn() {
    const email = this.localStorage.getItem('email');
    const token = this.localStorage.getItem('token');
    return email !== 'null' && email !== null && token !== 'null' && token !== null;
  }
}
