import { Injectable } from '@angular/core';
import { CanActivate, ActivatedRouteSnapshot, RouterStateSnapshot, UrlTree, Router } from '@angular/router';
import { Observable } from 'rxjs';
import { NotificationService } from '../services/notification.service';
import { LoginService } from '../services/login.service'

@Injectable({
  providedIn: 'root'
})
export class AuthGuard implements CanActivate {

  constructor(private router: Router, private notification: NotificationService, private loginService: LoginService) {}

  // canActivate(
  //   next: ActivatedRouteSnapshot,
  //   state: RouterStateSnapshot): Observable<boolean | UrlTree> | Promise<boolean | UrlTree> | boolean | UrlTree {
      
  //     // Basic authentication based on if the user provided email id.

  //     // If email is set in localStorage then page is accessible otherwise redirected to the login page

  //     const email = localStorage.getItem('email');
  //     const token = localStorage.getItem('token');
  //     console.log('Auth: ', email);
  //     if (email !== null && email !== 'null'){
  //       return true;
  //     }
  //     this.notification.presentToast('Please login first!')
  //     this.router.navigateByUrl('/login');
  //     return false;
  // }

  canActivate(
    next: ActivatedRouteSnapshot,
    state: RouterStateSnapshot): Observable<boolean | UrlTree> | Promise<boolean | UrlTree> | boolean | UrlTree {
    if (this.loginService.isLoggedIn()) {
      return true;
    }
    else {
      this.router.navigate(['login'], { queryParams: { loginRedirect: state.url } });
      return false;
    }
  }
  
}
