import { Injectable } from '@angular/core';
import { LoadingService } from './loading.service';
import {
  HttpInterceptor,
  HttpRequest,
  HttpHandler,
  HttpEvent,
  HttpHeaders,
  HttpErrorResponse,
  HttpResponse
} from '@angular/common/http';

import { EMPTY, Observable, of, throwError } from 'rxjs';
import { catchError, map } from 'rxjs/operators';
import { LoginService } from './login.service';

@Injectable()
export class MyHttpLogInterceptor implements HttpInterceptor {

  constructor(private loginService: LoginService, private _loading: LoadingService) { }

  handleAuthError(err: HttpErrorResponse): Observable<any> {
    
    // Handling unauthorized error by logging the user out and asking to login again
    if (err.status === 401){
      localStorage.removeItem('email');
      localStorage.removeItem('token');
      this.loginService.logout();
      return of(EMPTY);
    }
    return throwError(err);
  }
  intercept(request: HttpRequest<any>, next: HttpHandler): Observable<HttpEvent<any>> {

    // Intercepting the HTTP request and adding the email id of the user in the header
    let show_loader = 'false';
    if (request.params.get('loading')){
      show_loader = request.params.get('loading')
    }

    const token = localStorage.getItem('token') || 'null';
    const email = localStorage.getItem('email') || 'null';
    if (!email) {
      return next.handle(request);
    }
    const headers = new HttpHeaders({ email, token });
    const customReq = request.clone({ headers });

    if (show_loader === 'true'){
      this._loading.setLoading(true, request.url);
      return next.handle(customReq)
        .pipe(catchError((err) => {
          this._loading.setLoading(false, request.url);
          this.handleAuthError(err);
          return err;
        }))
        .pipe(map<HttpEvent<any>, any>((evt: HttpEvent<any>) => {
          if (evt instanceof HttpResponse) {
            this._loading.setLoading(false, request.url);
          }
          return evt;
        }));
    }
    
    return next.handle(customReq).pipe(catchError(x => this.handleAuthError(x)));
  }
}
