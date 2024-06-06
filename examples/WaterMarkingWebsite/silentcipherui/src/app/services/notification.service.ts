import { Injectable } from '@angular/core';
// import { ToastController } from '@ionic/angular';
import { ToastrService } from 'ngx-toastr';

@Injectable({
providedIn: 'root'
})
export class NotificationService {
// constructor(public toastController: ToastController) {
// }
constructor(private toastr: ToastrService) {}
unlimitedToast = null;
presenting = false;
// ToDo change the CSS class of the notification bar
dangerCSS = 'bg-danger';

async presentToastSuccess(text: string) {
    this.toastr.success(text, '', {
        timeOut: 3000,
    });
}

async presentToastError(text: string) {
    this.toastr.error(text, '', {
        timeOut: 3000,
    });
}
async presentUnlimitedToast(text: string) {
    // const position = 'top';
    // this.unlimitedToast = await this.toastController.create({
    //     position,
    //     message: text,
    //     duration: 2000000
    // });
    // this.unlimitedToast.present();
    this.presenting = true;
    
    this.toastr.error(text, text, {
        timeOut: 3000000000,
    });
}
}
