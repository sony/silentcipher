import { Component, OnInit , ViewChild, ElementRef, HostListener  } from '@angular/core';

@Component({
  selector: 'app-new',
  templateUrl: './new.page.html',
  styleUrls: ['./new.page.scss'],
})
export class NewPage implements OnInit  {

  @ViewChild('container', {read: ElementRef}) container: ElementRef;
  @ViewChild('header', {read: ElementRef}) header: ElementRef;

  @HostListener('window:resize', ['$event'])
  onResize(event?) {
    this.container.nativeElement.style.height = window.innerHeight - this.header.nativeElement.offsetHeight + 'px';
  }

  constructor() { }

  ngOnInit(){

  }

  ionViewDidEnter() {
    this.onResize();
  }

}
