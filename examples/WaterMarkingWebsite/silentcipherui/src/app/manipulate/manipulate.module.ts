import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

import { ManipulatePageRoutingModule } from './manipulate-routing.module';

import { ManipulatePage } from './manipulate.page';

import { HeaderComponentModule } from '../reusable/header/header.module';
import { FontAwesomeModule } from '@fortawesome/angular-fontawesome';

@NgModule({
  imports: [
    CommonModule,
    FormsModule,
    ManipulatePageRoutingModule,
    HeaderComponentModule,
    FontAwesomeModule
  ],
  declarations: [ManipulatePage]
})
export class ManipulatePageModule {}
