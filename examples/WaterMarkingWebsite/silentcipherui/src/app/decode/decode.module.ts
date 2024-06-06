import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

import { DecodePageRoutingModule } from './decode-routing.module';

import { DecodePage } from './decode.page';

import { HeaderComponentModule } from '../reusable/header/header.module';
import { FontAwesomeModule } from '@fortawesome/angular-fontawesome';

@NgModule({
  imports: [
    CommonModule,
    FormsModule,
    DecodePageRoutingModule,
    HeaderComponentModule,
    FontAwesomeModule
  ],
  declarations: [DecodePage]
})
export class DecodePageModule {}
