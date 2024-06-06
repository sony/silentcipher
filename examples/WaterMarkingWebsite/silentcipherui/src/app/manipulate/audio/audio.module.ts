import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

import { AudioPageRoutingModule } from './audio-routing.module';

import { AudioPage } from './audio.page';

import { HeaderComponentModule } from '../../reusable/header/header.module';
import { FontAwesomeModule } from '@fortawesome/angular-fontawesome';

@NgModule({
  imports: [
    CommonModule,
    FormsModule,
    AudioPageRoutingModule,
    HeaderComponentModule,
    FontAwesomeModule
  ],
  declarations: [AudioPage]
})
export class AudioPageModule {}
