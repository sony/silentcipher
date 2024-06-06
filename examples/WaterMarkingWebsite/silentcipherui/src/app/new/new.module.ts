import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

import { NewPageRoutingModule } from './new-routing.module';

import { NewPage } from './new.page';

import { HeaderComponentModule } from '../reusable/header/header.module';
import { ProjectbarComponentModule } from '../reusable/projectbar/projectbar.module';
import { FontAwesomeModule } from '@fortawesome/angular-fontawesome';

@NgModule({
  imports: [
    CommonModule,
    FormsModule,
    NewPageRoutingModule,
    ProjectbarComponentModule,
    HeaderComponentModule,
    FontAwesomeModule
  ],
  declarations: [NewPage]
})
export class NewPageModule {}
