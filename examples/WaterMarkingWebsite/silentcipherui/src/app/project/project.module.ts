import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';


import { ProjectPageRoutingModule } from './project-routing.module';

import { HeaderComponentModule } from '../reusable/header/header.module';

import { ProjectPage } from './project.page';
import { ProjectbarComponentModule } from '../reusable/projectbar/projectbar.module';
import { FontAwesomeModule } from '@fortawesome/angular-fontawesome';

@NgModule({
  imports: [
    CommonModule,
    FormsModule,
    HeaderComponentModule,
    ProjectPageRoutingModule,
    ProjectbarComponentModule,
    FontAwesomeModule
  ],
  declarations: [ProjectPage]
})
export class ProjectPageModule {}
