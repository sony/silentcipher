import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ProjectbarComponent } from './projectbar.component';
import { RouterModule } from '@angular/router';

@NgModule({
  imports: [
    CommonModule,
    FormsModule,
    RouterModule
  ],
  declarations: [ProjectbarComponent],
  exports: [ProjectbarComponent],
})
export class ProjectbarComponentModule {}
