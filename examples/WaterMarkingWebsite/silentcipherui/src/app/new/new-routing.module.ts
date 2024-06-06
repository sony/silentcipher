import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';

import { NewPage } from './new.page';

const routes: Routes = [
  {
    path: '',
    component: NewPage
  },
  {
    path: 'audio',
    loadChildren: () => import('./audio/audio.module').then( m => m.AudioPageModule)
  }
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule],
})
export class NewPageRoutingModule {}
