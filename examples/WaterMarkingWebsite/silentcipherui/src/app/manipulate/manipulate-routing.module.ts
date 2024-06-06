import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';

import { ManipulatePage } from './manipulate.page';

const routes: Routes = [
  {
    path: '',
    component: ManipulatePage
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
export class ManipulatePageRoutingModule {}
