import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';

import { DecodePage } from './decode.page';

const routes: Routes = [
  {
    path: '',
    component: DecodePage
  },
  {
    path: 'audio',
    loadChildren: () => import('./audio/audio.module').then( m => m.AudioPageModule)
  },
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule],
})
export class DecodePageRoutingModule {}
