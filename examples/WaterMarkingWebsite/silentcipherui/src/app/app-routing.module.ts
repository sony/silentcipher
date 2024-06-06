import { NgModule } from '@angular/core';
import { PreloadAllModules, RouterModule, Routes } from '@angular/router';
import { AuthGuard } from './guard/auth.guard';

const routes: Routes = [
  {
    path: '',
    redirectTo: 'login',
    pathMatch: 'full'
  },
  {
    path: 'login',
    loadChildren: () => import('./login/login.module').then( m => m.LoginPageModule)
  },
  // {
  //   path: 'main',
  //   loadChildren: () => import('./main/main.module').then( m => m.MainPageModule),
  //   canActivate: [AuthGuard]
  // },
  {
    path: 'main',
    redirectTo: 'new/audio',
    pathMatch: 'full',
  },
  {
    path: 'project',
    loadChildren: () => import('./project/project.module').then( m => m.ProjectPageModule),
    canActivate: [AuthGuard]
  },
  {
    path: 'new',
    loadChildren: () => import('./new/new.module').then( m => m.NewPageModule),
    canActivate: [AuthGuard]
  },
  {
    path: 'decode',
    loadChildren: () => import('./decode/decode.module').then( m => m.DecodePageModule),
    canActivate: [AuthGuard]
  },
  {
    path: 'manipulate',
    loadChildren: () => import('./manipulate/manipulate.module').then( m => m.ManipulatePageModule),
    canActivate: [AuthGuard]
  }
];

@NgModule({
  imports: [
    RouterModule.forRoot(routes, { preloadingStrategy: PreloadAllModules })
  ],
  exports: [RouterModule]
})
export class AppRoutingModule { }
