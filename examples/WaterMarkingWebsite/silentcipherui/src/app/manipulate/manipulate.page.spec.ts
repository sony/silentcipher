import { async, ComponentFixture, TestBed } from '@angular/core/testing';
import { IonicModule } from '@ionic/angular';

import { ManipulatePage } from './manipulate.page';

describe('ManipulatePage', () => {
  let component: ManipulatePage;
  let fixture: ComponentFixture<ManipulatePage>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ ManipulatePage ],
      imports: [IonicModule.forRoot()]
    }).compileComponents();

    fixture = TestBed.createComponent(ManipulatePage);
    component = fixture.componentInstance;
    fixture.detectChanges();
  }));

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
