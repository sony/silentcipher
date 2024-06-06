import { async, ComponentFixture, TestBed } from '@angular/core/testing';
import { IonicModule } from '@ionic/angular';

import { DecodePage } from './decode.page';

describe('DecodePage', () => {
  let component: DecodePage;
  let fixture: ComponentFixture<DecodePage>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ DecodePage ],
      imports: [IonicModule.forRoot()]
    }).compileComponents();

    fixture = TestBed.createComponent(DecodePage);
    component = fixture.componentInstance;
    fixture.detectChanges();
  }));

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
