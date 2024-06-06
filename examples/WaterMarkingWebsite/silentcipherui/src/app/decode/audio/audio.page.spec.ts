import { async, ComponentFixture, TestBed } from '@angular/core/testing';
import { IonicModule } from '@ionic/angular';

import { AudioPage } from './audio.page';

describe('AudioPage', () => {
  let component: AudioPage;
  let fixture: ComponentFixture<AudioPage>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ AudioPage ],
      imports: [IonicModule.forRoot()]
    }).compileComponents();

    fixture = TestBed.createComponent(AudioPage);
    component = fixture.componentInstance;
    fixture.detectChanges();
  }));

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
