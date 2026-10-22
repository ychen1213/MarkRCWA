"""Top-view V geometry and local straight-arm frames, not a full 2D solver."""
from dataclasses import dataclass
import numpy as np
from mark_rcwa.model import positive
from .profiles import rectangle
from .operations import translate


@dataclass(frozen=True)
class ChevronPlan:
    period_y_um: float = 8.
    arm_angle_deg: float = 45.
    duty: float = .4
    mark_width_x_um: float = 24.
    padding_x_um: float = 8.
    depth_um: float = .12
    ridge: str = 'Si3N4'
    groove: str = 'air'
    top: str = 'air'
    bottom: str = 'glass_demo'

    def __post_init__(self):
        for key in ('period_y_um','mark_width_x_um','depth_um'):
            positive(getattr(self,key),key)
        positive(self.padding_x_um,'padding_x_um',allow_zero=True)
        if not np.isfinite(self.arm_angle_deg) or not 0 < self.arm_angle_deg < 89:
            raise ValueError('arm_angle_deg must lie in (0,89), measured from the x axis')
        if not np.isfinite(self.duty) or not 0 < self.duty < 1:
            raise ValueError('duty must lie in (0,1)')
        for key in ('ridge','groove','top','bottom'):
            if not isinstance(getattr(self,key),str) or not getattr(self,key):
                raise ValueError(f'{key} must be a material identifier')

    @property
    def slope(self):
        return float(np.tan(np.radians(self.arm_angle_deg)))

    @property
    def period_x_um(self):
        return self.mark_width_x_um+2*self.padding_x_um

    @property
    def normal_pitch_um(self):
        return self.period_y_um/np.hypot(1,self.slope)

    def normal(self,arm):
        if arm not in ('left','right'):
            raise ValueError('arm must be left or right')
        return np.array([self.slope if arm=='left' else -self.slope,1.])/np.hypot(1,self.slope)

    def reciprocal(self,arm):
        return 2*np.pi/self.normal_pitch_um*self.normal(arm)

    def mask(self,x_um,y_um,displacement_um=(0.,0.)):
        """True is ridge; band centres obey y=s*abs(x)+j*Py. x is not wrapped."""
        shift=np.asarray(displacement_um,dtype=float)
        if shift.shape!=(2,) or not np.isfinite(shift).all():
            raise ValueError('displacement_um must be a finite [x,y] pair')
        x,y=np.broadcast_arrays(np.asarray(x_um,float),np.asarray(y_um,float))
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('mask coordinates must be finite')
        x,y=x-shift[0],y-shift[1]
        phase=(y-self.slope*np.abs(x)+self.period_y_um/2)%self.period_y_um-self.period_y_um/2
        distance=abs(phase)-self.duty*self.period_y_um/2
        # Exclude exact-edge ties consistently after floating modulo operations.
        # This affects only roundoff-scale boundary classification, not the CD.
        tolerance=32*np.finfo(float).eps*self.period_y_um
        distance=np.where(abs(distance)<=tolerance,0.,distance)
        return (abs(x)<self.mark_width_x_um/2)&(distance<0)

    def local_stack(self,arm,displacement_um=(0.,0.)):
        """Infinite straight-arm surrogate centred at local u=0, away from apex."""
        shift=np.asarray(displacement_um,dtype=float)
        if shift.shape!=(2,) or not np.isfinite(shift).all():
            raise ValueError('displacement_um must be a finite [x,y] pair')
        stack=rectangle(pitch_um=self.normal_pitch_um,depth_um=self.depth_um,duty=self.duty,
            ridge=self.ridge,groove=self.groove,top=self.top,bottom=self.bottom)
        return translate(stack,-self.normal_pitch_um/2+float(self.normal(arm)@shift))

    def recover_displacement(self,left_pair_phase,right_pair_phase,order=1):
        """Physical geometry shift from *test-reference* pair phases; branch limited."""
        if isinstance(order,bool) or not isinstance(order,int) or order<=0:
            raise ValueError('order must be a positive integer')
        phase=np.asarray([left_pair_phase,right_pair_phase],float)
        if not np.isfinite(phase).all():
            raise ValueError('pair phases must be finite')
        operators=np.vstack([self.reciprocal('left'),self.reciprocal('right')])
        return np.linalg.solve(-2*order*operators,phase)
