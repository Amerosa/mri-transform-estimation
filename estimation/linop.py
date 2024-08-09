import sigpy.linop as linop
import sigpy.backend as backend

def get_indicies(index):
    return (index + 1) % 3, (index + 2) % 3

def Rotation(index, img_shape, factors_tan, factors_sin):
    l1, l2 = get_indicies(index)
    l1 -= 3
    l2 -= 3
    Fl1 = linop.FFT(img_shape, axes=(l1,))
    Fl2 = linop.FFT(img_shape, axes=(l2,))
    Tan = linop.Multiply(img_shape, factors_tan)
    Sin = linop.Multiply(img_shape, factors_sin)
    return linop.Compose([Fl1.H, Tan, Fl1,
                          Fl2.H, Sin, Fl2,
                          Fl1.H, Tan, Fl1])

def Translation(img_shape, factors):
    return linop.Compose([linop.IFFT(img_shape, axes=(-1,-2,-3)),
                          linop.Multiply(img_shape, factors),
                          linop.FFT(img_shape, axes=(-1,-2,-3))])

class RTransform(linop.Linop):
        def __init__(self, img_shape, fwd_factors, inv_factors):
            self.fwd_factors = fwd_factors
            self.inv_factors = inv_factors
            self.img_shape = img_shape
            super().__init__(img_shape, img_shape, repr_str="RigidTransform")
        
        def _apply(self, input):
            Q  = Translation(self.img_shape, self.fwd_factors.trans)
            R1 = Rotation(0, self.img_shape, self.fwd_factors.tan[0], self.fwd_factors.sin[0])
            R2 = Rotation(1, self.img_shape, self.fwd_factors.tan[1], self.fwd_factors.sin[1])
            R3 = Rotation(2, self.img_shape, self.fwd_factors.tan[2], self.fwd_factors.sin[2])
            return Q * R1 * R2 * R3 * input
        
        def _adjoint_linop(self):
            Q  = Translation(self.img_shape, self.inv_factors.trans)
            R1 = Rotation(0, self.img_shape, self.inv_factors.tan[0], self.inv_factors.sin[0])
            R2 = Rotation(1, self.img_shape, self.inv_factors.tan[1], self.inv_factors.sin[1])
            R3 = Rotation(2, self.img_shape, self.inv_factors.tan[2], self.inv_factors.sin[2])
            return R3 * R2 * R1 * Q 


class RigidTransform(linop.Linop):
    def __init__(self, parameters, img_shape, device):
        self.parameters = parameters
        self.img_shape = img_shape
        self.img_dims = len(img_shape)
        self.device = device

        self.k1, self.k2, self.k3 = self.generate_kgrid()
        self.r1, self.r2, self.r3 = self.generate_rgrid()
        self.rkgrid_tan, self.rkgrid_sin = self.generate_rkgrid()
        
        self.kgrid = [self.k1, self.k2, self.k3]
        #self.rgrid = [self.r1, self.r2, self.r3]
        xp = device.xp
        self.rgrid = xp.stack([self.r1, self.r2, self.r3], axis=0)

        self.factors_tan = None
        self.factors_sin = None
        self.factors_u   = None
        self.factors_dtan = None
        self.factors_dsin = None
        self.factors_du = None
        self.factors_ddtan = None
        self.factors_ddsin = None
        self.factors_ddu = None

        self.factors_inv_tan = None
        self.factors_inv_sin = None
        self.factors_inv_u   = None

        #Common LinearOps
        self.Tan   = None 
        self.DTan  = None
        self.DDTan = None 
        self.Sin   = None
        self.DSin  = None
        self.DDsin = None
        self.update_state(self.parameters)

        self.R1     = self.Rotation(0)
        self.R2     = self.Rotation(1)
        self.R3     = self.Rotation(2)
        self.DR1    = self.DRotation(0)
        self.DR2    = self.DRotation(1)
        self.DR3    = self.DRotation(2)
        self.DDR1   = self.DDRotation(0)
        self.DDR2   = self.DDRotation(1)
        self.DDR3   = self.DDRotation(2)
        super().__init__(img_shape, img_shape)
    
    def _apply(self, input):
        with backend.get_device(input):
            return self.Transform() * input

    def _adjoint_linop(self):
        R1 = self.Rotation(0, direct=False)
        R2 = self.Rotation(1, direct=False)
        R3 = self.Rotation(2, direct=False)
        Q  = self.Translation(direct=False)
        return R3.H * R2.H * R1.H * Q.H
    
    def Shear(self, factors, axis):
        return linop.Compose([linop.IFFT(self.img_shape, axes=(axis,)),
                              linop.Multiply(self.img_shape, factors),
                              linop.FFT(self.img_shape, axes=(axis,))])
    
    def Rotation(self, index, direct=True):
        axis_one = ((index + 1) % 3) - 3
        axis_two = ((index + 2) % 3) - 3
        if direct:
            factors_tan = self.factors_tan
            factors_sin = self.factors_sin
        else:
            factors_tan = self.factors_inv_tan
            factors_sin = self.factors_inv_sin
        return linop.Compose([self.Shear(factors_tan[index], axis_one),
                              self.Shear(factors_sin[index], axis_two),
                              self.Shear(factors_tan[index], axis_one)])
    
    def Translation(self, direct=True):
        if direct:
            factors = self.factors_u
        else:
            factors = self.factors_inv_u
        return linop.Compose([linop.IFFT(self.img_shape, axes=(-1,-2,-3)),
                              linop.Multiply(self.img_shape, factors),
                              linop.FFT(self.img_shape, axes=(-1,-2,-3))])
    
    def DRotation(self, idx):
        l1, l2  = get_indicies(idx) 
        F = [linop.FFT(self.img_shape, axes=(i,)) for i in range(3)]
        Tan = self.Tan
        Sin = self.Sin
        DSin = self.DSin
        DTan = self.DTan

        InnerOP = linop.Add([DTan[idx]*F[l1]*F[l2].H*Sin[idx]*F[l2]*F[l1].H*Tan[idx],
                             Tan[idx]*F[l1]*F[l2].H*DSin[idx]*F[l2]*F[l1].H*Tan[idx],
                             Tan[idx]*F[l1]*F[l2].H*Sin[idx]*F[l2]*F[l1].H*DTan[idx]])
        return F[l1].H * InnerOP * F[l1]

    def DDRotation(self, idx):
        l1, l2  = get_indicies(idx) 
        F = [linop.FFT(self.img_shape, axes=(i,)) for i in range(3)]
        Tan = self.Tan
        Sin = self.Sin
        DSin = self.DSin
        DTan = self.DTan
        DDTan = self.DDTan
        DDSin = self.DDSin

        InnerOp = linop.Add([
                            DDTan[idx] * F[l1] * F[l2].H * Sin[idx] * F[l2] * F[l1].H * Tan[idx],
                            Tan[idx] * F[l1] * F[l2].H * DDSin[idx] * F[l2] * F[l1].H * Tan[idx],
                            Tan[idx] * F[l1] * F[l2].H * Sin[idx] * F[l2] * F[l1].H * DDTan[idx],
                            2 * DTan[idx] * F[l1] * F[l2].H * DSin[idx] * F[l2] * F[l1].H * Tan[idx],
                            2 * Tan[idx] * F[l1] * F[l2].H * DSin[idx] * F[l1] * F[l1].H * DTan[idx],
                            2 * DTan[idx] * F[l1] * F[l2].H * Sin[idx] * F[l2] * F[l1].H * DTan[idx]
                            ])
        return F[l1].H * InnerOp * F[l1]
    
    def generate_factors_linops(self):
        self.Tan   = [linop.Multiply(self.img_shape, factors) for factors in self.factors_tan] 
        self.DTan  = [linop.Multiply(self.img_shape, factors) for factors in self.factors_dtan]
        self.DDTan = [linop.Multiply(self.img_shape, factors) for factors in self.factors_ddtan]
        self.Sin   = [linop.Multiply(self.img_shape, factors) for factors in self.factors_sin]
        self.DSin  = [linop.Multiply(self.img_shape, factors) for factors in self.factors_dsin]
        self.DDSin = [linop.Multiply(self.img_shape, factors) for factors in self.factors_ddsin]

    def Transform(self):
        return self.Translation() * self.R1 * self.R2 * self.R3
    
    def generate_kgrid(self):
        xp = self.device.xp
        sx, sy, sz = self.img_shape
        sxt = (-(sx//2), -(sx//-2)) if sx > 1 else (0,1)
        syt = (-(sy//2), -(sy//-2)) if sy > 1 else (0,1)
        szt = (-(sz//2), -(sz//-2)) if sz > 1 else (0,1)
        k1, k2, k3 = xp.mgrid[sxt[0]:sxt[1], syt[0]:syt[1], szt[0]:szt[1]].astype(xp.float64)
        k1 *= (2 * xp.pi / sx) 
        k2 *= (2 * xp.pi / sy) 
        k3 *= (2 * xp.pi / sz) 
        return k1, k2, k3
    
    def generate_rgrid(self):
        xp = self.device.xp
        sx, sy, sz = self.img_shape
        r1, r2, r3 = xp.mgrid[:sx, :sy, :sz].astype(xp.float64)
        centroid = tuple(map(lambda x: -(x//-2), self.img_shape))
        r1 -= centroid[0]
        r2 -= centroid[1]
        r3 -= centroid[2]
        return r1, r2, r3
    
    def generate_rkgrid(self):
        return ([self.k2 * self.r3, self.k3 * self.r1, self.k1 * self.r2],
            [self.k3 * self.r2, self.k1 * self.r3, self.k2 * self.r1] )
        
    def calc_factors(self):
        #TODO accomodate for shape of [params, shots]
        xp = self.device.xp
        t1, t2, t3  = self.parameters[:3]
        self.factors_u = t1 * self.k1 + t2 * self.k2 + t3 * self.k3
        self.factors_u = xp.exp(-1j * self.factors_u)
        
        tan_rot = 1j * xp.tan(self.parameters[3:]/2)
        sin_rot = -1j * xp.sin(self.parameters[3:])
        
        self.factors_tan = [xp.exp(tan_rot[idx] * self.rkgrid_tan[idx]) for idx in range(3)]
        self.factors_sin = [xp.exp(sin_rot[idx] * self.rkgrid_sin[idx]) for idx in range(3)]
    
    def calc_inv_factors(self):
        xp = self.device.xp
        t1, t2, t3  = self.parameters[:3]
        self.factors_inv_u = t1 * self.k1 + t2 * self.k2 + t3 * self.k3
        self.factors_inv_u = xp.exp(1j * self.factors_u)

        tan_rot = -1j * xp.tan(self.parameters[3:]/2)
        sin_rot =  1j * xp.sin(self.parameters[3:])
        
        self.factors_inv_tan = [xp.exp(tan_rot[idx] * self.rkgrid_tan[idx]) for idx in range(3)]
        self.factors_inv_sin = [xp.exp(sin_rot[idx] * self.rkgrid_sin[idx]) for idx in range(3)]

    
    def calc_first_order_factors(self):
        xp = self.device.xp
        rotations = self.parameters[:3]

        #self.factors_dtan = [1j*( (1 + xp.tan(rotations[idx]/2) ** 2) / 2) * self.kgrid[(idx+1)%3] * self.rgrid[(idx+2)%3] * self.factors_tan[idx] for idx in range(3)]
        tan_rot = (1 + (xp.tan(rotations/2) ** 2)) / 2
        cos_rot = xp.cos(rotations)

        self.factors_dtan = []
        self.factors_dsin = []
        for idx in range(3):
            l1, l2 = get_indicies(idx)
            self.factors_dtan.append( 1j * tan_rot[idx] * self.kgrid[l1] * self.rgrid[l2] * self.factors_tan[idx])
        #self.factors_dsin = [-1j * xp.cos(rotations[idx] * self.kgrid[(idx+2)%3] * self.rgrid[(idx+1)%3] * self.factors_sin[idx] for idx in range(3))]
        for idx in range(3):
            l1, l2 = get_indicies(idx)
            self.factors_dsin.append( -1j * cos_rot[idx] * self.kgrid[l2] * self.rgrid[l1] * self.factors_sin[idx])
        
        du = [-1j*self.kgrid[idx] * self.factors_u for idx in range(3)]
        self.factors_du = xp.stack(du, axis=0)

    def calc_second_order_factors(self):
        xp = self.device.xp
        rotations = self.parameters[:3]
        self.factors_ddtan = [ (xp.tan(rotations[idx]/2) + 1j*((1+xp.tan(rotations[idx]/2)**2)/2) * self.kgrid[(idx+1)%3] * self.rgrid[(idx+1)%3]) * self.factors_dtan[idx] for idx in range(3) ]
        self.factors_ddsin = [ -1*(xp.tan(rotations[idx] + 1j*xp.cos(rotations[idx] * self.kgrid[(idx+2)%3] * self.rgrid[(idx+1)%3]))) * self.factors_dsin[idx] for idx in range(3)]

        u_part1 = [-self.kgrid[0] * self.kgrid[idx] for idx in range(3)]
        u_part2 = [-self.kgrid[1] * self.kgrid[idx] for idx in range(3)]
        u_part3 = [-self.kgrid[2] * self.kgrid[idx] for idx in range(3)]
        self.factors_ddu = xp.stack([xp.stack(u_part1), xp.stack(u_part2), xp.stack(u_part3)])

    def update_factors(self, parameters, partials=True):
        self.parameters = parameters
        self.calc_factors()
        self.calc_inv_factors()
        if partials:
            self.calc_first_order_factors()
            self.calc_second_order_factors()
    
    def update_state(self, parameters, partials=True):
        self.parameters = parameters
        self.update_factors(parameters, partials=partials)
        self.generate_factors_linops()
        self.R1     = self.Rotation(0)
        self.R2     = self.Rotation(1)
        self.R3     = self.Rotation(2)
        self.DR1    = self.DRotation(0)
        self.DR2    = self.DRotation(1)
        self.DR3    = self.DRotation(2)
        self.DDR1   = self.DDRotation(0)
        self.DDR2   = self.DDRotation(1)
        self.DDR3   = self.DDRotation(2)

    def DTranslation(self, partial_idx):
        F = linop.FFT(self.img_shape, axes=list(range(-self.img_dims, 0)))
        U = linop.Multiply(self.img_shape, self.factors_du[partial_idx])
        return F.H * U * F

    def DDTranslation(self, fst_partial_idx, snd_partial_idx):
        F = linop.FFT(self.img_shape, axes=list(range(-self.img_dims, 0)))
        U = linop.Multiply(self.img_shape, self.factors_ddu[fst_partial_idx, snd_partial_idx])
        return F.H * U * F
        
    def DTransform(self, partial_idx):
        if 0 <= partial_idx <= 2:
            return self.DTranslation(partial_idx) * self.R1 * self.R2 * self.R3
        elif partial_idx == 3:
            return self.Translation() * self.DR1 * self.R2 * self.R3
        elif partial_idx == 4:
            return self.Translation() * self.R1 * self.DR2 * self.R3
        elif partial_idx == 5:
            return self.Translation() * self.R1 * self.R2 * self.DR3
        else:
            raise RuntimeError("Wrong index for the Gradient of the transform!")
        
    def DDTransform(self, fp_idx, sp_idx):
        if 0 <= fp_idx <= sp_idx <= 2:
            return self.DDTranslation(fp_idx, sp_idx) * self.R1 * self.R2 * self.R3
        elif 0 <= fp_idx <= 2 and sp_idx == 3:
            return self.DTranslation(fp_idx) * self.DR1 * self.R2 * self.R3 
        elif 0 <= fp_idx <= 2 and sp_idx == 4:
            return self.DTranslation(fp_idx) * self.R1 * self.DR2 * self.R3 
        elif 0 <= fp_idx <= 2 and sp_idx == 5:
             return self.DTranslation(fp_idx) * self.R1 * self.R2 * self.DR3 
        elif fp_idx == 3 and sp_idx == 3:
           return self.Translation() * self.DDR1 * self.R2 * self.R3
        elif fp_idx == 3 and sp_idx == 4:
            return self.Translation() * self.DR1 * self.DR2 * self.R3 
        elif fp_idx == 3 and sp_idx == 5:
            return self.Translation() * self.DR1 * self.R2 * self.DR3 
        elif fp_idx == 4 and sp_idx == 4:
            return self.Translation() * self.R1 * self.DDR2 * self.R3 
        elif fp_idx == 4 and sp_idx == 5:
            return self.Translation() * self.R1 * self.DR2 * self.DR3 
        elif fp_idx == 5 and sp_idx == 5:
            return self.Translation() * self.R1 * self.R2 * self.DDR3
        else:
            raise RuntimeError("Wrong indices for the Hessian of the transform!")
    