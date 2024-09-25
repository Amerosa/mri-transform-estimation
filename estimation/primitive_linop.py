import sigpy as sp

def Rotation(index, img_shape, factors_tan, factors_sin, is_img_input=False):
    #factors are tan/sin are a list of 3 arrays with shape (shots, x, y, z)
    l1 = (index + 1) % 3 - 3
    l2 = (index + 2) % 3 - 3
    num_shots = factors_tan[0].shape[0]
    pipeline_shape = [num_shots] + list(img_shape)
    
    Fl1 = sp.linop.FFT(pipeline_shape, axes=(l1,))
    Fl2 = sp.linop.FFT(pipeline_shape, axes=(l2,))
    Tan = sp.linop.Multiply(pipeline_shape, factors_tan[index])
    Sin = sp.linop.Multiply(pipeline_shape, factors_sin[index])

    if is_img_input:
        A = Fl1.H * Tan * Fl1 * Fl2.H * Sin * Fl2 * Fl1.H * sp.linop.Multiply(img_shape, factors_tan[index]) * sp.linop.FFT(img_shape, axes=(l1,))
    else:
        A = Fl1.H * Tan * Fl1 * Fl2.H * Sin * Fl2 * Fl1.H * Tan * Fl1

    A.repr_str = f'Rotation Axis {index}'
    return A

def GradRotation(axis, img_shape, factors_tan, factors_sin, grad_factors, is_img_input=False):
    l1 = (axis + 1) % 3 - 3
    l2 = (axis + 2) % 3 - 3 
    num_shots = factors_tan[0].shape[0]
    pipeline_shape = [num_shots] + list(img_shape)
    Fl1 = sp.linop.FFT(pipeline_shape, axes=(l1,))
    Fl2 = sp.linop.FFT(pipeline_shape, axes=(l2,))
    
    Tan   = sp.linop.Multiply(pipeline_shape, factors_tan[axis])
    Sin   = sp.linop.Multiply(pipeline_shape, factors_sin[axis])
    D_Tan = sp.linop.Multiply(pipeline_shape, grad_factors['tan'][axis])
    D_Sin = sp.linop.Multiply(pipeline_shape, grad_factors['sin'][axis])

    if not is_img_input:
        InnerOp = sp.linop.Add([D_Tan * Fl1 * Fl2.H * Sin * Fl2 * Fl1.H * Tan,
                         Tan * Fl1 * Fl2.H * D_Sin * Fl2 * Fl1.H * Tan,
                         Tan * Fl1 * Fl2.H * Sin * Fl2 * Fl1.H * D_Tan])
        return Fl1.H * InnerOp * Fl1

    #If the image is input then we need to adjust the first transform to take in appropriate shape
    InnerOp = sp.linop.Add([D_Tan * Fl1 * Fl2.H * Sin * Fl2 * Fl1.H * sp.linop.Multiply(img_shape, factors_tan[axis]),
                         Tan * Fl1 * Fl2.H * D_Sin * Fl2 * Fl1.H * sp.linop.Multiply(img_shape, factors_tan[axis]),
                         Tan * Fl1 * Fl2.H * Sin * Fl2 * Fl1.H * sp.linop.Multiply(img_shape, grad_factors['tan'][axis])])

    return Fl1.H * InnerOp * sp.linop.FFT(img_shape, axes=(l1,))

def HessRotation(axis, img_shape, factors_tan, factors_sin, grad_factors, hess_factors, is_img_input=False):
        l1 = (axis + 1) % 3 - 3
        l2 = (axis + 2) % 3 - 3
        num_shots = factors_tan[0].shape[0]
        pipeline_shape = [num_shots] + list(img_shape)
        
        Fl1 = sp.linop.FFT(pipeline_shape, axes=(l1,))
        Fl2 = sp.linop.FFT(pipeline_shape, axes=(l2,))
    
        Tan    = sp.linop.Multiply(pipeline_shape, factors_tan[axis])
        Sin    = sp.linop.Multiply(pipeline_shape, factors_sin[axis])
        D_Tan  = sp.linop.Multiply(pipeline_shape, grad_factors['tan'][axis])
        D_Sin  = sp.linop.Multiply(pipeline_shape, grad_factors['sin'][axis])
        D2_Tan = sp.linop.Multiply(pipeline_shape, hess_factors['tan'][axis])
        D2_Sin = sp.linop.Multiply(pipeline_shape, hess_factors['sin'][axis])
        
        if not is_img_input:
            InnerOp = sp.linop.Add([
                            D2_Tan * Fl1 * Fl2.H * Sin * Fl2 * Fl1.H * Tan,
                            Tan * Fl1 * Fl2.H * D2_Sin * Fl2 * Fl1.H * Tan,
                            Tan * Fl1 * Fl2.H * Sin * Fl2 * Fl1.H * D2_Tan,
                            2 * D_Tan * Fl1 * Fl2.H * D_Sin * Fl2 * Fl1.H * Tan,
                            2 * Tan * Fl1 * Fl2.H * D_Sin * Fl2 * Fl1.H * D_Tan,
                            2 * D_Tan * Fl1 * Fl2.H * Sin * Fl2 * Fl1.H * D_Tan
                            ])
            return Fl1.H * InnerOp * Fl1

        InnerOp = sp.linop.Add([
                        D2_Tan * Fl1 * Fl2.H * Sin * Fl2 * Fl1.H * sp.linop.Multiply(img_shape, factors_tan[axis]),
                        Tan * Fl1 * Fl2.H * D2_Sin * Fl2 * Fl1.H * sp.linop.Multiply(img_shape, factors_tan[axis]),
                        Tan * Fl1 * Fl2.H * Sin * Fl2 * Fl1.H * sp.linop.Multiply(img_shape, hess_factors['tan'][axis]),
                        2 * D_Tan * Fl1 * Fl2.H * D_Sin * Fl2 * Fl1.H * sp.linop.Multiply(img_shape, factors_tan[axis]),
                        2 * Tan * Fl1 * Fl2.H * D_Sin * Fl2 * Fl1.H * sp.linop.Multiply(img_shape, grad_factors['tan'][axis]),
                        2 * D_Tan * Fl1 * Fl2.H * Sin * Fl2 * Fl1.H * sp.linop.Multiply(img_shape, grad_factors['tan'][axis])
                        ])
        return Fl1.H * InnerOp * sp.linop.FFT(img_shape, axes=(l1,))

def Translation(img_shape, factors, is_img_input=False):
    num_shots = factors.shape[0]
    pipeline_shape = [num_shots] + list(img_shape)
    if is_img_input:
        F = sp.linop.FFT(img_shape, axes=(-1,-2,-3))
        U = sp.linop.Multiply(img_shape, factors)
    else:
        F = sp.linop.FFT(pipeline_shape, axes=(-1,-2,-3))
        U = sp.linop.Multiply(pipeline_shape, factors)

    A = sp.linop.IFFT(pipeline_shape, axes=(-1,-2,-3)) * U * F
    A.repr_str = 'Translation'
    return A

def GradTranslation(axis, img_shape, grad_factors, is_img_input=False):
    A = Translation(img_shape, grad_factors['trans'][axis], is_img_input)
    A.repr_str = f'Grad Translation wrt Axis {axis}'
    return A

def HessTranslation(first_partial, second_partial, img_shape, hess_factors, is_img_input=False):
    A = Translation(img_shape, hess_factors['trans'][first_partial][second_partial])
    A.repr_str = f'Hess Translation wrt Partial: {first_partial}|{second_partial}'
    return A