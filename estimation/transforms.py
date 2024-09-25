from .primitive_linop import *

def RigidTransform(img_shape, factors_trans, factors_tan, factors_sin):
    U  = Translation(img_shape, factors_trans)
    R0 = Rotation(0, img_shape, factors_tan, factors_sin)
    R1 = Rotation(1, img_shape, factors_tan, factors_sin)
    R2 = Rotation(2, img_shape, factors_tan, factors_sin, is_img_input=True)
    A =  U * R0 * R1 * R2
    A.repr_str = 'RigidTransform'
    return A

def GradRigidTransform(partial_idx, img_shape, factors_trans, factors_tan, factors_sin, grad_factors):
    U  = Translation(img_shape, factors_trans)
    R0 = Rotation(0, img_shape, factors_tan, factors_sin)
    R1 = Rotation(1, img_shape, factors_tan, factors_sin)
    R2 = Rotation(2, img_shape, factors_tan, factors_sin, True)
    
    if 0 <= partial_idx <= 2:
        return GradTranslation(partial_idx, img_shape, grad_factors) * R0 * R1 * R2
    elif partial_idx == 3:
        return U * GradRotation(0, img_shape, factors_tan, factors_sin, grad_factors) * R1 * R2
    elif partial_idx == 4:
        return U * R0 * GradRotation(1, img_shape, factors_tan, factors_sin, grad_factors) * R2
    elif partial_idx == 5:
        return U * R0 * R1 * GradRotation(2, img_shape, factors_tan, factors_sin, grad_factors, True) #special case where we need img input
    else:
        raise RuntimeError("Wrong index for the Gradient of the transform!")
    
def HessRigidTransform(partial_first, partial_second, img_shape, factors_trans, factors_tan, factors_sin, grad_factors, hess_factors):
    U  = Translation(img_shape, factors_trans)
    R0 = Rotation(0, img_shape, factors_tan, factors_sin)
    R1 = Rotation(1, img_shape, factors_tan, factors_sin)
    R2 = Rotation(2, img_shape, factors_tan, factors_sin, True)
    D_R0 = GradRotation(0, img_shape, factors_tan, factors_sin, grad_factors)
    D_R1 = GradRotation(1, img_shape, factors_tan, factors_sin, grad_factors)
    D_R2 = GradRotation(2, img_shape, factors_tan, factors_sin, grad_factors, True)
    D2_R0 = HessRotation(0, img_shape, factors_tan, factors_sin, grad_factors, hess_factors)
    D2_R1 = HessRotation(1, img_shape, factors_tan, factors_sin, grad_factors, hess_factors)
    D2_R2 = HessRotation(2, img_shape, factors_tan, factors_sin, grad_factors, hess_factors, True)
    
    if 0 <= partial_first <= partial_second <= 2:
        return HessTranslation(partial_first, partial_second, img_shape, hess_factors) * R0 * R1 * R2
    elif 0 <= partial_first <= 2 and partial_second == 3:
        return GradTranslation(partial_first, img_shape, grad_factors) * D_R0 * R1 * R2 
    elif 0 <= partial_first <= 2 and partial_second == 4:
        return GradTranslation(partial_first, img_shape, grad_factors) * R0 * D_R1 * R2 
    elif 0 <= partial_first <= 2 and partial_second == 5:
        return GradTranslation(partial_first, img_shape, grad_factors) * R0 * R1 * D_R2 
    elif partial_first == 3 and partial_second == 3:
        return U * D2_R0 * R1 * R2
    elif partial_first == 3 and partial_second == 4:
        return U * D_R0 * D_R1 * R2 
    elif partial_first == 3 and partial_second == 5:
        return U * D_R0 * R1 * D_R2 
    elif partial_first == 4 and partial_second == 4:
        return U * R0 * D2_R1 * R2 
    elif partial_first == 4 and partial_second == 5:
        return U * R0 * D_R1 * D_R2 
    elif partial_first == 5 and partial_second == 5:
        return U * R0 * R1 * D2_R2
    else:
        raise RuntimeError("Wrong indices for the Hessian of the transform!")