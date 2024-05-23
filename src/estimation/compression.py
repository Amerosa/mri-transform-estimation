import numpy as np

def coilArrayCompression(S, y, perc, gpu):
    reg = 0.001
    
    if perc != 1:
        # Initialize
        if y is not None and len(y):
            NY = np.shape(y)
        else:
            NY = np.shape(S)
        
        # Compute P
        if gpu:
            S = np.array(S, dtype=np.complex64)
        Sconj = np.conj(S) #(128, 128, 1, 32)
        normal = np.sum(Sconj * S, axis=3) + reg
        normal = np.reciprocal(normal) #(128,128, 1)
        Saux = S * normal[..., np.newaxis]

        
        Sconj = np.reshape(Sconj, (128,128,1,1,32))
        #np.transpose(Sconj, (0, 1, 2, 4, 3))
        P = np.zeros((NY[3], NY[3]), dtype=np.complex64)
        for s in range(NY[2]):
            Sbis = Saux[:, :, s, :, np.newaxis] * Sconj[:, :, s, :, :]
            P += np.sum(np.sum(Sbis, axis=1), axis=0)

        # Compute F
        U, F, _ = np.linalg.svd(P)
        Ftot = np.sum(F)
        for m in range(NY[3]):
            if (np.sum(F[:m]) / Ftot) >= perc:
                M = m
                break
        print(M)
        
        # Compressing matrix
        A = np.conj(U).T
        A = np.transpose(A[0:11,:])
        A = np.expand_dims(A, axis=(0,1,2))

        # Compress sensitivities
        NS = np.shape(S)
        temp_shape = (NS[0], NS[1], NS[2], M)
        SC = np.zeros(temp_shape, dtype=np.complex64)

        for m in range(M):
            SC[..., m] = np.sum(S * A[0,0,0,:,m], axis=3)

        
        if y is not None and len(y):
            # Compress data
            temp_shape = (NY[0], NY[1], NY[2], M)
            yC = np.zeros(temp_shape, dtype=np.complex64)
            if gpu:
                yC = np.array(yC)
                y = np.array(y)
            for m in range(M):
                yC[..., m] = np.sum(y * A[0,0,0,:, m], axis=3)
            if gpu:
                yC = np.array(yC)
        else:
            yC = None
    else:
        SC = S
        yC = y
    
    SC = np.moveaxis(SC, -1, 0)
    return SC, yC
