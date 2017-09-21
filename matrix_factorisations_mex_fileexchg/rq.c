/*
 * RQ factorization
 *
 * R = rq(A)
 * [R,Q] = rq(A)
 * [R,Q] = rq(A,0)
 *
 * example compile command (see also make_factor.m):
 * mex -O rq.c libmwlapack.lib
 * or
 * mex -O rq.c libmwblas.lib libmwlapack.lib (>= R2007B)
 *
 * calls the SGERQF/DGERQF/CGERQF/ZGERQF and
 * SORGRQ/DORGRQ/CUNGRQ/ZUNGRQ named LAPACK functions
 *
 * Ivo Houtzager
 *
 * Delft Center of Systems and Control
 * The Netherlands, 2010
 */

#include "mex.h"
#include "factor.h"
#include "matrix.h"

void rq_double(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    ptrdiff_t lwork, info = 1;
    double *Qpr, *Rpr, *Ipr, *Qpi, *Rpi, *Ipi, *Ap, *ptau, *pwork, *pwork2, *psize, *psize2;
    size_t m, n, min_mn, lda, m2, element_size = sizeof(double), econ = 0, cplx = 0, dc = 1;
    mwIndex i, j, start, limit;
    mxClassID classid = mxDOUBLE_CLASS;
    mxComplexity cplxflag = mxREAL;

    /* check complex */
    if (mxIsComplex(prhs[0])) {
        cplxflag = mxCOMPLEX;
        cplx = 1;
        dc = 2;
    }
    
    /* get matrix data */
    m = mxGetM(prhs[0]);
    n = mxGetN(prhs[0]);
    if (m == 0 || n == 0) {
        if (nlhs == 1) {
            plhs[0] = mxCreateNumericMatrix(m,n,classid,cplxflag);
        }
        else {
            plhs[0] = mxCreateNumericMatrix(m,n,classid,cplxflag);
            plhs[1] = mxCreateNumericMatrix(n,n,classid,cplxflag);
            if (n != 0) {
                Qpr = mxGetData(plhs[0]);
                for (j=0; j<n; j++) {
                    Qpr[j*n + j] = 1;
                }
            }
        }
        return;
    }
    if (nrhs == 2) {
        if (mxGetScalar(prhs[1]) == 0) {
            econ = 1;
        }
    }

    /* allocate tau */
    min_mn = min(m,n);
    ptau = mxMalloc(dc*min_mn*element_size);

    /* allocate A matrix */
    if ((m < n) && (econ != 1)) {
        Ap = mxMalloc(dc*n*n*element_size);
        lda = n;
    }
    else {
        Ap = mxMalloc(dc*m*n*element_size);
        lda = m;
    }

    /* copy input to A matrix */
    if (cplx) {
        Ipr = mxGetData(prhs[0]);
        Ipi = mxGetImagData(prhs[0]);
        for (i=0; i<m; i++) {
            for (j=0; j<n; j++) {
                Ap[j*2*lda+2*i] = Ipr[j*m+i];
                Ap[j*2*lda+2*i+1] = Ipi[j*m+i];
            }
        }
    }
    else {
        Ipr = mxGetData(prhs[0]);
        for (i=0; i<m; i++) {
            for (j=0; j<n; j++) {
                Ap[j*lda+i] = Ipr[j*m+i];
            }
        }
    }

    /* determine blocksize */
    lwork = -1;
    psize = mxMalloc(dc*element_size);
    if (cplx) {
        zgerqf(&m, &n, Ap, &lda, ptau, psize, &lwork, &info);
    }
    else {
        dgerqf(&m, &n, Ap, &lda, ptau, psize, &lwork, &info);
    }
    lwork = psize[0];
    mxFree(psize);

    /* allocate workspace */
    pwork = mxMalloc(dc*lwork*element_size);

    /* calls the DGERQF function */
    if (cplx) {
        zgerqf(&m, &n, Ap, &lda, ptau, pwork, &lwork, &info);
    }
    else {
        dgerqf(&m, &n, Ap, &lda, ptau, pwork, &lwork, &info);
    }
    mxFree(pwork);
    if (info != 0) {
        mxFree(ptau);
        mxFree(Ap);
        if (cplx) {
            mexErrMsgTxt("ZGERQF not successful");
        }
        else {
            mexErrMsgTxt("DGERQF not successful");
        }
    }

    /* extract lower triangular part */
    if ((econ == 1) && m < n) {
        plhs[0] = mxCreateNumericMatrix(m,m,classid,cplxflag);
        if (cplx) {
            Rpr = mxGetData(plhs[0]);
            Rpi = mxGetImagData(plhs[0]);
            for (j=0; j<m; j++) {
                limit = j<m-1 ? j:m-1;
                for (i=0; i<=limit; i++) {
                    Rpr[j*m+i] = Ap[(j+n-m)*2*lda+2*i];
                    Rpi[j*m+i] = Ap[(j+n-m)*2*lda+2*i+1];
                }
            }
        }
        else {
            Rpr = mxGetData(plhs[0]);
            for (j=0; j<m; j++) {
                limit = j<m-1 ? j:m-1;
                for (i=0; i<=limit; i++) {
                    Rpr[j*m+i] = Ap[(j+n-m)*lda+i];
                }
            }
        }
    }
    else {
        plhs[0] = mxCreateNumericMatrix(m,n,classid,cplxflag);
        if (cplx) {
            Rpr = mxGetData(plhs[0]);
            Rpi = mxGetImagData(plhs[0]);
            if (m < n) {
                for (j=0; j<m; j++) {
                    limit = j<m-1 ? j:m-1;
                    for (i=0; i<=limit; i++) {
                        Rpr[(j+n-m)*m+i] = Ap[(j+n-m)*2*lda+2*i];
                        Rpi[(j+n-m)*m+i] = Ap[(j+n-m)*2*lda+2*i+1];
                    }
                }
            }
            else {
                for (j=0; j<n; j++) {
                    limit = j < min_mn-1 ? j : min_mn-1;
                    for (i=0; i<=limit+m-n; i++) {
                        Rpr[j*m+i] = Ap[j*2*lda+2*i];
                        Rpi[j*m+i] = Ap[j*2*lda+2*i+1];
                    }
                }
            }
        }
        else {
            Rpr = mxGetData(plhs[0]);
            if (m < n) {
                for (j=0; j<m; j++) {
                    limit = j<m-1 ? j:m-1;
                    for (i=0; i<=limit; i++) {
                        Rpr[(j+n-m)*m+i] = Ap[(j+n-m)*lda+i];
                    }
                }
            }
            else {
                for (j=0; j<n; j++) {
                    limit = j < min_mn-1 ? j : min_mn-1;
                    for (i=0; i<=limit+m-n; i++) {
                        Rpr[j*m+i] = Ap[j*lda+i];
                    }
                }
            }
        }
    }

    if (nlhs == 2) {
        m2 = (econ == 1) ? min_mn : n;
        
        if (cplx) {
            if (m > n) {
                for (j=0; j<n; j++) {
                    start = j<n ? j:n;
                    for (i=start+m-n; i<m; i++) {
                        Ap[j*2*lda+2*i-2*m+2*n] = Ap[j*2*lda+2*i];
                        Ap[j*2*lda+2*i-2*m+2*n+1] = Ap[j*2*lda+2*i+1];
                    }
                }
            }
            else if ((m < n) && (econ != 1)) {
                for (j=0; j<n-1; j++) {
                    start = j<n-m ? 0:j-n+m;
                    for (i=m; i>start; i--) {
                        Ap[j*2*lda+2*(i-1)+2*n-2*m] = Ap[j*2*lda+2*(i-1)];
                        Ap[j*2*lda+2*(i-1)+2*n-2*m+1] = Ap[j*2*lda+2*(i-1)+1];
                    }
                }
            }
        }
        else {
            if (m > n) {
                for (j=0; j<n; j++) {
                    start = j<n ? j:n;
                    for (i=start+m-n; i<m; i++) {
                        Ap[j*lda+i-m+n] = Ap[j*lda+i];
                    }
                }
            }
            else if ((m < n) && (econ != 1)) {
                for (j=0; j<n-1; j++) {
                    start = j<n-m ? 0:j-n+m;
                    for (i=m; i>start; i--) {
                        Ap[j*lda+(i-1)+n-m] = Ap[j*lda+(i-1)];
                    }
                }
            }
        }
        
        /* determine blocksize */
        lwork = -1;
        psize2 = mxMalloc(dc*element_size);
        if (cplx) {
            zungrq(&m2, &n, &min_mn, Ap, &lda, ptau, psize2, &lwork, &info);
        }
        else {
            dorgrq(&m2, &n, &min_mn, Ap, &lda, ptau, psize2, &lwork, &info);
        }
        lwork = psize2[0];
        mxFree(psize2);

        /* allocate workspace */
        pwork2 = mxMalloc(dc*lwork*element_size);

        /* calls the DORGRQ function */
        if (cplx) {
            zungrq(&m2, &n, &min_mn, Ap, &lda, ptau, pwork2, &lwork, &info);
        }
        else {
            dorgrq(&m2, &n, &min_mn, Ap, &lda, ptau, pwork2, &lwork, &info);
        }
        mxFree(pwork2);
        if (info != 0) {
            mxFree(ptau);
            mxFree(Ap);
            if (cplx) {
                mexErrMsgTxt("ZUNGRQ not successful");
            }
            else {
                mexErrMsgTxt("DORGRQ not successful");
            }
        }

        /* allocate Q matrix */
        plhs[1] = mxCreateNumericMatrix(m2,n,classid,cplxflag);
        if (cplx) {
            Qpr = mxGetData(plhs[1]);
            Qpi = mxGetImagData(plhs[1]);
            
            /* copy Ap to Qp */
            for (i=0; i<m2; i++) {
                for (j=0; j<n; j++) {
                    Qpr[j*m2+i] = Ap[j*2*lda+2*i];
                    Qpi[j*m2+i] = Ap[j*2*lda+2*i+1];
                }
            }
        }
        else {
            Qpr = mxGetData(plhs[1]);
            
            /* copy Ap to Qp */
            for (i=0; i<m2; i++) {
                for (j=0; j<n; j++) {
                    Qpr[j*m2+i] = Ap[j*lda+i];
                }
            }
        }
    }

    mxFree(ptau);
    mxFree(Ap);
}


void rq_single(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    ptrdiff_t lwork, info = 1;
    float *Qpr, *Rpr, *Ipr, *Qpi, *Rpi, *Ipi, *Ap, *ptau, *pwork, *pwork2, *psize, *psize2;
    size_t m, n, min_mn, lda, m2, element_size = sizeof(float), econ = 0, cplx = 0, dc = 1;
    mwIndex i, j, start, limit;
    mxClassID classid = mxSINGLE_CLASS;
    mxComplexity cplxflag = mxREAL;

    /* check complex */
    if (mxIsComplex(prhs[0])) {
        cplxflag = mxCOMPLEX;
        cplx = 1;
        dc = 2;
    }
    
    /* get matrix data */
    m = mxGetM(prhs[0]);
    n = mxGetN(prhs[0]);
    if (m == 0 || n == 0) {
        if (nlhs == 1) {
            plhs[0] = mxCreateNumericMatrix(m,n,classid,cplxflag);
        }
        else {
            plhs[0] = mxCreateNumericMatrix(m,n,classid,cplxflag);
            plhs[1] = mxCreateNumericMatrix(n,n,classid,cplxflag);
            if (n != 0) {
                Qpr = mxGetData(plhs[0]);
                for (j=0; j<n; j++) {
                    Qpr[j*n + j] = 1;
                }
            }
        }
        return;
    }
    if (nrhs == 2) {
        if (mxGetScalar(prhs[1]) == 0) {
            econ = 1;
        }
    }

    /* allocate tau */
    min_mn = min(m,n);
    ptau = mxMalloc(dc*min_mn*element_size);

    /* allocate A matrix */
    if ((m < n) && (econ != 1)) {
        Ap = mxMalloc(dc*n*n*element_size);
        lda = n;
    }
    else {
        Ap = mxMalloc(dc*m*n*element_size);
        lda = m;
    }

    /* copy input to A matrix */
    if (cplx) {
        Ipr = mxGetData(prhs[0]);
        Ipi = mxGetImagData(prhs[0]);
        for (i=0; i<m; i++) {
            for (j=0; j<n; j++) {
                Ap[j*2*lda+2*i] = Ipr[j*m+i];
                Ap[j*2*lda+2*i+1] = Ipi[j*m+i];
            }
        }
    }
    else {
        Ipr = mxGetData(prhs[0]);
        for (i=0; i<m; i++) {
            for (j=0; j<n; j++) {
                Ap[j*lda+i] = Ipr[j*m+i];
            }
        }
    }

    /* determine blocksize */
    lwork = -1;
    psize = mxMalloc(dc*element_size);
    if (cplx) {
        cgerqf(&m, &n, Ap, &lda, ptau, psize, &lwork, &info);
    }
    else {
        sgerqf(&m, &n, Ap, &lda, ptau, psize, &lwork, &info);
    }
    lwork = psize[0];
    mxFree(psize);

    /* allocate workspace */
    pwork = mxMalloc(dc*lwork*element_size);

    /* calls the DGERQF function */
    if (cplx) {
        cgerqf(&m, &n, Ap, &lda, ptau, pwork, &lwork, &info);
    }
    else {
        sgerqf(&m, &n, Ap, &lda, ptau, pwork, &lwork, &info);
    }
    mxFree(pwork);
    if (info != 0) {
        mxFree(ptau);
        mxFree(Ap);
        if (cplx) {
            mexErrMsgTxt("CGERQF not successful");
        }
        else {
            mexErrMsgTxt("SGERQF not successful");
        }
    }

    /* extract lower triangular part */
    if ((econ == 1) && m < n) {
        plhs[0] = mxCreateNumericMatrix(m,m,classid,cplxflag);
        if (cplx) {
            Rpr = mxGetData(plhs[0]);
            Rpi = mxGetImagData(plhs[0]);
            for (j=0; j<m; j++) {
                limit = j<m-1 ? j:m-1;
                for (i=0; i<=limit; i++) {
                    Rpr[j*m+i] = Ap[(j+n-m)*2*lda+2*i];
                    Rpi[j*m+i] = Ap[(j+n-m)*2*lda+2*i+1];
                }
            }
        }
        else {
            Rpr = mxGetData(plhs[0]);
            for (j=0; j<m; j++) {
                limit = j<m-1 ? j:m-1;
                for (i=0; i<=limit; i++) {
                    Rpr[j*m+i] = Ap[(j+n-m)*lda+i];
                }
            }
        }
    }
    else {
        plhs[0] = mxCreateNumericMatrix(m,n,classid,cplxflag);
        if (cplx) {
            Rpr = mxGetData(plhs[0]);
            Rpi = mxGetImagData(plhs[0]);
            if (m < n) {
                for (j=0; j<m; j++) {
                    limit = j<m-1 ? j:m-1;
                    for (i=0; i<=limit; i++) {
                        Rpr[(j+n-m)*m+i] = Ap[(j+n-m)*2*lda+2*i];
                        Rpi[(j+n-m)*m+i] = Ap[(j+n-m)*2*lda+2*i+1];
                    }
                }
            }
            else {
                for (j=0; j<n; j++) {
                    limit = j < min_mn-1 ? j : min_mn-1;
                    for (i=0; i<=limit+m-n; i++) {
                        Rpr[j*m+i] = Ap[j*2*lda+2*i];
                        Rpi[j*m+i] = Ap[j*2*lda+2*i+1];
                    }
                }
            }
        }
        else {
            Rpr = mxGetData(plhs[0]);
            if (m < n) {
                for (j=0; j<m; j++) {
                    limit = j<m-1 ? j:m-1;
                    for (i=0; i<=limit; i++) {
                        Rpr[(j+n-m)*m+i] = Ap[(j+n-m)*lda+i];
                    }
                }
            }
            else {
                for (j=0; j<n; j++) {
                    limit = j < min_mn-1 ? j : min_mn-1;
                    for (i=0; i<=limit+m-n; i++) {
                        Rpr[j*m+i] = Ap[j*lda+i];
                    }
                }
            }
        }
    }

    if (nlhs == 2) {
        m2 = (econ == 1) ? min_mn : n;
        
        if (cplx) {
            if (m > n) {
                for (j=0; j<n; j++) {
                    start = j<n ? j:n;
                    for (i=start+m-n; i<m; i++) {
                        Ap[j*2*lda+2*i-2*m+2*n] = Ap[j*2*lda+2*i];
                        Ap[j*2*lda+2*i-2*m+2*n+1] = Ap[j*2*lda+2*i+1];
                    }
                }
            }
            else if ((m < n) && (econ != 1)) {
                for (j=0; j<n-1; j++) {
                    start = j<n-m ? 0:j-n+m;
                    for (i=m; i>start; i--) {
                        Ap[j*2*lda+2*(i-1)+2*n-2*m] = Ap[j*2*lda+2*(i-1)];
                        Ap[j*2*lda+2*(i-1)+2*n-2*m+1] = Ap[j*2*lda+2*(i-1)+1];
                    }
                }
            }
        }
        else {
            if (m > n) {
                for (j=0; j<n; j++) {
                    start = j<n ? j:n;
                    for (i=start+m-n; i<m; i++) {
                        Ap[j*lda+i-m+n] = Ap[j*lda+i];
                    }
                }
            }
            else if ((m < n) && (econ != 1)) {
                for (j=0; j<n-1; j++) {
                    start = j<n-m ? 0:j-n+m;
                    for (i=m; i>start; i--) {
                        Ap[j*lda+(i-1)+n-m] = Ap[j*lda+(i-1)];
                    }
                }
            }
        }
        
        /* determine blocksize */
        lwork = -1;
        psize2 = mxMalloc(dc*element_size);
        if (cplx) {
            cungrq(&m2, &n, &min_mn, Ap, &lda, ptau, psize2, &lwork, &info);
        }
        else {
            sorgrq(&m2, &n, &min_mn, Ap, &lda, ptau, psize2, &lwork, &info);
        }
        lwork = psize2[0];
        mxFree(psize2);

        /* allocate workspace */
        pwork2 = mxMalloc(dc*lwork*element_size);

        /* calls the DORGRQ function */
        if (cplx) {
            cungrq(&m2, &n, &min_mn, Ap, &lda, ptau, pwork2, &lwork, &info);
        }
        else {
            sorgrq(&m2, &n, &min_mn, Ap, &lda, ptau, pwork2, &lwork, &info);
        }
        mxFree(pwork2);
        if (info != 0) {
            mxFree(ptau);
            mxFree(Ap);
            if (cplx) {
                mexErrMsgTxt("CUNGRQ not successful");
            }
            else {
                mexErrMsgTxt("SORGRQ not successful");
            }
        }

        /* allocate Q matrix */
        plhs[1] = mxCreateNumericMatrix(m2,n,classid,cplxflag);
        if (cplx) {
            Qpr = mxGetData(plhs[1]);
            Qpi = mxGetImagData(plhs[1]);
            
            /* copy Ap to Qp */
            for (i=0; i<m2; i++) {
                for (j=0; j<n; j++) {
                    Qpr[j*m2+i] = Ap[j*2*lda+2*i];
                    Qpi[j*m2+i] = Ap[j*2*lda+2*i+1];
                }
            }
        }
        else {
            Qpr = mxGetData(plhs[1]);
            
            /* copy Ap to Qp */
            for (i=0; i<m2; i++) {
                for (j=0; j<n; j++) {
                    Qpr[j*m2+i] = Ap[j*lda+i];
                }
            }
        }
    }

    mxFree(ptau);
    mxFree(Ap);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* check for proper number of arguments */
    if (nrhs != 1 && nrhs != 2) {
        mexErrMsgTxt("RQ requires one or two input arguments.");
    }
    if (nlhs > 2) {
        mexErrMsgTxt("Too many output arguments.");
    }
    if (!mxIsNumeric(prhs[0]) || mxIsSparse(prhs[0])) {
        mexErrMsgTxt( "Input must be a full matrix." );
    }
    if (mxIsDouble(prhs[0])) {
        rq_double(nlhs, plhs, nrhs, prhs);
    }
    else if (mxIsSingle(prhs[0])) {
        rq_single(nlhs, plhs, nrhs, prhs);
    }
    else {
        mexErrMsgTxt( "Class is not supported." );
    }
}
