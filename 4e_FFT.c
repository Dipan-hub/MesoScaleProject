#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <complex.h>

#include <fftw3.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main(void){

FILE *fpw;
FILE *FP;
int i1, i2, i3, i4, i5, i6, i7, i8, i9, i10,J;
int half_nx, half_ny;
double denom;
double kx, ky, delta_kx, delta_ky;
double kx2, ky2, k2, k4;

double *vec_kx, *vec_ky;
char NAME[100];
char name[100];
int INDEX;

double a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,eta;
a0 = 0.0; a1 = 0.0; a2 = 8.072789087; a3 = -81.24549382; a4 = 408.0297321; a5 = -1244.129167; a6 = 2444.046270; a7 = -3120.635139; a8 = 2506.663551; a9 = -1151.003178; a10 = 230.2006355;
printf("%le\n",a7);
double noise_str = 1.0e-03;
double random_n;

int time_steps = 200000;
int file_counter = 500;

double Af = 0.1;
double kappa = 0.29/2;
//double D = 1.0;
double M = 5.0;
//double L = 1.0;
double delta_x = 1; // change the deltax,deltay,deltat
double delta_y = 1;
double delta_t = 0.01;
double n1,n2;
int n_x = 400;
int n_y = 400;
int nxny = n_x*n_y;
double inv_nxny = 1.0/nxny;

//system("rm *.dat");
system("mkdir -p val_folder2");


double *c;
double ave_comp;

double *Omega11;
double *Omega12;
double *Omega21;
double *Omega22;

double c11m, c12m, c44m;
double c11p, c12p, c44p;
double eps0_11,eps0_12,eps0_21,eps0_22;
double sig0_11,sig0_12,sig0_22;

c11m = 250.0;
c12m = 150.0;
c44m = 100.0;

c11p = 250.0;
c12p = 150.0;
c44p = 100.0;

eps0_11 = 0.005;
eps0_12 = 0.0;
eps0_21 = 0.0;
eps0_22 = 0.005;

sig0_11 = 0.0;
sig0_12 = 0.0;
sig0_22 = 0.0;


double Ceff[3][3][3][3];
double DeltaC[3][3][3][3];
double S[3][3][3][3];
double Cprecip[3][3][3][3];
double Cmatrix[3][3][3][3];

double A, nu;			/* Zener ansiotropy and Poisson's ratio */

int nrl = 1, nrh = 2, ncl = 1, nch = 2; 
int nrow = nrh-nrl+1, ncol = nch-ncl+1;
double **E;
double **strain;
double **sig_app;
double **epsilon_T;
double **sigma_T;

int n_beta = 100000;
double *beta;
double *beta_prime;

double onebyf;
double jnk;
int factor;
int index_beta;
double steepness_factor = 5.0;

size_t tmp;

double vol_fraction;
double realc;
double b, bp;

double Omega_inv11, Omega_inv12, Omega_inv21, Omega_inv22;
double det_Omega_inv;
double n[3];



fftw_complex *comp;
fftw_complex *g;
fftw_complex *u1, *u2;
fftw_complex *u1_old, *u2_old;
fftw_complex *eps_star11, *eps_star12, *eps_star22;
fftw_complex *Beta;
fftw_complex *mu_el;

fftw_plan planF, planB;

comp = fftw_malloc(nxny*sizeof(fftw_complex));
g = fftw_malloc(nxny*sizeof(fftw_complex));
u1 = fftw_malloc(nxny*sizeof(fftw_complex));
u2 = fftw_malloc(nxny*sizeof(fftw_complex));
u1_old = fftw_malloc(nxny*sizeof(fftw_complex));
u2_old = fftw_malloc(nxny*sizeof(fftw_complex));
eps_star11 = fftw_malloc(nxny* sizeof(fftw_complex));
eps_star12 = fftw_malloc(nxny* sizeof(fftw_complex));
eps_star22 = fftw_malloc(nxny* sizeof(fftw_complex));
Beta = fftw_malloc(nxny* sizeof(fftw_complex));
mu_el = fftw_malloc(nxny* sizeof(fftw_complex));

fftw_plan_with_nthreads(8);

planF = fftw_plan_dft_2d(n_x,n_y,comp,comp, FFTW_FORWARD, FFTW_ESTIMATE);
planB = fftw_plan_dft_2d(n_x,n_y,comp,comp, FFTW_BACKWARD, FFTW_ESTIMATE);

half_nx = (int) n_x/2;
half_ny = (int) n_y/2;

delta_kx = (2.0*M_PI)/(n_x*delta_x);
delta_ky = (2.0*M_PI)/(n_y*delta_y);


c = (double *) malloc((size_t) nxny* sizeof(double));
Omega11 = (double *) malloc((size_t) nxny* sizeof(double));
Omega12 = (double *) malloc((size_t) nxny* sizeof(double));
Omega21 = (double *) malloc((size_t) nxny* sizeof(double));
Omega22 = (double *) malloc((size_t) nxny* sizeof(double));




vec_kx = (double *) malloc((size_t) n_x* sizeof(double));
vec_ky = (double *) malloc((size_t) n_y* sizeof(double));


E=(double **) malloc((size_t)((nrow+1)*sizeof(double*)));
	E += 1;
	E -= nrl;
E[nrl]=(double *) malloc((size_t)((nrow*ncol+1)*sizeof(double)));
	E[1] += 1;
	E[1] -= ncl;
for(i1=nrl+1;i1<=nrh;i1++) E[i1]=E[i1-1]+ncol;


strain=(double **) malloc((size_t)((nrow+1)*sizeof(double*)));
	strain += 1;
	strain -= nrl;
strain[nrl]=(double *) malloc((size_t)((nrow*ncol+1)*sizeof(double)));
	strain[1] += 1;
	strain[1] -= ncl;
for(i1=nrl+1;i1<=nrh;i1++) strain[i1]=strain[i1-1]+ncol;

sig_app=(double **) malloc((size_t)((nrow+1)*sizeof(double*)));
	sig_app += 1;
	sig_app -= nrl;
sig_app[nrl]=(double *) malloc((size_t)((nrow*ncol+1)*sizeof(double)));
	sig_app[1] += 1;
	sig_app[1] -= ncl;
for(i1=nrl+1;i1<=nrh;i1++) sig_app[i1]=sig_app[i1-1]+ncol;

epsilon_T=(double **) malloc((size_t)((nrow+1)*sizeof(double*)));
	epsilon_T += 1;
	epsilon_T -= nrl;
epsilon_T[nrl]=(double *) malloc((size_t)((nrow*ncol+1)*sizeof(double)));
	epsilon_T[1] += 1;
	epsilon_T[1] -= ncl;
for(i1=nrl+1;i1<=nrh;i1++) epsilon_T[i1]=epsilon_T[i1-1]+ncol;

sigma_T=(double **) malloc((size_t)((nrow+1)*sizeof(double*)));
	sigma_T += 1;
	sigma_T -= nrl;
sigma_T[nrl]=(double *) malloc((size_t)((nrow*ncol+1)*sizeof(double)));
	sigma_T[1] += 1;
	sigma_T[1] -= ncl;
for(i1=nrl+1;i1<=nrh;i1++) sigma_T[i1]=sigma_T[i1-1]+ncol;


/* Initial Profile Declaration */

ave_comp = 0.0;
for(i1 = 0; i1< n_x; ++i1){
for(i2 = 0; i2< n_y; ++i2){
/*
random_n = (0.5 - rand()/(double) RAND_MAX);
__real__ comp[i2 + i1*n_y] = 0.5 + noise_str*(random_n);
*/
double dx = i1 - n_x/2.0;               /* distance from centre (pixels) */
double dy = i2 - n_y/2.0;

double a_maj = 20.0/0.9;                /* a ≃ 22.222 px  (20 nm / 0.9)  */
double b_min = 20.0*0.9;                /* b  = 18.0 px   (20 nm*0.9)    */

double ell = (dx*dx)/(a_maj*a_maj) + (dy*dy)/(b_min*b_min);

if(ell <= 1.0){
__real__ comp[i2 + i1*n_y] = 1.0;
__imag__ comp[i2 + i1*n_y] = 0.0;
}

else{
__real__ comp[i2 + i1*n_y] = 0.0065;
__imag__ comp[i2 + i1*n_y] = 0.0;
}


ave_comp = ave_comp + __real__ comp[i2 + i1*n_y];
}}

ave_comp = 1.*ave_comp/nxny;

INDEX = 0;
sprintf(NAME,"val_folder/time_%d.dat",INDEX);
fpw = fopen(NAME,"w");
for(i1 = 0; i1< n_x; ++i1){
for(i2 = 0; i2< n_y; ++i2){
fprintf(fpw,"%d %d %le\n",i1,i2,__real__ comp[i2 + i1*n_y]);
}
fprintf(fpw,"\n");
}
fclose(fpw);

/* Interpolation beta and beta_prime Declaration */

onebyf = 1.0/n_beta;
jnk = -0.1;
factor = (int) (n_beta/10);

index_beta = n_beta+n_beta+n_beta+1;

beta = (double *) malloc((size_t) index_beta* sizeof(double));

beta_prime = (double *) malloc((size_t) index_beta* sizeof(double));

sprintf(NAME,"beta.dat");
fpw = fopen(NAME,"w");
for(i1=0; i1 < n_beta+2.*factor+1; ++i1){
//	beta[i1] = 	jnk - ave_comp;
//	beta_prime[i1] = 	1.0;

		beta[i1] = 0.5
			*(1.0+(tanh(2.0*steepness_factor*jnk-steepness_factor)));
			if(jnk < 0.0) beta[i1] = 0.0;
			else if(jnk > 1.0) beta[i1] = 1.0;

	  bp = 1.0/(cosh(2.0*steepness_factor*jnk-steepness_factor));
		beta_prime[i1] = steepness_factor*bp*bp;
			if(jnk < 0.0) beta_prime[i1] = 0.0;
			else if(jnk > 1.0) beta_prime[i1] = 0.0;


	jnk = jnk + onebyf;

fprintf(fpw,"%d %le\n",i1-factor,beta[i1]);
}
fclose(fpw);

/* Elastic Tensor Generation */

for(i1=1; i1 < 3; ++i1){
for(i2=1; i2 < 3; ++i2){
for(i3=1; i3 < 3; ++i3){
for(i4=1; i4 < 3; ++i4){
	Cprecip[i1][i2][i3][i4] = 0.0;
	Cmatrix[i1][i2][i3][i4] = 0.0;
	Ceff[i1][i2][i3][i4] = 0.0;
	DeltaC[i1][i2][i3][i4] = 0.0;
	S[i1][i2][i3][i4] = 0.0;
}}}}

// Calculate the matrix and precipitate elastic constants 

Cmatrix[1][1][1][1] = c11m;
Cmatrix[1][1][2][2] = c12m;
Cmatrix[1][2][1][2] = c44m;
Cmatrix[1][2][2][1] = Cmatrix[2][1][1][2] = Cmatrix[2][1][2][1]  = Cmatrix[1][2][1][2];
Cmatrix[2][2][1][1] = Cmatrix[1][1][2][2];
Cmatrix[2][2][2][2] = Cmatrix[1][1][1][1];

Cprecip[1][1][1][1] = c11p;
Cprecip[1][1][2][2] = c12p;
Cprecip[1][2][1][2] = c44p;
Cprecip[1][2][2][1] = Cprecip[2][1][1][2] = Cprecip[2][1][2][1]  = Cprecip[1][2][1][2];
Cprecip[2][2][1][1] = Cprecip[1][1][2][2];
Cprecip[2][2][2][2] = Cprecip[1][1][1][1];

	for(i1=1; i1 < 3; ++i1){
	for(i2=1; i2 < 3; ++i2){
	for(i3=1; i3 < 3; ++i3){
	for(i4=1; i4 < 3; ++i4){
		Ceff[i1][i2][i3][i4] = 
			ave_comp*Cprecip[i1][i2][i3][i4] 
				+ (1.0-ave_comp)*Cmatrix[i1][i2][i3][i4];
		DeltaC[i1][i2][i3][i4] = 
			Cprecip[i1][i2][i3][i4] - Cmatrix[i1][i2][i3][i4];
printf("Ceff[%d][%d][%d][%d] = %lf\n",i1,i2,i3,i4,Ceff[i1][i2][i3][i4]);
	}}}}




// Calculate the non-zero compliance tensor components 

nu = 0.5*Ceff[1][1][2][2]/(Ceff[1][1][2][2]+Ceff[1][2][1][2]);
A = 2.0*Ceff[1][2][1][2]/(Ceff[1][1][1][1]-Ceff[1][1][2][2]);

printf("%lf %lf\n",A,nu);

S[1][1][1][1] = (3.0+A-4.0*nu)/(8.0*Ceff[1][2][1][2]);
S[1][1][2][2] = (1.0-A-4.0*nu)/(8.0*Ceff[1][2][1][2]);
S[1][2][1][2] = (1.0+A)/(4.0*A*Ceff[1][2][1][2]);

S[2][2][2][2] = S[1][1][1][1];
S[2][2][1][1] = S[1][1][2][2];
S[1][2][2][1] = S[2][1][1][2] = S[1][2][1][2];
S[2][1][2][1] = S[1][2][1][2];

// Applied Stress initialization
sig_app[1][1] = 0.0;
sig_app[1][2] = 0.0;
sig_app[2][1] = 0.0;
sig_app[2][2] = 0.0;

// Applied Stress 
sig_app[1][1] = sig0_11;
sig_app[1][2] = sig0_12;
sig_app[2][2] = sig0_22;

sig_app[2][1] = sig_app[1][2];


// Eigen Stress initialization
sigma_T[1][1] = 0.0;
sigma_T[1][2] = 0.0;
sigma_T[2][1] = 0.0;
sigma_T[2][2] = 0.0;

// Eigenstrain tensor

epsilon_T[1][1] = eps0_11;
epsilon_T[2][2] = eps0_22;
epsilon_T[1][2] = eps0_12;
epsilon_T[2][1] = eps0_21;

/// Eigenstress - Corresponding to Ceff 

for(i1=1; i1<3; ++i1){
for(i2=1; i2<3; ++i2){
for(i3=1; i3<3; ++i3){
for(i4=1; i4<3; ++i4){
sigma_T[i1][i2] = sigma_T[i1][i2] 
	+ Ceff[i1][i2][i3][i4]*epsilon_T[i3][i4];
}}}}

// Homogeneous Strain Calculation

vol_fraction = 0.0;
for(i1=0; i1<n_x; ++i1){
for(i2=0; i2<n_y; ++i2){
J = i2+n_y*i1;
realc = creal(comp[J]);
b = beta[(int) (n_beta*realc)];
vol_fraction = vol_fraction + b;
}}
vol_fraction = vol_fraction*inv_nxny;

for(i1=1; i1<3; ++i1){
for(i2=1; i2<3; ++i2){
E[i1][i2] = epsilon_T[i1][i2]*vol_fraction;
}}

for(i1=1; i1<3; ++i1){
for(i2=1; i2<3; ++i2){
for(i3=1; i3<3; ++i3){
for(i4=1; i4<3; ++i4){
E[i1][i2] = E[i1][i2] + S[i1][i2][i3][i4]*sig_app[i3][i4];
}}}}

// Wave number venctor declaration to calculate Acoustic Tensor Omega


for(i1=0; i1 < n_x; ++i1){
if(i1 < half_nx) vec_kx[i1] = i1*delta_kx;
else vec_kx[i1] = (i1-n_x)*delta_kx;
}
for(i2=0; i2 < n_y; ++i2){
if(i2 < half_ny) vec_ky[i2] = i2*delta_ky;
else vec_ky[i2] = (i2-n_y)*delta_ky;
}



double el;
double Elastic;





/*** TIME LOOP  ***/
double time = 0.0;
for(INDEX = 1; time<=300; ++INDEX) {
	time = INDEX * delta_t;
    double bulk    = 0.0;
    double grad    = 0.0;   /* we’ll fill this later */
    double Elastic = 0.0;

for(i1=0;i1<n_x;++i1){
for(i2=0;i2<n_y;++i2){
eta = __real__ comp[i2 + i1*n_y];
__real__ g[i2 + i1*n_y] = Af*(a1+2*a2*eta+3*a3*pow(eta,2)+4*a4*pow(eta,3)+5*a5*pow(eta,4)+6*a6*pow(eta,5)+7*a7*pow(eta,6)+8*a8*pow(eta,7)+9*a9*pow(eta,8)+10*a10*pow(eta,9));
//__real__ g[i2 + i1*n_y] = 2.*Af*__real__ comp[i2 + i1*n_y]*(1. - __real__ comp[i2 + i1*n_y])*(1. - 2.*__real__ comp[i2 + i1*n_y]);
__imag__ g[i2 + i1*n_y] = 0.0;

}}

if(INDEX%file_counter == 0 ){
printf("%d\n",INDEX);}
// Acoustic Tensor Calculation
if(INDEX == 1){

for(i1=0; i1<n_x; ++i1){
	 n[1] = vec_kx[i1];
for(i2=0; i2<n_y; ++i2){
	 n[2] = vec_ky[i2];
		J = i2+n_y*i1;

		Omega_inv11 = Ceff[1][1][1][1]*n[1]*n[1]+
			Ceff[1][1][2][1]*n[1]*n[2]+
			Ceff[1][2][1][1]*n[2]*n[1]+
			Ceff[1][2][2][1]*n[2]*n[2];
		Omega_inv12 = Ceff[1][1][1][2]*n[1]*n[1]+
			Ceff[1][1][2][2]*n[1]*n[2]+
			Ceff[1][2][1][2]*n[2]*n[1]+
			Ceff[1][2][2][2]*n[2]*n[2];
		Omega_inv21 = Ceff[2][1][1][1]*n[1]*n[1]+
			Ceff[2][1][2][1]*n[1]*n[2]+
			Ceff[2][2][1][1]*n[2]*n[1]+
			Ceff[2][2][2][1]*n[2]*n[2];
		Omega_inv22 = Ceff[2][1][1][2]*n[1]*n[1]+
			Ceff[2][1][2][2]*n[1]*n[2]+
			Ceff[2][2][1][2]*n[2]*n[1]+
			Ceff[2][2][2][2]*n[2]*n[2];

//printf("%lf %lf %lf %lf %lf %lf %lf \n",Omega_inv11,Omega_inv12,Omega_inv21,Omega_inv22, Ceff[1][1][1][1],n[1],n[2]);			
	det_Omega_inv = Omega_inv11*Omega_inv22 - Omega_inv12*Omega_inv21;
	if(det_Omega_inv != 0.0){
		Omega11[J] = Omega_inv22/det_Omega_inv;  
		Omega22[J] = Omega_inv11/det_Omega_inv;  
		Omega12[J] = -Omega_inv12/det_Omega_inv;  
		Omega21[J] = -Omega_inv21/det_Omega_inv;  
	}
	else{
		Omega11[J] = Omega_inv22;
		Omega22[J] = Omega_inv11;
		Omega12[J] = -1.0*Omega_inv12;
		Omega21[J] = -1.0*Omega_inv21;
	}

}}

}

/** Displacement u_zero Calculation **/

// Fourier transform of the eigenstrain interpolation function 

for(i3=0; i3<n_x; ++i3){
for(i4=0; i4<n_y; ++i4){
	J = i4+n_y*i3;
	u1[J] = 0.0;
	u2[J] = 0.0;
	realc = __real__ comp[J];
    if (realc < 0.0) realc = 0.0;
    if (realc > 1.0) realc = 1.0;
//printf("realc %lf\n",realc);
if(__real__ comp[J] > 2.2 || __real__ comp[J] < -2.2){
printf("Check realc %lf %d. Exiting\n",realc,(int) (n_beta*realc));
//exit(0);
}

//if((int) (n_beta*realc) >=0){ index_beta = factor + (int) (n_beta*realc);}
//else if((int) (n_beta*realc) <0){ index_beta = factor + (int) (n_beta*realc);}
//	Beta[J] = beta[(int) index_beta];
n1 = creal(comp[J]);
	//Beta[J] = beta[(int) (n_beta*creal(comp[J]))] + _Complex_I*0.0;
Beta[J] = realc*realc*realc*(10-15*n1 + 6*n1*n1) + _Complex_I*0.0;

}}
fftw_execute_dft(planF,Beta,Beta);

for(i1=0; i1 < n_x; ++i1){
if(i1 < half_nx) n[1] =  i1*delta_kx;
else n[1] = (i1-n_x)*delta_kx;
for(i2=0; i2 < n_y; ++i2){
if(i2 < half_ny) n[2] =  i2*delta_ky;
else n[2] = (i2-n_y)*delta_ky;
	J = i2+n_y*i1;
	for(i4=1; i4<3; ++i4){
		u1[J] = u1[J] 
		- _Complex_I*Omega11[J]*n[i4]*sigma_T[1][i4]*Beta[J]
		- _Complex_I*Omega21[J]*n[i4]*sigma_T[2][i4]*Beta[J];
		u2[J] = u2[J] 
		- _Complex_I*Omega12[J]*n[i4]*sigma_T[1][i4]*Beta[J]
		- _Complex_I*Omega22[J]*n[i4]*sigma_T[2][i4]*Beta[J];
	}
}}

for(i1=0; i1 < n_x; ++i1){
for(i2=0; i2 < n_y; ++i2){
	J = i2+n_y*i1;
	eps_star11[J] = _Complex_I*vec_kx[i1]*u1[J];
	eps_star22[J] = _Complex_I*vec_ky[i2]*u2[J];
	eps_star12[J] = 0.5*_Complex_I*(vec_kx[i1]*u2[J]+vec_ky[i2]*u1[J]);
}}

// Taking strain back to the real space 

fftw_execute_dft(planB,eps_star11,eps_star11);
fftw_execute_dft(planB,eps_star12,eps_star12);
fftw_execute_dft(planB,eps_star22,eps_star22);
for(i3=0; i3<n_x; ++i3){
for(i4=0; i4<n_y; ++i4){
J = i4+n_y*i3;
eps_star11[J] = eps_star11[J]*inv_nxny;
eps_star22[J] = eps_star22[J]*inv_nxny;
eps_star12[J] = eps_star12[J]*inv_nxny;
}}

Elastic = 0.0;
/* mu_el Calculation */

for(i1=0; i1 < n_x; ++i1){
for(i2=0; i2 < n_y; ++i2){
	J = i2+n_y*i1;
	realc = creal(comp[J]);
    if (realc < 0.0) realc = 0.0;
    if (realc > 1.0) realc = 1.0;
	mu_el[J] = 0.0;
	el = 0.0;
//if((int) (n_beta*realc) >=0){ index_beta = factor + (int) (n_beta*realc);}
//else if((int) (n_beta*realc) <0){ index_beta = factor + (int) (n_beta*realc);}
int idx = (int)(n_beta * realc);
if (idx < 0) idx = 0;
if (idx >= index_beta) idx = index_beta - 1;
b  = beta[idx];
bp = beta_prime[idx];
        n2 = realc;
        if(n2 <=0) {b = 0.;
        bp = 0.;
        }
        else if (n2>1) {b = 1.;
        bp = 0.;
        }
        else
        {
	b = n2*n2*n2*(10-15*n2 + 6*n2*n2);
        bp = 30*n2*n2*(1-2*n2 +n2*n2);
        }
	
	strain[1][1] = creal(eps_star11[J]) - epsilon_T[1][1]*b;
	//[1][1] + creal(eps_star11[J]) - epsilon_T[1][1]*b;
	strain[1][2] = creal(eps_star12[J]) - epsilon_T[1][2]*b;
	//E[1][2] + creal(eps_star12[J]) - epsilon_T[1][2]*b;
	strain[2][1] = creal(eps_star12[J]) - epsilon_T[2][1]*b;
	//E[2][1] + creal(eps_star12[J]) - epsilon_T[2][1]*b;
	strain[2][2] = creal(eps_star22[J]) - epsilon_T[2][2]*b;
        //E[2][2] + creal(eps_star22[J]) - epsilon_T[2][2]*b;

	for(i7=1; i7 < 3; ++i7){
	for(i8=1; i8 < 3; ++i8){
	for(i9=1; i9 < 3; ++i9){
	for(i10=1; i10 < 3; ++i10){
	el += 0.5*Ceff[i7][i8][i9][i10]*strain[i9][i10]*strain[i7][i8];
	}}}}
        
	for(i3=1; i3 < 3; ++i3){
	for(i4=1; i4 < 3; ++i4){
	for(i5=1; i5 < 3; ++i5){
	for(i6=1; i6 < 3; ++i6){
	mu_el[J] = mu_el[J]
		-bp*epsilon_T[i3][i4]*strain[i5][i6]*Ceff[i3][i4][i5][i6];
	}}}}
        Elastic += el;
}}

// === PATCH bulk_energy BEGIN
for (i1 = 0; i1 < n_x; ++i1){
    for (i2 = 0; i2 < n_y; ++i2){
        J = i2 + n_y*i1;
        double eta_local = creal(comp[J]);
        if (eta_local < 0.0) eta_local = 0.0;
        if (eta_local > 1.0) eta_local = 1.0;

        double fbulk =
              a0  + a1*eta_local
            + a2*pow(eta_local,2)  + a3*pow(eta_local,3)
            + a4*pow(eta_local,4)  + a5*pow(eta_local,5)
            + a6*pow(eta_local,6)  + a7*pow(eta_local,7)
            + a8*pow(eta_local,8)  + a9*pow(eta_local,9)
            + a10*pow(eta_local,10);

        bulk += Af * fbulk;          /* Af = well height = 0.1 */
    }
}
// === PATCH bulk_energy END


printf("%le\n",Elastic);
/** Let us take the comp in Fourier space **/

/* comp,g,mu_el are now in Fourier space */
fftw_execute_dft(planF,comp,comp);
fftw_execute_dft(planF,g,g);
fftw_execute_dft(planF,mu_el,mu_el);

/* === PATCH gradient_energy BEGIN */
grad = 0.0;
for (i1 = 0; i1 < n_x; ++i1){
    /* kx */
    if (i1 < half_nx) kx =  i1 * delta_kx;
    else               kx = (i1 - n_x) * delta_kx;
    for (i2 = 0; i2 < n_y; ++i2){
        /* ky */
        if (i2 < half_ny) ky =  i2 * delta_ky;
        else              ky = (i2 - n_y) * delta_ky;

        k2 = kx*kx + ky*ky;
        J  = i2 + n_y*i1;

        /* |η_k|^2 = real^2 + imag^2 */
        double mod2 = creal(comp[J])*creal(comp[J]) +
                      cimag(comp[J])*cimag(comp[J]);

        grad += k2 * mod2;
    }
}
/* prefactor κ and 1/N (already in inv_nxny) */
grad *= kappa * inv_nxny;
/* === PATCH gradient_energy END */


for(i1=0;i1<n_x;++i1){
if(i1 < half_nx) kx = i1*delta_kx;
else kx = (i1-n_x)*delta_kx;

for(i2=0;i2<n_y;++i2){
if(i2 < half_ny) ky = i2*delta_ky;
else ky = (i2-n_y)*delta_ky;

k2 = kx*kx + ky*ky;
k4 = k2*k2;

	J = i2+n_y*i1;

denom = (1 + 2*kappa*k4*delta_t);

//if(i1 == half_nx && i2 == half_nx){
//printf("%lf %lf\n",denom, k4);}

comp[J] = 1.*(comp[J] - M*k2*delta_t*(g[J] + mu_el[J]))/denom;

}}


fftw_execute_dft(planB,comp,comp);

ave_comp = 0.0;
for(i1=0;i1<n_x;++i1){
for(i2=0;i2<n_y;++i2){
__real__ comp[i2 +
 i1*n_y] = 1.*comp[i2 + i1*n_y]/(n_x*n_y);
__imag__ comp[i2 + i1*n_y] = 0.0;

ave_comp = ave_comp + __real__ comp[i2 + i1*n_y];
}}

ave_comp = 1.*ave_comp/nxny;
if(ave_comp > 1.1 || ave_comp< -0.2){
printf("Average composition out of the boud. Exiting!!");
exit(0);
}
//printf("ave_comp = %lf\n",ave_comp);



if(INDEX%file_counter == 0){
sprintf(NAME,"val_folder/time_%d.dat",INDEX);
fpw = fopen(NAME,"w");
for(i1 = 0; i1< n_x; ++i1){
for(i2 = 0; i2< n_y; ++i2){
fprintf(fpw,"%d %d %le\n",i1,i2,__real__ comp[i2 + i1*n_y]);
}
fprintf(fpw,"\n");
}
fclose(fpw);
}
FP = fopen("energy2.dat","a");
//fprintf(FP,"%d \t %le \n",INDEX,Elastic);
// === PATCH csv_output BEGIN
fprintf(FP,
    "%.6lf,%.6lf,%.6lf,%.6lf,%.6lf\n",
    time,             /* time                   */
    Elastic,          /* elastic energy         */
    bulk,             /* bulk energy            */
    grad,             /* gradient (0 for now)   */
    Elastic + bulk + grad);   /* total               */
// === PATCH csv_output END

fclose(FP);
}
fftw_free(comp);
fftw_free(g);
fftw_free(u1);
fftw_free(u2);
fftw_free(u1_old);
fftw_free(u2_old);
fftw_free(eps_star11);
fftw_free(eps_star12);
fftw_free(eps_star22);
fftw_free(Beta);
fftw_free(mu_el);

fftw_destroy_plan(planF);
fftw_destroy_plan(planB);

free(c);
free(Omega11);
free(Omega12);
free(Omega21);
free(Omega22);

free(beta);
free(beta_prime);

free(vec_kx);
free(vec_ky);

	free(E[nrl]+ncl-1);
	free(E+nrl-1);

	free(strain[nrl]+ncl-1);
	free(strain+nrl-1);

	free(sig_app[nrl]+ncl-1);
	free(sig_app+nrl-1);

	free(epsilon_T[nrl]+ncl-1);
	free(epsilon_T+nrl-1);

	free(sigma_T[nrl]+ncl-1);
	free(sigma_T+nrl-1);


return 0;
}
