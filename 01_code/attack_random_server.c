// attack_random_server.c
//
// Compile:
// gcc -O2 -Wall -Wextra -o build/attack_random_server attack_random_server.c mt19937-64.c -lm
//
// Run:
// ./attack_random_server N sam_num a Kr_input T target_node REPEAT [EARLY_EXIT] [seed]
//
// Strategy:
//   Pre-scan: 5 anchor points — (±PI, ±W_MAX) × 4 + (0, 0).
//     If both E_safe_max and E_danger_min are found, skip to Phase 2.
//   Phase 1 (random init): uniformly sample (theta, w) until both
//     E_safe_max and E_danger_min are found (if anchor scan was insufficient).
//   Phase 2 (uniform boundary): sample E_temp uniformly (in log-space, i.e.
//     geometrically) around the boundary pair, using the energy-safety
//     concept's geometric mean/deviation:
//       m_ES     = sqrt(E_safety * E_danger)   (geometric mean)
//       sigma_ES = sqrt(E_danger / E_safety)   (multiplicative deviation)
//     where E_safety = min(E_safe_max, E_danger_min),
//           E_danger = max(E_safe_max, E_danger_min).
//     Search range extends each known boundary outward by one more
//     factor of sigma_ES (multiplicatively, not additively):
//       [E_safety / sigma_ES,  E_danger * sigma_ES]
//     which is symmetric in log-space and always non-negative.
//     E_temp is then sampled log-uniformly within this range
//     (i.e., uniform in log(E_temp)), consistent with the geometric-mean
//     framing of energy safety.
//     theta sampled uniformly in (-PI, PI), w inverted from E_temp.
//     Continues for remaining REPEAT budget.
//
// Safe criterion: num_synced == N  AND  no link phase-difference crossed +-PI.
// EARLY_EXIT=1: stop simulation as soon as crossing detected (faster, approx).
// EARLY_EXIT=0: full simulation — desync/danger/safe distinguished by num_synced.
//
// Output: result_raw3/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

/* -----------------------------------------------------------------------
 * Tunable parameters
 * ----------------------------------------------------------------------- */
#define H              0.001
#define PI             3.141592653589793238462643383279502884
#define SYNC_TOL       5e-2
#define W_MAX          20.0
#define PROGRESS_EVERY 1000
#define EARLY_FRAC     0.01     /* for survivability */

/* -----------------------------------------------------------------------
 * MT RNG
 * ----------------------------------------------------------------------- */
void init_genrand64(unsigned long long seed);
double genrand64_real2(void);   /* [0,1) */

/* -----------------------------------------------------------------------
 * Network struct
 * ----------------------------------------------------------------------- */
typedef struct {
    int     N;
    int    *degree;
    int   **nn;
    double **K;
    double *P;
    double *mass;
    double *gamma;
} Network;

/* -----------------------------------------------------------------------
 * Utilities
 * ----------------------------------------------------------------------- */
static void die(const char *msg) {
    fprintf(stderr, "ERROR: %s\n", msg);
    exit(EXIT_FAILURE);
}
static void *xmalloc(size_t size) {
    void *ptr = malloc(size);
    if (!ptr) die("malloc failed");
    return ptr;
}
static FILE *xfopen(const char *path, const char *mode) {
    FILE *fp = fopen(path, mode);
    if (!fp) { fprintf(stderr,"ERROR: failed to open: %s\n",path); exit(EXIT_FAILURE); }
    return fp;
}
static void ensure_dir(const char *d) {
    if (mkdir(d,0755)!=0 && errno!=EEXIST) {
        fprintf(stderr,"ERROR: mkdir '%s': %s\n",d,strerror(errno));
        exit(EXIT_FAILURE);
    }
}
static double wrap_to_pi(double x) {
    x = fmod(x+PI, 2.0*PI);
    if (x<0.0) x+=2.0*PI;
    return x-PI;
}

/* -----------------------------------------------------------------------
 * Energy functions
 * ----------------------------------------------------------------------- */
static double potential_diff(const Network *net, int target,
                              double theta_trial, const double *theta_eq)
{
    double delta_theta = wrap_to_pi(theta_trial - theta_eq[target]);
    double V_trial=0.0, V_eq=0.0;
    for (int j=0;j<net->degree[target];j++) {
        int    nb  = net->nn[target][j];
        double kij = net->K[target][j];
        V_trial -= kij*cos(theta_trial      - theta_eq[nb]);
        V_eq    -= kij*cos(theta_eq[target] - theta_eq[nb]);
    }
    V_trial -= net->P[target]*delta_theta;
    return V_trial - V_eq;
}

static double local_energy_diff(const Network *net, int target,
                                 double theta_trial, double w_trial,
                                 const double *theta_eq)
{
    return 0.5*net->mass[target]*w_trial*w_trial
           + potential_diff(net,target,theta_trial,theta_eq);
}

/* -----------------------------------------------------------------------
 * Local mean-field UEP estimate for a single node.
 *
 * Equilibrium condition (with omega=0 so damping drops out):
 *   P_i = sum_j K_ij * sin(theta_i - theta_j)
 * Treating neighbors as fixed at their equilibrium angles, combine them
 * into one local mean-field phasor:
 *   R_i * exp(-i*psi_i) = sum_j K_ij * exp(-i*theta_eq[j])
 * so that P_i / R_i = sin(theta_i - psi_i), which has two roots in
 * [0, 2*PI): the known stable equilibrium theta_eq[target], and the
 * other root is the (local-mean-field-approximated) UEP.
 *
 * Returns the UEP's delta-theta (relative to theta_eq[target], wrapped
 * to (-PI, PI]) via *out_dtheta, and sets *out_valid=1 on success.
 * *out_valid=0 (out_dtheta undefined) if the local mean-field has no
 * real solution (|P_i/R_i| > 1) or is degenerate (R_i ~= 0) — in that
 * case the caller should fall back to the original unrestricted scheme.
 * ----------------------------------------------------------------------- */
static double compute_uep_dtheta(const Network *net, int target,
                                  const double *theta_eq, int *out_valid)
{
    double sum_cos = 0.0, sum_sin = 0.0;
    for (int j=0;j<net->degree[target];j++) {
        int    nb  = net->nn[target][j];
        double kij = net->K[target][j];
        sum_cos += kij*cos(theta_eq[nb]);
        sum_sin += kij*sin(theta_eq[nb]);
    }
    double R = sqrt(sum_cos*sum_cos + sum_sin*sum_sin);
    if (R < 1e-9) { *out_valid = 0; return 0.0; }
    double psi   = atan2(sum_sin, sum_cos);
    double ratio = net->P[target] / R;
    if (ratio > 1.0 || ratio < -1.0) { *out_valid = 0; return 0.0; }

    double base_asin = asin(ratio);          /* in [-PI/2, PI/2] */
    double sol1 = psi + base_asin;
    double sol2 = psi + (PI - base_asin);

    double cand1 = wrap_to_pi(sol1);
    double cand2 = wrap_to_pi(sol2);
    double d1 = fabs(wrap_to_pi(cand1 - theta_eq[target]));
    double d2 = fabs(wrap_to_pi(cand2 - theta_eq[target]));

    double uep_abs   = (d1 < d2) ? cand2 : cand1;   /* the OTHER root */
    double uep_dtheta = wrap_to_pi(uep_abs - theta_eq[target]);

    if (fabs(uep_dtheta) < 1e-9) { *out_valid = 0; return 0.0; } /* degenerate */

    *out_valid = 1;
    return uep_dtheta;
}

/* dth를 (-PI, PI]에서 균등샘플링하되, UEP 반대쪽(먼 쪽) 금지구간은 제외.
 * uep_valid=0이면 제한 없이 전체 범위에서 샘플링(기존과 동일). */
static double sample_dtheta_restricted(int uep_valid, double uep_dtheta)
{
    if (!uep_valid) {
        return (2.0*genrand64_real2() - 1.0) * PI;
    }
    if (uep_dtheta < 0.0) {
        /* 금지구간: (-PI, uep_dtheta) -> 허용: [uep_dtheta, PI] */
        return uep_dtheta + genrand64_real2() * (PI - uep_dtheta);
    } else {
        /* 금지구간: (uep_dtheta, PI) -> 허용: [-PI, uep_dtheta] */
        return -PI + genrand64_real2() * (uep_dtheta + PI);
    }
}

/* -----------------------------------------------------------------------
 * Network I/O
 * ----------------------------------------------------------------------- */
static void free_network(Network *net) {
    if (!net) return;
    if (net->nn) { for(int i=0;i<net->N;i++) free(net->nn[i]); free(net->nn); }
    if (net->K)  { for(int i=0;i<net->N;i++) free(net->K[i]);  free(net->K);  }
    free(net->degree); free(net->P); free(net->mass); free(net->gamma);
    net->nn=NULL; net->K=NULL; net->degree=NULL; net->P=NULL;
    net->mass=NULL; net->gamma=NULL;
}

static void load_network(Network *net, int N,
                          const char *node_file, const char *edge_file, double Kr)
{
    net->N=N;
    net->degree=(int*)   xmalloc(sizeof(int)*N);
    net->P     =(double*)xmalloc(sizeof(double)*N);
    net->mass  =(double*)xmalloc(sizeof(double)*N);
    net->gamma =(double*)xmalloc(sizeof(double)*N);
    net->nn=NULL; net->K=NULL;
    for(int i=0;i<N;i++){net->degree[i]=0;net->P[i]=0.0;net->mass[i]=1.0;net->gamma[i]=0.1;}

    FILE *fp=xfopen(node_file,"r");
    for(int i=0;i<N;i++){
        int id; double p;
        double m=1.0,g=0.1;
        int nread=fscanf(fp,"%d %lf %lf %lf",&id,&p,&m,&g);
        if(nread<2){fclose(fp);die("node file read error");}
        if(id<0||id>=N){fclose(fp);die("node id out of range");}
        net->P[id]=p;
        if(nread>=3) net->mass[id]=m;
        if(nread>=4) net->gamma[id]=g;
    }
    fclose(fp);

    fp=xfopen(edge_file,"r");
    int c1,c2; double c3;
    while(fscanf(fp,"%d %d %lf",&c1,&c2,&c3)==3){
        if(c1<0||c1>=N||c2<0||c2>=N){fclose(fp);die("edge node id out of range");}
        net->degree[c1]++; net->degree[c2]++;
    }
    fclose(fp);

    net->nn=(int**)xmalloc(sizeof(int*)*N);
    net->K=(double**)xmalloc(sizeof(double*)*N);
    int *tmp=(int*)xmalloc(sizeof(int)*N);
    for(int i=0;i<N;i++){
        net->nn[i]=(int*)xmalloc(sizeof(int)*net->degree[i]);
        net->K[i]=(double*)xmalloc(sizeof(double)*net->degree[i]);
        tmp[i]=0;
    }
    fp=xfopen(edge_file,"r");
    while(fscanf(fp,"%d %d %lf",&c1,&c2,&c3)==3){
        int i1=tmp[c1]++, i2=tmp[c2]++;
        net->nn[c1][i1]=c2; net->K[c1][i1]=Kr*c3;
        net->nn[c2][i2]=c1; net->K[c2][i2]=Kr*c3;
    }
    fclose(fp);
    free(tmp);
}

static void load_initial_condition(int N, const char *init_file,
                                    double *theta_init, double *w_init)
{
    FILE *fp=xfopen(init_file,"r");
    for(int i=0;i<N;i++){
        int id; double wi,thi;
        if(fscanf(fp,"%d %lf %lf",&id,&wi,&thi)!=3){fclose(fp);die("init file read error");}
        if(id<0||id>=N){fclose(fp);die("init node id out of range");}
        w_init[id]=wi; theta_init[id]=thi;
    }
    fclose(fp);
}

/* -----------------------------------------------------------------------
 * RK4 integrator
 * ----------------------------------------------------------------------- */
static void rk4_step(const Network *net,
                     double *theta, double *w,
                     double *k1_th, double *k2_th, double *k3_th, double *k4_th,
                     double *k1_w,  double *k2_w,  double *k3_w,  double *k4_w,
                     double *th2, double *th3, double *th4)
{
    int N=net->N;
    for(int i=0;i<N;i++){
        double c=0;
        for(int j=0;j<net->degree[i];j++) c+=net->K[i][j]*sin(theta[net->nn[i][j]]-theta[i]);
        k1_w[i] =H*(net->P[i]-net->gamma[i]*w[i]+c)/net->mass[i];
        k1_th[i]=H*w[i];
    }
    for(int i=0;i<N;i++) th2[i]=theta[i]+0.5*k1_th[i];
    for(int i=0;i<N;i++){
        double c=0;
        for(int j=0;j<net->degree[i];j++) c+=net->K[i][j]*sin(th2[net->nn[i][j]]-th2[i]);
        k2_w[i] =H*(net->P[i]-net->gamma[i]*(w[i]+0.5*k1_w[i])+c)/net->mass[i];
        k2_th[i]=H*(w[i]+0.5*k1_w[i]);
    }
    for(int i=0;i<N;i++) th3[i]=theta[i]+0.5*k2_th[i];
    for(int i=0;i<N;i++){
        double c=0;
        for(int j=0;j<net->degree[i];j++) c+=net->K[i][j]*sin(th3[net->nn[i][j]]-th3[i]);
        k3_w[i] =H*(net->P[i]-net->gamma[i]*(w[i]+0.5*k2_w[i])+c)/net->mass[i];
        k3_th[i]=H*(w[i]+0.5*k2_w[i]);
    }
    for(int i=0;i<N;i++) th4[i]=theta[i]+k3_th[i];
    for(int i=0;i<N;i++){
        double c=0;
        for(int j=0;j<net->degree[i];j++) c+=net->K[i][j]*sin(th4[net->nn[i][j]]-th4[i]);
        k4_w[i] =H*(net->P[i]-net->gamma[i]*(w[i]+k3_w[i])+c)/net->mass[i];
        k4_th[i]=H*(w[i]+k3_w[i]);
    }
    for(int i=0;i<N;i++){
        w[i]    +=(k1_w[i] +2*k2_w[i] +2*k3_w[i] +k4_w[i] )/6.0;
        theta[i]+=(k1_th[i]+2*k2_th[i]+2*k3_th[i]+k4_th[i])/6.0;
    }
}

/* -----------------------------------------------------------------------
 * Diagnostics
 * ----------------------------------------------------------------------- */
static double order_parameter(int N, const double *theta){
    double sc=0,ss=0;
    for(int i=0;i<N;i++){sc+=cos(theta[i]);ss+=sin(theta[i]);}
    return sqrt(sc*sc+ss*ss)/(double)N;
}
static int count_synced_nodes(int N,const double *w,double tol){
    int c=0; for(int i=0;i<N;i++) if(fabs(w[i])<tol) c++; return c;
}
static double max_abs_w(int N,const double *w){
    double m=0; for(int i=0;i<N;i++) if(fabs(w[i])>m) m=fabs(w[i]); return m;
}
static double max_residual(const Network *net,const double *theta){
    double mr=0;
    for(int i=0;i<net->N;i++){
        double c=0;
        for(int j=0;j<net->degree[i];j++) c+=net->K[i][j]*sin(theta[net->nn[i][j]]-theta[i]);
        double r=fabs(net->P[i]+c); if(r>mr) mr=r;
    }
    return mr;
}

/* -----------------------------------------------------------------------
 * Single simulation — returns safe flag (synced AND no link crossed +-PI)
 * early_exit=1: stop rk4 immediately on crossing (num_synced reflects
 *               mid-simulation state — likely < N, classified as desync).
 * early_exit=0: full T steps — num_synced reflects final convergence.
 * ----------------------------------------------------------------------- */
static int run_one(const Network *net, int N, int target, int T,
                   const double *theta_init, const double *w_init,
                   double dtheta, double dw,
                   double *theta, double *w,
                   double *k1_th,double *k2_th,double *k3_th,double *k4_th,
                   double *k1_w, double *k2_w, double *k3_w, double *k4_w,
                   double *th2, double *th3, double *th4,
                   int n_edges,
                   const int *edge_i, const int *edge_j,
                   double *edge_prev,
                   int early_exit,
                   int *out_ns, double *out_op, double *out_mw, double *out_mr,
                   double *out_mw_early)
{
    for(int i=0;i<N;i++){theta[i]=theta_init[i]; w[i]=w_init[i];}
    theta[target] += dtheta;
    w[target]     += dw;

    for(int e=0;e<n_edges;e++)
        edge_prev[e] = wrap_to_pi(theta[edge_i[e]] - theta[edge_j[e]]);

    int crossed = 0;
    int early_steps = (int)(EARLY_FRAC * (double)T);
    double mw_early = max_abs_w(N, w);   /* t=0 시점도 포함 */

    for(int t=0;t<T;t++){
        rk4_step(net,theta,w,k1_th,k2_th,k3_th,k4_th,k1_w,k2_w,k3_w,k4_w,th2,th3,th4);
        if(t < early_steps){
            double mw_now = max_abs_w(N, w);
            if(mw_now > mw_early) mw_early = mw_now;
        }
        if(!crossed){
            for(int e=0;e<n_edges;e++){
                double curr = wrap_to_pi(theta[edge_i[e]] - theta[edge_j[e]]);
                double prev = edge_prev[e];
                if(prev*curr < 0.0 && fabs(prev) > PI*0.5){
                    crossed = 1;
                    if(early_exit) goto sim_done;
                    break;
                }
                edge_prev[e] = curr;
            }
        }
    }

    sim_done:
    *out_ns = count_synced_nodes(N, w, SYNC_TOL);
    *out_op = order_parameter(N, theta);
    *out_mw = max_abs_w(N, w);
    *out_mr = max_residual(net, theta);
    *out_mw_early = mw_early;

    return (*out_ns == N && !crossed);
}

/* -----------------------------------------------------------------------
 * write_row
 * ----------------------------------------------------------------------- */
static void write_row(FILE *fp, int N, int target, double Kr,
                      int ns, double dw, double dth, double de,
                      double op, double mw, double mr, int safe,
                      const double *w, const double *theta,
                      int T, int repeat, unsigned long long seed,
                      double E_temp, double E_safe_max, double E_danger_min,
                      double mw_early,
                      const char *phase_label)
{
    fprintf(fp,"%d,%.6f,%d,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%d",
            target,Kr,ns,dw,dth,de,op,mw,mr,safe);
    for(int i=0;i<N;i++) fprintf(fp,",%.10f",w[i]);
    for(int i=0;i<N;i++) fprintf(fp,",%.10f",theta[i]);
    fprintf(fp,",%d,%d,%llu,%.10f,%.10f,%.10f,%.10f,%s\n",
            T,repeat,seed,E_temp,E_safe_max,E_danger_min,mw_early,phase_label);
}

/* -----------------------------------------------------------------------
 * main
 * ----------------------------------------------------------------------- */
int main(int argc, char *argv[]) {
    if(argc<7||argc>11){
        fprintf(stderr,
            "Usage: %s N sam_num Kr_input T target REPEAT [EARLY_EXIT] [seed] [INIT_ONLY] [apply_uep]\n"
            "  mass/gamma loaded per-node from node file.\n"
            "  EARLY_EXIT: 1=stop sim on crossing (default=1), 0=full sim\n"
            "  seed: RNG seed (default=time-based)\n"
            "  apply_uep: 1=UEP 반대편(먼 쪽) 배제구간 적용(default), 0=전체범위(-PI~PI) 섭동\n",
            argv[0]);
        return EXIT_FAILURE;
    }

    int    N       = atoi(argv[1]);
    int    sam_num = atoi(argv[2]);
    double Kr      = atof(argv[3])*0.01;
    int    T       = atoi(argv[4]);
    int    target  = atoi(argv[5]);
    int    REPEAT  = atoi(argv[6]);
    int    EARLY_EXIT = 1;
    unsigned long long seed = 0;
    int seed_set = 0;

    if(argc>=8)  EARLY_EXIT = atoi(argv[7]);
    if(argc>=9){ seed = strtoull(argv[8],NULL,10); seed_set = 1; }
    int    INIT_ONLY = 0;
    if(argc>=10) INIT_ONLY = atoi(argv[9]);
    int    apply_uep = 1;   /* 기본값: UEP 배제구간 적용 (기존 동작과 동일) */
    if(argc==11) apply_uep = atoi(argv[10]);
    if(!seed_set){
        seed = (unsigned long long)time(NULL)
             + 1000003ULL*(unsigned long long)sam_num
             + 10007ULL*(unsigned long long)target
             + 101ULL*(unsigned long long)N
             + (unsigned long long)(1000.0*Kr);
    }

    if(N<=0)      die("N must be positive");
    if(T<=0)      die("T must be positive");
    if(REPEAT<=0) die("REPEAT must be positive");
    if(target<0||target>=N) die("target_node out of range");

    init_genrand64(seed);
    char result_dir[32];
    if (INIT_ONLY) {
        /* apply_uep=0(전체범위 섭동) -> result_raw2, apply_uep=1(UEP 제한) -> result_raw1 */
        snprintf(result_dir, sizeof(result_dir), apply_uep ? "result_raw1" : "result_raw2");
    } else {
        snprintf(result_dir, sizeof(result_dir), "result_raw3");
    }
    ensure_dir(result_dir);

    char node_file[256], edge_file[256], init_file[256], result_file[256];
    snprintf(node_file,   sizeof(node_file),   "network/%d_sample%d_nodes.txt",N,sam_num);
    snprintf(edge_file,   sizeof(edge_file),   "network/%d_sample%d_edges.txt",N,sam_num);
    snprintf(init_file,   sizeof(init_file),   "init/%d_sample%d_init_Kr_%.2f.txt",N,sam_num,Kr);
    snprintf(result_file, sizeof(result_file), "%s/%d_analysis%d_n_%d_Kr_%.2f.csv", result_dir,
             N,sam_num,target,Kr);

    fprintf(stderr,"INFO: N=%d sam_num=%d Kr=%.4f T=%d target=%d "
                   "REPEAT=%d EARLY_EXIT=%d seed=%llu\n",
            N,sam_num,Kr,T,target,REPEAT,EARLY_EXIT,seed);

    Network net;
    load_network(&net,N,node_file,edge_file,Kr);

    double *theta      = (double*)xmalloc(sizeof(double)*N);
    double *w          = (double*)xmalloc(sizeof(double)*N);
    double *theta_init = (double*)xmalloc(sizeof(double)*N);
    double *w_init     = (double*)xmalloc(sizeof(double)*N);
    load_initial_condition(N,init_file,theta_init,w_init);

    double *k1_w  = (double*)xmalloc(sizeof(double)*N);
    double *k2_w  = (double*)xmalloc(sizeof(double)*N);
    double *k3_w  = (double*)xmalloc(sizeof(double)*N);
    double *k4_w  = (double*)xmalloc(sizeof(double)*N);
    double *k1_th = (double*)xmalloc(sizeof(double)*N);
    double *k2_th = (double*)xmalloc(sizeof(double)*N);
    double *k3_th = (double*)xmalloc(sizeof(double)*N);
    double *k4_th = (double*)xmalloc(sizeof(double)*N);
    double *th2   = (double*)xmalloc(sizeof(double)*N);
    double *th3   = (double*)xmalloc(sizeof(double)*N);
    double *th4   = (double*)xmalloc(sizeof(double)*N);

    int n_edges = 0;
    for(int i=0;i<N;i++) n_edges += net.degree[i];
    n_edges /= 2;
    int    *edge_i    = (int*)   xmalloc(sizeof(int)   *n_edges);
    int    *edge_j    = (int*)   xmalloc(sizeof(int)   *n_edges);
    double *edge_prev = (double*)xmalloc(sizeof(double)*n_edges);
    {
        int e=0;
        for(int i=0;i<N;i++)
            for(int k=0;k<net.degree[i];k++){
                int j=net.nn[i][k];
                if(j>i){ edge_i[e]=i; edge_j[e]=j; edge_prev[e]=0.0; e++; }
            }
    }

    FILE *fp = xfopen(result_file,"w");
    setvbuf(fp,NULL,_IOFBF,1<<20);

    fprintf(fp,"target,Kr,num_synced,attack_w,attack_theta,delta_e,"
               "order_parameter,max_abs_w,max_residual,safe");
    for(int i=0;i<N;i++) fprintf(fp,",w_%d",i);
    for(int i=0;i<N;i++) fprintf(fp,",theta_%d",i);
    fprintf(fp,",steps,repeat,seed,E_temp,E_safe_max,E_danger_min,max_abs_w_early,phase\n");

    double E_safe_max   = NAN;
    double E_danger_min = NAN;
    int repeat_used = 0;

    /* -----------------------------------------------------------------------
     * Local mean-field UEP estimate (한 번만 계산).
     * UEP의 부호에 따라 anchor 3점과 이후 theta 샘플링 금지구간을 정한다.
     * ----------------------------------------------------------------------- */
    int    uep_valid;
    double uep_dtheta = compute_uep_dtheta(&net, target, theta_init, &uep_valid);
    if (!apply_uep) {
        uep_valid = 0;  /* apply_uep=0: UEP 제한을 끄고 전체범위(-PI~PI) 섭동 */
    }
    if (!apply_uep) {
        fprintf(stderr,"INFO: apply_uep=0, UEP 배제구간 미적용 (전체범위 섭동)\n");
    } else if (uep_valid) {
        fprintf(stderr,"INFO: UEP dtheta = %.6f (%s side)\n",
                uep_dtheta, (uep_dtheta < 0.0) ? "negative" : "positive");
    } else {
        fprintf(stderr,"INFO: UEP estimate invalid/degenerate, "
                       "falling back to unrestricted sampling\n");
    }

    /* -----------------------------------------------------------------------
     * Pre-scan: anchor points covering extremes of the (theta, omega) space.
     * 집중탐색(INIT_ONLY=false)에서만 수행한다 — 초기 E_safe_max/E_danger_min
     * 경계를 빨리 확보해 phase2의 경계 집중 탐색을 시작할 수 있게 하기 위함.
     * 균등탐색(INIT_ONLY=true)은 anchor 없이 처음부터 바로 phase1을 수행한다.
     * UEP가 유효하면 UEP 반대편(먼 쪽) ±PI 두 점 + 원점, 총 3점만 사용한다.
     * UEP가 무효하면 기존과 동일하게 5점((±PI,±W_MAX)×4 + (0,0))을 사용한다.
     * Each is written to CSV with phase label "anchor".
     * ----------------------------------------------------------------------- */
    if (!INIT_ONLY) {
        double anchors5[5][2] = {
            { PI,  W_MAX}, {-PI,  W_MAX},
            { PI, -W_MAX}, {-PI, -W_MAX},
            { 0.0,  0.0 }
        };
        double anchors3[3][2];
        int n_anchors;
        double (*anchors)[2];

        if (uep_valid) {
            double far_theta = (uep_dtheta < 0.0) ? PI : -PI;
            anchors3[0][0] = 0.0;       anchors3[0][1] = 0.0;
            anchors3[1][0] = far_theta; anchors3[1][1] = W_MAX;
            anchors3[2][0] = far_theta; anchors3[2][1] = -W_MAX;
            anchors = anchors3;
            n_anchors = 3;
        } else {
            anchors = anchors5;
            n_anchors = 5;
        }

        for(int ai = 0; ai < n_anchors; ai++){
            double dth = anchors[ai][0];
            double dw  = anchors[ai][1];
            double theta_trial = theta_init[target] + dth;
            double E_temp = local_energy_diff(&net, target, theta_trial, dw, theta_init);
            double de = E_temp;

            int ns; double op, mw, mr, mw_early;
            int safe = run_one(&net,N,target,T,theta_init,w_init,
                               dth,dw,theta,w,
                               k1_th,k2_th,k3_th,k4_th,
                               k1_w,k2_w,k3_w,k4_w,
                               th2,th3,th4,
                               n_edges,edge_i,edge_j,edge_prev,
                               EARLY_EXIT,&ns,&op,&mw,&mr,&mw_early);

            if(safe){
                if(isnan(E_safe_max) || E_temp > E_safe_max) E_safe_max = E_temp;
            } else {
                if(isnan(E_danger_min) || E_temp < E_danger_min) E_danger_min = E_temp;
            }

            write_row(fp,N,target,Kr,ns,dw,dth,de,op,mw,mr,safe,
                      w,theta,T,repeat_used,seed,E_temp,E_safe_max,E_danger_min,mw_early,"anchor");
            repeat_used++;
        }
        fprintf(stderr,"INFO: anchor scan done  E_safe_max=%.4f  E_danger_min=%.4f\n",
                E_safe_max, E_danger_min);
    } else {
        fprintf(stderr,"INFO: INIT_ONLY=1 (균등탐색), anchor scan 생략 -> phase1부터 바로 시작\n");
    }

    /* -----------------------------------------------------------------------
     * Phase 1: random uniform sampling until both boundaries found
     * ----------------------------------------------------------------------- */
    while(repeat_used < REPEAT){
        if(!isnan(E_safe_max) && !isnan(E_danger_min) && !INIT_ONLY) break;

        double dth = sample_dtheta_restricted(uep_valid, uep_dtheta);
        double dw  = (2.0*genrand64_real2() - 1.0) * W_MAX;

        double theta_trial = theta_init[target] + dth;
        double E_temp = local_energy_diff(&net, target, theta_trial, dw, theta_init);
        double de = E_temp;

        int ns; double op, mw, mr, mw_early;
        int safe = run_one(&net,N,target,T,theta_init,w_init,
                           dth,dw,theta,w,
                           k1_th,k2_th,k3_th,k4_th,
                           k1_w,k2_w,k3_w,k4_w,
                           th2,th3,th4,
                           n_edges,edge_i,edge_j,edge_prev,
                           EARLY_EXIT,&ns,&op,&mw,&mr,&mw_early);

        if(safe){
            if(isnan(E_safe_max) || E_temp > E_safe_max) E_safe_max = E_temp;
        } else {
            if(isnan(E_danger_min) || E_temp < E_danger_min) E_danger_min = E_temp;
        }

        write_row(fp,N,target,Kr,ns,dw,dth,de,op,mw,mr,safe,
                  w,theta,T,repeat_used,seed,E_temp,E_safe_max,E_danger_min,mw_early,"init");

        repeat_used++;
        if(repeat_used%PROGRESS_EVERY==0)
            fprintf(stderr,"INFO: [init] repeat %d / %d  E_safe_max=%.4f  E_danger_min=%.4f\n",
                    repeat_used,REPEAT,E_safe_max,E_danger_min);
    }

    /* -----------------------------------------------------------------------
     * Phase 2: sampling around the boundary pair using the energy-safety
     * geometric mean/deviation concept:
     *   lo = min(E_safe_max, E_danger_min), hi = max(E_safe_max, E_danger_min)
     *   r  = sqrt(hi/lo)                    (= sigma_ES, multiplicative deviation)
     *   range = [lo/r, hi*r]                (symmetric in log-space, >= 0)
     * (Sampling within the resulting range remains linear-uniform, as before.)
     * ----------------------------------------------------------------------- */
    if(!isnan(E_safe_max) && !isnan(E_danger_min) && repeat_used < REPEAT && !INIT_ONLY){
        fprintf(stderr,"INFO: entering phase2  E_safe_max=%.4f  E_danger_min=%.4f\n",
                E_safe_max, E_danger_min);

        while(repeat_used < REPEAT){
            /* 기하평균(m_ES) / 편차(sigma_ES) 기반 탐색 범위.
             * lo = min(E_safe_max, E_danger_min), hi = max(...)
             * r  = sqrt(hi/lo)  (= sigma_ES)
             * 확장 범위: [lo/r, hi*r]  (로그 스케일에서 대칭, 항상 >=0)
             * 범위 내부 샘플링도 로그 균등(log-uniform)으로 수행한다. */
            double lo = (E_safe_max < E_danger_min) ? E_safe_max : E_danger_min;
            double hi = (E_safe_max < E_danger_min) ? E_danger_min : E_safe_max;

            double lo_safe = (lo > 1e-9) ? lo : 1e-9; /* r 계산용 (0-division 방지) */
            double r = sqrt(hi / lo_safe);
            const double R_MIN = 1.0 + 1e-6; /* lo≈hi일 때 최소 확장폭 보장 */
            if(r < R_MIN) r = R_MIN;

            double range_lo = lo / r;   /* lo=0이면 range_lo=0 (에너지 음수 방지) */
            double range_hi = hi * r;

            /* 영역(전범위 / L / R) 선택은 이번 repeat 시도 동안 단 한 번만 한다.
             * theta 탐색 실패로 E_temp를 다시 뽑아야 할 때도 이 선택은 유지한다.
             * (그렇지 않으면 영역별 탐색 성공률 차이 때문에, 실제 기록되는
             *  샘플 비율이 의도한 len_L:len_R 길이 비례에서 벗어나게 된다.) */
            double reg_log_lo, reg_log_hi;
            if(E_safe_max <= E_danger_min){
                /* 초기: 전범위 연속 탐색 (로그 균등) */
                reg_log_lo = log(range_lo > 1e-9 ? range_lo : 1e-9);
                reg_log_hi = log(range_hi);
            } else {
                /* 수렴 후: 죽은 구간 [E_danger_min, E_safe_max] 제외,
                 * 로그폭에 비례한 확률로 L/R 중 하나를 이번 시도의 영역으로 확정 */
                double lo_L = range_lo > 1e-9 ? range_lo : 1e-9, hi_L = E_danger_min;
                double lo_R = E_safe_max,                        hi_R = range_hi;

                double logL_lo = log(lo_L), logL_hi = log(hi_L > lo_L ? hi_L : lo_L);
                double logR_lo = log(lo_R), logR_hi = log(hi_R > lo_R ? hi_R : lo_R);

                double len_L = logL_hi - logL_lo;
                double len_R = logR_hi - logR_lo;
                double total = len_L + len_R;
                if(total < 1e-12) total = 1e-12;

                if(genrand64_real2() * total < len_L){
                    reg_log_lo = logL_lo; reg_log_hi = logL_hi;
                } else {
                    reg_log_lo = logR_lo; reg_log_hi = logR_hi;
                }
            }

            double dth, dw, E_temp;
            int found = 0;
            const int MAX_E_RETRY = 200; /* 같은 영역 안에서 E_temp를 재추첨하는 최대 횟수 */
            for(int eri=0; eri<MAX_E_RETRY && !found; eri++){
                E_temp = exp(reg_log_lo + genrand64_real2() * (reg_log_hi - reg_log_lo));

                for(int tri=0; tri<1000 && !found; tri++){
                    double th_try = sample_dtheta_restricted(uep_valid, uep_dtheta);
                    double theta_trial = theta_init[target] + th_try;
                    double V = potential_diff(&net, target, theta_trial, theta_init);
                    double kinetic = E_temp - V;
                    if(kinetic < 0.0) continue;
                    double w_abs = sqrt(2.0*kinetic/net.mass[target]);
                    if(w_abs > W_MAX) continue;
                    double w_sign = (genrand64_real2() < 0.5) ? 1.0 : -1.0;
                    dth = th_try; dw = w_sign * w_abs; found = 1;
                }
            }
            if(!found) continue; /* 이 영역에서 정말로 유효점을 못 찾음 (극히 드묾) */

            double theta_trial = theta_init[target] + dth;
            double de = local_energy_diff(&net,target,theta_trial,dw,theta_init);

            int ns; double op, mw, mr, mw_early;
            int safe = run_one(&net,N,target,T,theta_init,w_init,
                               dth,dw,theta,w,
                               k1_th,k2_th,k3_th,k4_th,
                               k1_w,k2_w,k3_w,k4_w,
                               th2,th3,th4,
                               n_edges,edge_i,edge_j,edge_prev,
                               EARLY_EXIT,&ns,&op,&mw,&mr,&mw_early);

            if(safe){
                if(E_temp > E_safe_max) E_safe_max = E_temp;
            } else {
                if(E_temp < E_danger_min) E_danger_min = E_temp;
            }

            write_row(fp,N,target,Kr,ns,dw,dth,de,op,mw,mr,safe,
                      w,theta,T,repeat_used,seed,E_temp,E_safe_max,E_danger_min,mw_early,"uniform");

            repeat_used++;
            if(repeat_used%PROGRESS_EVERY==0)
                fprintf(stderr,"INFO: [uniform] repeat %d / %d  E_safe_max=%.4f  E_danger_min=%.4f\n",
                        repeat_used,REPEAT,E_safe_max,E_danger_min);
        }
    } else if(repeat_used < REPEAT){
        fprintf(stderr,"WARNING: phase2 not entered (E_safe_max=%.4f E_danger_min=%.4f). "
                       "Increase REPEAT.\n", E_safe_max, E_danger_min);
    }

    fclose(fp);
    free(k1_w);free(k2_w);free(k3_w);free(k4_w);
    free(k1_th);free(k2_th);free(k3_th);free(k4_th);
    free(th2);free(th3);free(th4);
    free(theta);free(w);free(theta_init);free(w_init);
    free(edge_i);free(edge_j);free(edge_prev);
    free_network(&net);

    fprintf(stderr,"INFO: finished. total repeats used: %d\n",repeat_used);
    return EXIT_SUCCESS;
}