// attack_power_server.c
//
// Compile:
// gcc -O2 -Wall -Wextra -o build/attack_power_server attack_power_server.c mt19937-64.c -lm
//
// Run:
// ./attack_power_server N sam_num Kr_input T target_node REPEAT [EARLY_EXIT] [P_MAX_FACTOR] [P_MIN_FACTOR] [T_MAX_SEC] [seed]
//
// Strategy:
//   Perturbation: target node receives P + delta_P power for t in [0, delta_t],
//                 then returns to P for t in [delta_t, T].
//   E_temp = delta_P * delta_t  (impulse approximation)
//   Phase 1 only: uniformly sample (delta_P, delta_t) for all REPEAT trials.
//     E_safe_max and E_danger_min are tracked throughout.
//
// Safe criterion: num_synced == N  AND  no link phase-difference crossed +-PI.
// EARLY_EXIT=1: stop simulation as soon as crossing detected.
// EARLY_EXIT=0: full simulation — desync/danger/safe distinguished by num_synced.
//
// mass and gamma are loaded per-node from the node file (4-column format).
//
// Output: result_raw4/

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
#define PROGRESS_EVERY 1000

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
        double m=1.0, g=0.1;
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
 * Uses per-node mass and gamma from net->mass[i] and net->gamma[i].
 * ----------------------------------------------------------------------- */
static void rk4_step(const Network *net,
                     double *theta, double *w,
                     double *k1_th, double *k2_th, double *k3_th, double *k4_th,
                     double *k1_w,  double *k2_w,  double *k3_w,  double *k4_w,
                     double *th2, double *th3, double *th4,
                     int target, double extra_P)
{
    int N=net->N;

    /* k1 */
    for(int i=0;i<N;i++){
        double c=0.0;
        for(int j=0;j<net->degree[i];j++)
            c+=net->K[i][j]*sin(theta[net->nn[i][j]]-theta[i]);
        double p_eff = net->P[i] + (i==target ? extra_P : 0.0);
        k1_w[i] =H*(p_eff - net->gamma[i]*w[i] + c)/net->mass[i];
        k1_th[i]=H*w[i];
    }
    /* k2 */
    for(int i=0;i<N;i++) th2[i]=theta[i]+0.5*k1_th[i];
    for(int i=0;i<N;i++){
        double c=0.0;
        for(int j=0;j<net->degree[i];j++)
            c+=net->K[i][j]*sin(th2[net->nn[i][j]]-th2[i]);
        double p_eff = net->P[i] + (i==target ? extra_P : 0.0);
        k2_w[i] =H*(p_eff - net->gamma[i]*(w[i]+0.5*k1_w[i]) + c)/net->mass[i];
        k2_th[i]=H*(w[i]+0.5*k1_w[i]);
    }
    /* k3 */
    for(int i=0;i<N;i++) th3[i]=theta[i]+0.5*k2_th[i];
    for(int i=0;i<N;i++){
        double c=0.0;
        for(int j=0;j<net->degree[i];j++)
            c+=net->K[i][j]*sin(th3[net->nn[i][j]]-th3[i]);
        double p_eff = net->P[i] + (i==target ? extra_P : 0.0);
        k3_w[i] =H*(p_eff - net->gamma[i]*(w[i]+0.5*k2_w[i]) + c)/net->mass[i];
        k3_th[i]=H*(w[i]+0.5*k2_w[i]);
    }
    /* k4 */
    for(int i=0;i<N;i++) th4[i]=theta[i]+k3_th[i];
    for(int i=0;i<N;i++){
        double c=0.0;
        for(int j=0;j<net->degree[i];j++)
            c+=net->K[i][j]*sin(th4[net->nn[i][j]]-th4[i]);
        double p_eff = net->P[i] + (i==target ? extra_P : 0.0);
        k4_w[i] =H*(p_eff - net->gamma[i]*(w[i]+k3_w[i]) + c)/net->mass[i];
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
        for(int j=0;j<net->degree[i];j++)
            c+=net->K[i][j]*sin(theta[net->nn[i][j]]-theta[i]);
        double r=fabs(net->P[i]+c); if(r>mr) mr=r;
    }
    return mr;
}

/* -----------------------------------------------------------------------
 * Single simulation with power perturbation
 * ----------------------------------------------------------------------- */
static int run_one(const Network *net, int N, int target, int T,
                   const double *theta_init, const double *w_init,
                   double delta_P, int delta_t_steps,
                   double *theta, double *w,
                   double *k1_th,double *k2_th,double *k3_th,double *k4_th,
                   double *k1_w, double *k2_w, double *k3_w, double *k4_w,
                   double *th2, double *th3, double *th4,
                   int n_edges,
                   const int *edge_i, const int *edge_j,
                   double *edge_prev,
                   int early_exit,
                   int *out_ns, double *out_op, double *out_mw, double *out_mr,
                   double *out_w_at_dt, double *out_theta_at_dt)
{
    for(int i=0;i<N;i++){theta[i]=theta_init[i]; w[i]=w_init[i];}

    for(int e=0;e<n_edges;e++)
        edge_prev[e] = wrap_to_pi(theta[edge_i[e]] - theta[edge_j[e]]);

    int crossed = 0;
    int recorded_dt = 0;

    for(int t=0;t<T;t++){
        double ep = (t < delta_t_steps) ? delta_P : 0.0;
        rk4_step(net,theta,w,k1_th,k2_th,k3_th,k4_th,k1_w,k2_w,k3_w,k4_w,
                 th2,th3,th4,target,ep);

        if(!recorded_dt && t == delta_t_steps - 1){
            for(int i=0;i<N;i++){
                out_w_at_dt[i]     = w[i];
                out_theta_at_dt[i] = theta[i];
            }
            recorded_dt = 1;
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

    if(!recorded_dt){
        for(int i=0;i<N;i++){
            out_w_at_dt[i]     = w[i];
            out_theta_at_dt[i] = theta[i];
        }
    }

    sim_done:
    *out_ns = count_synced_nodes(N, w, SYNC_TOL);
    *out_op = order_parameter(N, theta);
    *out_mw = max_abs_w(N, w);
    *out_mr = max_residual(net, theta);

    return (*out_ns == N && !crossed);
}

/* -----------------------------------------------------------------------
 * write_row
 * ----------------------------------------------------------------------- */
static void write_row(FILE *fp, int N, int target, double Kr,
                      int ns, double attack_P, double attack_t,
                      double de, double op, double mw, double mr, int safe,
                      double p_max, double p_min,
                      const double *w, const double *theta,
                      const double *w_at_dt, const double *theta_at_dt,
                      int T, int repeat, unsigned long long seed,
                      double E_temp, double E_safe_max, double E_danger_min,
                      const char *phase_label)
{
    fprintf(fp,"%d,%.6f,%d,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%d,%.6f,%.6e",
            target,Kr,ns,attack_P,attack_t,de,op,mw,mr,safe,p_max,p_min);
    for(int i=0;i<N;i++) fprintf(fp,",%.10f",w[i]);
    for(int i=0;i<N;i++) fprintf(fp,",%.10f",theta[i]);
    for(int i=0;i<N;i++) fprintf(fp,",%.10f",w_at_dt[i]);
    for(int i=0;i<N;i++) fprintf(fp,",%.10f",theta_at_dt[i]);
    fprintf(fp,",%d,%d,%llu,%.10f,%.10f,%.10f,%s\n",
            T,repeat,seed,E_temp,E_safe_max,E_danger_min,phase_label);
}

/* -----------------------------------------------------------------------
 * main
 * ----------------------------------------------------------------------- */
int main(int argc, char *argv[]) {
    if(argc<7||argc>12){
        fprintf(stderr,
            "Usage: %s N sam_num Kr_input T target REPEAT [EARLY_EXIT] [P_MAX_FACTOR] [P_MIN_FACTOR] [T_MAX_SEC] [seed]\n"
            "  mass and gamma are loaded per-node from the node file.\n"
            "  EARLY_EXIT:    1=stop sim on crossing (default=1), 0=full sim\n"
            "  P_MAX_FACTOR: P_max = factor * sum(K[target]) (default=5.0)\n"
            "  P_MIN_FACTOR: P_min = P_max * factor (default=1e-3)\n"
            "                delta_P sampled log-uniformly in [P_min, P_max]\n"
            "  T_MAX_SEC:    max delta_t in seconds (default=10.0)\n"
            "  seed: RNG seed (default=time-based)\n", argv[0]);
        return EXIT_FAILURE;
    }

    int    N       = atoi(argv[1]);
    int    sam_num = atoi(argv[2]);
    double Kr      = atof(argv[3])*0.01;
    int    T       = atoi(argv[4]);
    int    target  = atoi(argv[5]);
    int    REPEAT  = atoi(argv[6]);
    int    EARLY_EXIT = 1;
    double P_MAX_FACTOR_val = 5.0;
    double P_MIN_FACTOR_val = 1e-3;
    unsigned long long seed = 0;
    int seed_set = 0;

    double T_MAX_SEC = 10.0;
    if(argc>=8)  EARLY_EXIT = atoi(argv[7]);
    if(argc>=9)  P_MAX_FACTOR_val = atof(argv[8]);
    if(argc>=10) P_MIN_FACTOR_val = atof(argv[9]);
    if(argc>=11) T_MAX_SEC = atof(argv[10]);
    if(argc==12){ seed = strtoull(argv[11],NULL,10); seed_set = 1; }
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
    ensure_dir("result_raw4");

    char node_file[256], edge_file[256], init_file[256], result_file[256];
    snprintf(node_file,   sizeof(node_file),   "network/%d_sample%d_nodes.txt",N,sam_num);
    snprintf(edge_file,   sizeof(edge_file),   "network/%d_sample%d_edges.txt",N,sam_num);
    snprintf(init_file,   sizeof(init_file),   "init/%d_sample%d_init_Kr_%.2f.txt",N,sam_num,Kr);
    snprintf(result_file, sizeof(result_file), "result_raw4/%d_analysis%d_n_%d_Kr_%.2f.csv",
             N,sam_num,target,Kr);

    fprintf(stderr,"INFO: N=%d sam_num=%d Kr=%.4f T=%d target=%d "
                   "REPEAT=%d EARLY_EXIT=%d P_MAX_FACTOR=%.2f P_MIN_FACTOR=%.2e "
                   "T_MAX_SEC=%.2f seed=%llu\n"
                   "      (mass and gamma loaded per-node from node file)\n",
            N,sam_num,Kr,T,target,REPEAT,EARLY_EXIT,
            P_MAX_FACTOR_val,P_MIN_FACTOR_val,T_MAX_SEC,seed);

    Network net;
    load_network(&net,N,node_file,edge_file,Kr);

    double *theta      = (double*)xmalloc(sizeof(double)*N);
    double *w          = (double*)xmalloc(sizeof(double)*N);
    double *theta_init = (double*)xmalloc(sizeof(double)*N);
    double *w_init     = (double*)xmalloc(sizeof(double)*N);
    double *w_at_dt    = (double*)xmalloc(sizeof(double)*N);
    double *theta_at_dt= (double*)xmalloc(sizeof(double)*N);
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

    /* CSV header */
    fprintf(fp,"target,Kr,num_synced,attack_P,attack_t,delta_e,"
               "order_parameter,max_abs_w,max_residual,safe,p_max,p_min");
    for(int i=0;i<N;i++) fprintf(fp,",w_%d",i);
    for(int i=0;i<N;i++) fprintf(fp,",theta_%d",i);
    for(int i=0;i<N;i++) fprintf(fp,",w_at_dt_%d",i);
    for(int i=0;i<N;i++) fprintf(fp,",theta_at_dt_%d",i);
    fprintf(fp,",steps,repeat,seed,E_temp,E_safe_max,E_danger_min,phase\n");

    /* compute P_max and P_min for target node */
    double sum_K = 0.0;
    for(int j=0;j<net.degree[target];j++) sum_K += net.K[target][j];
    double p_max = sum_K * P_MAX_FACTOR_val;
    double p_min = sum_K * P_MIN_FACTOR_val;
    if(p_max <= 0.0) p_max = 10.0;
    if(P_MIN_FACTOR_val <= 0.0){
        fprintf(stderr,"WARNING: P_MIN_FACTOR=%.2e must be > 0, reset to 1e-3\n",
                P_MIN_FACTOR_val);
        P_MIN_FACTOR_val = 1e-3;
        p_min = sum_K * P_MIN_FACTOR_val;
    }
    if(p_min >= p_max){
        fprintf(stderr,"WARNING: P_min(%.4e) >= P_max(%.4e), check P_MIN/MAX_FACTOR\n",
                p_min, p_max);
    }
    fprintf(stderr,"INFO: sum_K=%.4f  P_max=%.4f  P_min=%.4e  for target=%d\n",
            sum_K, p_max, p_min, target);

    double E_safe_max   = NAN;
    double E_danger_min = NAN;
    int repeat_used = 0;

    /* -----------------------------------------------------------------------
     * Phase 1 only: random uniform sampling for all REPEAT trials
     * ----------------------------------------------------------------------- */
    while(repeat_used < REPEAT){
        /* delta_t: log-uniform in [H, T_MAX_SEC] */
        double log_dt = log10(H) + (log10(T_MAX_SEC) - log10(H)) * genrand64_real2();
        double delta_t = pow(10.0, log_dt);
        if(delta_t < H)         delta_t = H;
        if(delta_t > T_MAX_SEC) delta_t = T_MAX_SEC;
        int delta_t_steps = (int)(delta_t / H);
        if(delta_t_steps < 1) delta_t_steps = 1;
        if(delta_t_steps > T) delta_t_steps = T;
        /* delta_P: log-uniform in [P_MIN_FACTOR * sum_K, P_MAX_FACTOR * sum_K], random sign */
        double log_p_min     = log10(p_min);
        double log_p_max_val = log10(p_max);
        double log_dp_val    = log_p_min + (log_p_max_val - log_p_min) * genrand64_real2();
        double delta_P = pow(10.0, log_dp_val);
        delta_P *= (genrand64_real2() < 0.5) ? 1.0 : -1.0;

        double E_temp = delta_P * delta_t;
        double de     = E_temp;

        int ns; double op, mw, mr;
        int safe = run_one(&net,N,target,T,theta_init,w_init,
                           delta_P, delta_t_steps,
                           theta,w,
                           k1_th,k2_th,k3_th,k4_th,
                           k1_w,k2_w,k3_w,k4_w,
                           th2,th3,th4,
                           n_edges,edge_i,edge_j,edge_prev,
                           EARLY_EXIT,
                           &ns,&op,&mw,&mr,
                           w_at_dt, theta_at_dt);

        if(safe){
            if(isnan(E_safe_max) || E_temp > E_safe_max) E_safe_max = E_temp;
        } else {
            if(isnan(E_danger_min) || E_temp < E_danger_min) E_danger_min = E_temp;
        }

        write_row(fp,N,target,Kr,ns,delta_P,delta_t,de,op,mw,mr,safe,p_max,p_min,
                  w,theta,w_at_dt,theta_at_dt,
                  T,repeat_used,seed,E_temp,E_safe_max,E_danger_min,"init");

        repeat_used++;
        if(repeat_used%PROGRESS_EVERY==0)
            fprintf(stderr,"INFO: [init] repeat %d / %d  E_safe_max=%.4f  E_danger_min=%.4f\n",
                    repeat_used,REPEAT,E_safe_max,E_danger_min);
    }

    fclose(fp);
    free(k1_w);free(k2_w);free(k3_w);free(k4_w);
    free(k1_th);free(k2_th);free(k3_th);free(k4_th);
    free(th2);free(th3);free(th4);
    free(theta);free(w);free(theta_init);free(w_init);
    free(w_at_dt);free(theta_at_dt);
    free(edge_i);free(edge_j);free(edge_prev);
    free_network(&net);

    fprintf(stderr,"INFO: finished. total repeats used: %d\n",repeat_used);
    return EXIT_SUCCESS;
}