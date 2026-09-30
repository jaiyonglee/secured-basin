// get_init.c
//
// Compile:
// gcc -O2 -Wall -Wextra -o build/get_init get_init.c mt19937-64.c -lm
//
// Run:
// ./get_init N sam_num a Kr_input T
//
// Example:
// ./get_init 6 0 0.1 100 500000
//
// Input files:
//   network/{N}_sample{sam_num}_nodes.txt
//   network/{N}_sample{sam_num}_edges.txt
//
// Output file:
//   init/{N}_sample{sam_num}_init_Kr_{Kr}.txt
//
// Important behavior:
//   This program now saves the init file only if the final state is a valid
//   synchronized equilibrium. Otherwise it exits with failure and does not save.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>

#define H 0.001
#define SYNC_TOL 5e-2
#define RESIDUAL_TOL 1e-3
#define PI 3.141592653589793238462643383279502884

typedef struct {
    int N;
    int *degree;
    int **nn;
    double **K;
    double *P;
    double *mass;   /* 노드별 질량 m_i */
    double *gamma;  /* 노드별 감쇠 γ_i */
} Network;

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
    if (!fp) {
        fprintf(stderr, "ERROR: failed to open file: %s\n", path);
        exit(EXIT_FAILURE);
    }
    return fp;
}

static void ensure_dir(const char *dirname) {
    if (mkdir(dirname, 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "ERROR: failed to create directory '%s': %s\n",
                dirname, strerror(errno));
        exit(EXIT_FAILURE);
    }
}

static double wrap_to_pi(double x) {
    while (x > PI) x -= 2.0 * PI;
    while (x <= -PI) x += 2.0 * PI;
    return x;
}

static void canonicalize_theta(int N, double *theta) {
    double ref = theta[0];
    for (int i = 0; i < N; i++) {
        theta[i] = wrap_to_pi(theta[i] - ref);
    }
    theta[0] = 0.0;
}

static double order_parameter(int N, const double *theta) {
    double sum_cos = 0.0;
    double sum_sin = 0.0;
    for (int i = 0; i < N; i++) {
        sum_cos += cos(theta[i]);
        sum_sin += sin(theta[i]);
    }
    return sqrt(sum_cos * sum_cos + sum_sin * sum_sin) / (double)N;
}

static double max_abs_w(int N, const double *w) {
    double m = 0.0;
    for (int i = 0; i < N; i++) {
        double aw = fabs(w[i]);
        if (aw > m) m = aw;
    }
    return m;
}

static int count_synced_nodes(int N, const double *w, double tol) {
    int cnt = 0;
    for (int i = 0; i < N; i++) {
        if (fabs(w[i]) < tol) cnt++;
    }
    return cnt;
}

static double max_residual(const Network *net, const double *theta) {
    double max_r = 0.0;
    for (int i = 0; i < net->N; i++) {
        double coupling = 0.0;
        for (int j = 0; j < net->degree[i]; j++) {
            int nb = net->nn[i][j];
            coupling += net->K[i][j] * sin(theta[nb] - theta[i]);
        }
        /* 평형 조건: P_i + Σ K_ij sin(θj-θi) = 0 */
        double r = fabs(net->P[i] + coupling);
        if (r > max_r) max_r = r;
    }
    return max_r;
}

static void free_network(Network *net) {
    if (!net) return;

    if (net->nn) {
        for (int i = 0; i < net->N; i++) free(net->nn[i]);
        free(net->nn);
    }
    if (net->K) {
        for (int i = 0; i < net->N; i++) free(net->K[i]);
        free(net->K);
    }
    free(net->degree);
    free(net->P);
    free(net->mass);
    free(net->gamma);

    net->nn = NULL;
    net->K = NULL;
    net->degree = NULL;
    net->P = NULL;
    net->mass = NULL;
    net->gamma = NULL;
}

static void load_network(Network *net, int N, const char *node_file, const char *edge_file, double Kr) {
    net->N = N;
    net->degree = (int *)   xmalloc(sizeof(int)    * N);
    net->P      = (double *)xmalloc(sizeof(double) * N);
    net->mass   = (double *)xmalloc(sizeof(double) * N);
    net->gamma  = (double *)xmalloc(sizeof(double) * N);
    net->nn = NULL;
    net->K = NULL;

    for (int i = 0; i < N; i++) {
        net->degree[i] = 0;
        net->P[i]      = 0.0;
        net->mass[i]   = 1.0;   /* default: M=1 */
        net->gamma[i]  = 0.1;   /* default: γ=0.1 */
    }

    FILE *fp_node = xfopen(node_file, "r");
    for (int i = 0; i < N; i++) {
        int node_id;
        double p, m, g;
        /* 포맷 1: "id power mass gamma" (새 형식) */
        /* 포맷 2: "id power"            (구 형식, mass/gamma는 default 유지) */
        int nread = fscanf(fp_node, "%d %lf %lf %lf", &node_id, &p, &m, &g);
        if (nread < 2) {
            fclose(fp_node);
            die("failed to read node file");
        }
        if (node_id < 0 || node_id >= N) {
            fclose(fp_node);
            die("node id out of range in node file");
        }
        net->P[node_id] = p;
        if (nread >= 3) net->mass[node_id]  = m;
        if (nread >= 4) net->gamma[node_id] = g;
    }
    fclose(fp_node);

    FILE *fp_edge = xfopen(edge_file, "r");
    int c1, c2;
    double c3;
    while (fscanf(fp_edge, "%d %d %lf", &c1, &c2, &c3) == 3) {
        if (c1 < 0 || c1 >= N || c2 < 0 || c2 >= N) {
            fclose(fp_edge);
            die("node id out of range in edge file");
        }
        net->degree[c1]++;
        net->degree[c2]++;
    }
    fclose(fp_edge);

    net->nn = (int **)xmalloc(sizeof(int *) * N);
    net->K = (double **)xmalloc(sizeof(double *) * N);
    int *temp_degree = (int *)xmalloc(sizeof(int) * N);

    for (int i = 0; i < N; i++) {
        net->nn[i] = (int *)xmalloc(sizeof(int) * net->degree[i]);
        net->K[i] = (double *)xmalloc(sizeof(double) * net->degree[i]);
        temp_degree[i] = 0;
    }

    fp_edge = xfopen(edge_file, "r");
    while (fscanf(fp_edge, "%d %d %lf", &c1, &c2, &c3) == 3) {
        int idx1 = temp_degree[c1]++;
        int idx2 = temp_degree[c2]++;

        net->nn[c1][idx1] = c2;
        net->nn[c2][idx2] = c1;
        net->K[c1][idx1] = Kr * c3;
        net->K[c2][idx2] = Kr * c3;
    }
    fclose(fp_edge);

    free(temp_degree);
}

static void rk4_step(
    const Network *net,
    double *theta,
    double *w,
    double *k1_th, double *k2_th, double *k3_th, double *k4_th,
    double *k1_w,  double *k2_w,  double *k3_w,  double *k4_w
) {
    int N = net->N;

    /* k1 */
    for (int i = 0; i < N; i++) {
        double coupling = 0.0;
        for (int j = 0; j < net->degree[i]; j++) {
            int nb = net->nn[i][j];
            coupling += net->K[i][j] * sin(theta[nb] - theta[i]);
        }
        k1_w[i]  = H * (net->P[i] - net->gamma[i] * w[i] + coupling) / net->mass[i];
        k1_th[i] = H * w[i];
    }

    /* k2 */
    for (int i = 0; i < N; i++) {
        double coupling = 0.0;
        double th_i = theta[i] + 0.5 * k1_th[i];
        for (int j = 0; j < net->degree[i]; j++) {
            int nb = net->nn[i][j];
            double th_nb = theta[nb] + 0.5 * k1_th[nb];
            coupling += net->K[i][j] * sin(th_nb - th_i);
        }
        k2_w[i]  = H * (net->P[i] - net->gamma[i] * (w[i] + 0.5 * k1_w[i]) + coupling) / net->mass[i];
        k2_th[i] = H * (w[i] + 0.5 * k1_w[i]);
    }

    /* k3 */
    for (int i = 0; i < N; i++) {
        double coupling = 0.0;
        double th_i = theta[i] + 0.5 * k2_th[i];
        for (int j = 0; j < net->degree[i]; j++) {
            int nb = net->nn[i][j];
            double th_nb = theta[nb] + 0.5 * k2_th[nb];
            coupling += net->K[i][j] * sin(th_nb - th_i);
        }
        k3_w[i]  = H * (net->P[i] - net->gamma[i] * (w[i] + 0.5 * k2_w[i]) + coupling) / net->mass[i];
        k3_th[i] = H * (w[i] + 0.5 * k2_w[i]);
    }

    /* k4 */
    for (int i = 0; i < N; i++) {
        double coupling = 0.0;
        double th_i = theta[i] + k3_th[i];
        for (int j = 0; j < net->degree[i]; j++) {
            int nb = net->nn[i][j];
            double th_nb = theta[nb] + k3_th[nb];
            coupling += net->K[i][j] * sin(th_nb - th_i);
        }
        k4_w[i]  = H * (net->P[i] - net->gamma[i] * (w[i] + k3_w[i]) + coupling) / net->mass[i];
        k4_th[i] = H * (w[i] + k3_w[i]);
    }

    for (int i = 0; i < N; i++) {
        w[i]     += (k1_w[i]  + 2.0 * k2_w[i]  + 2.0 * k3_w[i]  + k4_w[i])  / 6.0;
        theta[i] += (k1_th[i] + 2.0 * k2_th[i] + 2.0 * k3_th[i] + k4_th[i]) / 6.0;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        fprintf(stderr,
            "Usage: %s N sam_num Kr_input T\n"
            "Example: %s 54 0 100 500000\n"
            "  mass and gamma are read per-node from the node file.\n",
            argv[0], argv[0]
        );
        return EXIT_FAILURE;
    }

    int N = atoi(argv[1]);
    int sam_num = atoi(argv[2]);
    double Kr = atof(argv[3]) * 0.01;
    int T = atoi(argv[4]);

    if (N <= 0) die("N must be positive");
    if (Kr <= 0.0) die("Kr must be positive");
    if (T <= 0) die("T must be positive");

    ensure_dir("init");

    char node_file[256], edge_file[256], result_file[256];
    snprintf(node_file, sizeof(node_file), "network/%d_sample%d_nodes.txt", N, sam_num);
    snprintf(edge_file, sizeof(edge_file), "network/%d_sample%d_edges.txt", N, sam_num);
    snprintf(result_file, sizeof(result_file), "init/%d_sample%d_init_Kr_%.2f.txt", N, sam_num, Kr);

    printf("Node file   : %s\n", node_file);
    printf("Edge file   : %s\n", edge_file);
    printf("Output file : %s\n", result_file);
    printf("Kr          : %.4f\n", Kr);

    Network net;
    load_network(&net, N, node_file, edge_file, Kr);

    double *theta = (double *)xmalloc(sizeof(double) * N);
    double *w = (double *)xmalloc(sizeof(double) * N);

    double *k1_w  = (double *)xmalloc(sizeof(double) * N);
    double *k2_w  = (double *)xmalloc(sizeof(double) * N);
    double *k3_w  = (double *)xmalloc(sizeof(double) * N);
    double *k4_w  = (double *)xmalloc(sizeof(double) * N);
    double *k1_th = (double *)xmalloc(sizeof(double) * N);
    double *k2_th = (double *)xmalloc(sizeof(double) * N);
    double *k3_th = (double *)xmalloc(sizeof(double) * N);
    double *k4_th = (double *)xmalloc(sizeof(double) * N);

    for (int i = 0; i < N; i++) {
        theta[i] = 0.0;
        w[i] = 0.0;
    }

    for (int t = 0; t < T; t++) {
        rk4_step(&net, theta, w, k1_th, k2_th, k3_th, k4_th, k1_w, k2_w, k3_w, k4_w);
    }

    canonicalize_theta(N, theta);

    int num_synced = count_synced_nodes(N, w, SYNC_TOL);
    double maxw = max_abs_w(N, w);
    double maxr = max_residual(&net, theta);
    double op = order_parameter(N, theta);

    int success = (num_synced == N) && (maxw < SYNC_TOL) && (maxr < RESIDUAL_TOL);

    printf("Num synced     : %d / %d\n", num_synced, N);
    printf("Max |w|        : %.10f\n", maxw);
    printf("Max residual   : %.10f\n", maxr);
    printf("Order parameter: %.10f\n", op);

    if (!success) {
        fprintf(stderr,
                "ERROR: final state is not a valid synchronized equilibrium.\n"
                "       Required: num_synced == N, max|w| < %.3e, max_residual < %.3e\n"
                "       Got     : num_synced = %d/%d, max|w| = %.10f, max_residual = %.10f\n"
                "       Increase T or Kr_input, or check the network/input files.\n",
                SYNC_TOL, RESIDUAL_TOL, num_synced, N, maxw, maxr);
        fprintf(stderr, "       (mass and gamma loaded per-node from node file)\n");

        free(k1_w);
        free(k2_w);
        free(k3_w);
        free(k4_w);
        free(k1_th);
        free(k2_th);
        free(k3_th);
        free(k4_th);
        free(theta);
        free(w);
        free_network(&net);
        return EXIT_FAILURE;
    }

    FILE *fp_out = xfopen(result_file, "w");
    for (int i = 0; i < N; i++) {
        fprintf(fp_out, "%d %.15f %.15f\n", i, w[i], theta[i]);
    }
    fclose(fp_out);

    free(k1_w);
    free(k2_w);
    free(k3_w);
    free(k4_w);
    free(k1_th);
    free(k2_th);
    free(k3_th);
    free(k4_th);

    free(theta);
    free(w);

    free_network(&net);

    printf("Saved synchronized equilibrium to %s\n", result_file);
    return EXIT_SUCCESS;
}