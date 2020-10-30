#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <vector>
#include "newton.h"

#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif

#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif

#ifdef __cplusplus
extern "C" {
#endif

extern double dnrm2_(int *, double *, int *);
extern double ddot_(int *, double *, int *, double *, int *);
extern int daxpy_(int *, double *, double *, int *, double *, int *);
extern int dscal_(int *, double *, double *, int *);

#ifdef __cplusplus
}
#endif

static void default_print(const char *buf)
{
	fputs(buf,stdout);
	fflush(stdout);
}

// On entry *f must be the function value of w
// On exit w is updated and *f is the new function value
double function::linesearch_and_update(double *w, double *s, double *f, double *g, double alpha)
{
	double gTs = 0;
	double eta = 0.01;
	int n = get_nr_variable();
	int max_num_linesearch = 20;
	std::vector<double> w_new(n);
	double fold = *f;

	for (int i=0;i<n;i++)
		gTs += s[i] * g[i];

	int num_linesearch = 0;
	for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
	{
		for (int i=0;i<n;i++)
			w_new[i] = w[i] + alpha*s[i];
		*f = fun(w_new.data());
		if (*f - fold <= eta * alpha * gTs)
			break;
		else
			alpha *= 0.5;
	}

	if (num_linesearch >= max_num_linesearch)
	{
		*f = fold;
		return 0;
	}
	else
		memcpy(w, w_new.data(), sizeof(double)*n);

	return alpha;
}

void NEWTON::info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*newton_print_string)(buf);
}

NEWTON::NEWTON(const function *fun_obj, double eps, double eps_cg, int max_iter)
{
	this->fun_obj=const_cast<function *>(fun_obj);
	this->eps=eps;
	this->eps_cg=eps_cg;
	this->max_iter=max_iter;
	newton_print_string = default_print;
}

NEWTON::~NEWTON()
{
}

void NEWTON::newton(double *w)
{
	int n = fun_obj->get_nr_variable();
	double init_step_size = 1;
	int inc = 1;
    std::vector<double> s(n);
    std::vector<double> r(n);
    std::vector<double> g(n);

	const double alpha_pcg = 0.01;
    std::vector<double> M(n, 0.0);

	// calculate gradient norm at w=0 for stopping condition.
	// the vector M has not been used yet, and has been filled
	// with zeros, so we have M == x_0 == 0 here.
    fun_obj->fun(M.data());
    fun_obj->grad(M.data(), g.data());

	double gnorm0 = dnrm2_(&n, g.data(), &inc);


	double f = fun_obj->fun(w);
	info("init f %5.3e\n", f);
	fun_obj->grad(w, g.data());
	double gnorm = dnrm2_(&n, g.data(), &inc);

	if (gnorm <= eps*gnorm0)
		return;


	for (int iter=1; iter <= max_iter; )
	{
		fun_obj->get_diag_preconditioner(M.data());
		for(int i=0; i<n; i++)
			M[i] = (1-alpha_pcg) + alpha_pcg*M[i];
		int cg_iter = pcg(g.data(), M.data(), s.data(), r.data());

		double fold = f;
        double step_size = fun_obj->linesearch_and_update(w, s.data(), &f, g.data(), init_step_size);

		if (step_size == 0)
		{
			info("WARNING: line search fails\n");
			break;
		}

		info("iter %2d f %5.3e |g| %5.3e CG %3d step_size %4.2e \n", iter, f, gnorm, cg_iter, step_size);

		double actred = fold - f;
		iter++;

		fun_obj->grad(w, g.data());

		gnorm = dnrm2_(&n, g.data(), &inc);
		if (gnorm <= eps*gnorm0)
			break;
		if (f < -1.0e+32)
		{
			info("WARNING: f < -1.0e+32\n");
			break;
		}
		if (fabs(actred) <= 1.0e-12*fabs(f))
		{
			info("WARNING: actred too small\n");
			break;
		}
	}
}

int NEWTON::pcg(double *g, double *M, double *s, double *r)
{
	int i, inc = 1;
	int n = fun_obj->get_nr_variable();
	double one = 1;
	std::vector<double> d(n);
	std::vector<double> Hd(n);
	std::vector<double> z(n);
	double zTr, znewTrnew, alpha, beta, cgtol;
	double Q = 0, newQ, Qdiff;

	for (i=0; i<n; i++)
	{
		s[i] = 0;
		r[i] = -g[i];
		z[i] = r[i] / M[i];
		d[i] = z[i];
	}

	zTr = ddot_(&n, z.data(), &inc, r, &inc);
	double gMinv_norm = sqrt(zTr);
	cgtol = min(eps_cg, sqrt(gMinv_norm));
	int cg_iter = 0;
	int max_cg_iter = max(n, 5);

	while (cg_iter < max_cg_iter)
	{
		cg_iter++;
		fun_obj->Hv(d.data(), Hd.data());

		alpha = zTr/ddot_(&n, d.data(), &inc, Hd.data(), &inc);
		daxpy_(&n, &alpha, d.data(), &inc, s, &inc);
		alpha = -alpha;
		daxpy_(&n, &alpha, Hd.data(), &inc, r, &inc);

		// Using quadratic approximation as CG stopping criterion
		newQ = -0.5*(ddot_(&n, s, &inc, r, &inc) - ddot_(&n, s, &inc, g, &inc));
		Qdiff = newQ - Q;
		if (newQ <= 0 && Qdiff <= 0)
		{
			if (cg_iter * Qdiff >= cgtol * newQ)
				break;
		}
		else
		{
			info("WARNING: quadratic approximation > 0 or increasing in CG\n");
			return cg_iter;
		}
		Q = newQ;

		for (i=0; i<n; i++)
			z[i] = r[i] / M[i];
		znewTrnew = ddot_(&n, z.data(), &inc, r, &inc);
		beta = znewTrnew/zTr;
		dscal_(&n, &beta, d.data(), &inc);
		daxpy_(&n, &one, z.data(), &inc, d.data(), &inc);
		zTr = znewTrnew;
	}

	if (cg_iter == max_cg_iter)
		info("WARNING: reaching maximal number of CG steps\n");

	return cg_iter;
}

void NEWTON::set_print_string(void (*print_string) (const char *buf))
{
	newton_print_string = print_string;
}
