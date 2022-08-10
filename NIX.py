import sys
import numpy as np
from scipy.stats import invgamma, norm, t, chi2
import matplotlib.pyplot as plt
from matplotlib import cm


class NIX:
    def __init__(self, m, k, s2, v):
        self.m = m  # mean
        self.k = k  # how strong we believe it
        self.s2 = s2  # variance
        self.v = v  # how strong we believe it
        self.sumx = 0
        self.sumx2 = 0
        self.n = 0  # number of data points

    # why are we doing this one at a time?
    def add(self, x):
        self.sumx += x
        self.sumx2 += x * x
        self.n += 1

    def remove(self, x):
        self.sumx -= x
        self.sumx2 -= x * x
        self.n -= 1

    # drawing from the prior (normal chi-squared)/initial observation
    def draw(self):
        v_t = inverse_x_squared(self.v, self.s2).rvs()
        m_t = norm(loc=self.m, scale=(v_t / self.k) ** (1 / 2)).rvs()

        return (m_t, v_t)

    def logPDF(self, m, var):
        logp_v = inverse_x_squared(self.v, self.s2).logpdf(var)
        logp_m = norm(loc=self.m, scale=(var / self.k) ** (1 / 2)).logpdf(m)

        return logp_m + logp_v

    def posterior(self):
        k_0 = self.k
        v_0 = self.v
        m_0 = self.m
        s2_0 = self.s2

        mean_data = self.sumx / self.n

        k_n = self.k + self.n
        v_n = self.v + self.n
        m_n = ((k_0 * m_0) + self.sumx) / k_n
        s2_n = (1 / v_n * (v_0 * s2_0 + (self.sumx2 - (self.sumx * (2 * mean_data)) + (self.sumx * mean_data)) + ((
                (self.n * k_0) / (k_0 + self.n)) * ((m_0 - mean_data) ** 2))))

        return NIX(m_n, k_n, s2_n, v_n)

    def postpred(self, x):
        # t-distribution
        nix_post = self.posterior()
        pdf = t.logpdf(x, df=nix_post.m, scale=((1 + nix_post.k) * nix_post.s2) / nix_post.k)

        return pdf


def inverse_x_squared(v, s2):
    shape = v/2
    scale = (v*s2)/2
    return invgamma(shape, scale=scale)

# no data inserted to calling logPDF -> gives us the prior loglikelihood
# insert data to NIX, call logPDF on posterior -> gives us log posterior likelihood


# with enough data, the model should converge to the true normal paramters
def synthetic_data():
    data = np.random.normal(0, 1, size=150)
    return data


def main():

    # setup
    m, k, s2, v = 0, 1, 1, 1
    # m, k, s2, v = 0, 5, 1, 1
    # m, k, s2, v = 0, 1, 1, 5
    # m, k, s2, v = 0.5, 5, 5, 0.5

    m_s = 50
    var_s = 100
    if m == 0:
        ms = np.linspace(-1, 1, m_s)
    elif m < 0:
        ms = np.linspace(m - 1, (-1 * m) + 1, m_s)
    elif m > 0:
        ms = np.linspace((-1 * m) - 1, m + 1, m_s)

    vars_ = np.linspace(0.1, s2, var_s)
    m0, vars0 = np.meshgrid(ms, vars_)
    data = synthetic_data()

    nix = NIX(m, k, s2, v)

    posterior_pdf = np.zeros((len(vars_), len(ms)))
    prior_pdf = np.zeros((len(vars_), len(ms)))

   # generate logp's
    [nix.add(x) for x in data]
    nix_posterior = nix.posterior()
    for i, var in enumerate(vars_):
        for j, m in enumerate(ms):
            prior_pdf[i, j] = nix.logPDF(m, var)
            posterior_pdf[i, j] = nix_posterior.logPDF(m, var)


    # reshaping for plotting
    prior_pdf = np.exp(np.array(prior_pdf))
    posterior_pdf = np.exp(np.array(posterior_pdf))

    # plotting
    figure = plt.figure()
    ax = figure.add_subplot(121, projection='3d')
    ax.view_init(20, -125)
    ax.plot_surface(m0, vars0, prior_pdf, cmap=cm.Spectral, rstride=1, cstride=1, antialiased=False)

    ax_post = figure.add_subplot(122, projection='3d')
    ax_post.view_init(20, -125)
    ax_post.plot_surface(m0, vars0, posterior_pdf, cmap=cm.Spectral, rstride=1, cstride=1, antialiased=False)

    plt.show()


if __name__ == "__main__":
    main()
