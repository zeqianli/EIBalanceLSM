""" Liquid State Machine (LSM) with E-I balanced neurons.

Author: Zeqian Li

==================

Jul 20, 2018: I wrote this a while ago. It worked but not very well. Codes are 
    kind of messy. 

    I'll just leave like that. Email me you have any questions.
"""

import numpy as np
import scipy.sparse as sp
import time
import matplotlib.pyplot as plt
from lzqtools import *
from config import *
import pickle


class LSM:
    def __init__(self, n, n_input=1, n_output=1,
                 p=0.1, p_input=0.1, module_level=0, rewire_probability_ex=0.99,
                 delta_g_ex=0.5, delta_g_inh=4,
                 dt=0.1, noise_level=0.1,
                 tau_kernel=30,
                 noise_method='spike', input_method='spike'):
        """Initialization.

        Network:
        n, n_input, n_output: Reservoir/input/output size
        p/p_input: reservoir/input-reservoir connnectivity
        module_level: (unimplemented) modulation level of reservoir
        rewire_probability_ex: (unimplemented) modulation related parameter
        
        Dynamics:
        delta_g_ex/delta_g_inh

        Others:
        dt: time step; default 0.1ms
        noise_level: noise level
        tau_kernal: \tau for kernal 
        noise_method/input_method: 'spike' or 'current'; 
            code noise/input as spikes or injected current.
        
        """
        # Basics
        self._initialized = False
        self._simulated = False
        self._trained = False
        self._noise_method = noise_method
        self._input_method = input_method
        self.t = 0  # time
        self.dt = dt
        self.simulate_data = None
        self.fit_result=None
        self.predict_data = None
        self.predict_result=None

        # Network parameters
        self.n_input = n_input  # number of input neuron
        self.n = n  # number of internal(reservoir) neurons
        self.n_output = n_output

        # Structure
        self.p = p  # reservoir sparsity
        self.p_input = p_input  # sparsity of input-reservoir connection
        # self.module_level = module_level  # level of modules of reservoir
        # self.rewire_probability_ex = rewire_probability_ex
        # self.rewire_probability_inh = 1
        self.w_input = None  # TO BE INITIALIZED
        self.connect_to_input = None  # TO BE INITIALIZED
        self.w = None  # TO BE INITIALIZED
        self.children_of = None  # TO BE INITIALIZED
        self.children_of_input = None  # TO BE INITIALIZED

        self.w_output = None  # initialized when trained

        # Dynamics parameters
        self.ex_ratio = 0.8
        self.inh_ratio = 1 - self.ex_ratio
        self.n_ex = int(self.n * self.ex_ratio)  # number of excitatory neurons
        self.n_inh = self.n - self.n_ex  # number of inhibitory neurons
        self.v_rest = -60
        self.v_ex = 0
        self.v_inh = -80
        self.v_threshold = -50
        self.tau = 20
        self.tau_ex = 5
        self.tau_inh = 10
        self.delta_gex = delta_g_ex  # should be fine tuned
        self.delta_ginh = delta_g_inh  # should be fine tuned
        self.refract_max = 5
        self.noise_level = noise_level  # TO BE TUNED

        # State parameters
        self.v_input = None  # input neuron voltage (state)
        self.v = None  # internal neuron voltage (state)
        self.v_output = None  # output neuron voltage (state)
        self.g_ex_input = None
        self.g_inh_input = None
        self.g_ex = None
        self.g_inh = None
        self.g_ex_output = None
        self.g_inh_output = None

        # Hidden parameters
        self._ex_index = None  # TO BE INITIALIZED
        self._inh_index = None  # TO BE INITIALIZED
        self._is_ex = None  # TO BE INITIALIZED
        self._is_inh = None  # TO BE INITIALIZED
        # self._in_refract = np.zeros(self.n, dtype=bool)
        # self._in_refract_input = np.zeros(self.n_input, dtype=bool)
        # self._in_refract_output = np.zeros(self.n_output, dtype=bool)
        self._refract_time = None
        self._refract_time_input = None
        self._refract_time_output = None

        self.tau_kernal = tau_kernel

        self._initialize()

    def _initialize(self):

        # Basics
        self._initialized = False
        self._simulated = False
        self._trained = False
        self.t = 0
        self.simulate_data = None

        # Structure (no module structure yet)
        # self.w_input = sp.rand(self.n_input, self.n, density=self.p_input, format='lil')
        # self.w_input[self.w_input.nonzero()] = 1  # comment this for random input weights
        # self.children_of_input = {i: [] for i in range(self.n_input)}
        # for p, c in zip(*self.w_input.nonzero()):
        #     self.children_of_input[p].append(c)
        if self._input_method == 'current':
            self.w_input = sp.rand(self.n_input, self.n, density=self.p_input, format='lil')
            self.w_input = np.asarray(self.w_input.todense())
            self.w_input[0, self._is_inh] = 0
        elif self._input_method == 'spike':
            temp = np.random.choice(self.n, int(self.n * self.p_input), replace=False)
            self.w_input = np.zeros((self.n_input, self.n))
            self.w_input[:, temp] = 1
            self.connect_to_input = self.w_input.copy()

        self.w = sp.rand(self.n, self.n, density=self.p, format='lil')
        self.w[self.w.nonzero()] = 1
        # self._top_down_build(0, self.n, self.module_level)
        self.children_of = {i: [] for i in range(self.n)}
        for p, c in zip(*self.w.nonzero()):
            self.children_of[p].append(c)

        # State parameters
        self.v_input = self.v_rest * np.ones(self.n_input)  # input neuron voltage (state)
        self.v = self.v_rest * np.ones(self.n)  # internal neuron voltage (state)
        self.v_output = self.v_rest * np.ones(self.n_output)  # output neuron voltage (state)
        self.g_ex_input = np.zeros(self.n_input)
        self.g_inh_input = np.zeros(self.n_input)
        self.g_ex = np.zeros(self.n)
        self.g_inh = np.zeros(self.n)
        self.g_ex_output = np.zeros(self.n_output)
        self.g_inh_output = np.zeros(self.n_output)

        # Hidden parameters
        self._is_inh = np.zeros(self.n, dtype=bool)
        self._is_inh[np.random.choice(self.n, self.n_inh, replace=False)] = True
        self._is_ex = np.invert(self._is_inh)
        self._ex_index = np.nonzero(self._is_ex)[0]
        self._inh_index = np.nonzero(self._is_inh)[0]
        # self._in_refract = np.zeros(self.n, dtype=bool)
        # self._in_refract_input = np.zeros(self.n_input, dtype=bool)
        # self._in_refract_output = np.zeros(self.n_output, dtype=bool)
        self._refract_time = np.zeros(self.n)
        self._refract_time_input = np.zeros(self.n_input)
        self._refract_time_output = np.zeros(self.n_output)

        self.w_input[0, self._is_inh] = 0

        self._initialized = True

    def _top_down_build(self, low, up, level, inspect=False):
        # TODO
        # """Top-down approach building the hierarchical modular reservoir."""
        # 
        # if inspect:
        #     print("rewiring: (%d, %d)" % (low, up))
        # if low >= up:
        #     raise ValueError("Too many levels of HMN.")
        # if level == 0:
        #     return
        # mid = int((low + up) / 2)

        # # Rewire
        # # A slight amount of connections are lost due to the vectorization method.
        # # This doesn't matter much, and could be fixed by increasing p a tiny little bit.
        # inter_i, inter_j = self.W[mid:up, low:mid].nonzero()
        # inter_i += mid
        # inter_j += low
        # rewire_ind = np.random.rand(len(inter_i)) < (self.R_ex * self.isex[inter_j] + self.R_inh * self.isinh[inter_j])
        # before_i, before_j = inter_i[rewire_ind], inter_j[rewire_ind]
        # after_i, after_j = np.random.choice(range(low, mid), len(before_i)), before_j
        # self.W[(after_i, after_j)] = self.W[(before_i, before_j)]
        # self.W[(before_i, before_j)] = 0

        # inter_i, inter_j = self.W[low:mid, mid:up].nonzero()
        # inter_i += low
        # inter_j += mid
        # rewire_ind = np.random.rand(len(inter_i)) < (self.R_ex * self.isex[inter_j] + self.R_inh * self.isinh[inter_j])
        # before_i, before_j = inter_i[rewire_ind], inter_j[rewire_ind]
        # after_i, after_j = np.random.choice(range(mid, up), len(before_i)), before_j
        # self.W[(after_i, after_j)] = self.W[(before_i, before_j)]
        # self.W[(before_i, before_j)] = 0

        # self._top_down_build(low, mid, level - 1)
        # self._top_down_build(mid, up, level - 1)
        pass

    def update(self, ipt):
        """Update one time step."""


        dt = self.dt

        # 1. update V; use Euler method for now for simplicity.
        i_ex = self.g_ex * (self.v_ex - self.v)
        i_inh = self.g_inh * (self.v_inh - self.v)
        i_leaky = self.v_rest - self.v

        if self._noise_method == 'current':
            i_noise = self.noise_level * np.random.rand(self.n)
        elif self._noise_method == 'spike':
            i_noise = 0
        else:
            raise NotImplementedError()

        if self._input_method == 'current':
            i_input = ipt
        elif self._input_method == 'spike':
            i_input = 0
        else:
            raise NotImplementedError()

        dV = dt / self.tau * (i_leaky + i_noise + i_ex + i_inh + i_input)

        # 2. refractory neuron remains same
        dV[self._refract_time > (dt / 2)] = 0
        self.v += dV

        # 3. update g

        fire = self.v > self.v_threshold

        if self._noise_method == 'spike':
            fire[np.random.rand(self.n) < self.noise_level / 1000 * dt] = True
        if self._input_method == 'spike':
            fire[ipt] = True

        rest = np.invert(fire)

        add_gex = np.zeros(self.n, dtype=int)
        add_ginh = np.zeros(self.n, dtype=int)
        decay_gex = np.ones(self.n, dtype=bool)
        decay_ginh = np.ones(self.n, dtype=bool)

        for i in np.where(fire)[0]:
            if self._is_ex[i]:
                add_gex[self.children_of[i]] += 1
                decay_gex[self.children_of[i]] = False
            else:
                add_ginh[self.children_of[i]] += 1
                decay_ginh[self.children_of[i]] = False

        # if self._noise_method == 'spike':
        #     ind = np.random.rand(self.n) < self.noise_level / 1000 * dt
        #     for i in np.where(ind)[0]:
        #         if self._is_ex[i]:
        #             add_gex[self.children_of[i]] += 1
        #             decay_gex[self.children_of[i]] = False
        #         else:
        #             add_ginh[self.children_of[i]] += 1
        #             decay_ginh[self.children_of[i]] = False
        #
        # if self._input_method == 'spike':
        #     # ipt is a n-dimention binary vector
        #     for i in np.where(ipt)[0]:
        #         if self._is_ex[i]:
        #             add_gex[self.children_of[i]] += 1
        #             decay_gex[self.children_of[i]] = False
        #         else:
        #             add_ginh[self.children_of[i]] += 1
        #             decay_ginh[self.children_of[i]] = False
        # if ipt.any():
        #     print("input.any()")

        # print(len(np.where(decay_gex)[0]))
        self.g_ex += self.delta_gex * add_gex
        self.g_ex[decay_gex] -= dt / self.tau_ex * self.g_ex[decay_gex]
        self.g_inh += self.delta_ginh * add_ginh
        self.g_inh[decay_ginh] -= dt / self.tau_inh * self.g_inh[decay_ginh]

        # 4. put fired neuron to refractory
        self.v[fire] = self.v_rest
        self._refract_time[fire] = self.refract_max
        self._refract_time[rest] = np.maximum(0, self._refract_time[rest] - dt)

        # 5. update t, output info
        self.t += dt

        return (fire, i_ex, i_inh, i_leaky)

    def save(self):
        # TODO: save
        pass

    @staticmethod
    def load():
        # TODO: load
        pass

    def simulate(self, ipts, inspect_time=100):
        """Simulate. Must call first, before self.train().

        ipt: ndarray(n_input, n_time)
            Assume scaled.
        inspect_time: print messages every inspect_time ms.
        
        Return: simulation data (also stored in self.simulation_data).
        """
        # ipts is assumed scaled
        dt = self.dt
        if not self._initialized:
            raise ValueError("LSM not initialized.")

        print("Simulating...")
        if len(np.shape(ipts)) != 2 or np.shape(ipts)[0] != self.n_input:
            raise ValueError("HMN fit dimension error: inputs")

        n_time = ipts.shape[1]

        v_collect = np.zeros((self.n, n_time))
        fire_collect = np.zeros((self.n, n_time))
        i_ex_collect = np.zeros((self.n, n_time))
        i_inh_collect = np.zeros((self.n, n_time))
        i_leaky_collect = np.zeros((self.n, n_time))

        if self._input_method == 'current':
            actual_input = self.w_input.T @ ipts
        elif self._input_method == 'spike':
            rates = self.connect_to_input.T @ ipts
            actual_input = np.random.rand(self.n, n_time) < (rates / 1000 * dt)
            print(sum(actual_input.flatten()))

        else:
            raise NotImplementedError("input_method must be current or spike.")

        for t in range(n_time):
            step_info = self.update(actual_input[:, t])
            v_collect[:, t] = self.v
            fire_collect[:, t] = step_info[0]
            i_ex_collect[:, t] = step_info[1]
            i_inh_collect[:, t] = step_info[2]
            i_leaky_collect[:, t] = step_info[3]

            if self.t - int(self.t / inspect_time) * inspect_time < dt:
                print("Time: %.1f" % self.t)

        self._simulated = True
        self.simulate_data = {'v': v_collect,
                              'fire': fire_collect,
                              'i_ex': i_ex_collect,
                              'i_inh': i_inh_collect,
                              'i_leaky': i_leaky_collect,
                              'input':ipts
                              }
        return self.simulate_data

    def train(self, opts, t_forget=100,regularization=0):
        """Training. Must call after self.simulate().

        opts: function to be fitted.
        t_forget: forget first t_forget ms.
        regularization: \lambda in linear regression.

        Return: fitted function by linear regression.
        """

        # 
        dt = self.dt

        if len(opts.shape) != 2 or opts.shape[0] != self.n_output:
            raise ValueError("Opt dimension error.")

        if not self._simulated:
            raise ValueError("LSM not simulated.")

        fire_collect = self.simulate_data['fire']
        S = self._add_kernel(fire_collect)[:, int(t_forget / dt):].T
        S = np.hstack((S, np.ones((S.shape[0],1))))
        D = opts[:, int(t_forget / dt):].T

        #self.w_output = np.linalg.pinv(S) @ D
        self.w_output=(D.T @ S @ np.linalg.inv(S.T@ S+regularization*np.eye(S.shape[1]))).T
        self._trained = True
        self.fit_result=(S @ self.w_output).T
        return self.fit_result

    def _add_kernel(self, fire_collection):
        dt = self.dt
        opt = np.zeros(fire_collection.shape)
        opt[:, 0] = fire_collection[:, 0]
        for t in range(1, opt.shape[1]):
            opt[:, t] = fire_collection[:, t] + np.exp(-dt / self.tau_kernal) * opt[:, t - 1]
        return opt

    def predict(self, ipts, inspect_time=100):
        """Predict. Much be called after training.

        ipts: input
        inspect_time: print message every inspect_time ms. 

        Return: predicted function
        """
        dt = self.dt
        if not self._trained:
            raise ValueError("LSM not trained.")

        print("Predicting...")
        if len(np.shape(ipts)) != 2 or np.shape(ipts)[0] != self.n_input:
            raise ValueError("HMN fit dimension error: inputs")

        n_time = ipts.shape[1]

        v_collect = np.zeros((self.n, n_time))
        fire_collect = np.zeros((self.n, n_time))
        i_ex_collect = np.zeros((self.n, n_time))
        i_inh_collect = np.zeros((self.n, n_time))
        i_leaky_collect = np.zeros((self.n, n_time))

        if self._input_method == 'current':
            actual_input = self.w_input.T @ ipts
        elif self._input_method == 'spike':
            rates = self.connect_to_input.T @ ipts
            actual_input = np.random.rand(self.n, n_time) < (rates / 1000 * dt)
            print(sum(actual_input.flatten()))

        else:
            raise NotImplementedError("input_method must be current or spike.")

        for t in range(n_time):
            step_info = self.update(actual_input[:, t])
            v_collect[:, t] = self.v
            fire_collect[:, t] = step_info[0]
            i_ex_collect[:, t] = step_info[1]
            i_inh_collect[:, t] = step_info[2]
            i_leaky_collect[:, t] = step_info[3]

            if self.t - int(self.t / inspect_time) * inspect_time < dt:
                print("Time: %.1f" % self.t)

        # self._simulated = True
        self.predict_data = {'v': v_collect,
                             'fire': fire_collect,
                             'i_ex': i_ex_collect,
                             'i_inh': i_inh_collect,
                             'i_leaky': i_leaky_collect
                             }

        S = self._add_kernel(fire_collect).T
        S = np.hstack((S, np.ones((S.shape[0],1))))
        self.predict_result=(S @ self.w_output).T
        # D = opts[:, int(t_forget / dt):].T
        #
        # self.w_output = np.linalg.pinv(S) @ D
        #
        # self._trained = True
        return self.predict_result


# def linear_regression_pseudoinverse(S, D):
#     """Pseudoinverse linear regression

#     S: ntime x (ninput+ninternal)
#     D: ntime x noutput
#     """
#     return (np.linalg.pinv(S) @ D).T


def poisson_spike(t, f, dt=0.1, dim=1):
    """ Generate a Poisson spike train.

    t: length
    f: frequency
    dt: time step; default 0.1ms
    """
    # dt, t in ms; f in Hz.
    return np.random.rand(dim, int(t / dt)) < (f * dt / 1000)


# def rate_of(spikes, resolution=100, dt=0.1, f='discrete'):
#     spikes = spikes.copy()
#     if spikes.ndim == 1:
#         spikes.reshape((1, -1))
#     n = spikes.shape[1]

#     n_resolution = int(resolution / dt)

#     n_bin = int(n / n_resolution)
#     bins = [[i * n_resolution, (i + 1) * n_resolution] for i in range(n_bin)]
#     bins.append([n_bin * n_resolution, n])

#     if f == 'discrete':
#         rate = np.zeros(spikes.shape)
#         for l, u in bins:
#             r = 1000 * np.count_nonzero(spikes[:, l:u], axis=1) / resolution
#             rate[:, l:u] = r
#     else:
#         raise NotImplemented()

#     return rate


def plot_spike(spikes, rates, dt=0.1):
    """Plot a single spike train with firing rates. """
    plt.figure()
    plt.subplot(211)
    plt.title("spikes")
    n, ts = spikes.nonzero()
    ts = ts * dt
    plt.scatter(ts, n, s=2)
    plt.subplot(212)
    plt.title("rates")
    plot_1d(rates, dx=dt)


def plot_fire_collect(fire_collect, dt=0.1, title="fire collect"):
    """Plot spike collection (lsm.simulate_data['fire'])"""
    plt.figure()
    plt.title(title)
    n, ts = fire_collect.nonzero()
    ts = ts * dt
    plt.scatter(ts, n, s=0.3)


def plot_gap_distribution(fire_collect, dt=0.1, title="gap distribution"):
    """Plot interval distribution between spikes. 
    
    SOC shows a power-law distribution.
    """
    have_spike = ~np.any(fire_collect, axis=0)
    ind_spike = np.where(have_spike)[0]
    gaps = ind_spike[1:] - ind_spike[:-1]
    gaps = gaps[gaps > 1]

    plt.figure()
    plt.title(title)

    if len(gaps) == 0:
        print("Error: len(gaps)==0")
    else:
        ps, bins = np.histogram(gaps, bins=50, density=True)
        xs = (bins[:-1] + bins[1:]) / 2
        _arg = ps > 0
        xs = xs[_arg]
        ys = ps[_arg]
        print(xs, ys)
        plt.scatter(xs, ys)
        plt.loglog()
        plt.ylim(ymax=1.2, ymin=min(ys) * 0.5)
        plt.xlim(xmin=1.5, xmax=max(xs * 1.5))


def plot_activity_distribution(fire_collect, dt=0.1, title="activity_distribution"):
    """Plot activity distribution of spikes. 
    
    SOC shows a power-law distribution.
    """
    # TODO
    pass


def plot_v(v, dt=0.1):
    """Plot voltage of a single neuron."""
    plt.figure()
    plt.title('v')
    nmax = len(v)
    plt.plot(np.arange(nmax) * dt, v)


def plot_v_collect(v_collect, dt=0.1):
    """Plot voltage of many neurons (lsm.simulate_data['v'])."""
    plt.figure()
    plt.title("v_collect")
    nmax = v_collect.shape[1]
    for i in range(len(v_collect)):
        plt.subplot(len(v_collect), 1, i + 1)
        plt.plot(np.arange(nmax) * dt, v_collect[i, :])


def plot_i(i_ex, i_inh, dt=0.1):
    """Plot currents of a single neuron."""
    plt.figure()
    plt.title("i")
    nmax = len(i_ex)
    plt.plot(np.arange(nmax) * dt, i_ex, label='i_ex')
    plt.plot(np.arange(nmax) * dt, i_inh, label='i_inh')
    plt.plot(np.arange(nmax) * dt, i_ex + i_inh, label='i_total')


def plot_i_collect(i_ex_collect, i_inh_collect, n=5, dt=0.1):
    """Plot currents of many neurons (lsm.simulate_data['i_ex','i_inh, 'i_ex'+'i_inh'])."""
    plt.figure()
    plt.title("i")
    nmax = i_ex_collect.shape[1]
    for i in range(n):
        plt.subplot(n, 1, i + 1)
        plt.plot(np.arange(nmax) * dt, i_ex_collect[i, :], label="i_ex", linewidth=1)
        plt.plot(np.arange(nmax) * dt, i_inh_collect[i, :], label='i_inh', linewidth=1)
        plt.plot(np.arange(nmax) * dt, i_ex_collect[i, :] + i_inh_collect[i, :], label="i_total", linewidth=1)
        plt.plot(np.arange(nmax) * dt, np.zeros(nmax), linestyle='--', linewidth=1)
        plt.legend()


def plot_simulate_data(simulate_data, n=10):
    """Combine a few things above."""
    print("Plotting simulation data...")
    plot_fire_collect(simulate_data['fire'])
    plot_v_collect(simulate_data['v'][:n, :])
    plot_i_collect(simulate_data['i_ex'][:n, :],
                   simulate_data['i_inh'][:n, :])


if __name__ == '__main__':
    # lsm = LSM(500, noise_level=0)
    # input_scale = 20

    # ipts = np.random.rand(1, 10000) * input_scale
    # lsm.simulate(ipts)

    # plot_v(lsm.simulate_data['v'][0])
    # plt.show(block=False)

    # plot_i(lsm.simulate_data['i_ex'][0], lsm.simulate_data['i_inh'][0])

    # plot_fire_collect(lsm.simulate_data['fire'])
    # plt.show()
    pass