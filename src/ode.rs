use crate::{TapeGraph, TapeOrientation};

/// Analytic solution for U(t)
fn u_analytic(lambda: f64, rho: f64, t: f64) -> f64 {
    let a = rho / (1. - rho);
    return 1. / (1. + a * (t * lambda).exp());
}

/// Solve the u ode system with RK4 scheme - with first order approximation
/// k_1     = f(u^n)
/// k_2     = f(u^n + k_1 dt/2)
/// k_3     = f(u^n + k_2 dt/2)
/// k_4     = f(u^n + k_3 dt)
/// u^n+1   = u^n + (k_1 + 2k_2 + 2k_3 + k_4) * dt/6
/// One rk4 step
#[derive(Default)]
pub(crate) struct Rk4Scratch {
    k1: Vec<f64>,
    k2: Vec<f64>,
    k3: Vec<f64>,
    k4: Vec<f64>,
    temp: Vec<f64>,
}

impl Rk4Scratch {
    fn ensure_len(&mut self, n: usize) {
        if self.k1.len() != n {
            self.k1.resize(n, 0.0);
            self.k2.resize(n, 0.0);
            self.k3.resize(n, 0.0);
            self.k4.resize(n, 0.0);
            self.temp.resize(n, 0.0);
        }
    }
}

pub(crate) fn rk4_d_ode(
    d: &mut [f64],
    scratch: &mut Rk4Scratch,
    dt: f64,
    lambda: f64,
    tau: f64,
    t: f64,
    rho: f64,
    eta: &[f64],
    graph: &TapeGraph,
) {
    let n = d.len();
    scratch.ensure_len(n);

    // k1
    d_ode(
        d,
        &mut scratch.k1,
        lambda,
        tau,
        eta,
        u_analytic(lambda, rho, t),
        graph,
    );

    // k2
    for i in 0..n {
        scratch.temp[i] = d[i] + (scratch.k1[i] * dt / 2.)
    }
    d_ode(
        &scratch.temp,
        &mut scratch.k2,
        lambda,
        tau,
        eta,
        u_analytic(lambda, rho, t + 0.5 * dt),
        graph,
    );

    // k3
    for i in 0..n {
        scratch.temp[i] = d[i] + (scratch.k2[i] * dt / 2.)
    }
    d_ode(
        &scratch.temp,
        &mut scratch.k3,
        lambda,
        tau,
        eta,
        u_analytic(lambda, rho, t + 0.5 * dt),
        graph,
    );

    // k4
    for i in 0..n {
        scratch.temp[i] = d[i] + (scratch.k3[i] * dt)
    }
    d_ode(
        &scratch.temp,
        &mut scratch.k4,
        lambda,
        tau,
        eta,
        u_analytic(lambda, rho, t + dt),
        graph,
    );

    for i in 0..n {
        d[i] += (scratch.k1[i] + 2. * scratch.k2[i] + 2. * scratch.k3[i] + scratch.k4[i]) * dt / 6.
    }
}

// solve for D_Ni(t) over i
fn d_ode(
    d: &[f64],
    dd: &mut [f64],
    lambda: f64,
    tau: f64,
    eta: &[f64],
    u_at_t: f64,
    graph: &TapeGraph,
) {
    let h: f64 = eta.iter().sum();

    for (i, state_i) in graph.states.iter().enumerate() {
        let divide_sum = graph.divide_targets[i]
            .iter()
            .map(|op_idx| op_idx.map(|idx| d[idx]).unwrap_or(0.))
            .sum::<f64>();

        match state_i.orientation {
            TapeOrientation::Even => {
                let mut edit_sum = 0.;
                for (e, idx) in graph.edit_targets[i].iter().enumerate() {
                    edit_sum += idx.map(|j| d[j] * eta[e]).unwrap_or(0.);
                }
                dd[i] = -(lambda + h) * d[i] + edit_sum + lambda / 2. * u_at_t * divide_sum;
            }
            TapeOrientation::Odd => {
                let transfer = graph.transfer_targets[i].map(|j| d[j]).unwrap_or(0.);
                dd[i] = -(lambda + tau) * d[i] + tau * transfer + lambda / 2. * u_at_t * divide_sum
            }
        }
    }
}
