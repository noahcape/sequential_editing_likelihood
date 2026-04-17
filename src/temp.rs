/// This file does nothing, just placing old functions here


// Solve the ODE system with rk4
pub fn rk4_fo() {
    let lambda = 1.0;
    let tau = 0.5;
    let h = 2.0;
    let rho = 0.1;

    let levels = 11;
    // initial condition (1 - rho)
    let mut u = vec![1. - rho; levels];

    let dt = 0.01;
    let t_max = 10.0;

    let mut t = 0.0;

    while t < t_max {
        rk4_step_fo(&mut u, dt, lambda, tau, h, u_ode_fo, t);
        t += dt;
        println!("{:#?}", u);
    }
}

pub fn rk4_so() {
    let lambda = 1.0;
    let tau = 0.5;
    let h = 2.0;
    let rho = 0.1;

    let levels = 15;
    let c_levels = levels * levels;
    let mut u = vec![1. - rho; levels];
    let mut c = vec![0.; c_levels];

    let dt = 0.01;
    let t_max = 4.0;

    let mut t = 0.0;

    while t < t_max {
        rk4_step_so(&mut u, &mut c, lambda, tau, h, dt);
        println!("{:#?}", u);
        t += dt;
    }
}

/// First order approximation of the U ODE system
/// with ansatz E[UiUj] = E[Ui]E[Uj] and
/// E[Ui]=E[Ul] for i,l \in L(k) (the same level)

/// U^{0}(t)                = \frac{1}{1 + \frac{\rho}{1 - \rho}e^{\lambda t}}
/// \dot{U^{2n + 1}}(t)     = -(\lambda  + \tau)U^{2n+1}(t) + \tau U^{2n}(t) + \lambda U^{2n}(t)U^{2(n+1)}(t)
/// \dot{U^{2n}}(t)         = -(\lambda + H)U^{2n}(t) + HU^{2n - 1}(t) + \lambda U^{2n}(t)^2
/// Implement simple ODE population scheme
fn u_ode_fo(u: &[f64], du: &mut [f64], lambda: f64, tau: f64, h: f64, t: f64) {
    let n = u.len();
    assert!(n % 2 == 1, "Length must be odd.");

    for d in 0..n {
        if d == 0 {
            du[d] = -lambda * u[d] + lambda * (u[d] * u[d])
        } else if d % 2 == 1 {
            du[d] = -(lambda + tau) * u[d] + tau * u[d - 1] + lambda * u[d - 1] * u[d + 1];
        } else {
            du[d] = -(lambda + h) * u[d] + h * u[d - 1] + lambda * u[d] * u[d]
        }
    }
}

/// Solve the u ode system with RK4 scheme - with first order approximation
/// k_1     = f(u^n)
/// k_2     = f(u^n + k_1 dt/2)
/// k_3     = f(u^n + k_2 dt/2)
/// k_4     = f(u^n + k_3 dt)
/// u^n+1   = u^n + (k_1 + 2k_2 + 2k_3 + k_4) * dt/6
/// One rk4 step
fn rk4_step_fo<T: Fn(&[f64], &mut [f64], f64, f64, f64, f64)>(
    u: &mut [f64],
    dt: f64,
    lambda: f64,
    tau: f64,
    h: f64,
    ode: T,
    t: f64,
) {
    let n = u.len();
    assert!(n % 2 == 1, "Length must be odd.");

    let mut k1 = vec![0.0; n];
    let mut k2 = vec![0.0; n];
    let mut k3 = vec![0.0; n];
    let mut k4 = vec![0.0; n];
    let mut temp = vec![0.0; n];

    // k1
    ode(u, &mut k1, lambda, tau, h, t);

    // k2
    for i in 0..n {
        temp[i] = u[i] + (k1[i] * dt / 2.)
    }
    ode(&temp, &mut k2, lambda, tau, h, t);

    // k3
    for i in 0..n {
        temp[i] = u[i] + (k2[i] * dt / 2.)
    }
    ode(&temp, &mut k3, lambda, tau, h, t);

    // k4
    for i in 0..n {
        temp[i] = u[i] + (k3[i] * dt)
    }
    ode(&temp, &mut k4, lambda, tau, h, t);

    for i in 0..n {
        u[i] += (k1[i] + 2. * k2[i] + 2. * k3[i] + k4[i]) * dt / 6.
    }
}

/// Second order approximation of the U ODEs
/// With the ansatz E[UiUj] = E[Ui]E[Uj] + Cov(E[Ui], E[Uj]) and
/// E[Ui]=E[Ul] for i,l \in L(k) (the same level)
fn u_ode_so(u: &[f64], du: &mut [f64], c: &[f64], dc: &mut [f64], lambda: f64, tau: f64, h: f64) {
    let n = u.len();
    assert!(n % 2 == 1, "Length must be odd.");

    // Jacobian diagonals
    let mut jd = vec![0.0; n];
    let mut jld = vec![0.0; n];
    let mut jud = vec![0.0; n];

    // set u system
    for d in 0..n {
        if d == 0 {
            du[d] = -lambda * u[d] + lambda * (u[d] * u[d] + c[d * n + d]);

            jd[d] = -lambda + 2. * lambda * u[d];
        } else if d % 2 == 1 {
            du[d] = -(lambda + tau) * u[d]
                + tau * u[d - 1]
                + lambda * (u[d - 1] * u[d + 1] + c[((d - 1) * n) + d + 1]);

            jld[d - 1] = tau + lambda * u[d + 1];
            jd[d] = -(lambda + tau);
            jud[d + 1] = lambda * u[d - 1];
        } else {
            du[d] = -(lambda + h) * u[d] + h * u[d - 1] + lambda * (u[d] * u[d] + c[d * n + d]);

            jld[d - 1] = h;
            jd[d] = -(lambda + h) + 2. * lambda * u[d];
        }
    }

    // set c system
    for i in 0..n {
        for j in 0..n {
            let idx = i * n + j;
            dc[idx] = 0.0;

            // Row contribution
            if i > 0 {
                dc[idx] += jld[i] * c[(i - 1) * n + j];
            }
            dc[idx] += jd[i] * c[i * n + j];
            if i < n - 1 {
                dc[idx] += jud[i] * c[(i + 1) * n + j];
            }

            // Column contribution
            if j > 0 {
                dc[idx] += jld[j] * c[i * n + j - 1];
            }
            dc[idx] += jd[j] * c[i * n + j];
            if j < n - 1 {
                dc[idx] += jud[j] * c[i * n + j + 1];
            }

            if i % 2 == 0 {
                dc[idx] += lambda * u[j] * c[i * n + i];
            } else {
                // odd i
                dc[idx] += lambda
                    * (u[i - 1] * c[((i + 1) * n) + j]
                        + u[i + 1] * c[((i - 1) * n) + j]
                        + u[j] * c[((i - 1) * n) + i + 1]);
            }

            if j % 2 == 0 {
                dc[idx] += lambda * u[i] * c[j * n + j];
            } else {
                // odd j
                dc[idx] += lambda
                    * (u[j - 1] * c[((j + 1) * n) + i]
                        + u[j + 1] * c[((j - 1) * n) + i]
                        + u[i] * c[((j - 1) * n) + j + 1]);
            }
        }
    }
}

fn rk4_step_so(u: &mut [f64], c: &mut [f64], lambda: f64, tau: f64, h: f64, dt: f64) {
    let n = u.len();
    let nn = c.len();

    let u0 = u.to_vec();
    let c0 = c.to_vec();

    let mut k1_u = vec![0.0; n];
    let mut k1_c = vec![0.0; nn];

    let mut k2_u = vec![0.0; n];
    let mut k2_c = vec![0.0; nn];

    let mut k3_u = vec![0.0; n];
    let mut k3_c = vec![0.0; nn];

    let mut k4_u = vec![0.0; n];
    let mut k4_c = vec![0.0; nn];

    let mut u_temp = vec![0.0; n];
    let mut c_temp = vec![0.0; nn];

    // k1
    u_ode_so(&u0, &mut k1_u, &c0, &mut k1_c, lambda, tau, h);

    // k2
    for i in 0..n {
        u_temp[i] = u0[i] + 0.5 * dt * k1_u[i];
    }
    for i in 0..nn {
        c_temp[i] = c0[i] + 0.5 * dt * k1_c[i];
    }
    u_ode_so(&u_temp, &mut k2_u, &c_temp, &mut k2_c, lambda, tau, h);

    // k3
    for i in 0..n {
        u_temp[i] = u0[i] + 0.5 * dt * k2_u[i];
    }
    for i in 0..nn {
        c_temp[i] = c0[i] + 0.5 * dt * k2_c[i];
    }
    u_ode_so(&u_temp, &mut k3_u, &c_temp, &mut k3_c, lambda, tau, h);

    // k4
    for i in 0..n {
        u_temp[i] = u0[i] + dt * k3_u[i];
    }
    for i in 0..nn {
        c_temp[i] = c0[i] + dt * k3_c[i];
    }
    u_ode_so(&u_temp, &mut k4_u, &c_temp, &mut k4_c, lambda, tau, h);

    // update
    for i in 0..n {
        u[i] = u0[i] + dt / 6.0 * (k1_u[i] + 2.0 * k2_u[i] + 2.0 * k3_u[i] + k4_u[i]);
    }

    for i in 0..nn {
        c[i] = c0[i] + dt / 6.0 * (k1_c[i] + 2.0 * k2_c[i] + 2.0 * k3_c[i] + k4_c[i]);
    }
}

/// Compute the expected number extant cells in a given level
fn d_ode_u1(d: &[f64], dd: &mut [f64], lambda: f64, tau: f64, h: f64, _: f64) {
    let n = d.len();
    assert!(n % 2 == 1, "Length must be odd.");

    for l in 0..n {
        if l == 0 {
            // flow in from birth
            dd[l] = lambda * (d[l] + d[l]) + tau * d[l + 1] + lambda * d[l + 1];
        } else if l == n - 1 {
            // flow out via edit
            dd[l] = -h * d[l] + 2. * lambda * d[l] + lambda * d[l - 1];
        } else if l % 2 == 1 {
            // flow in from even via h
            // flow out via lambda on all states
            dd[l] = h * d[l + 1] - 2. * lambda * d[l] - tau * d[l]
        } else {
            // flow in via transfer on odd and birth
            // flow out via h
            dd[l] = tau * d[l + 1] + lambda * (d[l] + d[l]) + lambda * d[l - 1] - h * d[l]
        }
    }
}

pub fn rk4_d_u1() {
    let lambda = 1.0;
    let tau = 0.1;
    let h = 4.0;

    let levels = 11;
    for l in [9, 10] {
        let mut d = vec![0.; levels];
        d[l] = 1.;

        let dt = 0.01;
        let t_max = 10.;

        let mut t = 0.0;

        while t < t_max {
            rk4_step_fo(&mut d, dt, lambda, tau, h, d_ode_u1, t);
            t += dt;
        }
        println!(
            "Expected number of lineages in levels starting with state in level {l}: {:#?}",
            d
        );
    }
}



/// Compute the initial density of each state which is the normalized expectation
/// of the number of extant leaves in that given level at the present time
pub fn f_i(h: f64, tau: f64, lambda: f64, b_len: f64, t_max: f64, m: usize) -> Vec<f64> {
    // compute initial probabilities after b_len for the root node
    // where the zero state is un-edited and last level is saturated
    let init_probs = level_init_prob(h, tau, b_len, m);
    // initiatlize empty expectations
    let mut expectation = vec![0.; (2 * m) + 1];

    for l in 0..(2 * m) + 1 {
        let mut d = vec![0.; (2 * m) + 1];
        d[l] = 1.;

        let dt = 0.01;
        let mut t = 0.0;

        while t < t_max {
            rk4_step_fo(&mut d, dt, lambda, tau, h, extant_distr, t);
            t += dt;
        }

        // d is the expected number of leaves at level i starting in level l
        // where 0 is saturated and 2*m is unedited
        for i in 0..d.len() {
            // combine to add to expected = probability * events
            expectation[i] += d[i] * init_probs[d.len() - i - 1]
        }
    }

    let sum: f64 = expectation.iter().sum();

    // normalize to convert each into a probability
    expectation.iter().map(|e| e / sum).collect()
}

fn level_init_prob(h: f64, tau: f64, b_len: f64, m: usize) -> Vec<f64> {
    let n = (2 * m) + 1;
    let mu: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { h } else { tau }).collect();

    let mut diag = vec![0.0; n];
    let mut superdiag = vec![0.0; n.saturating_sub(1)];

    for i in 0..n.saturating_sub(1) {
        diag[i] = -mu[i];
        superdiag[i] = mu[i];
    }

    exp_upper_bidiagonal_first_row(&diag, &superdiag, b_len, 1e-12)
}

/// codex function
fn exp_upper_bidiagonal_first_row(diag: &[f64], superdiag: &[f64], t: f64, tol: f64) -> Vec<f64> {
    let n = diag.len();
    assert_eq!(superdiag.len(), n.saturating_sub(1));

    if n == 0 {
        return Vec::new();
    }

    if t == 0.0 {
        let mut row = vec![0.0; n];
        row[0] = 1.0;
        return row;
    }

    // Uniformization: exp(Qt) = sum_k Pois(nu t, k) (I + Q / nu)^k.
    // For this upper-bidiagonal chain we only propagate one row vector, so each term is O(n).
    let nu = diag
        .iter()
        .zip(superdiag.iter().copied().chain(std::iter::once(0.0)))
        .map(|(d, s)| s.max(-*d))
        .fold(0.0_f64, f64::max);

    if nu == 0.0 {
        let mut row = vec![0.0; n];
        row[0] = 1.0;
        return row;
    }

    let lambda = nu * t;
    let mut poisson_weight = (-lambda).exp();

    let mut current = vec![0.0; n];
    let mut next = vec![0.0; n];
    current[0] = 1.0;

    let mut acc = current
        .iter()
        .map(|entry| poisson_weight * entry)
        .collect::<Vec<_>>();
    let mut poisson_tail = 1.0 - poisson_weight;

    let max_iters = ((lambda + 10.0 * lambda.sqrt()).ceil() as usize + n + 32).max(64);

    for k in 1..=max_iters {
        next.fill(0.0);

        for i in 0..n {
            next[i] += current[i] * (1.0 + diag[i] / nu);
            if i + 1 < n {
                next[i + 1] += current[i] * (superdiag[i] / nu);
            }
        }

        std::mem::swap(&mut current, &mut next);

        poisson_weight *= lambda / (k as f64);
        poisson_tail -= poisson_weight;

        for i in 0..n {
            acc[i] += poisson_weight * current[i];
        }

        if poisson_tail.max(0.0) * current.iter().sum::<f64>() < tol {
            break;
        }
    }

    acc
}

#[test]
fn init_prob() {
    let probs = level_init_prob(5.4, 0.2, 1.5, 3);

    assert_eq!(probs.len(), 6);
    assert!(probs.iter().all(|p| *p >= -1e-12));
    assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    assert!((probs[0] - (-5.4_f64 * 1.5).exp()).abs() < 1e-10);
}

#[test]
fn init_prob_equal_rates_matches_truncated_poisson_chain() {
    let mu = 1.7;
    let t = 0.8;
    let probs = level_init_prob(mu, mu, t, 2);
    let x = mu * t;
    let e = (-x).exp();

    assert!((probs[0] - e).abs() < 1e-10);
    assert!((probs[1] - x * e).abs() < 1e-10);
    assert!((probs[2] - 0.5 * x * x * e).abs() < 1e-10);
    assert!((probs[3] - (1.0 - e * (1.0 + x + 0.5 * x * x))).abs() < 1e-10);
}

#[test]
fn f_i_values() {
    let f_i = f_i(5.4, 0.2, 0.5, 4.2, 50.0, 10);
    println!("{:?}", f_i);
    println!("{:?}", f_i.iter().sum::<f64>());
}


// Solve the ODE system with rk4
pub fn u_temp(
    h: f64,
    lambda: f64,
    tau: f64,
    rho: f64,
    b_len: f64,
    t_max: f64,
    m: usize,
) -> Vec<Vec<f64>> {
    let f = f_i(h, tau, lambda, b_len, t_max, m);
    let mut u: Vec<f64> = (0..(2 * m) + 1).map(|i| (1. - rho) * f[i]).collect();

    let mut system = vec![u.clone()];

    let dt = 0.01;
    let mut t = 0.0;

    while t < t_max {
        rk4_step_fo(&mut u, dt, lambda, tau, h, u_ode_fo, t);
        system.push(u.clone());
        t += dt;
    }

    return system;
}

/// U^{0}(t)                = \frac{1}{1 + \frac{\rho}{1 - \rho}e^{\lambda t}}
/// \dot{U^{2n + 1}}(t)     = -(\lambda  + \tau)U^{2n+1}(t) + \tau U^{2n}(t) + \lambda U^{2n}(t)U^{2(n+1)}(t)
/// \dot{U^{2n}}(t)         = -(\lambda + H)U^{2n}(t) + HU^{2n - 1}(t) + \lambda U^{2n}(t)^2
/// Implement simple ODE population scheme
fn u_ode_fo_pop(u: &[f64], du: &mut [f64], lambda: f64, tau: f64, h: f64, _: f64) {
    let n = u.len();

    for d in 0..n {
        if d == 0 {
            du[d] = -lambda * u[d] + lambda * (u[d] * u[d])
        } else if d == n - 1 {
            du[d] = -(lambda + h) * u[d] + h * u[d - 1] + lambda * u[d] * u[d]
        } else if d % 2 == 1 {
            du[d] = -(lambda + tau) * u[d] + tau * u[d - 1] + lambda * u[d - 1] * u[d + 1];
        } else {
            du[d] = -(lambda + h) * u[d] + h * u[d - 1] + lambda * u[d] * u[d]
        }
    }
}

/// Compute the expected number extant cells in a given level at a given time
fn extant_distr(u: &[f64], du: &mut [f64], lambda: f64, tau: f64, h: f64, _: f64) {
    let n = u.len();

    for l in 0..n {
        if l == 0 {
            // flow in from birth
            du[l] = lambda * (u[l] + u[l]) + tau * u[l + 1] + lambda * u[l + 1];
        } else if l == n - 1 {
            // flow out via edit
            du[l] = -h * u[l] + 2. * lambda * u[l] + lambda * u[l - 1];
        } else if l % 2 == 1 {
            // flow in from even via h
            // flow out via lambda on all states
            du[l] = h * u[l + 1] - 2. * lambda * u[l] - tau * u[l]
        } else {
            // flow in via transfer on odd and birth
            // flow out via h
            du[l] = tau * u[l + 1] + lambda * (u[l] + u[l]) + lambda * u[l - 1] + lambda * u[l + 1]
                - h * u[l]
        }
    }
}