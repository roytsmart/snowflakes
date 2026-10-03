//! The snow crystal growth model of Gravner and Griffeath (2008), "Modeling
//! snow crystal growth II: A mesoscopic lattice map with plausible dynamics",
//! Physica D 237, 385, compiled to WebAssembly for the snowflakes web app.
//!
//! It computes exactly what the snowflakes Python package computes in single
//! precision, bit for bit: the same operations, on the same values, in the
//! same order as `snowflakes._numpy.step`. The package's tests build this
//! crate natively and check that the two agree.
//!
//! The grid is a triangular lattice stored on a square array: each cell
//! neighbors the cells above, below, left, and right of it, and the two along
//! the diagonal from lower left to upper right. Its edges wrap around. Every
//! field is padded by one cell on each side, filled from the opposite edge.
//!
//! The functions at the bottom are a C interface, which JavaScript calls in
//! the browser and Python calls in the tests.

/// A snow crystal growing on a periodic grid.
pub struct Crystal {
    /// The number of rows and columns of the grid.
    size: usize,
    /// The number of rows and columns of the padded fields.
    stride: usize,
    /// Whether each cell is attached to the crystal, as 0 or 1.
    a: Vec<u8>,
    /// The boundary mass, the quasi-liquid layer, of each cell.
    b: Vec<f32>,
    /// The crystal mass, the ice, of each cell.
    c: Vec<f32>,
    /// The diffusive mass, the vapor, of each cell.
    d: Vec<f32>,
    /// Space for the new vapor.
    e: Vec<f32>,
    /// The number of attached neighbors of each cell.
    count: Vec<u8>,
    /// The number of updates made so far.
    steps: u32,
    /// The distance, in cells, from the seed to the farthest attached cell.
    radius: u32,
    /// The state of the random number generator of the noise.
    rng: u64,
    /// Unpadded copies of the fields, for drawing.
    frame_a: Vec<u8>,
    frame_b: Vec<f32>,
    frame_c: Vec<f32>,
    frame_d: Vec<f32>,
    /// Space for sorting the crystal mass.
    scratch: Vec<f32>,
}

/// The parameters of the model, rounded to single precision once, as the
/// Python package rounds them.
struct Coefficients {
    alpha: f32,
    beta: f32,
    theta: f32,
    kappa: f32,
    one_minus_kappa: f32,
    mu: f32,
    one_minus_mu: f32,
    gamma: f32,
    one_minus_gamma: f32,
}

impl Coefficients {
    fn new(alpha: f64, beta: f64, gamma: f64, theta: f64, kappa: f64, mu: f64) -> Self {
        Coefficients {
            alpha: alpha as f32,
            beta: beta as f32,
            theta: theta as f32,
            kappa: kappa as f32,
            one_minus_kappa: (1.0 - kappa) as f32,
            mu: mu as f32,
            one_minus_mu: (1.0 - mu) as f32,
            gamma: gamma as f32,
            one_minus_gamma: (1.0 - gamma) as f32,
        }
    }
}

/// `x[p]`, without checking that `p` is in bounds.
///
/// The kernel only reads a cell inside the padding and its six neighbors,
/// which the padding guarantees are in bounds, so the checks Rust would make
/// cost time for nothing: half the speed of the kernel, as measured. Test
/// builds still check them.
#[inline(always)]
fn at<T: Copy>(x: &[T], p: usize) -> T {
    debug_assert!(p < x.len());
    // SAFETY: every caller passes a cell inside the padding or a neighbor of
    // one, which lies within the padded field.
    unsafe { *x.get_unchecked(p) }
}

/// `&mut x[p]`, without checking that `p` is in bounds, as for [`at`].
#[inline(always)]
fn at_mut<T>(x: &mut [T], p: usize) -> &mut T {
    debug_assert!(p < x.len());
    // SAFETY: as for `at`.
    unsafe { x.get_unchecked_mut(p) }
}

/// Fill the padding of `x` from the opposite edges of the grid.
fn wrap<T: Copy>(x: &mut [T], size: usize) {
    let s = size + 2;
    for j in 1..=size {
        x[j] = x[size * s + j];
        x[(size + 1) * s + j] = x[s + j];
    }
    for i in 0..s {
        x[i * s] = x[i * s + size];
        x[i * s + size + 1] = x[i * s + 1];
    }
}

impl Crystal {
    /// A single cell of ice in the middle of vapor of density `rho`.
    pub fn new(size: usize, rho: f64, seed: u64) -> Self {
        let stride = size + 2;
        let cells = stride * stride;
        let mut crystal = Crystal {
            size,
            stride,
            a: vec![0; cells],
            b: vec![0.0; cells],
            c: vec![0.0; cells],
            d: vec![0.0; cells],
            e: vec![0.0; cells],
            count: vec![0; cells],
            steps: 0,
            radius: 0,
            // SplitMix64 needs no particular seed, but zero is a poor state
            // for the xorshift generator it seeds.
            rng: seed ^ 0x9E37_79B9_7F4A_7C15,
            frame_a: vec![0; size * size],
            frame_b: vec![0.0; size * size],
            frame_c: vec![0.0; size * size],
            frame_d: vec![0.0; size * size],
            scratch: Vec::with_capacity(size * size),
        };
        let vapor = rho as f32;
        for i in 1..=size {
            for j in 1..=size {
                crystal.d[i * stride + j] = vapor;
            }
        }
        let center = (size / 2 + 1) * stride + size / 2 + 1;
        crystal.a[center] = 1;
        crystal.c[center] = 1.0;
        crystal.d[center] = 0.0;
        crystal
    }

    /// Advance the crystal by `num_steps` updates of the model.
    #[allow(clippy::too_many_arguments)]
    pub fn grow(
        &mut self,
        num_steps: u32,
        alpha: f64,
        beta: f64,
        gamma: f64,
        theta: f64,
        kappa: f64,
        mu: f64,
        sigma: f64,
    ) {
        let coefficients = Coefficients::new(alpha, beta, gamma, theta, kappa, mu);
        for _ in 0..num_steps {
            self.step(&coefficients);
            if sigma != 0.0 {
                self.noise(sigma);
            }
            self.steps += 1;
        }
    }

    /// Diffusion, freezing, attachment, and melting: steps i to iv.
    fn step(&mut self, k: &Coefficients) {
        let n = self.size;
        let s = self.stride;
        let center = (n / 2) as i64;
        let a = &mut self.a;
        let b = &mut self.b;
        let c = &mut self.c;
        let d = &mut self.d;
        let e = &mut self.e;
        let count = &mut self.count;

        wrap(a, n);
        wrap(d, n);

        // i. Diffusion: each unattached cell takes the average of itself and
        // its neighbors, with any attached neighbor reflecting the cell's own
        // vapor. Attached cells hold no vapor, so they add nothing to the
        // sum. ii. Freezing: on the boundary, proportion kappa of the vapor
        // becomes ice and the rest quasi-liquid.
        for i in 1..=n {
            for p in i * s + 1..=i * s + n {
                let attached = at(a, p - s)
                    + at(a, p - s + 1)
                    + at(a, p - 1)
                    + at(a, p + 1)
                    + at(a, p + s - 1)
                    + at(a, p + s);
                *at_mut(count, p) = attached;
                if at(a, p) != 0 {
                    *at_mut(e, p) = at(d, p);
                    continue;
                }
                let mut v = at(d, p) * (1 + attached) as f32;
                v += at(d, p - s);
                v += at(d, p - s + 1);
                v += at(d, p - 1);
                v += at(d, p + 1);
                v += at(d, p + s - 1);
                v += at(d, p + s);
                v /= 7.0;
                if attached > 0 {
                    *at_mut(b, p) += k.one_minus_kappa * v;
                    *at_mut(c, p) += k.kappa * v;
                    *at_mut(e, p) = 0.0;
                } else {
                    *at_mut(e, p) = v;
                }
            }
        }

        // iii. Attachment: the more attached neighbors a boundary cell has,
        // the less boundary mass it needs to join the crystal.
        wrap(e, n);
        for i in 1..=n {
            for j in 1..=n {
                let p = i * s + j;
                let attached = at(count, p);
                if at(a, p) != 0 || attached == 0 {
                    continue;
                }
                let mass = at(b, p);
                let attach = if attached <= 2 {
                    mass >= k.beta
                } else if attached == 3 {
                    if mass >= 1.0 {
                        true
                    } else {
                        let mut vapor = at(e, p);
                        vapor += at(e, p - s);
                        vapor += at(e, p - s + 1);
                        vapor += at(e, p - 1);
                        vapor += at(e, p + 1);
                        vapor += at(e, p + s - 1);
                        vapor += at(e, p + s);
                        vapor < k.theta && mass >= k.alpha
                    }
                } else {
                    true
                };
                if attach {
                    a[p] = 1;
                    c[p] += b[p];
                    b[p] = 0.0;
                    let r = i as i64 - 1 - center;
                    let q = j as i64 - 1 - center;
                    let distance = r.abs().max(q.abs()).max((q + r).abs()) as u32;
                    self.radius = self.radius.max(distance);
                }
            }
        }

        // iv. Melting: on the new boundary, proportions mu of the
        // quasi-liquid and gamma of the ice return to vapor.
        wrap(a, n);
        for i in 1..=n {
            for p in i * s + 1..=i * s + n {
                if at(a, p) != 0 {
                    continue;
                }
                let near = at(a, p - s)
                    | at(a, p - s + 1)
                    | at(a, p - 1)
                    | at(a, p + 1)
                    | at(a, p + s - 1)
                    | at(a, p + s);
                if near != 0 {
                    *at_mut(e, p) += k.mu * at(b, p) + k.gamma * at(c, p);
                    *at_mut(b, p) *= k.one_minus_mu;
                    *at_mut(c, p) *= k.one_minus_gamma;
                }
            }
        }

        std::mem::swap(&mut self.d, &mut self.e);
    }

    /// v. Noise: the vapor in each cell grows or shrinks by proportion
    /// sigma, each with probability one half.
    fn noise(&mut self, sigma: f64) {
        let lower = (1.0 - sigma) as f32;
        let upper = (1.0 + sigma) as f32;
        let n = self.size;
        let s = self.stride;
        let mut bits = 0u64;
        let mut remaining = 0;
        for i in 1..=n {
            for p in i * s + 1..=i * s + n {
                if remaining == 0 {
                    bits = self.random();
                    remaining = 64;
                }
                self.d[p] *= if bits & 1 == 0 { lower } else { upper };
                bits >>= 1;
                remaining -= 1;
            }
        }
    }

    /// The next 64 random bits, from SplitMix64.
    fn random(&mut self) -> u64 {
        self.rng = self.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Copy the fields, unpadded, into the frame, and return the crystal mass
    /// the brightest ice is drawn at: the 99th percentile over the crystal.
    pub fn frame(&mut self) -> f32 {
        let n = self.size;
        let s = self.stride;
        self.scratch.clear();
        for i in 0..n {
            for j in 0..n {
                let p = (i + 1) * s + j + 1;
                let q = i * n + j;
                self.frame_a[q] = self.a[p];
                self.frame_b[q] = self.b[p];
                self.frame_c[q] = self.c[p];
                self.frame_d[q] = self.d[p];
                if self.a[p] != 0 {
                    self.scratch.push(self.c[p]);
                }
            }
        }
        if self.scratch.len() < 2 {
            return 1.0;
        }
        let rank = (self.scratch.len() - 1) * 99 / 100;
        let (_, top, _) = self.scratch.select_nth_unstable_by(rank, f32::total_cmp);
        *top
    }
}

// The C interface. Each function takes the pointer `crystal_new` returned.

/// A new crystal: one cell of ice in the middle of a `size` by `size` grid of
/// vapor of density `rho`. `seed` seeds the noise.
#[unsafe(no_mangle)]
pub extern "C" fn crystal_new(size: u32, rho: f64, seed: u32) -> *mut Crystal {
    Box::into_raw(Box::new(Crystal::new(size as usize, rho, seed as u64)))
}

/// Free a crystal.
///
/// # Safety
/// `crystal` must come from `crystal_new` and not have been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crystal_free(crystal: *mut Crystal) {
    if !crystal.is_null() {
        drop(unsafe { Box::from_raw(crystal) });
    }
}

/// Advance a crystal by `num_steps` updates, with the parameters in the order
/// `snowflakes.step` takes them.
///
/// # Safety
/// `crystal` must come from `crystal_new` and not have been freed.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn crystal_grow(
    crystal: *mut Crystal,
    num_steps: u32,
    alpha: f64,
    beta: f64,
    gamma: f64,
    theta: f64,
    kappa: f64,
    mu: f64,
    sigma: f64,
) {
    let crystal = unsafe { &mut *crystal };
    crystal.grow(num_steps, alpha, beta, gamma, theta, kappa, mu, sigma);
}

/// The number of updates a crystal has had.
///
/// # Safety
/// `crystal` must come from `crystal_new` and not have been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crystal_steps(crystal: *const Crystal) -> u32 {
    unsafe { (*crystal).steps }
}

/// The distance, in cells, from the seed to the farthest attached cell.
///
/// # Safety
/// `crystal` must come from `crystal_new` and not have been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crystal_radius(crystal: *const Crystal) -> u32 {
    unsafe { (*crystal).radius }
}

/// Copy a crystal's fields into its frame, which `crystal_frame_a` and the
/// others point to, and return the crystal mass to draw at full brightness.
///
/// # Safety
/// `crystal` must come from `crystal_new` and not have been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crystal_frame(crystal: *mut Crystal) -> f32 {
    unsafe { (*crystal).frame() }
}

/// Whether each cell is attached, as 0 or 1, row by row, as of the last
/// `crystal_frame`.
///
/// # Safety
/// `crystal` must come from `crystal_new` and not have been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crystal_frame_a(crystal: *const Crystal) -> *const u8 {
    unsafe { (*crystal).frame_a.as_ptr() }
}

/// The boundary mass of each cell, row by row, as of the last
/// `crystal_frame`.
///
/// # Safety
/// `crystal` must come from `crystal_new` and not have been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crystal_frame_b(crystal: *const Crystal) -> *const f32 {
    unsafe { (*crystal).frame_b.as_ptr() }
}

/// The crystal mass of each cell, row by row, as of the last
/// `crystal_frame`.
///
/// # Safety
/// `crystal` must come from `crystal_new` and not have been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crystal_frame_c(crystal: *const Crystal) -> *const f32 {
    unsafe { (*crystal).frame_c.as_ptr() }
}

/// The vapor of each cell, row by row, as of the last `crystal_frame`.
///
/// # Safety
/// `crystal` must come from `crystal_new` and not have been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crystal_frame_d(crystal: *const Crystal) -> *const f32 {
    unsafe { (*crystal).frame_d.as_ptr() }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIG_11: (f64, f64, f64, f64, f64, f64) = (0.4, 1.6, 0.0005, 0.025, 0.005, 0.015);

    fn grown(size: usize, num_steps: u32, sigma: f64) -> Crystal {
        let (alpha, beta, gamma, theta, kappa, mu) = FIG_11;
        let mut crystal = Crystal::new(size, 0.635, 1);
        crystal.grow(num_steps, alpha, beta, gamma, theta, kappa, mu, sigma);
        crystal.frame();
        crystal
    }

    #[test]
    fn mass_is_nearly_conserved() {
        let size = 41;
        let total: f64 = 0.635f32 as f64 * (size * size - 1) as f64 + 1.0;
        let crystal = grown(size, 300, 0.0);
        let mass: f64 = (0..size * size)
            .map(|q| crystal.frame_b[q] as f64 + crystal.frame_c[q] as f64 + crystal.frame_d[q] as f64)
            .sum();
        assert!((mass - total).abs() < 1e-5 * total, "{mass} != {total}");
        assert!(crystal.frame_a.iter().map(|&x| x as u32).sum::<u32>() > 100);
    }

    #[test]
    fn six_fold_symmetry() {
        let n = 41;
        let crystal = grown(n, 300, 0.0);
        let half = (n / 2) as i64;
        for row in 0..n as i64 {
            for col in 0..n as i64 {
                let (q, r) = (col - half, row - half);
                // A turn of 60 degrees takes (q, r) to (-r, q + r).
                let turned_row = (q + r + half).rem_euclid(n as i64);
                let turned_col = (-r + half).rem_euclid(n as i64);
                let p = (row * n as i64 + col) as usize;
                let t = (turned_row * n as i64 + turned_col) as usize;
                assert_eq!(crystal.frame_a[p], crystal.frame_a[t]);
            }
        }
    }

    #[test]
    fn radius_is_tracked() {
        let n = 61;
        let crystal = grown(n, 300, 0.0);
        let half = (n / 2) as i64;
        let mut farthest = 0;
        for row in 0..n as i64 {
            for col in 0..n as i64 {
                if crystal.frame_a[(row * n as i64 + col) as usize] != 0 {
                    let (q, r) = (col - half, row - half);
                    farthest = farthest.max(q.abs().max(r.abs()).max((q + r).abs()));
                }
            }
        }
        assert_eq!(crystal.radius as i64, farthest);
    }

    #[test]
    fn noise_changes_the_vapor() {
        let quiet = grown(41, 100, 0.0);
        let noisy = grown(41, 100, 0.01);
        assert_ne!(quiet.frame_d, noisy.frame_d);
    }
}
