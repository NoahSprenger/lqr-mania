#![no_std]

extern crate alloc;
use alloc::{vec, vec::Vec};
use core::fmt;
use na::allocator::Allocator;
use na::base::DefaultAllocator;
use na::base::{Dim, DimMin, DimName};
use nalgebra as na;
use rand_core as rand;

#[derive(Debug)]
pub enum ControllerError {
    ZeroKi,
}

#[derive(Clone, Debug)]
/// Generic LQR optimal feedback controller
///
/// Templated via T,S,C which are numeric type, state and control dimensions
///
/// Riccatti equation solved via
/// https://www.tandfonline.com/doi/abs/10.1080/00207170410001714988
/// https://scicomp.stackexchange.com/questions/30757/discrete-time-algebraic-riccati-equation-dare-solver-in-c
pub struct LQRController<T, S, C>
where
    T: na::RealField,
    S: Dim + DimName + DimMin<S>, // State dimensions
    C: Dim + DimName + DimMin<C>, // Control dimensions
    DefaultAllocator:
        Allocator<T, S, S> + Allocator<T, C, C> + Allocator<T, S, C> + Allocator<T, C, S>,
{
    /// state cost
    pub q: Option<na::OMatrix<T, S, S>>,
    /// control cost
    pub r: Option<na::OMatrix<T, C, C>>,
    /// optimal gain
    pub k: Option<na::OMatrix<T, C, S>>,

    /// the controller also has an i-controller for removing steady state error in y dimension
    /// i-controller coefficient
    ki: T,
    /// accumulating integral error for the i-controller
    integral_error: T,
}

impl<T, S, C> LQRController<T, S, C>
where
    T: na::RealField + Copy,
    S: Dim + DimName + DimMin<S>,
    C: Dim + DimName + DimMin<C>,
    DefaultAllocator:
        Allocator<T, S, S> + Allocator<T, C, C> + Allocator<T, S, C> + Allocator<T, C, S>,
{
    // Instantiate controller in default mode without the i-controller
    pub fn new() -> Result<LQRController<T, S, C>, &'static str> {
        Ok(LQRController {
            q: None,
            r: None,
            k: None,
            ki: T::zero(),
            integral_error: T::zero(),
        })
    }
    // Initialise the i-controller with its ki coefficients. This enables the controller
    pub fn setup_i_controller(&mut self, ki: T) -> Result<(), ControllerError> {
        if ki != T::zero() {
            self.ki = ki;
        } else {
            Err(ControllerError::ZeroKi)?;
        }
        Ok(())
    }
    /// Computes and returns the optimal gain matrix K for the LQR controller
    ///
    /// # Arguments
    ///
    /// * `a` - state matrix of shape SxS
    /// * `b` - control matrix of shape SxC
    /// * `q` - state cost matrix of shape SxS
    /// * `r` - control cost amtrix of shape CxC
    /// * `epsilon` - small value to avoid division by 0; 1e-6 works nicely
    ///
    /// # Returns
    /// optimal feedback gain matrix `k` of shape CxS
    pub fn compute_gain(
        &mut self,
        a: &na::OMatrix<T, S, S>,
        b: &na::OMatrix<T, S, C>,
        q: &na::OMatrix<T, S, S>,
        r: &na::OMatrix<T, C, C>,
        epsilon: T,
        rng: &mut impl rand::RngCore,
    ) -> Result<na::OMatrix<T, C, S>, &'static str> {
        let mut a_k = a.clone(); // copy to here
        let mut g_k = b.clone()
            * r.clone().try_inverse().expect("Couldn't compute inverse")
            * b.clone().transpose();

        let mut h_k_1 = q.clone();
        let mut h_k = na::OMatrix::<T, S, S>::from_fn(|_, _| {
            T::from_u32(rng.next_u32()).expect("Couldn't cast to type T")
        });

        while {
            let error = (h_k_1.clone() - h_k).norm() / h_k_1.norm();
            h_k = h_k_1.clone();
            error >= epsilon
        } {
            let temp = (na::OMatrix::<T, S, S>::identity() + &g_k * &h_k)
                .try_inverse()
                .expect("Couldn't compute inverse");
            let a_k_1 = &a_k * &temp * &a_k;
            let g_k_1 = &g_k + &a_k * &temp * &g_k * &a_k.transpose();
            h_k_1 = &h_k + &a_k.transpose() * &h_k * &temp * &a_k;
            a_k = a_k_1;
            g_k = g_k_1;
        }

        // calculate final gain matrix
        self.k =
            Some(r.clone().try_inverse().expect("Couldn't compute inverse") * b.transpose() * h_k);
        return Ok(self.k.clone().unwrap());
    }

    /// Returns the optimal feedback control based on the desired and current state vectors.
    /// This should  be called only after compute_gain() has already been called.
    ///
    /// # Arguments
    ///
    /// * `current_state` - vector of length S
    /// * `desired_state` - vector of length S which we want to get to
    ///
    /// # Returns
    /// The feedback control gains. These are insufficient to control anything and have to be
    /// combined with the feedforward controls. Check examples
    pub fn compute_optimal_controls(
        &mut self,
        current_state: &na::OVector<T, S>,
        desired_state: &na::OVector<T, S>,
    ) -> Result<na::OVector<T, C>, &'static str>
    where
        T: na::RealField,
        DefaultAllocator: Allocator<T, S> + Allocator<T, C> + Allocator<T, C, S>,
    {
        let error = desired_state - current_state;
        let y_error = error[1];
        if y_error.abs() < T::from_f64(1e-2).unwrap() {
            self.integral_error = T::zero();
        } else {
            self.integral_error += y_error * self.ki.clone();
        }

        let mut controls = &self.k.clone().unwrap() * error;
        controls[0] -= self.integral_error.clone();

        Ok(controls)
    }
}

impl<T, S, C> fmt::Display for LQRController<T, S, C>
where
    T: na::RealField,
    S: Dim + DimName + DimMin<S>,
    C: Dim + DimName + DimMin<C>,
    DefaultAllocator:
        Allocator<T, S, S> + Allocator<T, C, C> + Allocator<T, S, C> + Allocator<T, C, S>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LQR controller
            Q: {:?}
            R: {:?}
            ki: {:}",
            self.q, self.r, self.ki,
        )
    }
}

pub struct ClosestIndexError;

pub fn find_closest_indices<T, S, N>(
    current_state: &na::OVector<T, S>,
    trajectory: &na::OMatrix<T, S, N>,
) -> Result<((usize, T), (usize, T)), ClosestIndexError>
where
    T: na::RealField + PartialOrd + Copy,
    S: Dim + DimName + DimMin<S>, // State dimensions
    N: Dim + DimName,             // Trajectory length
    DefaultAllocator: Allocator<T, S>
        + Allocator<T, S, N>
        + Allocator<T, na::Const<1_usize>, N>
        + nalgebra::allocator::Allocator<T, N>,
{
    if trajectory.ncols() < 2 {
        return Err(ClosestIndexError);
    }

    let mut state = Vec::with_capacity(current_state.len());
    for i in current_state.iter() {
        state.push(i.clone());
    }
    let state = vec![state; trajectory.ncols()].concat();

    let current_state_broadcasted = na::OMatrix::<T, S, N>::from_vec(state);
    let errors = trajectory - current_state_broadcasted;
    let errors = errors.map(|x| x.powi(2));
    let errors = errors.row_sum_tr();
    let (idx, error) = errors.argmin();

    // figure out lower and upper indices and their errors
    let (lower, upper) = if idx == 0 {
        ((0, error.sqrt()), (0, error.sqrt()))
    } else if idx == errors.len() - 1 {
        (
            (errors.len() - 1, error.sqrt()),
            (errors.len() - 1, error.sqrt()),
        )
    } else {
        if errors[idx + 1] > errors[idx - 1] {
            ((idx, error.sqrt()), (idx + 1, errors[idx + 1].sqrt()))
        } else {
            ((idx - 1, errors[idx - 1].sqrt()), (idx, error.sqrt()))
        }
    };

    // Sanity check
    if lower.0 != 0 && upper.0 != 0 && lower.0 > upper.0 {
        // Incorrectly detected trajectory indices
        return Err(ClosestIndexError);
    }

    Ok((lower, upper))
}

pub fn compute_target<T, S, N>(
    trajectory: &na::OMatrix<T, S, N>,
    lower: (usize, T),
    upper: (usize, T),
) -> na::OVector<T, S>
where
    T: na::RealField + Copy,
    S: Dim + DimName + DimMin<S>, // State dimensions
    N: Dim + DimName,             // Trajectory length
    DefaultAllocator: Allocator<T, S> + Allocator<T, S, N>,
{
    trajectory
        .column(lower.0)
        .component_mul(&na::OVector::from_element(upper.1 / (lower.1 + upper.1)))
        + trajectory
            .column(upper.0)
            .component_mul(&na::OVector::from_element(lower.1 / (lower.1 + upper.1)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_target_1() {
        let trajectory = na::Matrix5x2::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.1);
        let expected = na::Matrix5x1::new(0.5, 0.0, 0.0, 0.5, 0.05);
        let target = compute_target(&trajectory, (0, 0.5f64), (1, 0.5f64));
        assert_eq!(expected, target);
    }
    #[test]
    fn compute_target_2() {
        let trajectory = na::Matrix5x2::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.1);
        let expected = na::Matrix5x1::new(0.25, 0.0, 0.0, 0.25, 0.025);
        let target = compute_target(&trajectory, (0, 0.25), (1, 0.75));
        assert_eq!(target, expected);
    }

    #[test]
    fn compute_target_3() {
        let trajectory = na::Matrix5x2::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.1);
        let expected = na::Matrix5x1::new(0.75, 0.0, 0.0, 0.75, 0.07500000000000001);
        let target = compute_target(&trajectory, (0, 0.75), (1, 0.25));
        assert_eq!(target, expected);
    }

    #[test]
    fn compute_target_4() {
        let trajectory = na::Matrix5x2::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.1);
        let expected = na::Matrix5x1::new(0.0, 0.0, 0.0, 0.0, 0.0);
        let target = compute_target(&trajectory, (0, 0.0), (1, f64::MAX));
        assert_eq!(target, expected);
    }

    #[test]
    fn index_search() {
        let current_state = na::Matrix5x1::new(0.5, 0.0, 0.0, 0.0, 0.0);
        let trajectory = na::Matrix5x2::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

        match find_closest_indices(&current_state, &trajectory) {
            Ok((lower, upper)) => {
                assert_eq!(lower.0, 0);
                assert_eq!(lower.1, 0.5f64);
                assert_eq!(upper.0, 0);
                assert_eq!(upper.1, 0.5f64);
            }
            Err(_) => {
                assert!(false, "failed to find closest index")
            }
        }
    }

    #[test]
    fn index_search_before_path() {
        let current_state = na::Matrix5x1::new(-1.0, 0.0, 0.0, 0.0, 0.0);
        let trajectory = na::Matrix5x2::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

        match find_closest_indices(&current_state, &trajectory) {
            Ok((lower, upper)) => {
                assert_eq!(lower.0, 0);
                assert_eq!(lower.1, 1.0);
                assert_eq!(upper.0, 0);
                assert_eq!(upper.1, 1.0);
            }
            Err(_) => {
                assert!(false, "failed to find closest index")
            }
        }
    }

    #[test]
    fn index_search_after_path() {
        let current_state = na::Matrix5x1::new(2.0, 0.0, 0.0, 0.0, 0.0);
        let trajectory = na::Matrix5x2::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

        match find_closest_indices(&current_state, &trajectory) {
            Ok((lower, upper)) => {
                assert_eq!(lower.0, 1);
                assert_eq!(lower.1, 1.0);
                assert_eq!(upper.0, 1);
                assert_eq!(upper.1, 1.0);
            }
            Err(_) => {
                assert!(false, "failed to find closest index")
            }
        }
    }

    #[test]
    fn lqr_optimization() -> Result<(), ControllerError> {
        // Define state
        let _x: f64 = 2.0;
        let _y: f64 = 2.0;
        let theta: f64 = 0.34;
        let v: f64 = 3.0;

        // Define controls
        let delta: f64 = 0.0;
        let _acc: f64 = 0.0;

        // wheelbase
        let l = 2.0;

        // compute matrices for the LQR controller
        let a = na::Matrix4::<f64>::new(
            0.0,
            0.0,
            -v * theta.sin(),
            theta.cos(),
            0.0,
            0.0,
            v * theta.cos(),
            theta.sin(),
            0.0,
            0.0,
            0.0,
            delta.tan() / l,
            0.0,
            0.0,
            0.0,
            0.0,
        );
        let b = na::Matrix4x2::<f64>::new(
            0.0,
            0.0,
            0.0,
            0.0,
            v / (l * delta.cos().powf(2.0)),
            0.0,
            0.0,
            1.0,
        );
        let q = na::Matrix4::identity();
        let r = na::Matrix2::identity();

        let mut controller = LQRController::new().unwrap();
        

        controller.compute_gain(&a, &b, &q, &r, 1e-12, &mut rand_core::OsRng).unwrap();

        // hand calculated results
        let real_gain = na::Matrix2x4::<f64>::new(0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 0.0, 2.0);

        let mut gain_result = controller.k.unwrap().clone();

        // 0 out the gain
        gain_result.apply(|v| {
            if v.abs() < 1e-6 {
                *v = 0.0f64;
            }
        });

        assert_eq!(real_gain, gain_result);

        Ok(())
    }
}
