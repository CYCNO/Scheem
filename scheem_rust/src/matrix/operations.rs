use super::mat::Mat;
use crate::micrograd::ValueRef;
use std::ops::{Add, Mul, Sub};

impl Add<Mat> for Mat {
    type Output = Mat;

    fn add(self, other: Mat) -> Mat {
        if other.size[0] == 1 && self.size[1] == other.size[1] {
            let mut mat_data = vec![vec![ValueRef::new(0.0); self.size[1]]; self.size[0]];
            for i in 0..self.size[0] {
                for j in 0..self.size[1] {
                    mat_data[i][j] = self.data[i][j].clone() + other.data[0][j].clone();
                }
            }
            return Mat {
                data: mat_data,
                size: self.size,
            };
        }

        assert_eq!(self.size, other.size, "Matrix dimensions must match.");
        // create a vec and fill it with 0.0
        let mut mat_data = vec![vec![ValueRef::new(0.0); self.size[1]]; self.size[0]];

        for i in 0..self.size[0] {
            for j in 0..self.size[1] {
                mat_data[i][j] = self.data[i][j].clone() + other.data[i][j].clone();
            }
        }

        // output new Value
        Mat {
            data: mat_data,
            size: vec![self.size[0], self.size[1]],
        }
    }

}

impl Add<f64> for Mat {
    type Output = Mat;

    fn add(self, other: f64) -> Mat {
        let other_val = ValueRef::new(other);
        let mut mat_data = vec![vec![ValueRef::new(0.0); self.size[1]]; self.size[0]];

        for i in 0..self.size[0] {
            for j in 0..self.size[1] {
                mat_data[i][j] = self.data[i][j].clone() + other_val.clone();
            }
        }

        // output new Value
        Mat {
            data: mat_data,
            size: vec![self.size[0], self.size[1]],
        }
    }
}

impl Mul<Mat> for Mat {
    type Output = Mat;

    fn mul(self, other: Mat) -> Mat {
        assert_eq!(
            self.size[1], other.size[0],
            "Columns of first matrix must equal rows of second."
        );
        // create a vec and fill it with 0.0
        let mut mat_data = vec![vec![ValueRef::new(0.0); other.size[1]]; self.size[0]];

        for i in 0..self.size[0] {
            for j in 0..other.size[1] {
                let mut val = self.data[i][0].clone() * other.data[0][j].clone();
                for k in 1..self.size[1] {
                    val = val + self.data[i][k].clone() * other.data[k][j].clone();
                }
                mat_data[i][j] = val;
            }
        }

        // output new Value
        Mat {
            data: mat_data,
            size: vec![self.size[0], other.size[1]],
        }
    }
}

impl Mul<f64> for Mat {
    type Output = Mat;

    fn mul(self, other: f64) -> Mat {
        let other_val = ValueRef::new(other);
        let mut mat_data = vec![vec![ValueRef::new(0.0); self.size[1]]; self.size[0]];

        for i in 0..self.size[0] {
            for j in 0..self.size[1] {
                mat_data[i][j] = self.data[i][j].clone() * other_val.clone();
            }
        }

        // output new Value
        Mat {
            data: mat_data,
            size: vec![self.size[0], self.size[1]],
        }
    }
}

impl Sub<Mat> for Mat {
    type Output = Mat;

    fn sub(self, other: Mat) -> Mat {
        assert_eq!(self.size, other.size, "Matrix dimensions must match.");
        // create a vec and fill it with 0.0
        let mut mat_data = vec![vec![ValueRef::new(0.0); self.size[1]]; self.size[0]];

        for i in 0..self.size[0] {
            for j in 0..self.size[1] {
                mat_data[i][j] = self.data[i][j].clone() - other.data[i][j].clone();
            }
        }

        // output new Value
        Mat {
            data: mat_data,
            size: vec![self.size[0], self.size[1]],
        }
    }
}

impl Sub<f64> for Mat {
    type Output = Mat;

    fn sub(self, other: f64) -> Mat {
        let other_val = ValueRef::new(other);
        let mut mat_data = vec![vec![ValueRef::new(0.0); self.size[1]]; self.size[0]];

        for i in 0..self.size[0] {
            for j in 0..self.size[1] {
                mat_data[i][j] = self.data[i][j].clone() - other_val.clone();
            }
        }

        // output new Value
        Mat {
            data: mat_data,
            size: vec![self.size[0], self.size[1]],
        }
    }
}

impl Mat {
    pub fn sigmoid(&self) -> Mat {
        // create a vec and fill it with 0.0
        let mut mat_data = vec![vec![ValueRef::new(0.0); self.size[1]]; self.size[0]];

        for i in 0..self.size[0] {
            for j in 0..self.size[1] {
                mat_data[i][j] = self.data[i][j].clone().sigmoid();
            }
        }

        // output new Value
        Mat {
            data: mat_data,
            size: vec![self.size[0], self.size[1]],
        }
    }

    pub fn relu(&self) -> Mat {
        // create a vec and fill it with 0.0
        let mut mat_data = vec![vec![ValueRef::new(0.0); self.size[1]]; self.size[0]];

        for i in 0..self.size[0] {
            for j in 0..self.size[1] {
                mat_data[i][j] = self.data[i][j].clone().relu();
            }
        }

        // output new Value
        Mat {
            data: mat_data,
            size: vec![self.size[0], self.size[1]],
        }
    }
}
