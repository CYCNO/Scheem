use super::value::{ValueRef, Op};
use std::ops::{Add, Mul, Sub};

impl Add for ValueRef {
    type Output = ValueRef;

    // Fixed: `other` should be `ValueRef`, not `Value`
    fn add(self, other: ValueRef) -> ValueRef {
        // compute the output data
        let data = self.0.borrow().data + other.0.borrow().data;

        // add the children
        let prev = vec![self.clone(), other.clone()];

        // output new Value
        ValueRef::new_op(
            data,
            prev,
            Op::Add
        )
    }
}

impl Mul for ValueRef {
    type Output = ValueRef;

    fn mul(self, other: ValueRef) -> ValueRef {
        // compute the output data
        let data = self.0.borrow().data * other.0.borrow().data;

        // add the children
        let prev = vec![self.clone(), other.clone()];

        // output new Value
        ValueRef::new_op(
            data,
            prev,
            Op::Mul
        )
    }
}

impl Sub for ValueRef {
    type Output = ValueRef;

    fn sub(self, other: ValueRef) -> ValueRef {
        let data = self.0.borrow().data - other.0.borrow().data;
        let prev = vec![self.clone(), other.clone()];
        ValueRef::new_op(
            data,
            prev,
            Op::Sub
        )
    }
}


impl ValueRef {
    // operations
    pub fn pow(self, exponent: f64) -> ValueRef {
        let data = self.0.borrow().data.powf(exponent);

        let prev = vec![self.clone()];

        ValueRef::new_op(
            data,
            prev,
            Op::Pow(exponent)
        )
    }

    pub fn sigmoid(self) -> ValueRef {
        let data = 1.0 / (1.0 + (-self.0.borrow().data).exp()); // 1 / (1 + e^-x)

        let prev = vec![self.clone()];

        ValueRef::new_op(
            data,
            prev,
            Op::Sigmoid
        )
    }

    pub fn relu(self) -> ValueRef {
        let data = if self.0.borrow().data > 0.0 {
            self.0.borrow().data
        } else {
            0.0
        };

        let prev = vec![self.clone()];

        ValueRef::new_op(
            data,
            prev,
            Op::Relu
        )
    }
}
