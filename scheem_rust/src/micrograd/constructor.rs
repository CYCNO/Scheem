use std::rc::Rc;
use std::cell::RefCell;
use super::value::{ValueRef, Value, Op};

impl ValueRef {
    // constructor by user
    pub fn new(data: f64) -> Self {
        ValueRef(Rc::new(RefCell::new(Value {
            data: data,
            grad: 0.0,
            prev: Vec::new(),
            op: Op::None,
            visited: false,
        })))
    }

    // constructor while operation
    pub fn new_op(data: f64, prev: Vec<ValueRef>, op: Op) -> Self {
        ValueRef(Rc::new(RefCell::new(Value {
            data,
            grad: 0.0,
            prev,
            op,
            visited: false,
        })))
    }

    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }

    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }

    pub fn set_data(&self, val: f64) {
        self.0.borrow_mut().data = val;
    }

    pub fn set_grad(&self, val: f64) {
        self.0.borrow_mut().grad = val;
    }
}
