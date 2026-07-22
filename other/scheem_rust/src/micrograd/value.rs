use std::rc::Rc;
use std::cell::RefCell;

#[derive(Debug, Clone)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Pow(f64),
    Sigmoid,
    Relu,
    None,
}

// Added Debug here so we can print it in main
#[derive(Debug, Clone)]
pub struct ValueRef(pub(crate) Rc<RefCell<Value>>);

#[derive(Debug, Clone)]
pub struct Value {
    pub(crate) data: f64,
    pub(crate) grad: f64,
    pub(crate) prev: Vec<ValueRef>,
    pub(crate) op: Op,
    pub(crate) visited: bool,
}
