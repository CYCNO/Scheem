use crate::micrograd::ValueRef;

#[derive(Debug, Clone)]
pub struct Mat {
    pub data: Vec<Vec<ValueRef>>,
    pub size: Vec<usize>, // size[0]: rows, size[1]: columns
}
