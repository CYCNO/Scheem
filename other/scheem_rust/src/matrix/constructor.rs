use super::mat::Mat;
use crate::micrograd::ValueRef;

impl Mat {
    pub fn new(data: Vec<Vec<f64>>) -> Self {
        let mut mat_data: Vec<Vec<ValueRef>> = vec![];
        for rows in data {
            let mut row_data: Vec<ValueRef> = vec![];
            for i in rows {
                row_data.push(ValueRef::new(i));
            }
            mat_data.push(row_data);
        }
        let row = mat_data.len();
        let col = mat_data[0].len();
        Mat {
            data: mat_data,
            size: vec![row, col],
        }
    }

    pub fn new_empty(rows: usize, cols: usize) -> Self {
        let mut data = Vec::with_capacity(rows);

        for _ in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for _ in 0..cols {
                row.push(ValueRef::new(0.0));
            }
            data.push(row);
        }
        Mat {
            data: data,
            size: vec![rows, cols],
        }
    }

    pub fn size(&self) -> &[usize] {
        &self.size
    }
}
