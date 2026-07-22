use crate::matrix::Mat;
use crate::micrograd::ValueRef;
use pyo3::prelude::*;

#[pyclass(name = "ValueRef", unsendable, from_py_object)]
#[derive(Clone)]
pub struct PyValueRef(pub ValueRef);

#[pyclass(name = "Mat", unsendable, from_py_object)]
#[derive(Clone)]
pub struct PyMat(pub Mat);

#[pymethods]
impl PyValueRef {
    #[new]
    pub fn new(data: f64) -> Self {
        PyValueRef(ValueRef::new(data))
    }

    #[getter]
    pub fn data(&self) -> f64 {
        self.0.data()
    }

    #[setter]
    pub fn set_data(&self, val: f64) {
        self.0.0.borrow_mut().data = val;
    }

    #[getter]
    pub fn grad(&self) -> f64 {
        self.0.grad()
    }

    #[setter]
    pub fn set_grad(&self, val: f64) {
        self.0.0.borrow_mut().grad = val;
    }

    pub fn backward(&self) {
        self.0.backward();
    }

    pub fn update_params(&self, lr: f64) {
        let mut inner = self.0.0.borrow_mut();
        inner.data -= lr * inner.grad;
    }


    pub fn pow(&self, exponent: f64) -> PyValueRef {
        PyValueRef(self.0.clone().pow(exponent))
    }

    pub fn sigmoid(&self) -> PyValueRef {
        PyValueRef(self.0.clone().sigmoid())
    }

    pub fn relu(&self) -> PyValueRef {
        PyValueRef(self.0.clone().relu())
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyValueRef> {
        if let Ok(other_val) = other.extract::<PyValueRef>() {
            Ok(PyValueRef(self.0.clone() + other_val.0))
        } else if let Ok(val) = other.extract::<f64>() {
            Ok(PyValueRef(self.0.clone() + ValueRef::new(val)))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported type for addition",
            ))
        }
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyValueRef> {
        self.__add__(other)
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyValueRef> {
        if let Ok(other_val) = other.extract::<PyValueRef>() {
            Ok(PyValueRef(self.0.clone() * other_val.0))
        } else if let Ok(val) = other.extract::<f64>() {
            Ok(PyValueRef(self.0.clone() * ValueRef::new(val)))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported type for multiplication",
            ))
        }
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyValueRef> {
        self.__mul__(other)
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyValueRef> {
        let neg_one = ValueRef::new(-1.0);
        if let Ok(other_val) = other.extract::<PyValueRef>() {
            Ok(PyValueRef(self.0.clone() + (other_val.0 * neg_one)))
        } else if let Ok(val) = other.extract::<f64>() {
            Ok(PyValueRef(self.0.clone() + ValueRef::new(-val)))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported type for subtraction",
            ))
        }
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyValueRef> {
        let neg_self = self.0.clone() * ValueRef::new(-1.0);
        if let Ok(val) = other.extract::<f64>() {
            Ok(PyValueRef(ValueRef::new(val) + neg_self))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported type for subtraction",
            ))
        }
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyValueRef> {
        if let Ok(other_val) = other.extract::<PyValueRef>() {
            Ok(PyValueRef(self.0.clone() * other_val.0.pow(-1.0)))
        } else if let Ok(val) = other.extract::<f64>() {
            Ok(PyValueRef(self.0.clone() * ValueRef::new(val).pow(-1.0)))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported type for division",
            ))
        }
    }

    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyValueRef> {
        let self_inv = self.0.clone().pow(-1.0);
        if let Ok(val) = other.extract::<f64>() {
            Ok(PyValueRef(ValueRef::new(val) * self_inv))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported type for division",
            ))
        }
    }

    fn __repr__(&self) -> String {
        format!("ValueRef(data={}, grad={})", self.data(), self.grad())
    }
}

#[pymethods]
impl PyMat {
    #[new]
    pub fn new(data: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Try extracting PyMat
        if let Ok(other_mat) = data.extract::<PyMat>() {
            return Ok(other_mat);
        }

        // Try extracting Vec<Vec<f64>>
        if let Ok(data2d) = data.extract::<Vec<Vec<f64>>>() {
            return Ok(PyMat(Mat::new(data2d)));
        }

        // Try extracting Vec<Vec<PyValueRef>>
        if let Ok(data2d) = data.extract::<Vec<Vec<PyValueRef>>>() {
            let mut mat_data: Vec<Vec<ValueRef>> = vec![];
            for rows in data2d {
                let mut row_data: Vec<ValueRef> = vec![];
                for i in rows {
                    row_data.push(i.0.clone());
                }
                mat_data.push(row_data);
            }
            let row = mat_data.len();
            let col = mat_data[0].len();
            return Ok(PyMat(Mat {
                data: mat_data,
                size: vec![row, col],
            }));
        }

        // Try extracting Vec<PyValueRef> -> 1 x col
        if let Ok(data1d) = data.extract::<Vec<PyValueRef>>() {
            let mut row_data: Vec<ValueRef> = vec![];
            for i in data1d {
                row_data.push(i.0.clone());
            }
            let col = row_data.len();
            return Ok(PyMat(Mat {
                data: vec![row_data],
                size: vec![1, col],
            }));
        }

        // Try extracting Vec<f64> -> 1 x col
        if let Ok(data1d) = data.extract::<Vec<f64>>() {
            let mut row_data: Vec<ValueRef> = vec![];
            for i in data1d {
                row_data.push(ValueRef::new(i));
            }
            let col = row_data.len();
            return Ok(PyMat(Mat {
                data: vec![row_data],
                size: vec![1, col],
            }));
        }

        Err(pyo3::exceptions::PyTypeError::new_err(
            "Unsupported data type for Mat constructor",
        ))
    }

    #[staticmethod]
    pub fn empty(rows: usize, cols: usize) -> Self {
        PyMat(Mat::new_empty(rows, cols))
    }

    #[getter]
    pub fn size(&self) -> Vec<usize> {
        self.0.size.clone()
    }

    #[getter]
    pub fn rows(&self) -> usize {
        self.0.size[0]
    }

    #[getter]
    pub fn cols(&self) -> usize {
        self.0.size[1]
    }

    #[getter]
    pub fn data(&self) -> Vec<Vec<PyValueRef>> {
        let mut py_data: Vec<Vec<PyValueRef>> = vec![];
        for row in &self.0.data {
            let mut py_row: Vec<PyValueRef> = vec![];
            for val in row {
                py_row.push(PyValueRef(val.clone()));
            }
            py_data.push(py_row);
        }
        py_data
    }

    pub fn sigmoid(&self) -> PyMat {
        PyMat(self.0.sigmoid())
    }

    pub fn relu(&self) -> PyMat {
        PyMat(self.0.relu())
    }

    pub fn pow(&self, exponent: f64) -> PyMat {
        let mut mat_data = vec![vec![ValueRef::new(0.0); self.0.size[1]]; self.0.size[0]];
        for i in 0..self.0.size[0] {
            for j in 0..self.0.size[1] {
                mat_data[i][j] = self.0.data[i][j].clone().pow(exponent);
            }
        }
        PyMat(Mat {
            data: mat_data,
            size: self.0.size.clone(),
        })
    }

    pub fn emul(&self, other: &PyMat) -> PyResult<PyMat> {
        if self.0.size != other.0.size {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Matrix dimensions must match for element-wise multiplication.",
            ));
        }
        let mut mat_data = vec![vec![ValueRef::new(0.0); self.0.size[1]]; self.0.size[0]];
        for i in 0..self.0.size[0] {
            for j in 0..self.0.size[1] {
                mat_data[i][j] = self.0.data[i][j].clone() * other.0.data[i][j].clone();
            }
        }
        Ok(PyMat(Mat {
            data: mat_data,
            size: self.0.size.clone(),
        }))
    }


    pub fn random(&mut self) {
        self.0.random();
    }

    pub fn random_range(&mut self, lower: f64, upper: f64) {
        self.0.random_range(lower, upper);
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyMat> {
        if let Ok(other_mat) = other.extract::<PyMat>() {
            Ok(PyMat(self.0.clone() + other_mat.0))
        } else if let Ok(val) = other.extract::<f64>() {
            Ok(PyMat(self.0.clone() + val))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported type for addition",
            ))
        }
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyMat> {
        self.__add__(other)
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyMat> {
        if let Ok(other_mat) = other.extract::<PyMat>() {
            Ok(PyMat(self.0.clone() * other_mat.0))
        } else if let Ok(val) = other.extract::<f64>() {
            Ok(PyMat(self.0.clone() * val))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported type for multiplication",
            ))
        }
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyMat> {
        self.__mul__(other)
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyMat> {
        if let Ok(other_mat) = other.extract::<PyMat>() {
            Ok(PyMat(self.0.clone() - other_mat.0))
        } else if let Ok(val) = other.extract::<f64>() {
            Ok(PyMat(self.0.clone() - val))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported type for subtraction",
            ))
        }
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyMat> {
        if let Ok(val) = other.extract::<f64>() {
            let neg_self = self.0.clone() * -1.0;
            Ok(PyMat(neg_self + val))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported type for right-subtraction",
            ))
        }
    }

    fn __neg__(&self) -> PyMat {
        PyMat(self.0.clone() * -1.0)
    }

    fn __getitem__(&self, index: usize) -> PyResult<Vec<PyValueRef>> {
        if index < self.0.size[0] {
            let mut py_row: Vec<PyValueRef> = vec![];
            for val in &self.0.data[index] {
                py_row.push(PyValueRef(val.clone()));
            }
            Ok(py_row)
        } else {
            Err(pyo3::exceptions::PyIndexError::new_err(
                "Index out of range",
            ))
        }
    }

    fn __repr__(&self) -> String {
        format!("{}", self.0)
    }

    pub fn sum(&self) -> PyValueRef {
        if self.0.data.is_empty() || self.0.data[0].is_empty() {
            return PyValueRef(ValueRef::new(0.0));
        }
        let mut total = self.0.data[0][0].clone();
        let mut first = true;
        for row in &self.0.data {
            for val in row {
                if first {
                    first = false;
                    continue;
                }
                total = total + val.clone();
            }
        }
        PyValueRef(total)
    }
}

#[pyfunction]
pub fn free_graph(root: PyValueRef) {
    crate::micrograd::free_graph(root.0);
}
