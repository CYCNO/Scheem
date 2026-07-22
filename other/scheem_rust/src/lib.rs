use pyo3::prelude::*;

pub mod matrix;
pub mod micrograd;
pub mod py_bindings;

use py_bindings::{PyMat, PyValueRef, free_graph};

/// A Python module implemented in Rust.
#[pymodule]
fn scheem_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMat>()?;
    m.add_class::<PyValueRef>()?;
    m.add_function(wrap_pyfunction!(free_graph, m)?)?;
    Ok(())
}
