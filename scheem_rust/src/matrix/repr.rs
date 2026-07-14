use super::mat::Mat;
use std::fmt;

impl fmt::Display for Mat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Mat(")?;

        for row in &self.data {
            write!(f, "    [")?;

            for (i, val) in row.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:.4}", val.data())?;
            }

            writeln!(f, "]")?;
        }

        write!(f, ") size={:?}", &self.size)
    }
}
