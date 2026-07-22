use super::mat::Mat;
use rand::random_range;

impl Mat {
    pub fn random(&mut self) {
        for row in &self.data {
            for val in row {
                val.0.borrow_mut().data = random_range(-1.0..1.0);
            }
        }
    }

    pub fn random_range(&mut self, lower: f64, upper: f64) {
        for row in &self.data {
            for val in row {
                val.0.borrow_mut().data = random_range(lower..upper);
            }
        }
    }
}
