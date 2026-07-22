use super::value::{Op, ValueRef};
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

impl ValueRef {
    // backpropagation
    pub fn backward(&self) {
        let mut topo = vec![];
        build_topo(self.clone(), &mut topo);

        self.0.borrow_mut().grad = 1.0;

        topo.reverse();
        for v in &topo {
            let node = v.0.borrow();

            let op = node.op.clone();
            let self_grad = node.grad.clone();
            let self_data = node.data.clone();

            match op {
                Op::Add => {
                    node.prev[0].0.borrow_mut().grad += 1.0 * self_grad;
                    node.prev[1].0.borrow_mut().grad += 1.0 * self_grad;
                }
                Op::Sub => {
                    node.prev[0].0.borrow_mut().grad += 1.0 * self_grad;
                    node.prev[1].0.borrow_mut().grad -= 1.0 * self_grad;
                }
                Op::Mul => {
                    let a = node.prev[0].0.borrow().data;
                    let b = node.prev[1].0.borrow().data;

                    node.prev[0].0.borrow_mut().grad += b * self_grad;
                    node.prev[1].0.borrow_mut().grad += a * self_grad;
                }
                Op::Pow(component) => {
                    let n = node.prev[0].0.borrow().data;

                    node.prev[0].0.borrow_mut().grad +=
                        component * n.powf(component - 1.0) * self_grad;
                }
                Op::Sigmoid => {
                    node.prev[0].0.borrow_mut().grad += self_data * (1.0 - self_data) * self_grad;
                }
                Op::Relu => {
                    let x = node.prev[0].0.borrow().data;
                    node.prev[0].0.borrow_mut().grad += if x > 0.0 { 1.0 * self_grad } else { 0.0 }
                }
                Op::None => {}
            }
        }

        // Reset the visited flag for all traversed nodes so subsequent backward calls work correctly
        for v in topo {
            v.0.borrow_mut().visited = false;
        }
    }
}

impl PartialEq for ValueRef {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for ValueRef {}

impl Hash for ValueRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // We hash the memory address of the pointer, not the inner value
        std::ptr::hash(Rc::as_ptr(&self.0), state);
    }
}

fn build_topo(root: ValueRef, topo: &mut Vec<ValueRef>) {
    // The stack holds (Node, children_processed_flag)
    let mut stack = vec![(root, false)];

    while let Some((node, children_processed)) = stack.pop() {
        if children_processed {
            // We've already processed this node's children.
            // It's safe to add it to the final topo list.
            topo.push(node);
        } else {
            // First time seeing this node. Let's mark it as visited.
            let mut is_new = false;
            {
                let mut inner = node.0.borrow_mut();
                if !inner.visited {
                    inner.visited = true;
                    is_new = true;
                }
            }
            if is_new {
                // 1. Put the current node BACK on the stack, but mark it
                //    as true so it gets added to `topo` next time we see it.
                stack.push((node.clone(), true));

                // 2. Add all of its children to the stack to be processed
                for child in &node.0.borrow().prev {
                    // only add children we haven't visited yet
                    if !child.0.borrow().visited {
                        stack.push((child.clone(), false));
                    }
                }
            }
        }
    }
}

pub fn free_graph(root: ValueRef) {
    let mut stack = vec![root];
    let mut visited = HashSet::new();

    while let Some(node) = stack.pop() {
        // Only process nodes we haven't severed yet
        if visited.insert(node.clone()) {
            // .drain(..) removes all the children from this node's `prev` vector.
            // This physically breaks the connections between the parent and children!
            let children: Vec<ValueRef> = node.0.borrow_mut().prev.drain(..).collect();

            // Push the children onto the stack so we can sever their connections too
            for child in children {
                stack.push(child);
            }
        }
    }
}
