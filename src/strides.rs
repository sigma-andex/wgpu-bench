use crate::Shape;
use encase::impl_wrapper;

#[derive(Clone, PartialEq, Eq, Default, Hash)]
pub struct Strides(Vec<isize>);

impl_wrapper!(Strides; using);

impl Strides {
    pub fn inner(self) -> Vec<isize> {
        self.0
    }
}

impl std::fmt::Debug for Strides {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut shape = format!("[{}", self.0.first().unwrap_or(&0));
        for dim in self.0.iter().skip(1) {
            shape.push_str(&format!("x{}", dim));
        }
        write!(f, "{}]", shape)
    }
}

impl From<&Shape> for Strides {
    fn from(shape: &Shape) -> Self {
        let mut strides = vec![];
        let mut stride = 1;
        for size in shape.to_vec().iter().rev() {
            strides.push(stride);
            stride *= *size as isize;
        }
        strides.reverse();
        Self(strides)
    }
}

impl From<&Strides> for [u32; 4] {
    fn from(strides: &Strides) -> Self {
        assert!(strides.0.len() <= 4);
        let mut array = [0; 4];
        for (i, &stride) in strides.0.iter().enumerate() {
            array[i] = stride as u32;
        }
        array
    }
}

impl From<&Strides> for [u32; 3] {
    fn from(strides: &Strides) -> Self {
        assert!(strides.0.len() <= 3);
        let mut array = [0; 3];
        for (i, &stride) in strides.0.iter().enumerate() {
            array[i] = stride as u32;
        }
        array
    }
}

impl From<&Strides> for glam::UVec4 {
    fn from(strides: &Strides) -> Self {
        let array: [u32; 4] = strides.into();
        glam::UVec4::from(array)
    }
}

impl From<&Strides> for glam::UVec3 {
    fn from(strides: &Strides) -> Self {
        let array: [u32; 3] = strides.into();
        glam::UVec3::from(array)
    }
}

impl From<Strides> for glam::IVec3 {
    fn from(strides: Strides) -> Self {
        (&strides).into()
    }
}

impl From<&Strides> for glam::IVec3 {
    fn from(strides: &Strides) -> Self {
        glam::IVec3::new(strides.0[0] as _, strides.0[1] as _, strides.0[2] as _)
    }
}

#[cfg(test)]
mod tests {
    use crate::shape;

    #[test]
    fn test_strides() {
        use super::*;
        let shape = shape![2, 3, 4];
        let strides = Strides::from(&shape);
        assert_eq!(strides.inner(), vec![12, 4, 1]);
    }
}
