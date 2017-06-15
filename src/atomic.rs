use std::mem;
use std::panic::RefUnwindSafe;
use std::ops::{Add, BitAnd, BitOr, BitXor, Sub};
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};

pub trait IntCast : Copy + Eq + Add<Output=Self> + BitAnd<Output=Self>
    + BitOr<Output=Self> + BitXor<Output=Self> + Sub<Output=Self> {
    fn from(u: usize) -> Self;
    fn to(self) -> usize;
}

macro_rules! intcast {
    ($($type:ident)+) => {
        $(
            impl IntCast for $type {
                fn from(u: usize) -> Self {
                    u as Self
                }
                fn to(self) -> usize {
                    self as usize
                }
            }
        )+
    }
}
intcast! { u8 i8 u16 i16 u32 i32 u64 i64 }

pub struct Template<T> {
    v: UnsafeCell<T>,
}

impl<T: Default + IntCast> Default for Template<T> {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

// TODO: impl Debug

unsafe impl<T> Sync for Template<T> {}
impl<T> RefUnwindSafe for Template<T> {}

fn inject<T>(a: usize, b: usize, offset: usize) -> usize {
    let mask = ((1 << (mem::size_of::<T>() * 8)) - 1) << offset;
    (a & !mask) | (b << offset)
}

// straight from libcore's atomic.rs
#[inline]
fn strongest_failure_ordering(order: Ordering) -> Ordering {
    use self::Ordering::*;
    match order {
        Release => Relaxed,
        Relaxed => Relaxed,
        SeqCst => SeqCst,
        Acquire => Acquire,
        AcqRel => Acquire,
        _ => unreachable!(),
    }
}

impl<T: IntCast> Template<T> {
    #[inline]
    fn proxy(&self) -> (&AtomicUsize, usize) {
        let ptr = self.v.get() as usize;
        let aligned = ptr & !(mem::size_of::<usize>() - 1);
        (unsafe { &*(aligned as *const AtomicUsize) }, (ptr - aligned) * 8)
    }

    // TODO: make this const if const is stable first
    #[inline]
    pub /*const*/ fn new(v: T) -> Self {
        Template { v: UnsafeCell::new(v) }
    }

    #[inline]
    pub fn get_mut(&mut self) -> &mut T {
        unsafe { &mut *self.v.get() }
    }

    #[inline]
    pub fn into_inner(self) -> T {
        unsafe { self.v.into_inner() }
    }

    #[inline]
    pub fn load(&self, order: Ordering) -> T {
        let (p, o) = self.proxy();
        IntCast::from(p.load(order) >> o)
    }

    #[inline]
    fn op<F: Fn(T) -> Option<T>>(&self, f: F, order: Ordering) -> T {
        self.op_new(f, order, strongest_failure_ordering(order))
    }

    #[inline]
    fn op_new<F: Fn(T) -> Option<T>>(&self, f: F, success: Ordering, failure: Ordering) -> T {
        let (p, o) = self.proxy();
        let mut old = p.load(Ordering::Relaxed);
        loop {
            let old_t = IntCast::from(old >> o);
            let new_t = match f(old_t) {
                Some(x) => x,
                None => return old_t,
            };

            match Self::op_weak(p, o, old, new_t, success, failure) {
                Ok(()) => return IntCast::from(old >> o),
                Err(prev) => old = prev,
            };
        }
    }

    #[inline]
    fn op_weak(p: &AtomicUsize, o: usize, old: usize, new_t: T, success: Ordering, failure: Ordering) -> Result<(), usize> {
        let new = inject::<T>(old, new_t.to(), o);
        p.compare_exchange_weak(old, new, success, failure).map(|_| ())
    }
    
    #[inline]
    pub fn store(&self, val: T, order: Ordering) {
        self.op(|_| Some(val), order);
    }

    #[inline]
    pub fn swap(&self, val: T, order: Ordering) -> T {
        self.op(|_| Some(val), order)
    }

    #[inline]
    pub fn compare_and_swap(&self, current: T, new: T, order: Ordering) -> T {
        self.op(|x| if x == current { Some(new) } else { None }, order)
    }

    #[inline]
    pub fn compare_exchange(&self, current: T, new: T, success: Ordering, failure: Ordering) -> Result<T, T> {
        match self.op_new(|x| if x == current { Some(new) } else { None }, success, failure) {
            x if x == current => Ok(x),
            x => Err(x),
        }
    }

    #[inline]
    pub fn compare_exchange_weak(&self, current: T, new: T, success: Ordering, failure: Ordering) -> Result<T, T> {
        let (p, o) = self.proxy();
        let old = p.load(Ordering::Relaxed);
        let old_t = IntCast::from(old >> o);
        if old_t != current {
            return Err(old_t);
        }

        Self::op_weak(p, o, old, new, success, failure).map(|()| current).map_err(|x| IntCast::from(x >> o))
    }

    #[inline]
    pub fn fetch_add(&self, val: T, order: Ordering) -> T {
        self.op(|x| Some(x + val), order)
    }

    #[inline]
    pub fn fetch_sub(&self, val: T, order: Ordering) -> T {
        self.op(|x| Some(x - val), order)
    }

    #[inline]
    pub fn fetch_and(&self, val: T, order: Ordering) -> T {
        self.op(|x| Some(x & val), order)
    }

    #[inline]
    pub fn fetch_or(&self, val: T, order: Ordering) -> T {
        self.op(|x| Some(x | val), order)
    }

    #[inline]
    pub fn fetch_xor(&self, val: T, order: Ordering) -> T {
        self.op(|x| Some(x ^ val), order)
    }
}

pub type AtomicI8 = Template<i8>;
pub type AtomicU8 = Template<u8>;
pub type AtomicI16 = Template<i16>;
pub type AtomicU16 = Template<u16>;
pub type AtomicI32 = Template<i32>;
pub type AtomicU32 = Template<u32>;
pub type AtomicI64 = Template<i64>;
pub type AtomicU64 = Template<u64>;


#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;
    use std::u16;

    #[test]
    fn basics() {
        let v = AtomicU16::new(1337);
        let o = Ordering::Relaxed;
        assert_eq!(v.swap(42, o), 1337);
        assert_eq!(v.fetch_add(1, o), 42);
        assert_eq!(v.fetch_sub(1, o), 43);
        assert_eq!(v.fetch_and(0x20, o), 42);
        assert_eq!(v.fetch_or(0x0a, o), 0x20);
        assert_eq!(v.fetch_xor(42, o), 42);
        assert_eq!(v.fetch_sub(1, o), 0);
        assert_eq!(v.fetch_add(1, o), u16::MAX);
        assert_eq!(v.compare_and_swap(1, 2, o), 0);
        assert_eq!(v.compare_and_swap(0, 3, o), 0);
        assert_eq!(v.load(o), 3);
    }
}
