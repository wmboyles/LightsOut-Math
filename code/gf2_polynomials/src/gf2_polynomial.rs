use cached::proc_macro::cached;
use pyo3::prelude::*;

use itertools::Itertools;
use std::collections::HashSet;
use std::fmt;

#[derive(Clone, PartialEq, Eq)]
pub struct GF2Polynomial {
    degree: usize,
    degrees: HashSet<usize>,
}

// #[pymethods]
impl GF2Polynomial {
    /// Create a new polynomial with the given coefficients.
    pub fn new(degrees: HashSet<usize>) -> GF2Polynomial {
        GF2Polynomial {
            degree: *degrees.iter().max().unwrap_or(&0),
            degrees,
        }
    }

    /// Create the polynomial 0
    pub fn zero() -> Self {
        Self::new(HashSet::new())
    }

    /// Create the polynomial 1 = x^0
    pub fn one() -> Self {
        Self::new(HashSet::from([0]))
    }

    /// Returns true if the polynomial is 0
    pub fn is_zero(&self) -> bool {
        self.degree == 0 && !self.degrees.contains(&0)
    }

    /// Returns true if the polynomial is 1
    pub fn is_one(&self) -> bool {
        self.degree == 0 && self.degrees.contains(&0)
    }

    /// Returns the sum of two polynomials
    pub fn add(&self, other: &Self) -> Self {
        Self::new(HashSet::from_iter(
            self.degrees.symmetric_difference(&other.degrees).cloned(),
        ))
    }

    /// Returns the product of self and x^power
    pub fn multiply_by_x_power(&self, power: usize) -> Self {
        if power == 0 || self.is_zero() {
            return self.clone();
        }

        Self::new(HashSet::from_iter(self.degrees.iter().map(|&d| d + power)))
    }

    /// Returns the product of two polynomials
    pub fn multiply(&self, other: &Self) -> Self {
        self.degrees.iter().fold(Self::zero(), |acc, &d| {
            acc.add(&other.multiply_by_x_power(d))
        })
    }

    /// Returns the quotient and remainder of the division of self by other
    pub fn quotient_remainder(&self, divisor: &Self) -> (Self, Self) {
        // Easy cases
        if divisor.is_zero() {
            panic!("Division by zero");
        }

        let mut r = self.clone();
        let mut q = Self::zero();

        while (r.degree >= divisor.degree) && !r.is_zero() {
            let d_deg = r.degree - divisor.degree;

            // Quotient += x^d_deg
            q = q.add(&Self::from([d_deg]));

            // Remainder -= divisor * x^d_deg
            r = r.add(&divisor.multiply_by_x_power(d_deg));
        }

        (q, r)
    }

    /**
    If poly is x^a + x^b + ..., then this method returns x^(an) + x^(bn) + ... .
    Since (x^a + x^b)^n =  x^(an) + x^(bn) + ... when n is a power of 2,
    this method can help compute exponentiation usually faster than exponentiation by squaring.
    */
    fn distributive_pow(&self, power: usize) -> Self {
        Self::new(HashSet::from_iter(self.degrees.iter().map(|&d| d * power)))
    }

    /**
    (x^a ... + x^k)^i = (x^a ... + x^k)^(j1) * (x^a ... + x^k)^(j2) * ... * (x^a ... + x^k)^(jn)
    where j1, j2, ..., jn are the powers of 2 that sum to i (i.e. i's representation in binary).
    If i is a power of 2, then (x^a ... + x^k)^i = x^(ai) + ... + x^(ki)
    So, we can precompute each (x^a ... + x^k)^j1, j2, ..., jn and multiply them together.
    This is at least as fast as exponentiation by squaring b/c we have to do O(log2(i)) brute force multiplications.
    */
    pub fn pow(&self, exp: usize) -> Self {
        let mut prod = Self::one();
        let mut i = 1;
        while i <= exp {
            if (i & exp) != 0 {
                prod = prod.multiply(&self.distributive_pow(i));
            }
            i <<= 1;
        }

        prod
    }

    /**
    Returns the result of self composed with other
    If self is f(x) and other is g(x), then this method returns f(g(x))
    */
    pub fn compose(&self, other: &Self) -> Self {
        self.degrees
            .iter()
            .fold(Self::zero(), |acc, &d| acc.add(&other.pow(d)))
    }

    /// Returns the GCD of self and other
    pub fn gcd(&self, other: &Self) -> Self {
        let mut f = self.clone();
        let mut g = other.clone();

        while !g.is_zero() {
            let (_, r) = f.quotient_remainder(&g);
            f = g;
            g = r;
        }

        f
    }
}

impl From<usize> for GF2Polynomial {
    fn from(num: usize) -> GF2Polynomial {
        let mut i: usize = 1;
        let mut degrees = HashSet::new();
        while i < num {
            if (i & num).is_power_of_two() {
                degrees.insert(i.trailing_zeros() as usize);
            }
            i <<= 1;
        }

        GF2Polynomial::new(degrees)
    }
}
impl From<GF2Polynomial> for usize {
    fn from(poly: GF2Polynomial) -> usize {
        poly.degrees.iter().fold(0, |acc, &d| acc + (1 << d))
    }
}

impl FromIterator<usize> for GF2Polynomial {
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        GF2Polynomial::new(HashSet::from_iter(iter))
    }
}

impl<const N: usize> From<[usize; N]> for GF2Polynomial {
    fn from(arr: [usize; N]) -> Self {
        GF2Polynomial::new(HashSet::from(arr))
    }
}
impl fmt::Display for GF2Polynomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self
            .degrees
            .iter()
            .sorted()
            .rev()
            .fold("".to_string(), |acc, i| match acc.is_empty() {
                false => format!("{} + x^{}", acc, i),
                true => format!("x^{}", i),
            });

        match s.is_empty() {
            true => write!(f, "0"),
            false => write!(f, "{}", s),
        }
    }
}
impl core::fmt::Debug for GF2Polynomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl Default for GF2Polynomial {
    fn default() -> Self {
        Self::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::GF2Polynomial;

    #[test]
    fn usize_into_gf2poly() {
        let p = 0b10101usize;
        let expected = GF2Polynomial::from_iter(vec![0, 2, 4]);
        assert_eq!(GF2Polynomial::from(p), expected);
    }

    #[test]
    fn gf2poly_into_usize() {
        let p = GF2Polynomial::from_iter(vec![0, 2, 4]);
        let actual: usize = p.into();
        let expected = 0b10101usize;
        assert_eq!(actual, expected);
    }

    #[test]
    fn display() {
        let zero = GF2Polynomial::zero();
        assert_eq!(zero.to_string(), "0");

        let one = GF2Polynomial::one();
        assert_eq!(one.to_string(), "x^0");

        let x2x1 = GF2Polynomial::from([0, 1, 2]);
        assert_eq!(x2x1.to_string(), "x^2 + x^1 + x^0");

        let x3 = GF2Polynomial::from([3]);
        assert_eq!(x3.to_string(), "x^3");
    }

    #[test]
    fn eq() {
        let one1 = GF2Polynomial::one();
        let one2 = GF2Polynomial::one();
        assert!(one1.eq(&one2));
        assert_eq!(one1, one2);

        let zero = GF2Polynomial::zero();
        assert!(zero.eq(&zero));
        assert_eq!(zero, zero);
        assert!(!zero.eq(&one1));
        assert_ne!(zero, one1);
    }

    #[test]
    fn add() {
        let zero = GF2Polynomial::zero();
        let one = GF2Polynomial::one();

        // Test adding to self always gives 0
        let a = zero.add(&zero);
        let b = one.add(&one);
        assert!(a.is_zero());
        assert!(b.is_zero());

        // Test adding 0
        let one_plus_zero = one.add(&zero);
        assert_eq!(one_plus_zero, one);
        assert_eq!(one_plus_zero.degree, 0);

        // Test adding 2 polynomials of same degree
        let x2 = GF2Polynomial::from([2]);
        let x2_1 = GF2Polynomial::from([2, 0]);
        let x2_plus_x2_1 = x2.add(&x2_1);
        assert!(x2_plus_x2_1.is_one());

        // Bug we saw: (x^3 + x) + (x^4 + x^2) = x^4 + x^3 + x^2 + x
        let x3_x = GF2Polynomial::from([3, 1]);
        let x4_x2 = GF2Polynomial::from([4, 2]);
        let x3_x_plus_x4_x2 = x3_x.add(&x4_x2);
        let x3_x_plus_x4_x2_expected = GF2Polynomial::from([4, 3, 2, 1]);
        assert_eq!(x3_x_plus_x4_x2, x3_x_plus_x4_x2_expected);
    }

    #[test]
    fn multiply_by_x_power() {
        // multiplying by 0
        let zero = GF2Polynomial::zero();
        let zero_x2 = zero.multiply_by_x_power(2);
        assert!(zero_x2.is_zero());

        // multiplying by x^0 = 1
        let one = GF2Polynomial::one();
        let one_x0 = one.multiply_by_x_power(0);
        assert!(one_x0.is_one());

        // basic multiplication
        let x3_x2_x_1 = GF2Polynomial::from([3, 2, 1, 0]);

        let a = x3_x2_x_1.multiply_by_x_power(0);
        assert_eq!(a, x3_x2_x_1);

        let b = x3_x2_x_1.multiply_by_x_power(1);
        let b_expected = GF2Polynomial::from([1, 2, 3, 4]);
        assert_eq!(b, b_expected);

        let c = x3_x2_x_1.multiply_by_x_power(5);
        let c_expected = GF2Polynomial::from([5, 6, 7, 8]);
        assert_eq!(c, c_expected);

        // Bug we saw: (x^2 + x^0) * x^2 = x^4 + x^2
        let x2_x0 = GF2Polynomial::from([0, 2]);
        let x2_x0_x2 = x2_x0.multiply_by_x_power(2);
        let x2_x0_x2_expected = GF2Polynomial::from([2, 4]);
        assert_eq!(x2_x0_x2, x2_x0_x2_expected);
    }

    #[test]
    fn multiply() {
        let x4_x2_x = GF2Polynomial::from([1, 2, 4]);
        let x2_1 = GF2Polynomial::from([0, 2]);

        // multiply by 0
        let zero = GF2Polynomial::zero();
        let zero_x4_x2_x = zero.multiply(&x4_x2_x);
        assert!(zero_x4_x2_x.is_zero());
        let x4_x2_x_zero = x4_x2_x.multiply(&zero);
        assert!(x4_x2_x_zero.is_zero());

        // multiply by 1
        let one = GF2Polynomial::one();
        let one_x4_x2_x = one.multiply(&x4_x2_x);
        assert_eq!(one_x4_x2_x, x4_x2_x);
        let x4_x2_x_one = x4_x2_x.multiply(&one);
        assert_eq!(x4_x2_x_one, x4_x2_x);

        // Regular multiplication
        assert_eq!(x4_x2_x.degree, 4);
        assert_eq!(x2_1.degree, 2);
        let x6_x3_x2_x = x4_x2_x.multiply(&x2_1);
        let x6_x3_x2_x_expected = GF2Polynomial::from([1, 2, 3, 6]);
        assert_eq!(x6_x3_x2_x, x6_x3_x2_x_expected);
    }

    #[test]
    #[should_panic]
    fn quotient_remainder_divide_by_zero() {
        // Test division by 0 (should panic)
        let one = GF2Polynomial::one();
        let zero = GF2Polynomial::zero();
        one.quotient_remainder(&zero);
    }

    #[test]
    fn quotient_remainder() {
        // 0 / (anything, except 0) = 0
        let zero = GF2Polynomial::zero();
        let x4_x2_x = GF2Polynomial::from([1, 2, 4]);
        let (zero_quotient, zero_remainder) = zero.quotient_remainder(&x4_x2_x);
        assert!(zero_quotient.is_zero());
        assert!(zero_remainder.is_zero());

        // (x^4 + x^2 + x) / x = x^3 + x + x^0 (remainder 0)
        let x = GF2Polynomial::one().multiply_by_x_power(1);
        let (q, r) = x4_x2_x.quotient_remainder(&x);
        let q_expected = GF2Polynomial::from([0, 1, 3]);
        assert_eq!(q, q_expected);
        assert!(r.is_zero());

        // (x^4 + x^2 + x) / x^2 = x^2 + x^0 (remainder x)
        let x2 = GF2Polynomial::one().multiply_by_x_power(2);
        let (q, r) = x4_x2_x.quotient_remainder(&x2);
        let q_expected = GF2Polynomial::from([0, 2]);
        assert_eq!(q, q_expected);
        assert_eq!(r, x);
    }

    #[test]
    fn distributive_pow() {
        // (0)^(anything, except 0) = 0
        let zero = GF2Polynomial::zero();
        let zero_5 = zero.distributive_pow(5);
        assert!(zero_5.is_zero());

        // // 0^0 = 1
        // let zero_zero = zero.distributive_pow(0);
        // assert!(zero_zero.is_one());

        // anything^0 = 1
        let x4_x2_x = GF2Polynomial::from([1, 2, 4]);
        let x4_x2_x_0 = x4_x2_x.distributive_pow(0);
        assert!(x4_x2_x_0.is_one());

        // (x^4 + x^2 + x)^4 = x^16 + x^8 + x^4
        let x4_x2_x_4 = x4_x2_x.distributive_pow(4);
        let x4_x2_x_4_expected = GF2Polynomial::from([4, 8, 16]);
        assert_eq!(x4_x2_x_4, x4_x2_x_4_expected);
    }

    #[test]
    fn pow() {
        // 0^0 = 1
        let zero = GF2Polynomial::zero();
        let zero_zero = zero.pow(0);
        assert!(zero_zero.is_one());

        // 0 ^ (anything, except 0)= 0
        let zero_5 = zero.pow(5);
        assert!(zero_5.is_zero());

        // anything ^ 0 = 1
        let x4_x2_x = GF2Polynomial::from([1, 2, 4]);
        let x4_x2_x_zero = x4_x2_x.pow(0);
        assert!(x4_x2_x_zero.is_one());

        // anything ^ (power of 2) = distributive_pow
        let x4_x2_x_4 = x4_x2_x.pow(4);
        let x4_x2_x_4_expected = x4_x2_x.distributive_pow(4);
        assert_eq!(x4_x2_x_4, x4_x2_x_4_expected);

        // (x^4 + x^2 + x)^5 = (x^4 + x^2 + x)^4 * (x^4 + x^2 + x)
        // = (x^16 + x^8 + x^4) * (x^4 + x^2 + x)
        // = x^20 + x^18 + x^17  +  x^12 + x^10 + x^9  +  x^8 + x^6 + x^5
        let x4_x2_x_5 = x4_x2_x.pow(5);
        let x4_x2_x_5_expected = x4_x2_x_4.multiply(&x4_x2_x);
        assert_eq!(x4_x2_x_5, x4_x2_x_5_expected);
        let x4_x2_x_5_expected2 = GF2Polynomial::from([5, 6, 8, 9, 10, 12, 17, 18, 20]);
        assert_eq!(x4_x2_x_5, x4_x2_x_5_expected2);
    }

    #[test]
    fn compose() {
        // 0 @ (anything) = 0
        let zero = GF2Polynomial::zero();
        let x4_x2_x = GF2Polynomial::from([1, 2, 4]);
        let zero_composed = zero.compose(&x4_x2_x);
        assert!(zero_composed.is_zero());

        // anything @ 0 = 0 or 1, depending of whether anything is divisible by x
        let x4_x2_x_at_0 = x4_x2_x.compose(&zero);
        assert!(x4_x2_x_at_0.is_zero());

        let x4_x2_x_1 = GF2Polynomial::from([0, 1, 2, 4]);
        let x4_x2_x_1_at_0 = x4_x2_x_1.compose(&zero);
        assert!(x4_x2_x_1_at_0.is_one());

        // (x^4 + x^2 + x) @ (x^4 + x^2 + x) = (x^4 + x^2 + x)^4 + (x^4 + x^2 + x)^2 + (x^4 + x^2 + x)
        // = (x^16 + x^8 + x^4) + (x^8 + x^4 + x^2) + (x^4 + x^2 + x)
        // = x^16 + x^4 + x
        let x4_x2_x_composed = x4_x2_x.compose(&x4_x2_x);
        let x4_x2_x_composed_expected = GF2Polynomial::from([1, 4, 16]);
        assert_eq!(x4_x2_x_composed, x4_x2_x_composed_expected);
    }

    #[test]
    fn gcd() {
        // gcd(a,a) = a
        let x4_x2_x = GF2Polynomial::from([1, 2, 4]);
        let x4_x2_x_gcd = x4_x2_x.gcd(&x4_x2_x);
        assert_eq!(x4_x2_x_gcd, x4_x2_x);

        // gcd(x,x+1) = 1
        let x = GF2Polynomial::from([1]);
        let x_plus_1 = GF2Polynomial::from([0, 1]);
        let x_gcd = x.gcd(&x_plus_1);
        assert!(x_gcd.is_one());
        let x_gcd2 = x_plus_1.gcd(&x);
        assert!(x_gcd2.is_one());

        // gcd(x^4 + x^2 + x, x) = x
        let x4_x2_x_gcd = x4_x2_x.gcd(&x);
        assert_eq!(x4_x2_x_gcd, x);
        let x4_x2_x_gcd2 = x.gcd(&x4_x2_x);
        assert_eq!(x4_x2_x_gcd2, x);

        // gcd(x^4 + x^2 + 1, x^3 + x^2 + x) = x^2 + x + 1
        let x4_x2_1 = GF2Polynomial::from([0, 2, 4]);
        let x3_x2_x = GF2Polynomial::from([1, 2, 3]);
        let gcd = x4_x2_1.gcd(&x3_x2_x);
        let gcd_expected = GF2Polynomial::from([0, 1, 2]);
        assert_eq!(gcd, gcd_expected);
        let gcd2 = x3_x2_x.gcd(&x4_x2_1);
        assert_eq!(gcd2, gcd_expected);
    }
}

// ----------------------------------------------
#[cached]
fn find_bk(n: usize) -> (usize, usize) {
    let k = (n + 1).trailing_zeros() as usize;
    let b = (n + 1) >> k;
    (b, k)
}

#[cached]
fn brute_f1(y: usize) -> GF2Polynomial {
    GF2Polynomial::from_iter((0..=y).filter(|&i| ((y - i) & (2 * i + 1)) == 0))
}

#[cached]
fn grid_nullity(n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    // else if n.is_power_of_two() {
    //     return Self::grid_nullity(n-2) + match  & 1 == 1 {
    //         true => 4,
    //         false => 0,
    //     };
    // }

    let (b, k) = find_bk(n);
    if b == 1 {
        return 0;
    }

    let f1 = brute_f1(b - 1);
    let g = f1.gcd(&f1.compose(&GF2Polynomial::from([0, 1])));

    let delta = match n % 3 == 2 {
        true => 2,
        false => 0,
    };

    (g.degree + delta) * (1 << k) - delta
}

#[pyfunction]
fn grid_nullities(n: usize) -> Vec<usize> {
    (1..n).map(grid_nullity).collect()
}

#[pymodule]
fn gf2_polynomial(_py: Python, m: &PyModule) -> PyResult<()> {
    // m.add_class::<GF2Polynomial>()?;
    m.add_function(wrap_pyfunction!(grid_nullities, m)?)?;
    Ok(())
}
