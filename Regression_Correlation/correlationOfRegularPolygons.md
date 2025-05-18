# Correlation of Regular Polygons and Complex Roots of Unity

## Introduction

This document explores the fascinating relationship between regular polygons, complex roots of unity, and their correlation properties. We'll examine why regular polygons exhibit zero correlation between their x and y coordinates, and how this connects to the mathematical properties of complex roots of unity.

## Complex Roots of Unity

### Definition
The n-th roots of unity are the complex numbers that satisfy the equation:

$$ z^n = 1 $$

These roots can be expressed in the form:

$$ z_k = e^{i\frac{2\pi k}{n}}, \quad k = 0,1,\ldots,n-1 $$

where:
- $i$ is the imaginary unit
- $n$ is the number of roots
- $k$ is the index of each root

### Geometric Interpretation
When plotted on the complex plane:
- Each root lies on the unit circle
- The roots are evenly spaced, forming the vertices of a regular n-gon
- The first root (k=0) is always at (1,0)
- The roots are distributed at angles of $\frac{2\pi k}{n}$ from the positive real axis

### Key Properties
1. The sum of all n-th roots of unity is zero:
   $$ \sum_{k=0}^{n-1} e^{i\frac{2\pi k}{n}} = 0 $$

2. This implies that both the real and imaginary parts sum to zero:
   $$ \sum_{k=0}^{n-1} \cos\left(\frac{2\pi k}{n}\right) = 0 $$
   $$ \sum_{k=0}^{n-1} \sin\left(\frac{2\pi k}{n}\right) = 0 $$

## Correlation Analysis

### Coordinate Representation
For a regular n-gon, the vertices can be represented as:
$$ (x_k, y_k) = \left(\cos\left(\frac{2\pi k}{n}\right), \sin\left(\frac{2\pi k}{n}\right)\right) $$

### Correlation Coefficient
The correlation coefficient between x and y coordinates is given by:
$$ \rho_{x,y} = \frac{\text{Cov}(x,y)}{\sigma_x \sigma_y} = \frac{\sum_{k=0}^{n-1} (x_k - \bar{x})(y_k - \bar{y})}{\sqrt{\sum_{k=0}^{n-1} (x_k - \bar{x})^2 \sum_{k=0}^{n-1} (y_k - \bar{y})^2}} $$

### Proof of Zero Correlation

1. **Mean Values**
   From the properties of roots of unity:
   $$ \bar{x} = \frac{1}{n}\sum_{k=0}^{n-1} \cos\left(\frac{2\pi k}{n}\right) = 0 $$
   $$ \bar{y} = \frac{1}{n}\sum_{k=0}^{n-1} \sin\left(\frac{2\pi k}{n}\right) = 0 $$

2. **Simplified Correlation Formula**
   Since $\bar{x} = \bar{y} = 0$, the correlation formula simplifies to:
   $$ \rho_{x,y} = \frac{\sum_{k=0}^{n-1} x_k y_k}{\sqrt{\sum_{k=0}^{n-1} x_k^2 \sum_{k=0}^{n-1} y_k^2}} $$

3. **Trigonometric Identities**
   The numerator becomes:
   $$ \sum_{k=0}^{n-1} \cos\left(\frac{2\pi k}{n}\right) \sin\left(\frac{2\pi k}{n}\right) = 0 $$
   
   The denominator terms are equal:
   $$ \sum_{k=0}^{n-1} \cos^2\left(\frac{2\pi k}{n}\right) = \sum_{k=0}^{n-1} \sin^2\left(\frac{2\pi k}{n}\right) = \frac{n}{2} $$

4. **Final Result**
   Therefore:
   $$ \rho_{x,y} = \frac{0}{\sqrt{\frac{n}{2} \cdot \frac{n}{2}}} = 0 $$

## Implications

1. **Geometric Interpretation**
   - Regular polygons exhibit perfect symmetry
   - The x and y coordinates are uncorrelated
   - The regression line is always horizontal (y = 0)

2. **Statistical Significance**
   - This property holds for any regular n-gon (n ≥ 3)
   - The zero correlation is a direct consequence of the geometric symmetry
   - This provides a beautiful example of how geometric properties manifest in statistical measures

## Conclusion

The study of regular polygons and their correlation properties reveals a deep connection between geometry, complex numbers, and statistics. The zero correlation between coordinates is not a coincidence but a fundamental property arising from the mathematical structure of regular polygons and their relationship to complex roots of unity.

This analysis demonstrates how mathematical concepts from different domains (geometry, complex analysis, and statistics) can converge to reveal elegant and unexpected relationships. 