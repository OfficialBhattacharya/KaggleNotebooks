# Regression Properties of Rotating Equilateral Triangles

## Introduction

This document explores a fascinating property of equilateral triangles: when their vertices are used as data points for linear regression, the resulting regression line remains constant regardless of the triangle's rotation around its centroid. This leads to a profound insight about the relationship between the x and y coordinates of the vertices.

## The Problem

Consider an equilateral triangle in the XY plane:
- Three vertices serve as data points
- The triangle can rotate around its centroid
- We want to understand how the regression line changes with rotation

## Key Observations

### Initial Setup
For an equilateral triangle centered at the origin (0,0), the vertices can be represented as:
$$ (x_k, y_k) = \left(r\cos\left(\frac{2\pi k}{3} + \theta\right), r\sin\left(\frac{2\pi k}{3} + \theta\right)\right) $$
where:
- $r$ is the distance from the centroid to any vertex
- $\theta$ is the rotation angle
- $k = 0,1,2$ represents the three vertices

### Regression Line Properties

1. **Constant Slope**
   - The regression line is always parallel to the X-axis
   - The slope is always zero
   - This means $y = \bar{y}$ (the mean y-coordinate)

2. **Centroid Property**
   - The regression line always passes through the centroid
   - Since the triangle is centered at the origin, the line is $y = 0$

## Pearson's Correlation Coefficient

### Definition
Pearson's correlation coefficient ($\rho$) measures the linear correlation between two variables. For variables X and Y, it is defined as:

$$ \rho_{X,Y} = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y} = \frac{\mathbb{E}[(X-\mu_X)(Y-\mu_Y)]}{\sigma_X \sigma_Y} $$

where:
- $\text{Cov}(X,Y)$ is the covariance between X and Y
- $\sigma_X$ and $\sigma_Y$ are the standard deviations of X and Y
- $\mu_X$ and $\mu_Y$ are the means of X and Y
- $\mathbb{E}$ represents the expected value

### Properties
1. **Range**: $-1 \leq \rho \leq 1$
   - $\rho = 1$: Perfect positive linear correlation
   - $\rho = -1$: Perfect negative linear correlation
   - $\rho = 0$: No linear correlation

2. **Interpretation**
   - Measures the strength and direction of linear relationship
   - Independent of the scale of measurement
   - Sensitive to outliers

3. **Computation**
   For a sample of n points $(x_i, y_i)$:
   $$ \rho = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^n (y_i - \bar{y})^2}} $$

### Mathematical Proof

1. **Mean Values**
   For any rotation angle $\theta$:
   $$ \bar{x} = \frac{1}{3}\sum_{k=0}^{2} r\cos\left(\frac{2\pi k}{3} + \theta\right) = 0 $$
   $$ \bar{y} = \frac{1}{3}\sum_{k=0}^{2} r\sin\left(\frac{2\pi k}{3} + \theta\right) = 0 $$

2. **Correlation Coefficient**
   The correlation between x and y coordinates is:
   $$ \rho_{x,y} = \frac{\sum_{k=0}^{2} (x_k - \bar{x})(y_k - \bar{y})}{\sqrt{\sum_{k=0}^{2} (x_k - \bar{x})^2 \sum_{k=0}^{2} (y_k - \bar{y})^2}} = 0 $$

3. **Regression Line**
   Since $\rho_{x,y} = 0$, the regression line is:
   $$ y = \bar{y} = 0 $$

## Implications

### Geometric Interpretation
1. **Symmetry**
   - The equilateral triangle's perfect symmetry ensures equal distribution of points
   - Any rotation preserves this symmetry
   - The centroid remains fixed at the origin

2. **Statistical Independence**
   - The x and y coordinates are uncorrelated
   - There is no linear relationship between x and y
   - This holds true for any rotation angle

### Broader Significance

1. **Statistical Insight**
   - Perfect geometric symmetry leads to statistical independence
   - The lack of correlation is a fundamental property, not a coincidence
   - This property extends to regular polygons with any number of sides

2. **Practical Applications**
   - Understanding this property helps in:
     - Geometric data analysis
     - Pattern recognition
     - Statistical modeling of symmetric shapes

## Conclusion

The study of rotating equilateral triangles reveals a deep connection between geometry and statistics. The constant regression line, regardless of rotation, demonstrates that the x and y coordinates of the vertices are linearly independent. This property is not just a mathematical curiosity but a fundamental characteristic of symmetric geometric shapes.

This analysis shows how geometric properties can manifest in statistical measures, providing insights into the relationship between shape, symmetry, and statistical independence. 