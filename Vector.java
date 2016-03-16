package SingleLayerPerceptron;

/*
 * Vector class to allow n-dimensional input to Perceptron. Holds 
 * coefficient array to represent hyperplane bisecting the input domain.
 * Hyperplane would just be a line in 2D case.
 */

public class Vector {
	int[] coefficients;
	
	public Vector(int[] inputs) {
		this.coefficients = inputs;
	}
	
	/* Dot product method used on vector and trainer input arrays */
	public int dot_product(int[] input) {
		int sum = 0;
		for (int i = 0; i < coefficients.length; i++) {
			sum+=coefficients[i] * input[i];
		}
		return sum;
	}
	
	/* Dot product method used in weight adjustment calculation */
	public float dot_product(float [] input) {
		float sum = 0;
		for (int i = 0; i < coefficients.length; i++) {
			sum+=coefficients[i] * input[i];
		}
		return sum;
	}
}
