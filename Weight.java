package SingleLayerPerceptron;

import java.util.Random;

public class Weight {
	private float[] weights;
	private float min_weight = -1.0f;
	private float max_weight = 1.0f;
	
	public Weight(int n) {
		weights = new float[n];
		Random rand = new Random();
		for (int i = 0; i < weights.length; i++) {
			weights[i] = rand.nextFloat() * (max_weight - min_weight) + min_weight;
		}
	}
	
	public float get_w1() {
		return weights[0];
	}
	
	public float get_w2() {
		return weights[1];
	}
	
	public void set_w1(float w1) {
		weights[0] = w1;
	}
	
	public void set_w2(float w2) {
		weights[1] = w2;
	}
}
