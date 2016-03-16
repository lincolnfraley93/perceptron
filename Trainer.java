package SingleLayerPerceptron;

import java.util.Random;

public class Trainer {
	private int[] inputs;
	private int known_answer;
	
	public Trainer(int num_of_inputs) {
		inputs = new int[num_of_inputs];
		Random rand = new Random();
		for (int i = 0; i < num_of_inputs; i++) {
			inputs[i] = rand.nextInt((10 - (-10)) + 1) + (-10);
		}
	}
	
	public int[] get_inputs() {
		return inputs;
	}
	
	public void set_answer(int answer) {
		known_answer = answer;
	}
	
	public int get_answer() {
		return known_answer;
	}
}
