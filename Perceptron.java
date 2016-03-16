package SingleLayerPerceptron;
/*
 * Implementation of single layer perceptron. Classifies randomly
 * generated training set for some domain against vector input.
 */
import java.util.Random;

public class Perceptron {
	private float[] weights;
	private int[] vector;
	private Trainer[] trainers;
	private int bias = 1;
	private float bias_weight;
	private float learning_constant = 0.01f;
	private int num_of_inputs;

	public Perceptron(int[] vector) {
		num_of_inputs = vector.length;
		this.vector = vector;

		weights = new float[num_of_inputs];
		create_weights(weights);

		trainers = new Trainer[100];
		create_trainers(trainers);

		Random rand = new Random();
		bias_weight = rand.nextFloat() * (1 - (-1)) + (-1);
	}

	/*
	 * Create weight for each of the n inputs using random number
	 * generator.
	 */
	private void create_weights(float[] weights) {
		Random rand = new Random();
		for (int i = 0; i < weights.length; i++) {
			weights[i] = rand.nextFloat() * (1 - (-1)) + (-1);
		}
	}

	/*
	 * Populate trainer array and compute their known answers
	 */
	private void create_trainers(Trainer[] trainer) {
		Random rand = new Random();
		int output, known_answer;

		for (int i = 0; i < trainer.length; i++) {
			trainer[i] = new Trainer(num_of_inputs);
			output = dot_product(trainer[i].get_inputs(), vector) + bias;
			known_answer = (output >= 0) ? 1 : -1;
			trainer[i].set_answer(known_answer);
		}
	}

	/* Train perceptron by updating weights based on known answer */
	public void train() {
		int misclassifications = 0;
		int iterations = 0;
		boolean done = false;
		int output, known_answer, error;

		/* Iterate through trainer array until all input correctly classified */
		while (!done) {
			for (int i = 0; i < trainers.length; i++) {
				known_answer = trainers[i].get_answer();
				output = feed_forward(trainers[i]);

				if (output != known_answer) {
					error = known_answer;
					adjust_weights(trainers[i], error);
					bias_weight+=bias * error * learning_constant;
					misclassifications++;
				}
			}
			iterations++;
			if (misclassifications > 0) {
				System.out.format("Number of errors for iteration %d is %d%n", iterations, misclassifications);
				misclassifications = 0;
			} else {
				done = true;
			}
		}
		System.out.println("Number of iterations to classify data: " + iterations);
	}

	/* Given trainer input, produces output using sign function */
	public int feed_forward(Trainer trainer) {
		float sum = dot_product(trainer.get_inputs(), weights) + bias;
		return activation_function(sum);
	}

	/* Sign function */
	public int activation_function(float sum) {
		return (sum < 0) ? -1 : 1;
	}

	/* Adjusts weights for misclassified inputs */
	public void adjust_weights(Trainer trainer, int error) {
		int[] inputs = trainer.get_inputs();

		for (int i = 0; i < weights.length; i++) {
			weights[i]+=inputs[i] * error * learning_constant;
		}
	}

	/* Dot product method used on vector and trainer input arrays */
	private int dot_product(int[] array1, int[] array2) {
		int sum = 0;

		for (int i = 0; i < array1.length; i++) {
			sum+=array1[i] * array2[i];
		}
		return sum;
	}

	/* Dot product method used in weight adjustment calculation */
	private float dot_product(int[] array1, float[] array2) {
		float sum = 0;

		for (int i = 0; i < array1.length; i++) {
			sum+=array1[i] * array2[i];
		}
		return sum;
	}

	/* Test method for weights array */
	public void print_weights() {
		for (int i = 0; i < num_of_inputs; i++) {
			System.out.println(weights[i]);
		}
	}

	/* Test method for trainers array */
	public void print_trainers() {
		int[] trainer_inputs;
		for (int i = 0; i < trainers.length; i++) {
			trainer_inputs = trainers[i].get_inputs();
			for (int j = 0; j < num_of_inputs; j++) {
				System.out.format("Trainer %d, input %d: %d%n", i, j, trainer_inputs[j]);
			}
			System.out.format("Trainer %d known answer: %d%n", i, trainers[i].get_answer());
			System.out.println();
		}
	}

 	public static void main(String[] args) {
		int[] coefficients = {1, 1};
		Perceptron ptron = new Perceptron(coefficients);

		//  Test weights
		ptron.print_weights();

		//  Test trainers
		ptron.print_trainers();

		// Test train method
		ptron.train();
	}

}