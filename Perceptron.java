package SingleLayerPerceptron;
/*
 * Implementation of single layer perceptron. Classifies randomly
 * generated training set for some domain against vector input.
 */
import java.util.Random;

public class Perceptron {
	private float[] weights;
	private int[] inputs;
	private Trainer[] trainers;
	private int bias = 1;
	private float bias_weight;
	private float learning_constant = 0.0001f;
	private int num_of_inputs;

	public Perceptron(int[] inputs) {
		num_of_inputs = inputs.length;
		this.inputs = inputs;

		weights = new float[num_of_inputs];
		create_weights(weights);

		trainers = new Trainer[100];
		create_trainers(trainers);
	}

	/*
	 * Create weight for each of the n inputs and bias using random number
	 * generator.
	 */
	private void create_weights(float[] weights) {
		Random rand = new Random();
		for (int i = 0; i < weights.length; i++) {
			weights[i] = rand.nextFloat() * (1 - (-1)) + (-1);
		}
		bias_weight = rand.nextFloat() * (1 - (-1)) + (-1);
	}

	/*
	 * Populate trainer array and compute their known answers
	 */
	private void create_trainers(Trainer[] trainer) {
		Random rand = new Random();
		int output, known_answer;
		for (int i = 0; i < trainer.length; i++) {
			trainer[i] = new Trainer(num_of_inputs);
			output = dot_product(trainer[i].get_inputs(), inputs) + bias;
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
					error = (known_answer - output);
					adjust_weights(trainers[i], error);
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
		System.out.format("Number of iterations to classify data: %d%n", iterations);
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
		bias_weight+=bias * error * learning_constant;
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

 	public static void main(String[] args) {
 		int[] coefficients = new int[10];
 		Random rand = new Random();
 		for (int i = 0; i < 10; i++) {
 			coefficients[i] = rand.nextInt((10 - (-10)) + 1) + (-10);
 		}
 		
		Perceptron ptron = new Perceptron(coefficients);

		ptron.train();
	}

}