public class Sum {

    // Method to calculate the sum of numbers
    public static int calculateSum(int[] numbers) {
        int sum = 0;
        for (int num : numbers) {
            sum += num;
        }
        return sum;
    }

    // Method to calculate the average of numbers
    public static double calculateAverage(int[] numbers) {
        int sum = calculateSum(numbers);  // Calling calculateSum method
        return (double) sum / numbers.length;
    }

    public static void main(String[] args) {
        // Example array of numbers
        int[] numbers = {10, 20, 30, 40, 50};

        // Calculate sum and average
        int sum = calculateSum(numbers);
        double average = calculateAverage(numbers);

        // Output the results
        System.out.println("Sum: " + sum);
        System.out.println("Average: " + average);
    }
}

