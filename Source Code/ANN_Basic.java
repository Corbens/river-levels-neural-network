import java.io.*;

public class ANN {
    private Layer[] layers;

    public ANN(int[] layerSizes){
        Layer[] layers = new Layer[layerSizes.length-1];
        for(int i = 0; i < layerSizes.length-1; i++){
            int numWeights = layerSizes[i]*layerSizes[i+1];
            int numBiases = layerSizes[i+1];
            Layer currentLayer = new Layer(numWeights, numBiases, layerSizes[i+1], layerSizes[i]);
            layers[i] = currentLayer;
        }
        this.layers = layers;
    }

    private double[][] forwardPass(double[] predictors){
        double[][] allActivations = new double[this.layers.length][]; //save all the activation values
        for (int i = 0; i < this.layers.length; i++) {
            Layer currentLayer = this.layers[i];
            double[] activations = new double[currentLayer.getNumNodes()];
            for (int j = 0; j < currentLayer.getNumNodes(); j++) {
                double sum = currentLayer.getBiases()[j];
                for (int k = j * currentLayer.getNumInputs(); k < (j + 1) * currentLayer.getNumInputs(); k++) {
                    sum += currentLayer.getWeights()[k] * predictors[k % currentLayer.getNumInputs()];
                }
                double activation = (1 / (1 + Math.exp(sum * -1)));
                activations[j] = activation;
            }
            predictors = activations;
            allActivations[i] = activations;
        }
        return allActivations;
    }

    private double[][] backwardPass(double observedValue, double[][] activations){
        double[][] allDeltas = new double[this.layers.length][]; //save all the delta values
        for (int i = this.layers.length - 1; i >= 0; i--) {
            Layer currentLayer = this.layers[i];
            double[] deltas = new double[currentLayer.getNumNodes()];
            if (i == this.layers.length - 1) { //calculate delta if output node
                for (int j = 0; j < currentLayer.getNumNodes(); j++) {
                    double activation = activations[i][j];
                    double delta = (observedValue - activation) * (activation * (1 - activation));
                    deltas[j] = delta;
                }
            } else { //calculate delta if hidden node
                for (int j = 0; j < currentLayer.getNumNodes(); j++) {
                    Layer nextLayer = this.layers[i + 1];
                    double sumDeltasMultWeights = 0;
                    double activation = activations[i][j];
                    for (int k = 0; k < nextLayer.getNumNodes(); k++) {
                        sumDeltasMultWeights += (allDeltas[i + 1][k] * nextLayer.getWeights()[(k *currentLayer.getNumNodes()) + j]);
                    }
                    double delta = sumDeltasMultWeights * (activation * (1 - activation));
                    deltas[j] = delta;
                }
            }
            allDeltas[i] = deltas;
        }
        return allDeltas;
    }

    private void updateVariables(double lp, double[][] activations, double[][] deltas){
        for (int i = 0; i < this.layers.length; i++) {
            Layer currentLayer = this.layers[i];
            for (int j = 0; j < currentLayer.getWeights().length; j++) {
                double activation = activations[i][j % currentLayer.getNumNodes()];
                double delta = deltas[i][j % currentLayer.getNumNodes()];
                double currentWeightChange = (lp * delta * activation);
                currentLayer.getWeights()[j] += currentWeightChange;
            }
            for (int j = 0; j < currentLayer.getBiases().length; j++) {
                double activation = activations[i][j % currentLayer.getNumNodes()];
                double delta = deltas[i][j % currentLayer.getNumNodes()];
                double currentWeightChange = (lp * delta * activation);
                currentLayer.getBiases()[j] += currentWeightChange;
            }
        }
    }

    public void train(int epochs, double lp, double[][] trainingSet, double[][] validationSet, double[][] testSet, String[] testDates){
        double previousTrainingRMSE = Double.MAX_VALUE;
        double previousValidationRMSE = Double.MAX_VALUE;
        for(int i = 0; i < epochs; i++) {

            double trainingTotalError = 0;
            double trainingRMSE;
            for (int j = 0; j < trainingSet.length; j++) { //go through the training set
                double[] predictors = new double[trainingSet[j].length-1];
                for (int k = 0; k < trainingSet[j].length-1; k++){
                    predictors[k] = trainingSet[j][k];
                }
                double[][] activations = forwardPass(predictors); //forward pass
                double observedValue = trainingSet[j][trainingSet[j].length-1];
                double modelledValue = activations[activations.length-1][0];
                trainingTotalError += (modelledValue-observedValue)*(modelledValue-observedValue);
                double[][] deltas = backwardPass(observedValue, activations); //backward pass
                updateVariables(lp, activations, deltas); //update variables
            }
            trainingRMSE = Math.sqrt(trainingTotalError/trainingSet.length); //calculate training RMSE
            System.out.println("Epoch no: " + i + ", Training RMSE: " + trainingRMSE);

            try { //save the error and epoch to txt so can draw a graph showing how the error changed over time
                FileWriter writer = new FileWriter("ErrorVEpochLp" + lp +".txt", true);
                BufferedWriter bufferedWriter = new BufferedWriter(writer);
                bufferedWriter.write(i + ", " + trainingRMSE + "\n");
                bufferedWriter.close();
            } catch (IOException e) {
                e.printStackTrace();
            }

            if(i % 1000 == 0) { //check to see if training should stop every 1000 epochs
                double validationTotalError = 0;
                double validationRMSE;
                for (int j = 0; j < validationSet.length; j++) {
                    double[] predictors = new double[validationSet[j].length-1];
                    for (int k = 0; k < validationSet[j].length-1; k++){
                        predictors[k] = validationSet[j][k];
                    }
                    double[][] activations = forwardPass(predictors);
                    double observedValue = validationSet[j][validationSet[j].length-1];
                    double modelledValue = activations[activations.length-1][0];
                    validationTotalError += (modelledValue-observedValue)*(modelledValue-observedValue);
                }
                validationRMSE = Math.sqrt(validationTotalError/validationSet.length); //calculate validation RMSE
                if (validationRMSE > previousValidationRMSE && trainingRMSE < previousTrainingRMSE) { //check if validation error has increased but training error hasn't (over-training)
                    System.out.println("Terminated training after " + i + " epochs as validation error began to increase");
                    this.test(testSet, testDates);
                    return;
                }
                previousValidationRMSE = validationRMSE;
                previousTrainingRMSE = trainingRMSE;
            }

        } //completed all the epochs
        System.out.println("Completed Training");
        this.test(testSet, testDates);
    }

    private void test(double[][] testSet, String[] testDates){
        double totalTestError = 0;
        for (int j = 0; j < testSet.length; j++) {
            double[] predictors = new double[testSet[j].length-1];
            for (int k = 0; k < testSet[j].length-1; k++){
                predictors[k] = testSet[j][k];
            }
            String date = testDates[j];
            double observedValue = testSet[j][testSet[j].length-1];
            double modelledValue = predict(predictors);
            totalTestError += (modelledValue-observedValue)*(modelledValue-observedValue);
            try {
                FileWriter writer = new FileWriter("ObservedVsModelled.txt", true);
                BufferedWriter bufferedWriter = new BufferedWriter(writer);
                bufferedWriter.write(observedValue + ", " + modelledValue +  "," + date + "\n");
                bufferedWriter.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        double testRMSE = Math.sqrt(totalTestError/testSet.length);
        System.out.println("Test RMSE: " + testRMSE);
        System.out.println("Finished Testing");
    }

    public double predict(double[] predictors){
        double[][] activations = forwardPass(predictors);
        return activations[activations.length-1][0];
    }

}


