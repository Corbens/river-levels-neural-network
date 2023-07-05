public class Layer {
    private double[] weights;
    private double[] biases;
    private int numNodes;
    private int numInputs;

    public Layer(int numWeight, int numBiases, int numNodes, int numInputs) {
        double[] weights = new double[numWeight];
        double[] biases = new double[numBiases];
        double min = -2.0/numInputs;
        double max = 2.0/numInputs;
        for(int i = 0; i < numWeight; i++){
            weights[i] = min + (Math.random() * (max - min));
        }
        for(int i = 0; i < numBiases; i++){
            biases[i] = min + (Math.random() * (max - min));
        }
        this.weights = weights;
        this.biases = biases;
        this.numNodes = numNodes;
        this.numInputs = numInputs;
    }

    public double[] getWeights(){ return this.weights; }
    public double[] getBiases(){ return this.biases; }
    public int getNumNodes(){ return this.numNodes; }
    public int getNumInputs() { return this.numInputs; }

}
