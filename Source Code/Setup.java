import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class Setup {
    public static void main(String[] args) throws IOException {
        //read data from relevant text files and put into appropriate sets, change as appropriate to fit your data set
        double[][] trainingSet = new double[874][8];
        double[][] validationSet = new double[292][8];
        double[][] testSet = new double[292][8];
        String[] testDates = new String[292];
        BufferedReader br = new BufferedReader(new FileReader("Train&ValidationData.txt"));
        String line = null;
        int count = 0;
        while((line = br.readLine()) != null){
            String[] data = line.split(",");
            double row[] = {Double.parseDouble(data[0]),Double.parseDouble(data[1]),Double.parseDouble(data[2]),Double.parseDouble(data[3]),
                    Double.parseDouble(data[4]),Double.parseDouble(data[5]),Double.parseDouble(data[6]),Double.parseDouble(data[7])};
            if(count < 874){
                trainingSet[count] = row;
            }else{
                validationSet[count-874] = row;
            }
            count++;
        }
        br.close();
        br = new BufferedReader(new FileReader("TestData.txt"));
        count = 0;
        while((line = br.readLine()) != null){
            String[] data = line.split(",");
            double row[] = {Double.parseDouble(data[0]),Double.parseDouble(data[1]),Double.parseDouble(data[2]),Double.parseDouble(data[3]),
                    Double.parseDouble(data[4]),Double.parseDouble(data[5]),Double.parseDouble(data[6]),Double.parseDouble(data[7])};
            String date = data[8];
            testSet[count] = row;
            testDates[count] = date;
            count++;
        }
        br.close();

        /*an array with element 0 containing the amount of inputs, elements 1 to n-1 containing the amount
        of neurons in that respective hidden layer and element n containing the amount of output neurons*/
        int[] layerSizes = {7,8,1};
        ANN ann = new ANN(layerSizes); //create an ann
        ann.train(20000, 0.5, trainingSet, validationSet, testSet, testDates); //train the ann
        double[] x = {0.154107,0.148732,0.124928,0.164312,0.145753,0.129744,0.11042}; //some predictors
        System.out.println(ann.predict(x)[0]); //the prediction
    }

}
