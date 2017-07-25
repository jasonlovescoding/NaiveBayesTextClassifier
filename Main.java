import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.*;

// top-level unit, read the data given a path and use it for training & testing
public class Main {
    private static String ENCODING = "UTF-8";
    private static String DATA_PATH = ".\\data\\rawRequests.txt";

    public static Map.Entry<List<String>, List<String>> readSentencesWithTargets() {
        List<String> sentences = new ArrayList<>();
        List<String> targets = new ArrayList<>();
        try {
            File dataFile = new File(DATA_PATH);
            if (dataFile.isFile() && dataFile.exists()) {
                InputStreamReader inputStreamReader = new InputStreamReader(new FileInputStream(dataFile), ENCODING);
                BufferedReader bufferedReader = new BufferedReader(inputStreamReader);
                String line;
                while ((line = bufferedReader.readLine()) != null) {
                    String[] tokens = line.split(" \\|\\|\\| ");
                    sentences.add(tokens[0]);
                    targets.add(tokens[1]);
                    // - for debugging  -
                    //System.out.println(tokens[0]);
                    //System.out.println(tokens[1]);
                    // -    -   -   -   -
                }
                bufferedReader.close();
            }
        } catch (Exception e) {
            System.out.println("failed to read data");
        }
        return new AbstractMap.SimpleEntry<>(sentences, targets);
    }

    // scoreType: {"F1", "Recall", "Precision"}
    public static List<List<Double>> crossValidate(int fold, String scoreType) {
        Map.Entry<List<String>, List<String>> sentencesWithTargets = readSentencesWithTargets();
        List<String> sentences = sentencesWithTargets.getKey();
        List<String> targets = sentencesWithTargets.getValue();
        CrossValidator crossValidator = new CrossValidator(sentences, targets, fold);
        List<List<Double>> scoreLists = crossValidator.crossValidate(scoreType);
        for (int i = 0; i < scoreLists.size(); i++) {
            List<Double> scores = scoreLists.get(i);
            for (int j = 0; j < scores.size(); j++) {
                System.out.print(String.format("%.2f\t", scores.get(j)));
            }
            System.out.print("\n");
        }
        return scoreLists;
    }

    // do a simple train-test-split trial
    public static List<Double> trainTestTrial(double testRatio, String scoreType) {
        Map.Entry<List<String>, List<String>> sentencesWithTargets = readSentencesWithTargets();
        List<String> sentences = sentencesWithTargets.getKey();
        List<String> targets = sentencesWithTargets.getValue();
        TrainTestSplitter trainTestSplitter = new TrainTestSplitter(sentences, targets, testRatio);
        NaiveBayesTextClassifier classifier = new NaiveBayesTextClassifier();

        Map.Entry<List<String>, List<String>> trainSet = trainTestSplitter.trainSplit();
        List<String> trainSentences = trainSet.getKey();
        List<String> trainTargets = trainSet.getValue();
        classifier.fit(trainSentences, trainTargets);

        Map.Entry<List<String>, List<String>> testSet = trainTestSplitter.testSplit();
        List<String> testSentences = testSet.getKey();
        List<String> testTargets = testSet.getValue();
        List<Double> scores = classifier.score(testSentences, testTargets, scoreType);
        for (int i = 0; i < scores.size(); i++) {
            System.out.print(String.format("%.2f\t", scores.get(i)));
        }
        System.out.print("\n");
        return scores;
    }

    // do a simple train-and-predict trial
    public static void trainPredictTrial(double testRatio) {
        Map.Entry<List<String>, List<String>> sentencesWithTargets = readSentencesWithTargets();
        List<String> sentences = sentencesWithTargets.getKey();
        List<String> targets = sentencesWithTargets.getValue();
        TrainTestSplitter trainTestSplitter = new TrainTestSplitter(sentences, targets, testRatio);
        NaiveBayesTextClassifier classifier = new NaiveBayesTextClassifier();

        Map.Entry<List<String>, List<String>> trainSet = trainTestSplitter.trainSplit();
        List<String> trainSentences = trainSet.getKey();
        List<String> trainTargets = trainSet.getValue();
        classifier.fit(trainSentences, trainTargets);

        Map.Entry<List<String>, List<String>> testSet = trainTestSplitter.testSplit();
        List<String> testSentences = testSet.getKey();
        List<String> testTargets = testSet.getValue();
        List<String> predictions = classifier.predict(testSentences);
        for (int i = 0; i < predictions.size(); i++) {
            String trueCategory = testTargets.get(i);
            String predictedCategory = predictions.get(i);
            if (trueCategory.equals(predictedCategory)) System.out.print("√\t");
            else System.out.print("X\t");
            System.out.println(String.format("True: %s\tPredicted:%s", trueCategory, predictedCategory));
        }
        System.out.print("\n");
    }

    // do a combined trail of precision, recall and F1
    public static void combinedTrainTestTrial(double testRatio) {
        List<Double> precisions = trainTestTrial(testRatio, "precision");
        double averagePrecision = 0;
        for (double precision : precisions) {
            averagePrecision += precision;
        }
        averagePrecision /= precisions.size();
        System.out.println("Average precision: " + averagePrecision);

        List<Double> recalls = trainTestTrial(testRatio, "recall");
        double averagerecall = 0;
        for (double recall : recalls) {
            averagerecall += recall;
        }
        averagerecall /= recalls.size();
        System.out.println("Average recall: " + averagerecall);

        List<Double> F1s = trainTestTrial(testRatio, "F1");
        double averageF1 = 0;
        for (double F1 : F1s) {
            averageF1 += F1;
        }
        averageF1 /= F1s.size();
        System.out.println("Average F1: " + averageF1);
    }

    public static void main(String[] args) {
        // assign the path to the file containing the data
        // each line of the data must be in the form of "<sentence> ||| <category>"
        // e.g. "玉兰油 ||| 护肤"
        DATA_PATH = ".\\data\\rawRequests.txt";

        // 3 means 3-fold cross-validation
        //crossValidate(3, "Precision");

        // 0.2 means 20% of the data is used for testing
        //combinedTrainTestTrial(0.2);

        // 0.2 means 20% of the data is used for testing
        //trainTestTrial(0.2, "Precision");

        // 0.2 means 20% of the data is used for testing
        trainPredictTrial(0.2);
    }

}
