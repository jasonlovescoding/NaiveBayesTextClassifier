import java.util.*;

public class CrossValidator {
    private TrainTestSplitter trainTestSplitter;
    private int fold;

    public CrossValidator(List<String> sentences, List<String> targets, int fold) {
        assert sentences.size() == targets.size();
        this.trainTestSplitter = new TrainTestSplitter(sentences, targets, 1.0 / fold);
        this.fold = fold;
    }

    // scoreType: {"F1", "Recall", "Precision"}
    public List<List<Double>> crossValidate(String scoreType) {
        List<List<Double>> validationScores = new ArrayList<>();

        for (int i = 0; i < fold; i++) {
            NaiveBayesTextClassifier classifier = new NaiveBayesTextClassifier();

            Map.Entry<List<String>, List<String>>  trainSplit = trainTestSplitter.trainSplit(i);
            List<String> trainSentences = trainSplit.getKey();
            List<String> trainTargets = trainSplit.getValue();
            classifier.fit(trainSentences, trainTargets);

            Map.Entry<List<String>, List<String>>  testSplit = trainTestSplitter.testSplit(i);
            List<String> testSentences = testSplit.getKey();
            List<String> testTargets = testSplit.getValue();
            List<Double> scores = classifier.score(testSentences, testTargets, scoreType);

            validationScores.add(scores);
        }

        return validationScores;
    }

}
