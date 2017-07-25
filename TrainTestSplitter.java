import java.util.*;

public class TrainTestSplitter {
    private int testSetSize;
    private List<Map.Entry<String, String>> sentencesWithTargets;

    public TrainTestSplitter(List<String> sentences, List<String> targets, double testRatio) {
        assert sentences.size() == targets.size();

        this.testSetSize = (int) (testRatio * sentences.size());
        this.sentencesWithTargets = new ArrayList<>();
        for (int i = 0; i < sentences.size(); i++) {
            String sentence = sentences.get(i);
            String target = targets.get(i);
            Map.Entry<String, String> sentenceWithTarget = new AbstractMap.SimpleEntry(sentence, target);
            sentencesWithTargets.add(sentenceWithTarget);
        }
        Collections.shuffle(this.sentencesWithTargets);
    }

    // Return an entry
    // Entry key - train sentences
    // Entry value - train targets
    public Map.Entry<List<String>, List<String>> trainSplit() {
        List<String> trainSentences = new ArrayList<>();
        List<String> trainTargets = new ArrayList<>();

        for (int i = testSetSize; i < sentencesWithTargets.size(); i++) {
            trainSentences.add(sentencesWithTargets.get(i).getKey());
            trainTargets.add(sentencesWithTargets.get(i).getValue());
        }
        return new AbstractMap.SimpleEntry<>(trainSentences, trainTargets);
    }

    // Return an entry
    // Entry key - test sentences
    // Entry value - test targets
    public Map.Entry<List<String>, List<String>> testSplit() {
        List<String> testSentences = new ArrayList<>();
        List<String> testTargets = new ArrayList<>();

        // - for debugging  -
        //System.out.println(sentencesWithTargets);
        // -    -   -   -   -

        for (int i = 0; i < testSetSize; i++) {
            testSentences.add(sentencesWithTargets.get(i).getKey());
            testTargets.add(sentencesWithTargets.get(i).getValue());
        }
        return new AbstractMap.SimpleEntry<>(testSentences, testTargets);
    }

    // Return an entry
    // Entry key - train sentences
    // Entry value - train targets
    protected Map.Entry<List<String>, List<String>> trainSplit(int foldIndex) {
        assert sentencesWithTargets.size() > (foldIndex + 1) * testSetSize;

        List<String> trainSentences = new ArrayList<>();
        List<String> trainTargets = new ArrayList<>();

        for (int i = 0; i < sentencesWithTargets.size(); i++) {
            if (i >= foldIndex * testSetSize && i < (foldIndex + 1) * testSetSize) {
                continue;
            }
            trainSentences.add(sentencesWithTargets.get(i).getKey());
            trainTargets.add(sentencesWithTargets.get(i).getValue());
        }
        return new AbstractMap.SimpleEntry<>(trainSentences, trainTargets);
    }

    // Return an entry
    // Entry key - test sentences
    // Entry value - test targets
    protected Map.Entry<List<String>, List<String>> testSplit(int foldIndex) {
        assert sentencesWithTargets.size() > (foldIndex + 1) * testSetSize;

        List<String> testSentences = new ArrayList<>();
        List<String> testTargets = new ArrayList<>();

        for (int i = foldIndex * testSetSize; i < (foldIndex + 1) * testSetSize; i++) {
            testSentences.add(sentencesWithTargets.get(i).getKey());
            testTargets.add(sentencesWithTargets.get(i).getValue());
        }
        return new AbstractMap.SimpleEntry<>(testSentences, testTargets);
    }
}
