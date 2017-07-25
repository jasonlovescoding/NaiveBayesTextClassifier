import java.util.*;

public class NaiveBayesTextClassifier {
    // all the categories
    private Set<String> categories;
    // all the words
    private Set<String> words;
    // category -> term -> #occurrence of the term in this category
    private Map<String, Map<String, Integer>> termGivenCategoryToCount;
    // category -> #occurrence of the sentences in it
    private Map<String, Integer> categoryToSentenceCount;
    // category -> #terms in it
    private Map<String, Integer> categoryToTermCount;

    private long totalSentenceCount;

    private TermSegmentor termSegmentor;

    public NaiveBayesTextClassifier(TermSegmentor termSegmentor) {
        this.termGivenCategoryToCount = new HashMap<>();
        this.categoryToSentenceCount = new HashMap<>();
        this.categoryToTermCount = new HashMap<>();
        this.categories = new HashSet<>();
        this.words = new HashSet<>();

        this.totalSentenceCount = 0;
        this.termSegmentor = termSegmentor;
    }

    public NaiveBayesTextClassifier() {
        this.termGivenCategoryToCount = new HashMap<>();
        this.categoryToSentenceCount = new HashMap<>();
        this.categoryToTermCount = new HashMap<>();
        this.categories = new HashSet<>();
        this.words = new HashSet<>();

        this.totalSentenceCount = 0;
        this.termSegmentor = new TermSegmentor();
    }

    public void fit(List<String> sentences, List<String> targets) {
        assert sentences.size() == targets.size();

        this.totalSentenceCount = sentences.size();
        for (int i = 0; i < sentences.size(); i++) {
            List<String> terms = termSegmentor.segmentSentence(sentences.get(i));
            String category = targets.get(i);

            this.totalSentenceCount += 1;
            if (!this.categories.contains(category)) {
                // new category detected
                this.categories.add(category);
                // update termGivenCategoryToCount
                this.termGivenCategoryToCount.put(category, new HashMap<>());
                // update categoryToSentenceCount
                this.categoryToSentenceCount.put(category, 0);
                // update categoryToTermCount
                this.categoryToTermCount.put(category, 0);
            }

            for (String term : terms) {
                // - for debugging  -
                //System.out.print(term + '\t');
                // -    -   -   -   -
                if (!words.contains(term)) {
                    words.add(term);
                }
                if (!this.termGivenCategoryToCount.get(category).containsKey(term)) {
                    // new term under category found
                    this.termGivenCategoryToCount.get(category).put(term, 1);
                } else {
                    // this term has one more occurrence
                    int previousOccurrence = termGivenCategoryToCount.get(category).get(term);
                    this.termGivenCategoryToCount.get(category).put(term, previousOccurrence + 1);
                }
                this.categoryToTermCount.put(category, categoryToTermCount.get(category) + 1);
            }
            // this category has one more occurrence
            this.categoryToSentenceCount.put(category, categoryToSentenceCount.get(category) + 1);
        }
    }

    private double termGivenCategoryPosteriorProbability(String term, String category) {
        return ((double) termGivenCategoryToCount.get(category).get(term)) / categoryToTermCount.get(category);
    }

    private double categoryPriorProbability(String category) {
        return ((double) categoryToSentenceCount.get(category)) / totalSentenceCount;
    }

    // return the log-probability of a sentence
    private double sentenceLogProbability(String sentence, String category) {
        double logProbability = Math.log(categoryPriorProbability(category));
        // - for debugging  -
        //System.out.println("category prior probability:" + logProbability);
        // -    -   -   -   -
        List<String> terms = termSegmentor.segmentSentence(sentence);
        for (String term : terms) {
            double probability;
            if (!termGivenCategoryToCount.get(category).containsKey(term)) {
                // new term detected on the fly
                // smoothen it with a naive prior probability
                probability = 1.0 / (words.size() + 1);
            } else {
                probability = termGivenCategoryPosteriorProbability(term, category);
            }
            // - for debugging  -
            //System.out.println("probability:" + probability);
            // -    -   -   -   -
            logProbability += Math.log(probability);
        }
        // - for debugging  -
        //System.out.println("logProbability:" + logProbability);
        // -    -   -   -   -
        return logProbability;
    }

    public String predict(String sentence) {
        assert !categories.isEmpty();
        double maxLogProbability = -Double.MAX_VALUE;
        String bestPrediction = "";
        for (String category : categories) {
            double logProbability = sentenceLogProbability(sentence, category);
            // - for debugging  -
            //System.out.print(category + "\t" + logProbability + "\n");
            // -    -   -   -   -
            if (logProbability > maxLogProbability) {
                maxLogProbability = logProbability;
                bestPrediction = category;
            }
        }
        return bestPrediction;
    }

    public List<String> predict(List<String> sentences) {
        // - for debugging  -
        //System.out.println("sentences:" + sentences);
        // -    -   -   -   -
        List<String> predictions = new ArrayList<>();
        for (String sentence : sentences) {
            predictions.add(predict(sentence));
        }
        return predictions;
    }

    private List<Double> scoreF1(Map<String, Map<String, Integer>> confusionMap) {
        List<Double> F1s = new ArrayList<>();
        List<Double> recalls = scoreRecall(confusionMap);
        List<Double> precisions = scorePrecision(confusionMap);
        for (int i = 0; i < categories.size(); i++) {
            double recall = recalls.get(i);
            double precision = precisions.get(i);
            double F1 = 2 * recall * precision / (recall + precision);
            F1s.add(F1);
        }
        return F1s;
    }

    private List<Double> scoreRecall(Map<String, Map<String, Integer>> confusionMap) {
        List<Double> recalls = new ArrayList<>();
        for (String trueCategory : categories) {
            int truePositive = 0;
            int falseNegative = 0;
            Map<String, Integer> subConfusionMap = confusionMap.get(trueCategory);
            for (String predictedCategory : categories) {
                if (!predictedCategory.equals(trueCategory)) {
                    falseNegative += subConfusionMap.get(predictedCategory);
                } else {
                    truePositive += subConfusionMap.get(predictedCategory);
                }
            }
            recalls.add(((double) truePositive) / (truePositive + falseNegative));
        }
        return recalls;
    }

    private List<Double> scorePrecision(Map<String, Map<String, Integer>> confusionMap) {
        List<Double> precisions = new ArrayList<>();
        for (String predictedCategory : categories) {
            int truePositive = 0;
            int falsePositive = 0;
            for (String trueCategory : categories) {
                Map<String, Integer> subConfusionMap = confusionMap.get(trueCategory);
                if (!predictedCategory.equals(trueCategory)) {
                    falsePositive += subConfusionMap.get(predictedCategory);
                } else {
                    truePositive += subConfusionMap.get(predictedCategory);
                }
            }
            precisions.add(((double) truePositive) / (truePositive + falsePositive));
        }
        return precisions;
    }

    // compute the confusion matrix stored in a 2-level map format, so as to be indexed by category names
    private Map<String, Map<String, Integer>> computeConfusionMatrix(List<String> sentences, List<String> targets) {
        Map<String, Map<String, Integer>> confusionMap = new HashMap<>();
        // initialize the map (confusion matrix) of occurrences to be all zeros
        for (String trueCategory : categories) {
            Map<String, Integer> subConfusionMap = new HashMap<>();
            for (String predictedCategory : categories) {
                subConfusionMap.put(predictedCategory, 0);
            }
            confusionMap.put(trueCategory, subConfusionMap);
        }

        List<String> predictions = predict(sentences);
        for (int i = 0; i < sentences.size(); i++) {
            String trueCategory = targets.get(i);
            String predictedCategory = predictions.get(i);
            Map<String, Integer> subConfusionMap = confusionMap.get(trueCategory);
            subConfusionMap.put(predictedCategory, subConfusionMap.get(predictedCategory) + 1);
        }
        return confusionMap;
    }

    // scoreType: {"F1", "Recall", "Precision"}
    public List<Double> score(List<String> sentences, List<String> targets, String scoreType) {
        assert sentences.size() == targets.size();
        // true category -> predicted category -> #occurrence
        Map<String, Map<String, Integer>> confusionMatrix = computeConfusionMatrix(sentences, targets);

        if (scoreType.equalsIgnoreCase("F1") || scoreType.equalsIgnoreCase("F1-Score")) {
            return scoreF1(confusionMatrix);
        } else if (scoreType.equalsIgnoreCase("Recall")) {
            return scoreRecall(confusionMatrix);
        } else if (scoreType.equalsIgnoreCase("Precision")) {
            return scorePrecision(confusionMatrix);
        } else {
            System.out.println("Invalid score type. Returning precision by default.");
            return scorePrecision(confusionMatrix);
        }
    }

    public List<String> getCategories() {
        return new ArrayList<>(this.categories);
    }

    // unit test
    public static void main(String[] args) {
        List<String> sentences = new ArrayList<>();
        List<String> targets = new ArrayList<>();

        sentences.add("玉兰油");
        sentences.add("玉兰油水");
        targets.add("护肤");
        targets.add("颈部");

        NaiveBayesTextClassifier classifier = new NaiveBayesTextClassifier();
        classifier.fit(sentences, targets);
        System.out.println(classifier.predict("玉兰油水"));
    }
}
