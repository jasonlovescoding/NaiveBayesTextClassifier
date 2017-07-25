import org.ansj.domain.Term;
import org.ansj.splitWord.analysis.NlpAnalysis;
import org.ansj.splitWord.analysis.ToAnalysis;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.ArrayList;

// Overview:
// given a set of stop words,
// a instance of the word segmentor takes input of a list of strings,
// and for each string, produces output of a list of segmented terms
public class TermSegmentor {
    // stop words
    private Set<String> stopWords;
    private static final String ENCODING = "UTF-8";
    private static final String STOP_WORDS_PATH = ".\\data\\stopWords.txt";

    public TermSegmentor(Set<String> stopWords) {
        this.stopWords = new HashSet<String>(stopWords);
    }

    public TermSegmentor() {
        try {
            this.stopWords = new HashSet<>();
            File stopWordsFile = new File(STOP_WORDS_PATH);
            if (stopWordsFile.isFile() && stopWordsFile.exists()) {
                InputStreamReader inputStreamReader = new InputStreamReader(new FileInputStream(stopWordsFile), ENCODING);
                BufferedReader bufferedReader = new BufferedReader(inputStreamReader);
                String line;
                while ((line = bufferedReader.readLine()) != null) {
                    stopWords.add(line);
                }
                this.stopWords.add("");
                this.stopWords.add(" ");
                this.stopWords.add("\r");
                this.stopWords.add("\n");
                this.stopWords.add("\r\n");
                this.stopWords.add("nbsp");
                this.stopWords.add("\u0000");
                bufferedReader.close();
            }
        } catch (Exception e) {
            System.out.println("failed to read stop words");
        }
    }

    private List<String> filterStopWords(List<String> words) {
        List<String> filteredWords = new ArrayList<>();
        for (String word : words) {
            if (!stopWords.contains(word)) {
                filteredWords.add(word);
            }
        }
        return filteredWords;
    }

    public List<String> segmentSentence(String sentence) {
        List<String> terms = new ArrayList<>();
        try {
            List<Term> tokens = NlpAnalysis.parse(sentence).getTerms();
            for (Term term : tokens) {
                terms.add(term.getName());
            }
        } catch (Exception e) {
            System.out.println("segment failure");
        }
        return filterStopWords(terms);
    }

    public List<List<String>> segmentSentences(List<String> sentences) {
        List<List<String>> termsList = new ArrayList<>();
        for (String sentence : sentences) {
            List<String> terms = segmentSentence(sentence);
            termsList.add(terms);
        }
        return termsList;
    }
}
