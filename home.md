

### Problem Statement

 The goal of this exercise is to develop a fuzzy record matching program that takes records from “query” set and finds matching record in the “reference” set. 

** Data: ** In the attached excel file, there are two sheets:
 * “reference” sheet  contains a list of restaurants, its address and cuisine (“reference” set) 
 * “query” sheet contains again a list of restaurants, its address and cuisine, plus two additional columns (“query” set)
     * “reference_id” column needs to be populated using the fuzzy record matching program
     * “score(optional)” field can show to what degree the record matches or the confidence of the match. It is optional though
     
### Solution
Two solutions are explored for fuzzy match,
 * For baseline, 'fuzzywuzzy' python package is used to measure the parameters.
 * TD-IDF based approach is explored and compared with the baseline.

Demo contains two pages which can be selected from sidebar, 
  * Run Solution : Run end-to-end solution.
  * Parameter Fine-tuning : Interactive td-idf method's threshold fine-tuning.
 
 
 
*Note: Code is not optimised. Mostly copied from notebook file.*