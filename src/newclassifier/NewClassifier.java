/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package newclassifier;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSink;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;

//load data
//remove atribut
//filter:resample
//build classifier : NaiveBayes, DT
//testing model given test set
//10-fold cross validation, percentage split
//save/load model
//using model to classify one unseen data (input data)

/**
 *
 * @author Windy Amelia
 */
public class NewClassifier {

    private static Instances data;
    Classifier cls;
    
    public void readFile(String path) throws Exception {
        DataSource source = new DataSource(path);
        data = source.getDataSet();
        data.setClassIndex(data.numAttributes()-1);
    }
    
    public void removeAttribute(String[] options) throws Exception {
        Remove remove = new Remove();
        remove.setOptions(options); 
        data = Filter.useFilter(data, remove);
    }
    
    public void resample() throws Exception {
        Resample sampler = new Resample();
        //sampler.setOptions(options);
        data = Resample.useFilter(data, sampler);
    }
    
    public void setClassifierTree() throws Exception {
        cls = new J48();
        data.setClassIndex(data.numAttributes()-1);
        //cls.buildClassifier(data);
    }
    
    public void setClassifierBayes() throws Exception {
        cls = new NaiveBayes();
        data.setClassIndex(data.numAttributes()-1);
        //cls.buildClassifier(data);
    }
    
    public void crossValidation() throws Exception {
        cls.buildClassifier(data);
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(cls, data, 10, new Random(1));
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());
    }
    
    public void percentSplit(float percent) throws Exception {
        int trainSize = (int) Math.round(data.numInstances() * percent / 100);
        int testSize = data.numInstances() - trainSize;
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, testSize);
        cls.buildClassifier(train);
        
        data = new Instances(test);
        Evaluation eval = new Evaluation(data);
;       eval.evaluateModel(cls, data);
    }
    
    public void givenTestSet(String path) throws Exception {
        Instances test = DataSource.read(path);
        test.setClassIndex(test.numAttributes()-1);
        
        cls.buildClassifier(data);
        
        Evaluation eval = new Evaluation(data);
        eval.evaluateModel(cls, test);
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
	System.out.println(eval.toClassDetailsString());
	System.out.println(eval.toMatrixString());
    }
    
    public void saveModel() throws Exception {
        SerializationHelper.write("classifier.model", cls);
    }
    
    public void readModel() throws Exception {
        cls = (Classifier) SerializationHelper.read("classifier.model");
    }
    
    public void classify(String path) throws Exception {
        // load unlabeled data and set class attribute
        Instances unlabeled = DataSource.read(path);
        unlabeled.setClassIndex(unlabeled.numAttributes()-1);
        
        // copy
        Instances labeled = new Instances(unlabeled);
        
        // label instances
        for (int i = 0; i < unlabeled.numInstances(); i++) {
            double clsLabel = cls.classifyInstance(unlabeled.instance(i));
            labeled.instance(i).setClassValue(clsLabel);
        }
        
        // save labeled data
        DataSink.write("labeled.arff", labeled);
        
        // output prediction
        System.out.println("# - actual - predicted - distribution");
        for (int i = 0; i < labeled.numInstances(); i++) {
            double pred = cls.classifyInstance(labeled.instance(i));
            double[] dist = cls.distributionForInstance(labeled.instance(i));
            System.out.print((i+1) + " - ");
            System.out.print(labeled.instance(i).toString(labeled.classIndex()) + " - ");
            System.out.print(labeled.classAttribute().value((int)pred) + " - ");
            System.out.println(Utils.arrayToString(dist));
        }
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
    }
    
}
