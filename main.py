from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType
from numpy import array

spark = SparkSession.builder.appName('higgs').getOrCreate()

df = spark.read.csv("/home/user/HIGGS.csv",sep=",", inferSchema="true")

df.printSchema()

labelIndexer = StringIndexer(inputCol="_c0", outputCol="label").fit(df)

assembler = VectorAssembler(inputCols = ["_c1", "_c2", "_c3", "_c4","_c5", "_c6", "_c7", "_c8","_c9", "_c10", "_c11", "_c12","_c13", "_c14", "_c15", "_c16","_c17", "_c18", "_c19", "_c20","_c21", "_c22", "_c23", "_c24","_c25","_c26", "_c27", "_c28"], outputCol = "features")

(trainingData,testData) = df.randomSplit([0.7, 0.3])

#Creating model
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxDepth=5, maxBins=32)
print("Decisiontree created")

#Pipeline
pipeline = Pipeline(stages=[labelIndexer, assembler, dt])

#Fitting trainig data
model = pipeline.fit(trainingData)
print("Training Completed")

predictions = model.transform(testData)
print("Predictions Completed")

predictions.select("label", "features", 'rawPrediction', 'prediction', 'probability').show(10)

#Evaluate the model
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
accuracy = evaluator.evaluate(predictions)
print ("Test Error = %g" % (1-accuracy))

treeModel = model.stages[2]
print (treeModel)

#confusion matrix
print("===============================Start of Metrics===============================")
preds_and_labels = predictions.select(['label', 'features', 'rawPrediction', 'prediction', 'probability']).withColumn('label', F.col('label').cast(FloatType())).orderBy('prediction')
preds_and_labels = preds_and_labels.select(['prediction','label'])
metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))
print(metrics.confusionMatrix().toArray())
print("===============================End of Metrics===============================")


#Finding the best parameters for the model
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
paramGrid = (ParamGridBuilder()
             .addGrid(dt.maxDepth, [15,18,30])
             .addGrid(dt.maxBins, [32,68])
             .build())

# Run cross validations	 
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)
cvModel = cv.fit(trainingData)
#predictions = cvModel.transform(testData)
#evaluator.evaluate(predictions)

#Printing best model parametrs
model = cvModel.bestModel
java_model = model.stages[-1]._java_obj
{param.name: java_model.getOrDefault(java_model.getParam(param.name)) 
for param in paramGrid[0]}
