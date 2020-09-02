using ImageBinarizer;
using ImageClassifierSpatialPooler;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeoCortexApi;
using NeoCortexApi.Entities;
using NeoCortexApi.Utility;
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;


namespace UnitTestsProject.ImageClassifierSpatialPoolerTest
{
    [TestClass]
    public class ImageClassifierSpatialPoolerTests
    {
        /// <summary>
        /// Below test validates the implemented method of the Image classifier in spatial pooler
        /// <h3>It takes an input from the Trained and prediction images folder and generates the output</h3>
        /// </summary>
        [TestMethod]
        public void TestMethod_ToCheckImplementationOfSpatialPooler ()
        {
            var parameters = HelperImageClassifierSpatialPooler.GetDefaultParams(); // we will first initialize the parameters

            // Input Image parameters 
            var imageWidth = 50;
            var imageHeight = 50;

            // Output images parameter
            int outputWidth = 100;
            int outputHeight = 100;


            // setting the input space dimenstions and column space dimenstions for data

            parameters.setInputDimensions(new int[] { imageWidth, imageHeight });
            parameters.setColumnDimensions(new int[] { outputWidth, outputHeight });

            // setting the area of potential connections

            parameters.setNumActiveColumnsPerInhArea(0.02 * outputWidth * outputHeight);

            var sp = new SpatialPooler();
            var mem = new Connections();
            parameters.apply(mem);
            sp.init(mem);
            
            // giving defined value to trained active array  
            int[] activeArrayTrained = new int[outputWidth * outputHeight];

            // Images path
            String[] trainingImagesPath = Directory.GetFiles(Path.Combine(Directory.GetCurrentDirectory(), "TrainedImages"), "*.jpg");
            foreach (var trainingImagePath in trainingImagesPath)
            {
                String trainingImageName = Path.GetFileName(trainingImagePath);
                Debug.WriteLine(trainingImageName);


                // ************* Training ************* //

                int[] activeArray = new int[outputWidth * outputHeight];
                int[] inputVector = ReadImageData(trainingImagePath, imageHeight, imageWidth);
                int[] newActiveArray = new int[outputWidth * outputHeight];
                sp.compute(inputVector, activeArray, true); 
                var activeCols = ArrayUtils.IndexWhere(activeArray, (el) => el == 1);
                int[] oldActiveArray = activeArray;

                int isTrained = 0;

                while (isTrained == 0)
                {
                    // here the new active array is getting updated every time till we reach distance == 0
                    sp.compute(inputVector, newActiveArray, true);
                    activeCols = ArrayUtils.IndexWhere(newActiveArray, (el) => el == 1);

                    if (GetHammingDistance(oldActiveArray, newActiveArray) == 0)
                    {
                        isTrained = 1;
                    }
                    else
                    {
                        isTrained = 0;
                        oldActiveArray = newActiveArray;
                    }
                }

                var str = Helpers.StringifyVector(activeCols);
                Debug.WriteLine(str);

                activeArrayTrained = newActiveArray;
            }

            // ************* Prediction ************* 
            String[] predictionImagesPath = Directory.GetFiles(Path.Combine(Directory.GetCurrentDirectory(), "PredictionImages"), "*.jpg");
            foreach (var predictionImagePath in predictionImagesPath)
            {
                String predictionImageName = Path.GetFileName(predictionImagePath);
                Debug.WriteLine("Name of image for Prediction : "+predictionImageName);

                //Prediction

                int[] inputVectorPrediction = ReadImageData(predictionImagePath, imageHeight, imageWidth);
                int[] activeArrayPrediction = new int[outputWidth * outputHeight];
                sp.compute(inputVectorPrediction, activeArrayPrediction, false);
                var activeColsPrediction = ArrayUtils.IndexWhere(activeArrayPrediction, (el) => el == 1);
               
                var strPrediction = Helpers.StringifyVector(activeColsPrediction);

                Debug.WriteLine("Active Column: "+strPrediction);

                double predictionPercentage = GetMatchingPercentage(activeArrayTrained, activeArrayPrediction);
                
                Debug.WriteLine("prediction percentage related to trained data:{0}", predictionPercentage);
                if (predictionPercentage >= 90.00)
                {
                    Debug.WriteLine("Result: It is apple");
                }
                else
                {
                    Debug.WriteLine("Result: It is not Apple");
                }
                Debug.WriteLine("----------------------------------------------");
            }

            
        }
        /// <summary>
        /// Returns Binarized Image in integer array and creates file of Binarized Image
        /// </summary>
        /// <param name="imagePath">Name of Image to be binarized</param>
        /// <param name="height">Height of Binarized Image</param>
        /// <param name="width">Width of Binarized Image</param>
        /// <returns></returns>

        public int[] ReadImageData(String imagePath, int height, int width)
        {
            Binarizer bizer = new Binarizer(targetHeight: height, targetWidth: width);

             bizer.CreateBinary(imagePath, Path.Combine(Path.Combine(Directory.GetCurrentDirectory(),
                 $"OutputBinarizedImages"), $"bin-{Path.GetFileName(imagePath)}.txt"));

            var binaryString = bizer.GetBinary(imagePath); // we will get binary string from it

            int[] intArray = new int[height * width]; // we will get the int array from the string
            int j = 0;
            for (int i = 0; i < binaryString.Length; i++)
            {
                if (binaryString[i].Equals('0'))
                {
                    intArray[j] = 0;
                    j++;
                }
                else if (binaryString[i].Equals('1'))
                {
                    intArray[j] = 1;
                    j++;
                }
            }
            return intArray;
        }

        /// <summary>
        /// Returns Hamming Distance between two Arrays
        /// </summary>
        /// <param name="referenceArray">Array from which we have to compare</param>
        /// <param name="givenArray">Array which is given</param>
        /// <returns></returns>

        public int GetHammingDistance(int[] referenceArray, int[] givenArray)
        {
            int unmatchedIndex = 0;

            for (int i = 0; i < referenceArray.Length; i++)
            {
                if (referenceArray[i] != givenArray[i])
                {
                    unmatchedIndex++;
                }
            }

            return unmatchedIndex;
        }

        /// <summary>
        /// Returns percentage of Matching indexes
        /// </summary>
        /// <param name="trainedArray">Active array produced by training</param>
        /// <param name="predictionArray">Active array produced by prediction</param>
        /// <returns></returns>

        public double GetMatchingPercentage(int[] trainedArray, int[] predictionArray)
        {
            var activeColsTrained = ArrayUtils.IndexWhere(trainedArray, (el) => el == 1);

            var activeColsPredicted = ArrayUtils.IndexWhere(predictionArray, (el) => el == 1);

            var intersection = activeColsTrained.Intersect(activeColsPredicted).ToArray();

            double percentage = (double)intersection.Length * 100 / activeColsPredicted.Length;

            return percentage;
        }
    }
}
