package javaapplication1.rmbkscatter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;


import javaapplication1.pyramid.GaussianPyramid;
import javaapplication1.pyramid.LaplacianPyramid;
import javaapplication1.utils.ColorBalance;
import javaapplication1.utils.ImShow;

public class MainFunction {
	
	static {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}
	// pre-set parameters
	public static final int blkSize = 10 * 10;
	public static final int patchSize = 8;
	public static final double lambda = 10;
	public static final double gamma = 1.7;
	public static final int r = 10;
	public static final double eps = 1e-6;

	public static void main(String[] args) {
		String imgPath = "F:\\image enhancement\\imgbt2.jpeg";
		Mat image = Imgcodecs.imread(imgPath, Imgcodecs.CV_LOAD_IMAGE_COLOR);
		new ImShow("original").showImage(image); // show image
		image.convertTo(image, CvType.CV_32F);
		// image decomposition
		Mat[] decomposed = ImgDecomposition.decompose(image);
		Mat AL = decomposed[0];
		Mat RL = decomposed[1];
		// For RL
		RL = ColorBalance.SimplestColorBalance(RL, 5);
		
		double fTrans = 0.6;
		// Fusion process
		// calculate the weight
		Mat w1 = calWeight(AL);
		Mat w2 = calWeight(RL);
		// Fuse
		Mat fusion = pyramidFuse(w1, w2, AL, RL, 5);
		fusion.convertTo(fusion, CvType.CV_8UC1);
		new ImShow("fusion").showImage(fusion); // show fusion result
	}
	
	public static Mat pyramidFuse(Mat w1, Mat w2, Mat img1, Mat img2, int level) {
		// Normalized weight
		Mat sumW = new Mat();
		Core.add(w1, w2, sumW);
		Core.divide(w1, sumW, w1);
		Core.multiply(w1, new Scalar(2.0), w1);
		Core.divide(w2, sumW, w2);
		Core.multiply(w2, new Scalar(2.0), w2);
		// Pyramid decomposition
		// construct the Gaussian pyramid for weight
		Mat[] weight1 = GaussianPyramid.build(w1, level);
		Mat[] weight2 = GaussianPyramid.build(w2, level);
		// construct the Laplacian pyramid for input image channel
		img1.convertTo(img1, CvType.CV_32F);
		img2.convertTo(img2, CvType.CV_32F);
		List<Mat> bgr = new ArrayList<Mat>();
		Core.split(img1, bgr);
		Mat[] bCnl1 = LaplacianPyramid.build(bgr.get(0), level);
		Mat[] gCnl1 = LaplacianPyramid.build(bgr.get(1), level);
		Mat[] rCnl1 = LaplacianPyramid.build(bgr.get(2), level);
		bgr.clear();
		Core.split(img2, bgr);
		Mat[] bCnl2 = LaplacianPyramid.build(bgr.get(0), level);
		Mat[] gCnl2 = LaplacianPyramid.build(bgr.get(1), level);
		Mat[] rCnl2 = LaplacianPyramid.build(bgr.get(2), level);
		// fusion process
		Mat[] bCnl = new Mat[level];
		Mat[] gCnl = new Mat[level];
		Mat[] rCnl = new Mat[level];
		for (int i = 0; i < level; i++) {
			Mat cn = new Mat();
			Core.add(bCnl1[i].mul(weight1[i]), bCnl2[i].mul(weight2[i]), cn);
			bCnl[i] = cn.clone();
			Core.add(gCnl1[i].mul(weight1[i]), gCnl2[i].mul(weight2[i]), cn);
			gCnl[i] = cn.clone();
			Core.add(rCnl1[i].mul(weight1[i]), rCnl2[i].mul(weight2[i]), cn);
			rCnl[i] = cn.clone();
		}
		// reconstruct & output
		Mat bChannel = LaplacianPyramid.reconstruct(bCnl);
		Mat gChannel = LaplacianPyramid.reconstruct(gCnl);
		Mat rChannel = LaplacianPyramid.reconstruct(rCnl);
		Mat fusion = new Mat();
		Core.merge(new ArrayList<Mat>(Arrays.asList(bChannel, gChannel, rChannel)), fusion);
		return fusion;
	}
	
	public static Mat dehazeProcess(Mat img, Mat trans, double[] airlight) {
		Mat balancedImg = ColorBalance.SimplestColorBalance(img, 5);
		Mat bCnl = new Mat();
		Core.extractChannel(balancedImg, bCnl, 0);
		Mat gCnl = new Mat();
		Core.extractChannel(balancedImg, gCnl, 1);
		Mat rCnl = new Mat();
		Core.extractChannel(balancedImg, rCnl, 2);
		// get mean value
		double bMean = Core.mean(bCnl).val[0];
		double gMean = Core.mean(gCnl).val[0];
		double rMean = Core.mean(rCnl).val[0];
		// get transmission map for each channel
		Mat Tb = trans.clone();
		Core.multiply(Tb, new Scalar(Math.max(bMean, Math.max(gMean, rMean)) / bMean * 0.8), Tb);
		Mat Tg = trans.clone();
		Core.multiply(Tg, new Scalar(Math.max(bMean, Math.max(gMean, rMean)) / gMean * 0.9), Tg);
		Mat Tr = trans.clone();
		Core.multiply(Tr, new Scalar(Math.max(bMean, Math.max(gMean, rMean)) / rMean * 0.8), Tr);
		// dehaze by formula
		// blue channel
		Mat bChannel = new Mat();
		Core.subtract(bCnl, new Scalar(airlight[0]), bChannel);
		Core.divide(bChannel, Tb, bChannel);
		Core.add(bChannel, new Scalar(airlight[0]), bChannel);
		// green channel
		Mat gChannel = new Mat();
		Core.subtract(gCnl, new Scalar(airlight[1]), gChannel);
		Core.divide(gChannel, Tg, gChannel);
		Core.add(gChannel, new Scalar(airlight[1]), gChannel);
		// red channel
		Mat rChannel = new Mat();
		Core.subtract(rCnl, new Scalar(airlight[2]), rChannel);
		Core.divide(rChannel, Tr, rChannel);
		Core.add(rChannel, new Scalar(airlight[2]), rChannel);
		Mat dehazed = new Mat();
		Core.merge(new ArrayList<Mat>(Arrays.asList(bChannel, gChannel, rChannel)), dehazed);
		return dehazed;
	}
	
	public static Mat calWeight(Mat img) {
		Mat L = new Mat();
		img.convertTo(img, CvType.CV_8UC1);
		Imgproc.cvtColor(img, L, Imgproc.COLOR_BGR2GRAY);
		L.convertTo(L, CvType.CV_32F);
		Core.divide(L, new Scalar(255.0), L);
		// calculate Luminance weight
		Mat WC = FeatureWeight.LuminanceWeight(img, L);
		WC.convertTo(WC, L.type());
		// calculate the Saliency weight
		Mat WS = FeatureWeight.Saliency(img);
		WS.convertTo(WS, L.type());
		// calculate the Exposedness weight
		Mat WE = FeatureWeight.Exposedness(L);
		WE.convertTo(WE, L.type());
		// sum
		Mat weight = WC.clone();
		Core.add(weight, WS, weight);
		Core.add(weight, WE, weight);
		return weight;
	}

}
