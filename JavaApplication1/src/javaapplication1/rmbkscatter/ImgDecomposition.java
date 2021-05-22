package javaapplication1.rmbkscatter;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

public class ImgDecomposition {
	
	public static Mat[] decompose(Mat img) {
		List<Mat> AList = new ArrayList<Mat>();
		List<Mat> RList = new ArrayList<Mat>();
		List<Mat> bgr = new ArrayList<Mat>();
		Core.split(img, bgr);
		for (Mat cnl : bgr) {
			Mat alCnl = cnl.clone();
			Mat rlcnl = cnl.clone();
			double maxVal = Core.minMaxLoc(cnl).maxVal;
			Mat k = new Mat();
			Core.multiply(cnl, new Scalar(0.5 / maxVal), k);
			rlcnl = k.mul(rlcnl);
			Core.subtract(alCnl, rlcnl, alCnl);
			AList.add(alCnl);
			RList.add(rlcnl);
		}
		Mat Al = new Mat();
		Core.merge(AList, Al);
		Mat Rl = new Mat();
		Core.merge(RList, Rl);
		Mat[] result = {Al, Rl};
		return result;
	}
	
}
