package javaapplication1.pyramid;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class LaplacianPyramid {
	public static Mat[] build(Mat img, int level) {
		Mat[] lapPyr = new Mat[level];
		lapPyr[0] = img.clone();
		Mat tmpImg = img.clone();
		for (int i = 1; i < level; i++) {
			// resize image
			Imgproc.resize(tmpImg, tmpImg, new Size(), 0.5, 0.5, Imgproc.INTER_LINEAR);
			lapPyr[i] = tmpImg.clone();
		}
		// calculate the DoG
		for (int i = 0; i < level - 1; i++) {
			Mat tmpPyr = new Mat();
			Imgproc.resize(lapPyr[i + 1], tmpPyr, lapPyr[i].size(), 0, 0, Imgproc.INTER_LINEAR);
			Core.subtract(lapPyr[i], tmpPyr, lapPyr[i]);
		}
		return lapPyr;
	}

	public static Mat reconstruct(Mat[] pyramid) {
		int level = pyramid.length;
		for (int i = level - 1; i > 0; i--) {
			Mat tmpPyr = new Mat();
			Imgproc.resize(pyramid[i], tmpPyr, pyramid[i - 1].size(), 0, 0, Imgproc.INTER_LINEAR);
			Core.add(pyramid[i - 1], tmpPyr, pyramid[i - 1]);
		}
		return pyramid[0];
	}
}
