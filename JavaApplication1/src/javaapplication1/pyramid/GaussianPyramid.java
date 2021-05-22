package javaapplication1.pyramid;

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class GaussianPyramid {
	
	public static Mat[] build(Mat img, int level) {
		Mat[] gaussPyr = new Mat[level];
		Mat mask = filterMask(img);
		Mat tmp = new Mat();
		Imgproc.filter2D(img, tmp, -1, mask);
		gaussPyr[0] = tmp.clone();
		Mat tmpImg = img.clone();
		for (int i = 1; i < level; i++) {
			// resize image
			Imgproc.resize(tmpImg, tmpImg, new Size(), 0.5, 0.5, Imgproc.INTER_LINEAR);
			// blur image
			tmp = new Mat();
			Imgproc.filter2D(tmpImg, tmp, -1, mask);
			gaussPyr[i] = tmp.clone();
		}
		return gaussPyr;
	}
	
	private static Mat filterMask(Mat img) {
		double[] h = { 1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0 };
		Mat mask = new Mat(h.length, h.length, img.type());
		for (int i = 0; i < h.length; i++) {
			for (int j = 0; j < h.length; j++) {
				mask.put(i, j, h[i] * h[j]);
			}
		}
		return mask;
	}
	
}
