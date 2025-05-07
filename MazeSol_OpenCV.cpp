#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main() {
	Mat image = imread("TumorT1.jpg");

	if (image.empty()) {
		cerr << "Could not load the image!" << endl;
		return -1;
	}

	Mat resized;
	resize(image, resized, Size(512, 350));

	Mat gray;
	cvtColor(resized, gray, COLOR_BGR2GRAY);

	Mat blurred;
	GaussianBlur(gray, blurred, Size(5, 5), 0);

	Mat sharpened;
	Mat kernel = (Mat_<float>(3, 3) <<
		0, -1, 0,
		-1, 5, -1,
		0, -1, 0);
	filter2D(blurred, sharpened, blurred.depth(), kernel);

	int cropWidth = 80;
	int cropHeight = 80;
	int centerX = gray.cols / 2 - cropWidth / 2;
	int centerY = gray.rows / 2 - cropHeight / 2;
	Rect centralRegion(centerX, centerY, cropWidth, cropHeight);
	Mat csfRegion = sharpened(centralRegion);

	Scalar meanIntensity = mean(csfRegion);
	Scalar overallMean = mean(sharpened);
	double brightnessDiff = overallMean[0] - meanIntensity[0];

	int windowSize = 30;
	int stepSize = 10;
	int regionWidth = 120;
	int regionHeight = 120;
	int regionStartX = gray.cols / 2 - regionWidth / 2;
	int regionStartY = gray.rows / 2 - regionHeight / 2;

	double minPatchIntensity = 255;
	double maxPatchIntensity = 0;
	Point bestDarkSeed(-1, -1);
	Point bestBrightSeed(-1, -1);

	if (meanIntensity[0] < 130) {
		if (brightnessDiff > 20) {

			for (int y = regionStartY; y <= regionStartY + regionHeight - 70; y += stepSize) {
				for (int x = regionStartX; x <= regionStartX + regionWidth - 70; x += stepSize) {
					Rect window(x, y, 70, 70);

					if (x + 70 <= sharpened.cols && y + 70 <= sharpened.rows) {
						Mat patch = sharpened(window);
						double intensity = mean(patch)[0];
						if (intensity > maxPatchIntensity) {
							maxPatchIntensity = intensity;
							bestBrightSeed = Point(x + 35, y + 35);
						}
					}
				}
			}
		}
		else {
			for (int y = regionStartY; y <= regionStartY + regionHeight - 70; y += stepSize) {
				for (int x = regionStartX; x <= regionStartX + regionWidth - 70; x += stepSize) {
					Rect window(x, y, 70, 70);

					if (x + 70 <= sharpened.cols && y + 70 <= sharpened.rows) {
						Mat patch = sharpened(window);
						double intensity = mean(patch)[0];
						if (intensity < minPatchIntensity) {
							minPatchIntensity = intensity;
							bestDarkSeed = Point(x + 35, y + 35);
						}
					}
				}
			}
		}
	}
	else if (meanIntensity[0] > 115) {
		if (brightnessDiff > -20) {

			for (int y = regionStartY; y <= regionStartY + regionHeight - windowSize; y += stepSize) {
				for (int x = regionStartX; x <= regionStartX + regionWidth - windowSize; x += stepSize) {
					Rect window(x, y, windowSize, windowSize);

					if (x + windowSize <= sharpened.cols && y + windowSize <= sharpened.rows) {
						Mat patch = sharpened(window);
						double intensity = mean(patch)[0];
						if (intensity < minPatchIntensity) {
							minPatchIntensity = intensity;
							bestDarkSeed = Point(x + windowSize / 2, y + windowSize / 2);
						}
					}
				}
			}
		}
		else {

			int patchWidth = 70;
			int patchHeight = 70;

			for (int y = regionStartY; y <= regionStartY + regionHeight - patchHeight; y += stepSize) {
				for (int x = regionStartX; x <= regionStartX + regionWidth - patchWidth; x += stepSize) {
					Rect window(x, y, patchWidth, patchHeight);

					if (x + patchWidth <= sharpened.cols && y + patchHeight <= sharpened.rows) {
						Mat patch = sharpened(window);
						double intensity = mean(patch)[0];
						if (intensity > maxPatchIntensity) {
							maxPatchIntensity = intensity;
							bestBrightSeed = Point(x + patchWidth / 2, y + patchHeight / 2);
						}
					}
				}
			}
		}
	}

	Mat resultImage = resized.clone();
	if (bestDarkSeed.x != -1)
		circle(resultImage, bestDarkSeed, 30, Scalar(0, 0, 255), 2);
	if (bestBrightSeed.x != -1)
		circle(resultImage, bestBrightSeed, 30, Scalar(255, 0, 0), 2);

	imshow("Detected Tumor Region", resultImage);
	waitKey(0);
	return 0;
}