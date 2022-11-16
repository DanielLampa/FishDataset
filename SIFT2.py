import cv2 as cv
MAX_FEATURES = 5500
GOOD_MATCH_PERCENT = 0.03

def find_matches(im1, im2, detector_type: str, descriptor_type: str):
    img1=cv.imread(im1)
    img2=cv.imread(im2)
    
    # Convert images to grayscale
    im1Gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    im2Gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # init detector
    if detector_type == 'sift':
            detector = cv.SIFT_create(MAX_FEATURES)
    elif detector_type == 'fast':
            detector = cv.FastFeatureDetector_create()
    # ... ORB etc.
    # init descriptor
    if descriptor_type == 'sift':
            descriptor = cv.SIFT_create(MAX_FEATURES)

    keypoints1 = detector.detect(im1Gray, None)
    keypoints2 = detector.detect(im2Gray, None)

    # find descriptors with descriptor SIFT
    keypoints1, descriptors1 = descriptor.compute(im1Gray, keypoints1)
    keypoints2, descriptors2 = descriptor.compute(im2Gray, keypoints2)

    # BFMatcher object
    matcher = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
    matches = list(matcher.match(descriptors1, descriptors2))

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv.imwrite("matches.jpg", imMatches)
    cv.imshow('RESULT', imMatches)
    cv.waitKey(0)



image1="Image\Tilapia.jpg"
image2="Image\Tilapia2.jpeg"
find_matches(image1,image2,'sift','sift')