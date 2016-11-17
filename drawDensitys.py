from multiprocessing.pool import Pool

import MySQLdb as Mdb
import numpy as np
import matplotlib.pyplot as plt

# Hardcoded
DB_HOST = "localhost"
DB_USER = "frav"
DB_PASS = "VXxL4UOLvB6wc01Y3Cxi"
DB_NAME = "frav_ABC"

attributes = ["locateFace", "locateEyes", "backgroundUniformity", "isColor",
              "ISO_19794_5_EyesGazeFrontalBestPractice", "ISO_19794_5_EyesNotRedBestPractice",
              "ISO_19794_5_EyesOpenBestPractice", "ISO_19794_5_GoodExposure", "ISO_19794_5_GoodGrayScaleProfile",
              "ISO_19794_5_GoodVerticalFacePosition", "ISO_19794_5_HasNaturalSkinColour",
              "ISO_19794_5_HorizontallyCenteredFace", "ISO_19794_5_ImageWidthToHeightBestPractice",
              "ISO_19794_5_IsBackgroundUniformBestPractice", "ISO_19794_5_IsBestPractice",
              "ISO_19794_5_IsCompliant", "ISO_19794_5_IsFrontal", "ISO_19794_5_IsFrontalBestPractice",
              "ISO_19794_5_IsLightingUniform", "ISO_19794_5_IsSharp", "ISO_19794_5_LengthOfHead",
              "ISO_19794_5_LengthOfHeadBestPractice", "ISO_19794_5_MouthClosedBestPractice",
              "ISO_19794_5_NoHotSpots", "ISO_19794_5_NoTintedGlasses", "ISO_19794_5_OnlyOneFaceVisible",
              "ISO_19794_5_Resolution", "ISO_19794_5_ResolutionBestPractice", "ISO_19794_5_WidthOfHead",
              "ISO_19794_5_WidthOfHeadBestPractice", "Features_Ethnicity", "Features_Gender",
              "Features_WearsGlasses", "faceConfidence", "eye0Confidence", "eye1Confidence", "age", "chin", "crown",
              "deviationFromFrontalPose", "deviationFromUniformLighting", "ear0", "ear1", "ethnicityAsian",
              "ethnicityBlack", "ethnicityWhite", "exposure", "eye0X", "eye0Y", "eye0GazeFrontal", "eye0Open",
              "eye0Red", "eye0Tinted", "eye1X", "eye1Y", "eye1GazeFrontal", "eye1Open", "eye1Red", "eye1Tinted",
              "eyeDistance", "faceCenterX", "faceCenterY", "glasses", "grayScaleDensity", "height", "hotSpots",
              "isMale", "lengthOfHead", "mouthClosed", "naturalSkinColour", "numberOfFaces", "poseAngleRoll",
              "sharpness", "width", "widthOfHead", "camera", "light"]


def draw_density(attr):
    con = Mdb.connect(DB_HOST, DB_USER, DB_PASS, DB_NAME)
    cur = con.cursor()
    plt.figure()
    cur.execute("SELECT i." + attr + " FROM score_data s INNER JOIN imgs_data i ON s.id_img=i.id WHERE s.score >= 0 AND s.score <= 1 AND i.locateFace = 1 AND i.eye0Confidence >= 0 AND i.eye0Confidence <= 6 AND i.eye1Confidence >= 0 AND i.eye1Confidence <= 6 AND i.faceConfidence >= 0 AND i.faceConfidence <= 6 AND i.numberOfFaces = 1")
    data = cur.fetchall()
    data = np.asarray(data)
    plt.hist(data, 20, histtype='stepfilled', facecolor='g', alpha=0.75)
    plt.title(attr)
    plt.savefig("density/" + attr + ".png")
    print("Readed " + attr)
    plt.close()
    con.close()


pool = Pool(20)
pool.map(draw_density, attributes)
pool.close()
pool.join()

