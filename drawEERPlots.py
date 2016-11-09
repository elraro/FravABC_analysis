from multiprocessing.pool import Pool
import MySQLdb as Mdb
import numpy as np
import matplotlib.pyplot as plt

# Hardcoded
DB_HOST = "localhost"
DB_USER = "frav"
DB_PASS = "VXxL4UOLvB6wc01Y3Cxi"
DB_NAME = "frav_ABC"

attributes = ["ISO_19794_5_EyesGazeFrontalBestPractice", "ISO_19794_5_EyesNotRedBestPractice",
              "ISO_19794_5_EyesOpenBestPractice", "ISO_19794_5_GoodExposure", "ISO_19794_5_GoodGrayScaleProfile",
              "ISO_19794_5_GoodVerticalFacePosition", "ISO_19794_5_HasNaturalSkinColour",
              "ISO_19794_5_HorizontallyCenteredFace", "ISO_19794_5_ImageWidthToHeightBestPractice",
              "ISO_19794_5_IsBackgroundUniformBestPractice", "ISO_19794_5_IsBestPractice", "ISO_19794_5_IsCompliant",
              "ISO_19794_5_IsFrontal", "ISO_19794_5_IsFrontalBestPractice", "ISO_19794_5_IsLightingUniform",
              "ISO_19794_5_IsSharp", "ISO_19794_5_LengthOfHead", "ISO_19794_5_LengthOfHeadBestPractice",
              "ISO_19794_5_MouthClosedBestPractice", "ISO_19794_5_NoHotSpots", "ISO_19794_5_NoTintedGlasses",
              "ISO_19794_5_OnlyOneFaceVisible", "ISO_19794_5_Resolution", "ISO_19794_5_ResolutionBestPractice",
              "ISO_19794_5_WidthOfHead", "ISO_19794_5_WidthOfHeadBestPractice", "Features_Ethnicity", "Features_Gender",
              "Features_WearsGlasses"]

cameras = ["logitech", "microsoft"]

lights = ["fluorescent", "halogen", "led", "nir"]


def calculate_eer(attr):
    x = [0, 1]
    con = Mdb.connect(DB_HOST, DB_USER, DB_PASS, DB_NAME)
    cur = con.cursor()
    for camera in cameras:
        for light in lights:
            if camera == "microsoft" and light == "nir":
                continue
            plt.figure()
            for i in np.arange(2):
                cur.execute(
                    "SELECT p.clase, i.clase, s.score FROM score_data s INNER JOIN imgs_data i ON s.id_img = i.id INNER JOIN pass_data p ON s.id_pass = p.id WHERE i.camera = " + str(
                        cameras.index(camera) + 1) + " AND i.light = " + str(
                        lights.index(light) + 1) + " AND i." + attr + "=" + str(i))
                data = cur.fetchall()
                # clear data
                data = np.asarray(data)
                data = [x for x in data if x[2] >= 0]
                data = [x for x in data if x[2] <= 1]

                fn_rate = np.empty(shape=0)
                fp_rate = np.empty(shape=0)
                err_found = False
                err = 0
                for umbral in np.arange(0, 1.01, 0.01):
                    tp = 0
                    tn = 0
                    fp = 0
                    fn = 0
                    for d in data:
                        if d[2] >= umbral:
                            if d[0] == d[1]:
                                tp += 1
                            else:
                                fp += 1
                        else:
                            if d[0] == d[1]:
                                fn += 1
                            else:
                                tn += 1
                    try:
                        fn_rate_err = fn / (fn + tp)
                    except ZeroDivisionError:
                        fn_rate_err = 1
                    try:
                        fp_rate_err = fp / (fp + tn)
                    except ZeroDivisionError:
                        fp_rate_err = 1
                    fn_rate = np.append(fn_rate, fn_rate_err)
                    fp_rate = np.append(fp_rate, fp_rate_err)
                    if fn_rate_err > fp_rate_err and not err_found:
                        err = (fn_rate_err + fp_rate_err) / 2
                        err_found = True

                i_plot = 1 if i == 0 else 2
                plt.subplot(210 + i_plot)
                plt.plot(x, x, linestyle="dashed", color="red", linewidth=1)
                plt.plot(fn_rate, fp_rate, linewidth=1, color="blue", alpha=0.5,
                         label="EER=" + str(err))
                plt.xlabel("False Negative Rate")
                plt.ylabel("False Positive Rate")
                plt.legend(loc="lower right")
                plt.title(attr + "_" + str(i))
            plt.tight_layout()
            plt.savefig("EERPlots/" + attr + "_" + camera + "_" + light + ".png")
            print("Readed " + attr + " " + camera + " " + light)
            plt.close()
    con.close()

pool = Pool(20)
pool.map(calculate_eer, attributes)
pool.close()
pool.join()
