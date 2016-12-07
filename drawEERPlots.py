import threading
from multiprocessing.pool import Pool
import MySQLdb as Mdb
import numpy as np
import matplotlib.pyplot as plt

lock = threading.Lock()

# Hardcoded
DB_HOST = "localhost"
DB_USER = "frav"
DB_PASS = "VXxL4UOLvB6wc01Y3Cxi"
DB_NAME = "frav_ABC"

# 0 or 1
binary_attributes = ["locateFace", "locateEyes", "backgroundUniformity", "isColor",
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
                     "ISO_19794_5_WidthOfHeadBestPractice", "Features_Gender",
                     "Features_WearsGlasses", "numberOfFaces"]

standard_deviation_mean = ["faceConfidence", "eye0Confidence", "eye1Confidence", "chin", "crown",
                           "deviationFromFrontalPose", "deviationFromUniformLighting", "ear0", "ear1", "exposure",
                           "eye0GazeFrontal", "eye0Open", "eye0Tinted", "eye1GazeFrontal", "eye1Open", "eye1Tinted",
                           "eyeDistance", "faceCenterX", "faceCenterY", "grayScaleDensity", "lengthOfHead",
                           "poseAngleRoll", "widthOfHead"]

# umbral a partir de la media
umbral_mean = ["age", "eye0X", "eye0Y", "eye1X", "eye1Y"]

# umbral <0, >0
umbral_0 = ["ethnicityAsian", "ethnicityBlack", "ethnicityWhite", "glasses", "isMale", "mouthClosed"]

# umbral <0.5, >0.5
umbral_05 = ["naturalSkinColour", "sharpness"]

# umbral 0, resto
umbral_0_rest = ["eye0Red", "eye1Red", "hotSpots"]

rest = [("camera", [1, 2]), ("light", [1, 2, 3, 4]), ("Features_Ethnicity", [0, 1, 2]), ("height", [720, 1920]),
        ("width", [960, 1080])]


def calculate_eer_mean(attr):
    x = [0, 1]
    con = Mdb.connect(DB_HOST, DB_USER, DB_PASS, DB_NAME)
    cur = con.cursor()

    # primero vamos a calcular la media
    cur.execute(
        "SELECT AVG(i." + attr + ") FROM score_data s INNER JOIN imgs_data i ON s.id_img = i.id INNER JOIN pass_data p ON s.id_pass = p.id AND s.score >= 0 AND s.score <= 1 AND i.locateFace = 1 AND i.eye0Confidence >= 0 AND i.eye1Confidence >= 0 AND i.faceConfidence >= 0 AND i.numberOfFaces = 1")
    data = cur.fetchall()
    data = np.asarray(data)
    average = str(data[0][0])
    formules = ["<", ">"]
    plt.figure()
    for f in formules:
        cur.execute(
            "SELECT p.clase, i.clase, s.score FROM score_data s INNER JOIN imgs_data i ON s.id_img = i.id INNER JOIN pass_data p ON s.id_pass = p.id WHERE i." + attr + f + str(
                average) + " AND s.score >= 0 AND s.score <= 1 AND i.locateFace = 1 AND i.eye0Confidence >= 0 AND i.eye1Confidence >= 0 AND i.faceConfidence >= 0 AND i.numberOfFaces = 1 ")
        data = cur.fetchall()
        data = np.asarray(data)

        fn_rate = np.empty(shape=0)
        fp_rate = np.empty(shape=0)
        eer_found = False
        eer = 0
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
                fn_rate_eer = fn / (fn + tp)
            except ZeroDivisionError:
                fn_rate_eer = 1
            try:
                fp_rate_eer = fp / (fp + tn)
            except ZeroDivisionError:
                fp_rate_eer = 1
            fn_rate = np.append(fn_rate, fn_rate_eer)
            fp_rate = np.append(fp_rate, fp_rate_eer)
            if fn_rate_eer > fp_rate_eer and not eer_found:
                eer = (fn_rate_eer + fp_rate_eer) / 2
                eer_found = True

        i_plot = 1 if f == "<" else 2
        plt.subplot(210 + i_plot)
        plt.plot(x, x, linestyle="dashed", color="red", linewidth=1)
        plt.plot(fn_rate, fp_rate, linewidth=1, color="blue", alpha=0.5,
                 label="EER=" + str(eer))
        plt.xlabel("False Negative Rate")
        plt.ylabel("False Positive Rate")
        plt.legend(loc="lower right")
        if f == "<":
            plt.title(attr + " < " + str(average))
        else:
            plt.title(attr + " > " + str(average))
    plt.tight_layout()
    plt.savefig("EERPlots/" + attr + "_umbral_mean.png")
    print("Readed " + attr)
    plt.close()
    con.close()


def calculate_eer_binary(attr):
    x = [0, 1]
    con = Mdb.connect(DB_HOST, DB_USER, DB_PASS, DB_NAME)
    cur = con.cursor()

    plt.figure()
    for v in x:
        cur.execute(
            "SELECT p.clase, i.clase, s.score FROM score_data s INNER JOIN imgs_data i ON s.id_img = i.id INNER JOIN pass_data p ON s.id_pass = p.id WHERE i." + attr + "=" + str(
                v) + " AND s.score >= 0 AND s.score <= 1 AND i.locateFace = 1 AND i.eye0Confidence >= 0 AND i.eye1Confidence >= 0 AND i.faceConfidence >= 0 AND i.numberOfFaces = 1 ")
        data = cur.fetchall()
        data = np.asarray(data)

        fn_rate = np.empty(shape=0)
        fp_rate = np.empty(shape=0)
        eer_found = False
        eer = 0
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
                fn_rate_eer = fn / (fn + tp)
            except ZeroDivisionError:
                fn_rate_eer = 1
            try:
                fp_rate_eer = fp / (fp + tn)
            except ZeroDivisionError:
                fp_rate_eer = 1
            fn_rate = np.append(fn_rate, fn_rate_eer)
            fp_rate = np.append(fp_rate, fp_rate_eer)
            if fn_rate_eer > fp_rate_eer and not eer_found:
                eer = (fn_rate_eer + fp_rate_eer) / 2
                eer_found = True

        i_plot = 1 if v == 0 else 2
        plt.subplot(210 + i_plot)
        plt.plot(x, x, linestyle="dashed", color="red", linewidth=1)
        plt.plot(fn_rate, fp_rate, linewidth=1, color="blue", alpha=0.5,
                 label="EER=" + str(eer))
        plt.xlabel("False Negative Rate")
        plt.ylabel("False Positive Rate")
        plt.legend(loc="lower right")
        plt.title(attr + " = " + str(v))
    plt.tight_layout()
    plt.savefig("EERPlots/" + attr + "_binary.png")
    print("Readed " + attr)
    plt.close()
    con.close()


def calculate_eer_standard_desviation_mean(attr):
    x = [0, 1]
    con = Mdb.connect(DB_HOST, DB_USER, DB_PASS, DB_NAME)
    cur = con.cursor()

    # primero vamos a calcular la media y la desviacion tipica
    cur.execute(
        "SELECT i." + attr + " FROM score_data s INNER JOIN imgs_data i ON s.id_img = i.id INNER JOIN pass_data p ON s.id_pass = p.id AND s.score >= 0 AND s.score <= 1 AND i.locateFace = 1 AND i.eye0Confidence >= 0 AND i.eye1Confidence >= 0 AND i.faceConfidence >= 0 AND i.numberOfFaces = 1")
    data = cur.fetchall()
    data = np.asarray(data)
    mean = np.mean(data)
    deviation = np.std(data)
    top = mean + deviation
    down = mean - deviation
    lock.acquire()
    print("Media y desviacion tipica")
    print(attr)
    print(mean)
    print(deviation)
    print(top)
    print(down)
    print("------")
    lock.release()
    formules = [[">", "<"], ["<", ">"]]
    plt.figure()
    for f in formules:
        if f[0] == ">":
            cur.execute(
                "SELECT p.clase, i.clase, s.score FROM score_data s INNER JOIN imgs_data i ON s.id_img = i.id INNER JOIN pass_data p ON s.id_pass = p.id WHERE i." + attr +
                f[0] + str(
                    down) + " AND i." + attr + f[1] + str(
                    top) + "AND s.score >= 0 AND s.score <= 1 AND i.locateFace = 1 AND i.eye0Confidence >= 0 AND i.eye1Confidence >= 0 AND i.faceConfidence >= 0 AND i.numberOfFaces = 1 ")
            data = cur.fetchall()
            data = np.asarray(data)
        else:
            cur.execute(
                "SELECT p.clase, i.clase, s.score FROM score_data s INNER JOIN imgs_data i ON s.id_img = i.id INNER JOIN pass_data p ON s.id_pass = p.id WHERE i." + attr +
                f[0] + str(
                    down) + "AND s.score >= 0 AND s.score <= 1 AND i.locateFace = 1 AND i.eye0Confidence >= 0 AND i.eye1Confidence >= 0 AND i.faceConfidence >= 0 AND i.numberOfFaces = 1 ")
            data_aux = cur.fetchall()
            data_aux = np.asarray(data_aux)
            cur.execute(
                "SELECT p.clase, i.clase, s.score FROM score_data s INNER JOIN imgs_data i ON s.id_img = i.id INNER JOIN pass_data p ON s.id_pass = p.id WHERE i." + attr +
                f[1] + str(
                    top) + "AND s.score >= 0 AND s.score <= 1 AND i.locateFace = 1 AND i.eye0Confidence >= 0 AND i.eye1Confidence >= 0 AND i.faceConfidence >= 0 AND i.numberOfFaces = 1 ")
            data = cur.fetchall()
            data = np.asarray(data)
            data = np.concatenate((data, data_aux))

        fn_rate = np.empty(shape=0)
        fp_rate = np.empty(shape=0)
        eer_found = False
        eer = 0
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
                fn_rate_eer = fn / (fn + tp)
            except ZeroDivisionError:
                fn_rate_eer = 1
            try:
                fp_rate_eer = fp / (fp + tn)
            except ZeroDivisionError:
                fp_rate_eer = 1
            fn_rate = np.append(fn_rate, fn_rate_eer)
            fp_rate = np.append(fp_rate, fp_rate_eer)
            if fn_rate_eer > fp_rate_eer and not eer_found:
                eer = (fn_rate_eer + fp_rate_eer) / 2
                eer_found = True

        i_plot = 1 if f[0] == ">" else 2
        plt.subplot(210 + i_plot)
        plt.plot(x, x, linestyle="dashed", color="red", linewidth=1)
        plt.plot(fn_rate, fp_rate, linewidth=1, color="blue", alpha=0.5,
                 label="EER=" + str(eer))
        plt.xlabel("False Negative Rate")
        plt.ylabel("False Positive Rate")
        plt.legend(loc="lower right")
        if f[0] == ">":
            plt.title(attr + " dentro media")
        else:
            plt.title(attr + " fuera media")
    plt.tight_layout()
    plt.savefig("EERPlots/" + attr + "_standard_desviation_mean.png")
    print("Readed " + attr)
    plt.close()
    con.close()


def calculate_eer_0(attr):
    x = [0, 1]
    con = Mdb.connect(DB_HOST, DB_USER, DB_PASS, DB_NAME)
    cur = con.cursor()
    formules = [">", "<"]

    plt.figure()
    for f in formules:
        cur.execute(
            "SELECT p.clase, i.clase, s.score FROM score_data s INNER JOIN imgs_data i ON s.id_img = i.id INNER JOIN pass_data p ON s.id_pass = p.id WHERE i." + attr + f + str(
                0) + " AND s.score >= 0 AND s.score <= 1 AND i.locateFace = 1 AND i.eye0Confidence >= 0 AND i.eye1Confidence >= 0 AND i.faceConfidence >= 0 AND i.numberOfFaces = 1 ")
        data = cur.fetchall()
        data = np.asarray(data)

        fn_rate = np.empty(shape=0)
        fp_rate = np.empty(shape=0)
        eer_found = False
        eer = 0
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
                fn_rate_eer = fn / (fn + tp)
            except ZeroDivisionError:
                fn_rate_eer = 1
            try:
                fp_rate_eer = fp / (fp + tn)
            except ZeroDivisionError:
                fp_rate_eer = 1
            fn_rate = np.append(fn_rate, fn_rate_eer)
            fp_rate = np.append(fp_rate, fp_rate_eer)
            if fn_rate_eer > fp_rate_eer and not eer_found:
                eer = (fn_rate_eer + fp_rate_eer) / 2
                eer_found = True

        i_plot = 1 if f == ">" else 2
        plt.subplot(210 + i_plot)
        plt.plot(x, x, linestyle="dashed", color="red", linewidth=1)
        plt.plot(fn_rate, fp_rate, linewidth=1, color="blue", alpha=0.5,
                 label="EER=" + str(eer))
        plt.xlabel("False Negative Rate")
        plt.ylabel("False Positive Rate")
        plt.legend(loc="lower right")
        plt.title(attr + f + str(0))
    plt.tight_layout()
    plt.savefig("EERPlots/" + attr + "_umbral_0.png")
    print("Readed " + attr)
    plt.close()
    con.close()


def calculate_eer_05(attr):
    x = [0, 1]
    con = Mdb.connect(DB_HOST, DB_USER, DB_PASS, DB_NAME)
    cur = con.cursor()
    formules = [">", "<"]

    plt.figure()
    for f in formules:
        cur.execute(
            "SELECT p.clase, i.clase, s.score FROM score_data s INNER JOIN imgs_data i ON s.id_img = i.id INNER JOIN pass_data p ON s.id_pass = p.id WHERE i." + attr + f + str(
                0.5) + " AND s.score >= 0 AND s.score <= 1 AND i.locateFace = 1 AND i.eye0Confidence >= 0 AND i.eye1Confidence >= 0 AND i.faceConfidence >= 0 AND i.numberOfFaces = 1 ")
        data = cur.fetchall()
        data = np.asarray(data)

        fn_rate = np.empty(shape=0)
        fp_rate = np.empty(shape=0)
        eer_found = False
        eer = 0
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
                fn_rate_eer = fn / (fn + tp)
            except ZeroDivisionError:
                fn_rate_eer = 1
            try:
                fp_rate_eer = fp / (fp + tn)
            except ZeroDivisionError:
                fp_rate_eer = 1
            fn_rate = np.append(fn_rate, fn_rate_eer)
            fp_rate = np.append(fp_rate, fp_rate_eer)
            if fn_rate_eer > fp_rate_eer and not eer_found:
                eer = (fn_rate_eer + fp_rate_eer) / 2
                eer_found = True

        i_plot = 1 if f == ">" else 2
        plt.subplot(210 + i_plot)
        plt.plot(x, x, linestyle="dashed", color="red", linewidth=1)
        plt.plot(fn_rate, fp_rate, linewidth=1, color="blue", alpha=0.5,
                 label="EER=" + str(eer))
        plt.xlabel("False Negative Rate")
        plt.ylabel("False Positive Rate")
        plt.legend(loc="lower right")
        plt.title(attr + f + str(0.5))
    plt.tight_layout()
    plt.savefig("EERPlots/" + attr + "_umbral_05.png")
    print("Readed " + attr)
    plt.close()
    con.close()


def calculate_eer_0_rest(attr):
    x = [0, 1]
    con = Mdb.connect(DB_HOST, DB_USER, DB_PASS, DB_NAME)
    cur = con.cursor()
    formules = ["=", "!="]

    plt.figure()
    for f in formules:
        cur.execute(
            "SELECT p.clase, i.clase, s.score FROM score_data s INNER JOIN imgs_data i ON s.id_img = i.id INNER JOIN pass_data p ON s.id_pass = p.id WHERE i." + attr + f + str(
                0) + " AND s.score >= 0 AND s.score <= 1 AND i.locateFace = 1 AND i.eye0Confidence >= 0 AND i.eye1Confidence >= 0 AND i.faceConfidence >= 0 AND i.numberOfFaces = 1 ")
        data = cur.fetchall()
        data = np.asarray(data)

        fn_rate = np.empty(shape=0)
        fp_rate = np.empty(shape=0)
        eer_found = False
        eer = 0
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
                fn_rate_eer = fn / (fn + tp)
            except ZeroDivisionError:
                fn_rate_eer = 1
            try:
                fp_rate_eer = fp / (fp + tn)
            except ZeroDivisionError:
                fp_rate_eer = 1
            fn_rate = np.append(fn_rate, fn_rate_eer)
            fp_rate = np.append(fp_rate, fp_rate_eer)
            if fn_rate_eer > fp_rate_eer and not eer_found:
                eer = (fn_rate_eer + fp_rate_eer) / 2
                eer_found = True

        i_plot = 1 if f == "=" else 2
        plt.subplot(210 + i_plot)
        plt.plot(x, x, linestyle="dashed", color="red", linewidth=1)
        plt.plot(fn_rate, fp_rate, linewidth=1, color="blue", alpha=0.5,
                 label="EER=" + str(eer))
        plt.xlabel("False Negative Rate")
        plt.ylabel("False Positive Rate")
        plt.legend(loc="lower right")
        plt.title(attr + f + str(0))
    plt.tight_layout()
    plt.savefig("EERPlots/" + attr + "_umbral_0_rest.png")
    print("Readed " + attr)
    plt.close()
    con.close()


def calculate_eer_rest(attribute):
    x = [0, 1]
    con = Mdb.connect(DB_HOST, DB_USER, DB_PASS, DB_NAME)
    cur = con.cursor()

    attr = attribute[0]
    values = attribute[1]

    plt.figure()
    for value in values:
        cur.execute(
            "SELECT p.clase, i.clase, s.score FROM score_data s INNER JOIN imgs_data i ON s.id_img = i.id INNER JOIN pass_data p ON s.id_pass = p.id WHERE i." + attr + "=" + str(
                value) + " AND s.score >= 0 AND s.score <= 1 AND i.locateFace = 1 AND i.eye0Confidence >= 0 AND i.eye1Confidence >= 0 AND i.faceConfidence >= 0 AND i.numberOfFaces = 1 ")
        data = cur.fetchall()
        data = np.asarray(data)

        fn_rate = np.empty(shape=0)
        fp_rate = np.empty(shape=0)
        eer_found = False
        eer = 0
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
                fn_rate_eer = fn / (fn + tp)
            except ZeroDivisionError:
                fn_rate_eer = 1
            try:
                fp_rate_eer = fp / (fp + tn)
            except ZeroDivisionError:
                fp_rate_eer = 1
            fn_rate = np.append(fn_rate, fn_rate_eer)
            fp_rate = np.append(fp_rate, fp_rate_eer)
            if fn_rate_eer > fp_rate_eer and not eer_found:
                eer = (fn_rate_eer + fp_rate_eer) / 2
                eer_found = True

        i_plot = values.index(value) + 1
        if len(values) == 2:
            plt.subplot(210 + i_plot)
        else:
            plt.subplot(220 + i_plot)
        plt.plot(x, x, linestyle="dashed", color="red", linewidth=1)
        eer = float("{0:.4f}".format(eer))
        plt.plot(fn_rate, fp_rate, linewidth=1, color="blue", alpha=0.5,
                 label="EER=" + str(eer))
        plt.xlabel("False Negative Rate")
        plt.ylabel("False Positive Rate")
        plt.legend(loc="lower right")
        plt.title(attr + "=" + str(value))
    plt.tight_layout()
    plt.savefig("EERPlots/" + attr + ".png")
    print("Readed " + attr)
    plt.close()
    con.close()


pool = Pool(20)
# pool.map(calculate_eer_mean, umbral_mean)
# pool.map(calculate_eer_binary, binary_attributes)
pool.map(calculate_eer_standard_desviation_mean, standard_deviation_mean)
# pool.map(calculate_eer_0, umbral_0)
# pool.map(calculate_eer_05, umbral_05)
# pool.map(calculate_eer_0_rest, umbral_0_rest)
# pool.map(calculate_eer_rest, rest)
pool.close()
pool.join()