import MySQLdb as Mdb
import random
import pandas as pd
import matplotlib.pyplot as plt

# Hardcoded
DB_HOST = "localhost"
DB_USER = "frav"
DB_PASS = "VXxL4UOLvB6wc01Y3Cxi"
DB_NAME = "frav_ABC"
FOLDER = "/media/alberto/Datos/FRAV_ALBERTO/"

con = Mdb.connect(DB_HOST, DB_USER, DB_PASS, DB_NAME)
cur = con.cursor()

# cur.execute("SELECT s.score FROM score_data s INNER JOIN imgs_data i ON i.id = s.id_img WHERE i.camera=1 AND i.light=1")
# data = [item[0] for item in cur.fetchall()]
# data = [x for x in data if x >= 0]
# data = [x for x in data if x <= 1]
# data = random.sample(data, 200000)
# df = pd.DataFrame(data)
# df.plot.density(bw_method=0.5)
# # df.plot(kind='density')
# plt.show()