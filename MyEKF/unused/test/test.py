import ahrs
import datetime

wmm = ahrs.utils.WMM(datetime.date(2017, 5, 12), latitude=-33.9, longitude=151.18, height=0.05)
print(wmm.magnetic_elements['X'])
