import math
import numpy as np
"""adapted from old srt splot.java"""

def calctime(year,day,hour,min,sec):
    """convert time to seconds since New Year 1970"""
    secs = (year-1970)*31536000.0 + (day-1)*86400.0 + hour*3600.0+ min*60.0 + sec
    for i in range(1970,year):
        if (i % 4 == 0 and i % 100 != 0) or i % 400 == 0:
            secs = secs+ 86400.0
    if secs < 0:
        secs = 0.0
    return secs

def getGst(time):
    secs = (1999-1970)*31536000.0+17.0*3600.0+16.0*60.0+20.1948
    for i in range(1970,1999):
        if (i % 4 == 0 and i % 100 != 0) or i % 400 == 0:
            secs = secs + 86400.0
    gst = (time-secs)-(86164.09053*round((time-secs)/86164.09053))
    gst = gst/86164.09053*2*math.pi
    return gst

def dayOfYear(date):
      days = [0,31,28,31,30,31,30,31,31,30,31,30,31]
      d = list(map(int,date.split("-")))
      if d[0] % 400 == 0:
         days[2]+=1
      elif d[0]%4 == 0 and d[0]%100!=0:
         days[2]+=1
      for i in range(1,len(days)):
         days[i]+=days[i-1]
      return days[d[1]-1]+d[2]

def calc_vlsr(year,day,hour,min,sec,az,el,lat,lon):
    lat = lat/57.3
    lon = lon/57.3
    time = calctime(year,day,hour,min,sec)
    decc = -(28.0+56.0/60.0)*math.pi/180.0
    rac = (17.0+45.5/60.0)*math.pi/12.0
    dp = 27.1*math.pi/180.0
    rp = (12.0+51.4/60.0)*math.pi/12.0
    north = math.cos(az*math.pi/180.0)*math.cos(el*math.pi/180.0)
    west = -math.sin(az*math.pi/180.0)*math.cos(el*math.pi/180.0)
    zen = math.sin(el*math.pi/180.0)
    pole = north*math.cos(lat)+zen*math.sin(lat)
    rad = zen*math.cos(lat)-north*math.sin(lat)
    dec = math.atan2(pole,math.sqrt(rad**2+west**2))
    ha = math.atan2(west,rad)
    ra = -ha+getGst(time)-lon
    x0 = 20.0*math.cos(18.0*math.pi/12.0)*math.cos(30.0*math.pi/180.0)
    y0 = 20.0*math.sin(18.0*math.pi/12.0)*math.cos(30.0*math.pi/180.0)
    z0 = 20.0*math.sin(30.0*math.pi/180.0) #sun 20km/s toward ra=18h , dec=30.0 degrees
    vsun = -x0*math.cos(ra)*math.cos(dec)-y0*math.sin(ra)*math.cos(dec)-z0*math.sin(dec)
    x0 = math.cos(ra)*math.cos(dec)
    y0 = math.sin(ra)*math.cos(dec)
    z0 = math.sin(dec)
    x = x0
    y = y0*math.cos(23.5*math.pi/180.0)+z0*math.sin(23.5*math.pi/180.0)
    z = z0*math.cos(23.5*math.pi/180.0)-y0*math.sin(23.5*math.pi/180.0)
    soulat = math.atan2(z,math.sqrt(x**2+y**2))
    soulong = math.atan2(y,x)
    sunlong = (time*360.0/(365.25*86400.0)+280.0)*math.pi/180.0
    vel = vsun-30.0*math.cos(soulat)*math.sin(sunlong-soulong)
    return vel
