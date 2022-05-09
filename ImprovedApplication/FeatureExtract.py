import pandas as pd
import numpy as np

import parselmouth
from parselmouth.praat import call

features = ['name',
 'Median pitch',
 'Mean pitch',
 'Standard deviation',
 'Minimum pitch',
 'Maximum pitch',
 'Number of pulses',
 'Number of periods',
 'Mean period',
 'Standard deviation of period',
 'Fraction of locally unvoiced frames',
 'Number of voice breaks',
 'Degree of voice breaks',
 'Jitter (local)',
 'Jitter (local, absolute)',
 'Jitter (rap)',
 'Jitter (ppq5)',
 'Jitter (ddp)',
 'Shimmer (local)',
 'Shimmer (local, dB)',
 'Shimmer (apq3)',
 'Shimmer (apq5)',
 'Shimmer (apq11)',
 'Shimmer (dda)',
 'Mean autocorrelation',
 'Mean noise-to-harmonics ratio',
 'Mean harmonics-to-noise ratio']


def get_report(file, start=0, end=0):
    sound = parselmouth.Sound(file)
    pitch = sound.to_pitch()
    pulses = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")
    voice_report_str = parselmouth.praat.call([sound, pitch, pulses], "Voice report", start, end, 
                                              75, 500, 1.3, 1.6, 0.03, 0.45)

    return(voice_report_str)

def get_feats(report, name):
    feats = pd.DataFrame(columns=features)
    row = [name]
    report = report.replace('--undefined--', '100000')
    report_splt = report.split('\n')
    report_splt = report_splt[1:]
    for l in report_splt:
        l_splt=l.split(" ")
        
        
        nums = [num for num in l_splt if num.replace('.','',1).replace('%','',1).replace('E-','',1).isdigit()]
        if len(nums)>0:
            f_num = nums[0]
            f_num = f_num.replace("%", "")
            
            f_num = float(f_num)
            if f_num == 100000:
                row.append(None)
            else:
                row.append(f_num)
    
    feats.loc[len(feats)] = row
    
    return feats