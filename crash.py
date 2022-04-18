import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def umeyama(P, Q):
    assert P.shape == Q.shape
    n, dim = P.shape

    centeredP = P - P.mean(axis=0)
    centeredQ = Q - Q.mean(axis=0)

    C = np.dot(np.transpose(centeredP), centeredQ) / n

    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    Rot = np.dot(V, W)

    varP = np.var(P, axis=0).sum()
    scale = 1 / varP * np.sum(S)  # scale factor

    trans = Q.mean(axis=0) - P.mean(axis=0).dot(scale * Rot)

    return scale, Rot, trans


# ldk = np.random.random([68, 3])
# std_model = np.random.random([68, 3])
#
# scale, Rot, trans = umeyama(ldk, std_model)
# x = ldk.dot(scale * Rot) + trans
# x = (x - trans).dot(np.linalg.inv(scale * Rot))
#
# for y, x in zip(ldk, x):
#     print(np.sqrt((y[0] - x[0]) ** 2 + (y[1] - x[1]) ** 2 + (y[2] - x[2]) ** 2))

gt3d = [[-0.0280, 0.9302, 0.7591],
        [-0.0184, 0.9269, 0.6206],
        [-0.0137, 0.9223, 0.4887],
        [0.0070, 0.9030, 0.3615],
        [0.0654, 0.8745, 0.2245],
        [0.1782, 0.8246, 0.0930],
        [0.3181, 0.7615, 0.0139],
        [0.4428, 0.6526, -0.0607],
        [0.4971, 0.4814, -0.0773],
        [0.4447, 0.3047, -0.0471],
        [0.3179, 0.2035, 0.0186],
        [0.1768, 0.1262, 0.1101],
        [0.0636, 0.0736, 0.2229],
        [0.0050, 0.0379, 0.3633],
        [-0.0176, 0.0244, 0.4972],
        [-0.0262, 0.0207, 0.6358],
        [-0.0352, 0.0252, 0.7638],
        [0.4425, 0.9085, 0.9575],
        [0.5284, 0.8497, 0.9921],
        [0.5909, 0.7656, 0.9991],
        [0.6319, 0.6883, 0.9693],
        [0.6508, 0.6142, 0.9261],
        [0.6491, 0.4162, 0.9372],
        [0.6342, 0.3384, 0.9800],
        [0.5986, 0.2426, 1.0000],
        [0.5387, 0.1597, 0.9969],
        [0.4538, 0.0749, 0.9561],
        [0.6758, 0.5258, 0.8024],
        [0.7414, 0.5244, 0.7087],
        [0.8091, 0.5223, 0.6135],
        [0.8243, 0.5207, 0.5351],
        [0.6512, 0.6600, 0.4771],
        [0.6879, 0.6031, 0.4736],
        [0.7047, 0.5197, 0.4597],
        [0.6972, 0.4362, 0.4845],
        [0.6626, 0.3649, 0.4857],
        [0.4903, 0.9194, 0.7532],
        [0.5604, 0.8568, 0.8047],
        [0.5816, 0.6812, 0.8170],
        [0.5654, 0.6019, 0.7831],
        [0.5755, 0.6847, 0.7718],
        [0.5508, 0.8131, 0.7558],
        [0.5655, 0.4209, 0.7821],
        [0.5849, 0.3603, 0.8135],
        [0.5821, 0.1910, 0.8069],
        [0.5155, 0.0902, 0.7541],
        [0.5632, 0.2217, 0.7595],
        [0.5810, 0.3321, 0.7588],
        [0.5511, 0.7272, 0.2455],
        [0.6484, 0.7065, 0.2848],
        [0.7005, 0.5942, 0.3132],
        [0.7101, 0.5046, 0.3005],
        [0.7059, 0.4030, 0.3093],
        [0.6561, 0.3091, 0.2935],
        [0.5534, 0.2773, 0.2422],
        [0.6119, 0.3235, 0.1957],
        [0.6420, 0.4094, 0.1708],
        [0.6571, 0.5006, 0.1734],
        [0.6429, 0.5871, 0.1695],
        [0.6075, 0.6649, 0.1945],
        [0.5565, 0.7260, 0.2470],
        [0.6566, 0.5939, 0.2542],
        [0.6747, 0.4944, 0.2478],
        [0.6598, 0.4015, 0.2510],
        [0.5581, 0.2788, 0.2395],
        [0.6461, 0.4046, 0.2533],
        [0.6532, 0.4951, 0.2548],
        [0.6417, 0.5963, 0.2568]]
output = [[9.3059e-03, 1.0638e+00, 7.0675e-01],
          [-7.8968e-03, 1.0368e+00, 5.3823e-01],
          [1.1709e-03, 1.0257e+00, 4.1011e-01],
          [3.1913e-02, 1.0073e+00, 2.7024e-01],
          [9.8429e-02, 9.7306e-01, 1.3199e-01],
          [2.2682e-01, 9.0094e-01, 3.0410e-02],
          [4.1184e-01, 7.8654e-01, -2.7870e-02],
          [5.9481e-01, 6.3671e-01, -5.7638e-02],
          [6.7180e-01, 4.6244e-01, -6.3373e-02],
          [5.9979e-01, 2.8663e-01, -3.8178e-02],
          [4.1485e-01, 1.3350e-01, 4.1403e-03],
          [2.2080e-01, 9.9353e-03, 6.3094e-02],
          [8.6685e-02, -7.6606e-02, 1.6049e-01],
          [2.3115e-02, -1.2538e-01, 3.0480e-01],
          [-2.3478e-03, -1.5313e-01, 4.6672e-01],
          [-6.3686e-03, -1.7416e-01, 6.1461e-01],
          [-5.7271e-04, -1.9023e-01, 7.2792e-01],
          [3.8739e-01, 9.2907e-01, 6.7310e-01],
          [5.2486e-01, 8.7217e-01, 7.8211e-01],
          [6.1394e-01, 8.0092e-01, 8.3883e-01],
          [6.6159e-01, 7.3197e-01, 8.6020e-01],
          [6.7702e-01, 6.6883e-01, 8.5814e-01],
          [6.6126e-01, 3.2571e-02, 1.0389e+00],
          [6.3571e-01, -4.3173e-02, 1.0464e+00],
          [5.7689e-01, -1.2708e-01, 1.0330e+00],
          [4.7664e-01, -2.1480e-01, 9.8990e-01],
          [3.2658e-01, -2.8353e-01, 9.0198e-01],
          [6.6112e-01, 4.7722e-01, 7.3461e-01],
          [7.0283e-01, 4.9245e-01, 7.0985e-01],
          [7.4579e-01, 5.0719e-01, 6.8571e-01],
          [7.4142e-01, 5.1746e-01, 6.4936e-01],
          [5.5813e-01, 5.7282e-01, 5.4466e-01],
          [5.8895e-01, 5.5150e-01, 5.5495e-01],
          [6.0115e-01, 5.2016e-01, 5.5786e-01],
          [5.8500e-01, 4.8418e-01, 5.5924e-01],
          [5.5064e-01, 4.5613e-01, 5.5152e-01],
          [5.6352e-01, 7.6313e-01, 6.0152e-01],
          [6.4878e-01, 7.2462e-01, 6.5572e-01],
          [6.4629e-01, 6.6503e-01, 6.6038e-01],
          [6.0472e-01, 6.1286e-01, 6.2991e-01],
          [6.2274e-01, 6.6655e-01, 6.2405e-01],
          [6.0350e-01, 7.2601e-01, 6.1101e-01],
          [5.7732e-01, 9.5077e-02, 7.3786e-01],
          [6.0906e-01, 3.7722e-02, 7.7594e-01],
          [5.8601e-01, -2.4519e-02, 7.7202e-01],
          [5.1208e-01, -6.9404e-02, 7.3161e-01],
          [5.6084e-01, -1.7076e-02, 7.3580e-01],
          [5.8479e-01, 4.5561e-02, 7.4426e-01],
          [5.8915e-01, 5.9961e-01, 2.9139e-01],
          [6.4258e-01, 5.7400e-01, 3.2298e-01],
          [6.7283e-01, 5.4444e-01, 3.4238e-01],
          [6.7304e-01, 5.2903e-01, 3.3987e-01],
          [6.7057e-01, 5.1238e-01, 3.4017e-01],
          [6.3621e-01, 4.8597e-01, 3.2136e-01],
          [5.7743e-01, 4.6706e-01, 2.8753e-01],
          [6.0920e-01, 4.9113e-01, 2.8202e-01],
          [6.2492e-01, 5.0980e-01, 2.7588e-01],
          [6.2999e-01, 5.2812e-01, 2.7072e-01],
          [6.2824e-01, 5.4497e-01, 2.6569e-01],
          [6.1436e-01, 5.5793e-01, 2.6164e-01],
          [5.8734e-01, 5.6835e-01, 2.5853e-01],
          [6.4191e-01, 5.2743e-01, 2.8763e-01],
          [6.4898e-01, 5.0700e-01, 2.9023e-01],
          [6.3878e-01, 4.8636e-01, 2.8564e-01],
          [5.7649e-01, 4.4752e-01, 2.5739e-01],
          [6.2648e-01, 4.8781e-01, 2.6186e-01],
          [6.3104e-01, 5.0404e-01, 2.5738e-01],
          [6.2762e-01, 5.1891e-01, 2.5254e-01]]
gt2d = [[[29.6504, 65.4472],
         [29.2439, 90.6504],
         [30.4634, 108.9431],
         [33.3089, 130.8943],
         [38.5935, 156.0976],
         [46.6941, 174.9668],
         [58.3129, 189.4303],
         [70.3008, 200.8130],
         [103.2276, 206.9106],
         [133.6101, 200.2914],
         [148.7274, 187.9337],
         [159.8334, 171.2776],
         [167.1823, 151.5455],
         [172.3245, 128.0465],
         [174.2189, 106.0694],
         [174.6566, 83.5276],
         [173.7753, 62.6989],
         [33.7154, 41.8699],
         [42.2520, 36.5854],
         [52.4146, 34.5528],
         [81.2764, 43.4959],
         [87.7805, 51.2195],
         [110.1382, 50.4065],
         [118.2683, 43.0894],
         [145.9106, 37.3984],
         [158.5122, 40.6504],
         [166.6423, 44.3089],
         [99.9756, 64.2276],
         [99.9756, 80.4878],
         [99.9756, 97.9675],
         [100.3821, 115.8537],
         [78.0244, 119.1057],
         [88.1870, 119.5122],
         [100.2140, 121.7879],
         [114.6098, 119.1057],
         [123.9594, 119.1057],
         [37.3740, 69.9187],
         [46.7236, 66.2602],
         [79.2439, 67.0732],
         [84.9350, 70.3252],
         [71.5203, 75.2033],
         [52.8211, 74.7967],
         [116.2358, 73.5772],
         [121.9268, 67.8862],
         [148.3496, 67.4797],
         [160.9512, 71.1382],
         [146.7236, 73.9837],
         [128.8374, 74.7967],
         [0.0000, 0.0000],
         [67.0488, 154.8781],
         [88.9880, 148.3851],
         [103.6130, 150.4447],
         [117.8618, 148.3740],
         [126.3984, 150.8130],
         [135.3415, 153.6585],
         [131.2764, 163.4146],
         [118.4417, 170.4564],
         [103.8208, 170.3477],
         [89.6619, 170.9131],
         [76.6914, 166.5278],
         [67.4553, 154.8781],
         [88.6484, 157.3875],
         [103.2276, 157.3171],
         [119.8523, 157.6924],
         [132.4959, 156.5041],
         [0.0000, 0.0000],
         [0.0000, 0.0000],
         [0.0000, 0.0000]],
        [[9.1382, 98.6992],
         [7.5122, 141.7886],
         [6.6992, 168.6179],
         [11.1707, 192.6016],
         [19.7073, 212.9268],
         [31.9024, 226.7480],
         [47.3985, 233.7001],
         [68.3893, 238.2232],
         [99.8657, 238.2053],
         [123.3659, 238.9431],
         [142.4715, 236.0976],
         [159.9512, 229.1870],
         [174.9919, 219.0244],
         [188.0000, 199.5122],
         [194.9106, 174.7155],
         [196.5366, 144.2276],
         [194.9106, 96.2602],
         [17.6748, 55.2033],
         [33.1220, 41.3821],
         [54.6667, 36.5041],
         [72.5528, 39.3496],
         [84.7480, 47.4797],
         [113.6098, 46.6667],
         [129.0569, 37.7236],
         [146.9431, 35.6911],
         [166.4553, 42.6016],
         [181.3083, 54.3351],
         [96.9431, 54.7967],
         [97.3496, 68.2114],
         [96.1301, 81.6260],
         [95.7236, 97.0732],
         [70.4922, 116.8973],
         [81.1616, 115.2771],
         [96.3880, 116.9127],
         [111.3941, 113.7232],
         [123.8672, 116.2940],
         [23.0063, 76.2702],
         [42.0650, 64.5528],
         [71.3333, 61.3008],
         [81.1766, 67.8551],
         [64.4228, 71.4634],
         [43.2846, 73.0894],
         [117.6748, 66.5854],
         [129.0569, 59.6748],
         [155.0732, 62.5203],
         [176.2114, 75.9350],
         [156.6992, 71.8699],
         [129.8699, 70.6504],
         [0.0000, 0.0000],
         [56.2927, 153.9837],
         [89.2195, 139.3496],
         [99.7886, 143.0081],
         [107.5122, 138.5366],
         [130.2764, 148.2927],
         [140.4390, 163.7398],
         [133.6582, 169.2954],
         [118.4707, 170.9264],
         [102.2030, 168.9652],
         [86.4224, 170.1130],
         [72.0063, 167.9332],
         [59.9512, 163.7398],
         [83.1219, 155.2032],
         [103.5974, 155.3508],
         [120.1972, 156.3408],
         [140.0325, 164.1463],
         [119.4839, 156.8845],
         [103.2484, 155.6482],
         [0.0000, 0.0000]]]
transformation = [[[0.0036], [[0.0142, -0.9997, -0.0177],
                              [0.0609, 0.0185, -0.9980],
                              [0.9980, 0.0131, 0.0611]], [0.0220, 0.9532, 0.9275]],
                  [[0.0028], [[0.0734, -0.9971, -0.0192],
                              [-0.3466, -0.0074, -0.9380],
                              [0.9351, 0.0755, -0.3461]], [0.0948, 0.7796, 1.0657]]]
projected = [[[-29.5429, 61.3264],
              [-21.3226, 107.4389],
              [-17.5845, 142.9186],
              [-11.7095, 181.9707],
              [-1.2884, 221.0963],
              [19.6646, 250.9422],
              [52.3263, 269.5679],
              [94.6394, 280.1037],
              [143.1832, 282.0913],
              [191.4169, 273.0190],
              [232.8494, 257.4314],
              [265.9866, 237.2477],
              [288.9264, 207.6468],
              [301.4645, 166.4711],
              [308.2481, 121.1820],
              [313.3274, 80.1638],
              [317.2420, 48.8870],
              [9.3648, 76.3001],
              [25.1135, 48.2178],
              [44.8975, 33.6887],
              [64.0550, 28.2375],
              [81.5939, 28.7433],
              [256.6743, -24.7054],
              [277.4929, -27.5963],
              [300.5403, -25.3158],
              [324.6291, -15.5504],
              [343.4838, 5.8503],
              [135.1467, 61.6074],
              [131.2180, 69.2267],
              [127.4272, 76.6927],
              [124.7472, 86.7111],
              [109.2241, 112.8224],
              [115.1921, 110.3905],
              [123.8968, 109.6306],
              [133.7811, 108.7921],
              [141.4461, 110.2038],
              [56.3161, 98.1848],
              [67.0378, 84.4563],
              [83.4933, 82.8215],
              [97.9128, 90.2692],
              [83.1579, 92.4651],
              [66.6954, 96.0479],
              [240.5261, 57.3415],
              [256.3318, 47.0644],
              [273.4803, 47.4385],
              [285.8060, 57.1227],
              [271.4997, 57.0556],
              [254.2231, 55.4455],
              [103.1751, 183.4275],
              [110.3151, 175.4722],
              [118.5159, 170.4733],
              [122.7922, 171.0900],
              [127.3873, 170.8813],
              [134.6506, 175.3597],
              [139.8193, 183.6157],
              [133.3093, 185.7964],
              [128.2374, 187.8525],
              [123.2137, 189.4563],
              [118.5694, 190.9026],
              [114.9486, 191.8536],
              [111.9752, 192.3125],
              [123.3687, 184.9836],
              [129.0361, 184.2800],
              [134.7284, 185.2703],
              [145.3673, 191.8239],
              [134.3953, 191.6392],
              [129.9452, 193.0353],
              [125.8424, 194.3907]],
             [[-101.4122, 130.6313],
              [-91.0557, 189.5426],
              [-85.9579, 231.5533],
              [-77.6298, 274.8409],
              [-62.6646, 313.1786],
              [-32.7882, 331.5847],
              [13.4050, 328.4974],
              [72.0164, 316.1644],
              [136.4151, 308.9885],
              [197.2266, 309.9329],
              [246.8337, 319.0964],
              [285.5137, 323.7160],
              [312.2631, 307.8517],
              [327.0391, 267.3300],
              [335.1761, 216.0899],
              [341.5737, 166.8871],
              [346.6938, 128.0863],
              [-43.0516, 95.3084],
              [-19.8293, 41.6909],
              [7.6104, 11.7213],
              [33.3787, -1.2066],
              [56.3826, -2.2629],
              [282.2916, -59.4242],
              [308.6586, -58.5708],
              [337.2111, -46.5238],
              [366.2457, -19.3239],
              [387.4794, 29.0974],
              [125.3479, 41.7871],
              [121.1700, 44.8914],
              [117.1963, 47.6343],
              [113.6593, 60.3806],
              [89.7514, 118.2488],
              [98.1167, 111.0117],
              [109.6281, 108.5980],
              [122.0624, 110.2368],
              [131.2455, 117.1827],
              [21.4340, 97.9380],
              [37.0778, 69.2064],
              [58.2963, 68.1056],
              [76.0711, 83.6655],
              [57.3822, 83.2528],
              [35.6974, 89.8745],
              [259.7998, 52.1344],
              [280.8871, 35.5275],
              [302.5688, 39.8783],
              [316.9541, 62.7852],
              [299.4934, 55.1735],
              [277.6626, 49.1844],
              [82.7339, 199.5353],
              [93.0832, 182.3324],
              [104.3178, 172.1232],
              [109.8522, 172.9810],
              [115.7401, 173.2333],
              [124.4099, 183.9019],
              [129.8610, 202.6439],
              [122.1237, 200.4834],
              [115.9044, 200.5450],
              [109.5205, 201.6016],
              [103.4819, 203.4662],
              [98.5079, 206.5204],
              [94.0909, 210.9002],
              [109.9649, 194.4313],
              [117.4406, 192.7325],
              [124.5854, 195.6001],
              [137.0306, 212.9540],
              [123.9065, 205.1288],
              [118.2528, 206.0243],
              [112.8788, 208.0378]]]

def try_differences(gt3d, output, gt2ds, transformations, projecteds):
    gt3d, output, gt2ds, transformations, projecteds = np.asarray(gt3d), np.asarray(output), np.asarray(gt2ds), np.asarray(transformations), np.asarray(projecteds)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(2, 1, 1, projection='3d')
    X, Y, Z = gt3d.swapaxes(1, 0)
    ax.scatter(X, Y, Z, c='b', s=5)
    X, Y, Z = output.swapaxes(1, 0)
    ax.scatter(X, Y, Z, c='r', s=5)
    for i in range(2):
        transformation = transformations[i]
        gt2d = gt2ds[i]
        scale, rotation, translation = np.asarray(transformation[0]), np.asarray(transformation[1]), np.asarray(transformation[2])
        x_proj_tmp = (output - (translation)).dot(np.linalg.inv(1.6*scale * rotation))
        x_proj = x_proj_tmp[:, :2]
        ax = fig.add_subplot(2, 2, 3 + i)
        gt2d = np.where(gt2d > np.zeros(1), gt2d, np.zeros(1))
        X, Y = gt2d.swapaxes(1, 0)
        ax.scatter(X, Y, c='b', s=5)
        X, Y = x_proj.swapaxes(1, 0)
        ax.scatter(X, Y, c='r', s=5)
    plt.show()


def try_precomputed_ds(loader):
    for art in loader:
        y = art.y.numpy()
        gt3d = y[:68]
        gt3d = gt3d[:, :3]
        gt2d = y[68:]
        for id_view in range(len(gt2d) // 68):
            pt_y = gt2d[id_view * 68:(id_view + 1) * 68]
            scale, rotation, translation = pt_y[0, 2], pt_y[0, 3:12].reshape((3, 3)), pt_y[0, 12:]
            pt_y = pt_y[:, :2]
            x_proj = (gt3d - translation)
            tmp = np.linalg.inv(scale * rotation)
            x_proj = np.matmul(x_proj, tmp)
            x_proj = x_proj[:, :2]
            pt_y = np.where(pt_y > np.zeros(1), pt_y, np.zeros(1))
            X, Y = pt_y.swapaxes(1, 0)
            plt.scatter(X, Y, c='b', s=5)
            X, Y = x_proj.swapaxes(1, 0)
            plt.scatter(X, Y, c='r', s=5)
            plt.show()
            plt.close()


def try_ds(loader):
    for artifact in loader:
        gt3d = artifact['art_gt']
        gt2d = []
        for img in artifact['imgs']:
            scale, rotation, translation = img['img_rot']
            img_gt = img['img_gt']
            gt2d.append([scale, rotation, translation, img_gt])
        for img in gt2d:
            scale, rotation, translation, pt_y = img
            x_proj = (gt3d - translation)
            tmp = np.linalg.inv(scale * rotation)
            x_proj = np.matmul(x_proj, tmp)
            x_proj = x_proj[:, :2]
            pt_y = np.where(pt_y > np.zeros(1), pt_y, np.zeros(1))
            X, Y = pt_y.swapaxes(1, 0)
            plt.scatter(X, Y, c='b', s=5)
            X, Y = x_proj.swapaxes(1, 0)
            plt.scatter(X, Y, c='r', s=5)
            plt.show()
            plt.close()

def try_original_ds(ds):
    for artifact in ds:
        artifact.print_gt()
        # gt3d = artifact.gt
        # gt2d = []
        # for img in artifact.pictures:
        #     scale, rotation, translation = img.transformation
        #     img_gt = img['img_gt']
        #     gt2d.append([scale, rotation, translation, img_gt])
        # for img in gt2d:
        #     scale, rotation, translation, pt_y = img
        #     x_proj = (gt3d - translation)
        #     tmp = np.linalg.inv(scale * rotation)
        #     x_proj = np.matmul(x_proj, tmp)
        #     x_proj = x_proj[:, :2]
        #     pt_y = np.where(pt_y > np.zeros(1), pt_y, np.zeros(1))
        #     X, Y = pt_y.swapaxes(1, 0)
        #     plt.scatter(X, Y, c='b', s=5)
        #     X, Y = x_proj.swapaxes(1, 0)
        #     plt.scatter(X, Y, c='r', s=5)
        #     plt.show()
        #     plt.close()


if __name__ == '__main__':
    import face_alignment
    from TDDFA import TDDFA
    import yaml
    import os
    cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OMP_NUM_THREADS'] = '4'
    tddfa = TDDFA(**cfg)

    # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)
    # print(fa.face_alignment_net, fa.depth_prediciton_net)
    from generate_dataset import BuddhaDataset, Config, Artifact, Image
    original_ds = BuddhaDataset(Config('conf.json'))
    original_ds.load()
    original_ds = original_ds.artifacts
    # try_original_ds(original_ds[:10])
    images, cropped_images, gt, cropped_gt, bbox = [], [], [], [], []
    for artifact in original_ds[:100]:
        images.append([img.data for img in artifact.pictures])
        cropped_images.append([img.cropped_data for img in artifact.pictures])
        gt.append([img.precomputed_gt for img in artifact.pictures])
        cropped_gt.append([img.cropped_gt for img in artifact.pictures])
        bbox.append([img.bbox for img in artifact.pictures])
    art_i = 0
    for art_im, art_crop_im, art_gt, art_crop_gt, art_bbox in zip(images, cropped_images, gt, cropped_gt, bbox):
        art_path = 'output/3d_visu/art_{}'.format(art_i)
        if not os.path.exists(art_path):
            os.mkdir(art_path)
        im_i = 0
        for im, crop_im, y, crop_y, bbox in zip(art_im, art_crop_im, art_gt, art_crop_gt, art_bbox):
            param_lst, roi_box_lst = tddfa(im, [bbox])
            pred = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)
            pred = pred[0]
            fig = plt.figure()
            ax_3d = fig.add_subplot(1, 2, 1, projection='3d')
            ax_2d = fig.add_subplot(1, 2, 2)
            X, Y, Z = y.swapaxes(1, 0)
            ax_3d.scatter(X, Y, Z, c='b', s=5)
            ax_2d.scatter(X, Y, c='b', s=5)
            X, Y, Z = pred
            ax_3d.scatter(X, Y, Z, c='r', s=5)
            ax_2d.scatter(X, Y, c='r', s=5)
            plt.savefig(art_path + '/' + str(im_i) + '_gt_blue_pred_red.png')
            plt.close()
            fig = plt.figure()
            param_lst, roi_box_lst = tddfa(crop_im, [[0, 0, im.shape[0], im.shape[1]]])
            crop_pred = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)
            crop_pred = crop_pred[0]
            ax_3d = fig.add_subplot(1, 2, 1, projection='3d')
            ax_2d = fig.add_subplot(1, 2, 2)
            X, Y, Z = crop_y.swapaxes(1, 0)
            ax_3d.scatter(X, Y, Z, c='b', s=5)
            ax_2d.scatter(X, Y, c='b', s=5)
            X, Y, Z = crop_pred
            ax_3d.scatter(X, Y, Z, c='r', s=5)
            ax_2d.scatter(X, Y, c='r', s=5)
            plt.savefig(art_path + '/' + str(im_i) + '_gt_blue_pred_red_cropped.png')
            plt.close()
            plt.imsave(art_path + '/' + str(im_i) + '_photo.png', im)
            im_i += 1
        art_i += 1

    #
    # with open('ds_0_aug.pkl', 'rb') as f:
    #     # pickle dump generated in dataloader.py
    #     ds = pickle.load(f)
    # ds_train, ds_eval = ds
    # with open('ds_precomputed_graph.pkl', 'rb') as f:
    #     # pickle dump generated in graph_net.py
    #     loader_train, loader_eval = pickle.load(f)

    # try_precomputed_ds(loader_eval)
    # try_ds(ds_eval)

    # try_differences(gt3d, output, gt2d, transformation, projected)
    # try_differences(gt3d, gt3d, gt2d, transformation, projected)