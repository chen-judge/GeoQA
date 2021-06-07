import math


def g_equal(n1):  # 0
    return n1


def g_double(n1):  # 1
    return n1*2


def g_half(n1):  # 2
    return n1/2


def g_add(n1, n2):  # 3
    return n1 + n2


def g_minus(n1, n2):  # 4
    return math.fabs(n1 - n2)


def g_sin(n1):  # 5
    if n1 % 15 == 0 and 0 <= n1 <= 180:
        return math.sin(n1/180*math.pi)
    return False


def g_cos(n1):  # 6
    if n1 % 15 == 0 and 0 <= n1 <= 180:
        return math.cos(n1/180*math.pi)
    return False


def g_tan(n1):  # 7
    if n1 % 15 == 0 and 5 <= n1 <= 85:
        return math.tan(n1/180*math.pi)
    return False


def g_asin(n1):  # 8
    if -1 < n1 < 1:
        n1 = math.asin(n1)
        n1 = math.degrees(n1)
        return n1
    return False


def g_acos(n1):  # 9
    if -1 < n1 < 1:
        n1 = math.acos(n1)
        n1 = math.degrees(n1)
        return n1
    return False


def gougu_add(n1, n2):  # 13
    return math.sqrt(n1*n1+n2*n2)


def gougu_minus(n1, n2):  # 14
    if n1 != n2:
        return math.sqrt(math.fabs(n1*n1-n2*n2))
    return False


def g_bili(n1, n2, n3):  # 16
    if n1 > 0 and n2 > 0 and n3 > 0:
        return n1/n2*n3
    else:
        return False


def g_mul(n1, n2):  # 17
    return n1*n2


def g_divide(n1, n2):  # 18
    if n1 > 0 and n2 > 0:
        return n1/n2
    return False


def cal_circle_area(n1):  # 19
    return n1*n1*math.pi


def cal_circle_perimeter(n1):  # 20
    return 2*math.pi*n1


def cal_cone(n1, n2):  # 21
    return n1*n2*math.pi

