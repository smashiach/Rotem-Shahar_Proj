def fix_ratio_rotated_face(small, big):
    while big/small >= 1.05:
        small = small * 1.05
        big = big * 0.95

    return small, big