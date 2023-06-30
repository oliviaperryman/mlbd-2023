# t=0
def p_s0_is_1(p0):
    return p0


def p_s0_is_0(p0):
    return 1 - p0


def p_s0_given_o0_is_1(p0, ps, pg):
    return (1 - ps) * p0 / ((1 - ps) * p0 + (pg) * (1 - p0))


def p_s0_given_o0_is_0(p0, ps, pg):
    return ps * p0 / (ps * p0 + (1 - pg) * (1 - p0))


# t = 1....t


def p_st_given_otminus1(p_st_minus1, pf, pl):
    return (1 - pf) * p_st_minus1 + pl * (1 - p_st_minus1)


def p_st_given_otminus1_is_1(p_st, ps, pg):
    return (1 - ps) * p_st / ((1 - ps) * p_st + (pg) * (1 - p_st))


def p_st_given_otminus1_is_0(p_st, ps, pg):
    return ps * p_st / (ps * p_st + (1 - pg) * (1 - p_st))


def bkt_sequentially(po, ps, pg, pl, pf, observations):
    ps0 = p_s0_is_1(po)
    print("ps0", ps0)

    if observations[0]:
        p_s0_given_o0 = p_s0_given_o0_is_1(po, ps, pg)
    else:
        p_s0_given_o0 = p_s0_given_o0_is_0(po, ps, pg)

    print(f"p_s0_given_o0 is {observations[0]}", p_s0_given_o0)

    i = 1
    for obs in observations[1:]:
        p_st = p_st_given_otminus1(p_s0_given_o0, pf, pl)
        if obs:
            p_st_given_ot = p_st_given_otminus1_is_1(p_st, ps, pg)
        else:
            p_st_given_ot = p_st_given_otminus1_is_0(p_st, ps, pg)

        print(f"p_st where t = {i}", p_st)
        print(f"p_st_given_ot where t = {i} and obs = {obs}", p_st_given_ot)
        p_s0_given_o0 = p_st_given_ot
        i += 1


if __name__ == "__main__":
    po = 0.6
    ps = 0.1
    pg = 0.2
    pl = 0.3
    pf = 0.3

    # t=0
    _p_s0_is_1 = p_s0_is_1(po)
    _p_s0_given_o0_is_0 = p_s0_given_o0_is_0(po, ps, pg)
    _p_s0_given_o0_is_1 = p_s0_given_o0_is_1(po, ps, pg)

    print(_p_s0_is_1)
    print(_p_s0_given_o0_is_0)
    print(_p_s0_given_o0_is_1)

    # t=1
    p_s1_given_o0_is_0 = p_st_given_otminus1(_p_s0_given_o0_is_0, pf, pl)
    print(p_s1_given_o0_is_0)

    p_st_given_o0_is_0_o1_is_1 = p_st_given_otminus1_is_1(p_s1_given_o0_is_0, ps, pg)
    print(p_st_given_o0_is_0_o1_is_1)

    # t=2
    p_s2_given_o0_is_0_o1_is_1 = p_st_given_otminus1(p_st_given_o0_is_0_o1_is_1, pf, pl)
    print(p_s2_given_o0_is_0_o1_is_1)

    # mock exam
    bkt_sequentially(po, ps, pg, pl, pf, [0, 1, 0])

    # lecture slides
    bkt_sequentially(0.5, 0.2, 0.3, 0.4, 0, [0, 1, 1, 0])
