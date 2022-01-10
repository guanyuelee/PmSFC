def light(c, k=1.5):
    r = int(c[:2], base=16)
    g = int(c[2:4], base=16)
    b = int(c[4:], base=16)
    # print(f"before: {r},{g},{b}")
    r *= k
    r = int(r)
    r = min(r, 255)
    g *= k
    g = int(g)
    g = min(g, 255)
    b *= k
    b = int(b)
    b = min(b, 255)
    # print(f"after: {r},{g},{b}")
    res = f"{hex(r)[-2:]}{hex(g)[-2:]}{hex(b)[-2:]}"
    res = res.upper()
    print(res)


light('3484BA')
light('FF7F0E')
light('2B9F2B', 2)
light('D62728', 3)
