def formatTime(n):
    return f"0{n}" if n < 10 else f"{n}"

def calculateMinutesDiff(h1, m1, h2, m2):
    if h1 == h2:
        return m2 - m1
    else:
        return (h2 - h1 - 1) * 60 + 60 - m1 + m2

def addMinute2CurrentTime(h, m):
    return (h, m + 1) if m < 59 else (h + 1, 0)


if __name__ == "__main__":
    h1, m1 = 10, 23
    h2, m2 = 10, 25

    minutes_spent = 0
    diff = calculateMinutesDiff(h1, m1, h2, m2)
    while minutes_spent <= diff:
        print(f"{formatTime(h1)}:{formatTime(m1)}")
        h1, m1 = addMinute2CurrentTime(h1, m1)
        minutes_spent += 1





