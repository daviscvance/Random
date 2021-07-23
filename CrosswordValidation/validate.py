# Crossword Validation Testing

def CheckSequentialWhiteSpaces(crossword: list) -> bool:
    # Assume elements of crossword are either 1 or 0.
    count = 0
    total = 0
    white_spaces = []
    for i, row in enumerate(crossword):
        for j, column_element in enumerate(row):
            count += 1
            total += column_element
            if column_element == 0:
                white_spaces.append((i, j))

    num_white_spaces = count - total
    if num_white_spaces == 0 or num_white_spaces == 1:
        return True  # 0 or 1 white spaces, no neighbors to check.

    else:  # Check for full sequence of all white spaces.
        has_neighbors = []
        for i, j in white_spaces:
            neighbor_count = 0
            positions = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
            for neighbor_check in positions:
                if (i, j) in has_neighbors:
                    continue
                if neighbor_check in white_spaces:
                    neighbor_count += 1
                if num_white_spaces == 2 and neighbor_count == 1:
                    has_neighbors.append((i, j))
                    continue
                if neighbor_count > 1:
                    has_neighbors.append((i, j))
        return len(has_neighbors) == len(white_spaces)

def CheckSymmetry(crossword: list) -> bool:
    new_crossword = []
    for row in crossword:
        new_crossword.append(row[::-1])
    return (new_crossword == crossword and crossword[::-1] == crossword)

def TestCases(crossword: list) -> bool:
    return (CheckSequentialWhiteSpaces(crossword) and CheckSymmetry(crossword))


if __name__ == "__main__":
    passing_case1 = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]

    passing_case2 = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]

    passing_case3 = [
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0]
    ]

    passing_case4 = [
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1]
    ]

    failing_case1 = [
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ]

    failing_case2 = [
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
    ]

    failing_case3 = [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ]

passing_cases = [
    passing_case1, passing_case2, passing_case3, passing_case4
]

failing_cases = [
    failing_case1, failing_case2, failing_case3
]

for i, case in enumerate(passing_cases):
    print('Passing case {} scores {} for validity'.format(i+1, TestCases(case)))

for i, case in enumerate(failing_cases):
    print('Failing case {} scores {} for validity'.format(i+1, TestCases(case)))
