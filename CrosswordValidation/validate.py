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

  if total == count or total == count - 1:
    return True  # 0 or 1 white spaces to check for neighbors.

  else:
    has_neighbors = []
    neighbor_count = 0
    for i, j in white_spaces:
      if (i - 1, j) in white_spaces:
        neighbor_count += 1
      if (i + 1, j) in white_spaces:
        neighbor_count += 1
      if (i, j - 1) in white_spaces:
        neighbor_count += 1
      if (i, j + 1) in white_spaces:
        neighbor_count += 1
      if neighbor_count > 1:
        has_neighbors.append((i, j))
        neighbor_count = 0
      else:
        return False  # No neighbors found for a white space.
    if len(has_neighbors) == len(white_spaces):
      return True

def CheckSymmetry(crossword: list) -> bool:
  valid = False
  new_crossword = []
  for row in crossword:
    new_crossword.append(row[::-1])
  if new_crossword == crossword and \
     crossword[::-1] == crossword:
     valid = True
  return valid


def TestCases(crossword: list) -> bool:
  if CheckSequentialWhiteSpaces(crossword) and CheckSymmetry(crossword):
    return True
  else:
    return False

if __name__ == "__main__":
  passing_case1 = (
  [
    [0, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 0]
  ])

  passing_case2 = (
  [
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
  ])

  failing_case1 = (
  [
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0]
  ])

  failing_case2 = (
  [
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
  ])

  cases = [passing_case1, passing_case2, failing_case1, failing_case2]

  for i, case in enumerate(cases):
    print('Case {} scores {} for validity'.format(i, TestCases(case)))
