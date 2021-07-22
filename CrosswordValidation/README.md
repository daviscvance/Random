# Crossword Compliance

## Prompt
Given a `m x n` matrix with black and white spaces denoted by 1 and 0 respectively, test 2 conditions to ensure its a valid crossword template.

## Condition 1
Each white square denoted by a 0 needs to be sequentially connected to another white square. A single line could connect all of the white spaces and there should be no islands.

## Condition 2
A crossword can be flipped 180 degrees and have the same structure i.e. has symmetry
on each axis.

## Personal notes:

This is a timed test, there is no ability to copy/paste code and there is no code compilation / execution.

I cheated a bit though so detailing my mistakes here:

50 min, 2 google searches, I didnt need either of them probably wasted 5 min just reading.
8 executions:

-   1 error from bad assumption.
-   1 to check row, column unpacking - found I needed to enumerate.
-   1 to check if I can look up a tuple in a list appropriately (duh moment).
-   2 for when I thought I had solutions.
-   1 for solution complete at 45 min mark.

Implicit error in my design causing case 3 to return True.
-   2 more for final testing.

Corrected and executed final final solution for 8 executions total @ 50min.
