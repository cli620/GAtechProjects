from testing_suite_ice_rover_for_viz import State, Submission

import math

import matplotlib
import matplotlib.pyplot as plt


PI = math.pi

# params = {'test_case': 1,
#                   'area_map': ['LLLLLLLL',
#                                'L..L...L',
#                                'L......L',
#                                'L..L...L',
#                                'L......L',
#                                'L.....@L',
#                                'LLLLLLLL'],
#                   'todo': [(1.5, -1.5),
#                            (5.0, -3.5)],
#                   'horizon_distance': 4.0,
#                   'max_distance': 2.0,
#                   'max_steering': PI / 2. + 0.01}


# params = {'test_case': 2,
#           'area_map': ['LLLLLLLL',
#                        'L..L..@L',
#                        'L......L',
#                        'L..LLLLL',
#                        'L..L...L',
#                        'L......L',
#                        'LLLLLLLL'],
#           'todo': [(4.5, -4.5),
#                    (1.5, -2.0)],
#           'horizon_distance': 4.0,
#           'max_distance': 2.0,
#           'max_steering': PI / 2. + 0.01}


# params = {'test_case': 3,
#           'area_map': ['LLLLLLLL',
#                        'L..L...L',
#                        'L......L',
#                        'L..LLLLL',
#                        'L..L..LL',
#                        'L.....@L',
#                        'LLLLLLLL'],
#           'todo': [(5.5, -2.0),
#                    (4.5, -4.5)],
#           'horizon_distance': 4.0,
#           'max_distance': 2.0,
#           'max_steering': PI / 2. + 0.01}

#
# params = {'test_case': 4,
#           'area_map': ['LLLLLLLLL',
#                        'L@.L...LL',
#                        'LL...L..L',
#                        'LLLLLLLLL'],
#           'todo': [(5.5, -1.5),
#                    (7.0, -2.5)],
#           'horizon_distance': 4.0,
#           'max_distance': 2.0,
#           'max_steering': PI / 2. + 0.01}


# params = {'test_case': 5,
#           'area_map': ['LLLLLLL',
#                        'LL.@.LL',
#                        'L..L..L',
#                        'LLLLLLL'],
#           'todo': [(1.5, -2.5),
#                    (5.5, -2.5)],
#           'horizon_distance': 4.0,
#           'max_distance': 2.0,
#           'max_steering': PI / 2. + 0.01}



# params = {'test_case': 6,
#           'area_map': ['LLLLLLL',
#                        'LL.@.LL',
#                        'LLLLLLL'],
#           'todo': [(2.5, -1.5),
#                    (4.5, -1.5)],
#           'horizon_distance': 4.0,
#           'max_distance': 2.0,
#           'max_steering': PI / 2. + 0.01}
#
# params = {'test_case': 7,
#             'area_map': ['@......',
#                         '...L...',
#                         '.......',
#                         '.....L.',
#                         'L......'],
#             'todo': [(0.5, -0.5),
#                     (1.5, -0.5),
#                     (2.5, -0.5),
#                     (3.5, -0.5),
#                     (4.5, -0.5),
#                     (5.5, -0.5),
#                     (6.5, -0.5),
#                     (0.5, -1.5),
#                     (1.5, -1.5),
#                     (5.5, -1.5),
#                     (6.5, -1.5),
#                     (0.5, -2.5),
#                     (1.5, -2.5),
#                     (5.5, -2.5),
#                     (6.5, -2.5),
#                     (0.5, -3.5),
#                     (1.5, -3.5),
#                     (5.5, -3.5),
#                     (6.5, -3.5),
#                     (0.5, -4.5),
#                     (1.5, -4.5),
#                     (2.5, -4.5),
#                     (3.5, -4.5),
#                     (4.5, -4.5),
#                     (5.5, -4.5),
#                     (6.5, -4.5)],
#             'horizon_distance': 4.0,
#             'max_distance': 2.0,
#             'max_steering': PI / 2. + 0.01}

params = {'test_case': 7,
# 'area_map': ['.....................................................',
#              '..LLLLLL..LLLLLLL..L.....L..LLLLLLLL..................',
#              '..L....L.....L.....L.....L..L.......L.................',
#              '..L....L.....L.....L.....L..L.......L.................',
#              '..L....L.....L.....LLLLLLL..L..LLLLL..................',
#              '..LLLLLL.....L...........L..L...L.....................',
#              '..L....L.....L...........L..L.....L...................',
#              '..L....L.....L...........L..L......L..................',
#              'LLL....L..LLLLLLL........L..L.......LLLLLLLLLLLLLLLL..',
#              '@L....................................................'],
'area_map': ['.......................................................',
             'LLLLLLL...LLLLLLLL..LLLLLLL..LLLLLLLL..LLLLLLL.LLLLLLLL.',
             'L......L..L.......L.L....LL..L.......L.L.......L.......L',
             'L......L..L.......L.L....LL..L.......L.L.......L.......L',
             'L......L..L..LLLLL..L....LL..L..LLLLL..L..LLLL.L..LLLLL.',
             'L......L..L...L.....LLLLLLL..L.........L.......L...L....',
             'L......L..L.....L...L....LL..L.........L.......L.....L..',
             'L......L..L......L..L....LL..L.........L.......L......L.',
             'LLLLLLL...L.......L.L....LL..L.........LLLLLLL.L.......L',
             '@.......................................................'],
'todo': [(4.0, -3.0), (53.0, -10.0), (32.0, -4.0), (7.0, -7.0), (53, 53)],
'horizon_distance': 3.0,
'max_distance': 1.0,
'max_steering': PI / 2. + 0.01}


if __name__ == "__main__":
    sub = Submission()
    spiral_path = sub.execute_student_plan(params['area_map'], params['todo'], params['max_distance'], params['max_steering'], params['horizon_distance'])

    x_sample, y_sample = zip(*params['todo'])

    x = spiral_path[1:, 0]
    y = spiral_path[1:, 1]

    fig, ax = plt.subplots()
    ax.plot(x, y)

    map = params['area_map']
    # print(map)
    landmarks = []
    for i in range(len(map)):
        for j in range(len(map[0])):
            # print(map[i][j])
            if map[i][j] =='L':
                landmarks.append((j,-i))
    # print(landmarks)
    x_landmarks, y_landmarks = zip(*landmarks)
        # for j in range(len(map[0][0])):
        #     print(map[i][])
    ax.scatter(x_landmarks, y_landmarks, color='green')

    ax.scatter(x_sample, y_sample, color='red', marker='X')
    ax.scatter(x[0], y[0], color='black')
    ax.scatter(x[-1], y[-1], color='blue')

    ax.grid()
    plt.show()
