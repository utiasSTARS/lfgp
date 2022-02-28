import copy

# CONFIG
max_iters = 5000
alpha = .1
stop_tol = 1e-5
max_action_q_learning = False  # if true, q learning instead of SARSA
initial_q = 0

q_table = {
    (1, None): 0,
    (2, None): 0,
    (3, None): 0,
    (4, None): 0,
    (5, None): 0,
    (1, 2): initial_q,
    (1, 5): initial_q,
    (2, 3): initial_q,
    (2, 6): initial_q,
    (3, 4): initial_q,
    (3, 6): initial_q,
    (4, 5): initial_q,
    (4, 6): initial_q,
    (5, 1): initial_q,
    (5, 5): initial_q,
    (6, 1): initial_q,
    (6, 6): initial_q
}
q_new = copy.deepcopy(q_table)

r_table = {
    (1, 2): 1,
    (1, 5): -1,
    (2, 3): 1,
    (2, 6): -1,
    (3, 4): 1,
    (3, 6): -1,
    (4, 5): 1,
    (4, 6): -1,
    (5, 1): -1,
    (5, 5): 1,
    (6, 1): -1,
    (6, 6): -1
}

buffer = [
    ((1, 5), (5, 5)),  # ep 2
    ((5, 5), (5, 5)),
    ((5, 5), (5, 5)),
    ((5, 5), (5, 5)),
    ((5, 5), (5, None)),
    ((1, 2), (2, 6)),  # ep 1
    ((2, 6), (6, 1)),
    ((6, 1), (1, 5)),
    ((1, 5), (5, 5)),
    ((5, 5), (5, None)),
    ((1, 2), (2, 3)),  # ep 3
    ((2, 3), (3, 6)),
    ((3, 6), (6, 1)),
    ((6, 1), (1, 5)),
    ((1, 5), (5, None)),
]

valid_update_states = [pair[0] for pair in buffer]

for ep_i in range(len(buffer) // 5):

    for i in range(max_iters):

        short_buffer = buffer[:(ep_i + 1) * 5]
        for (state_act, next_state_act) in short_buffer:

            if next_state_act[1] is None:  # equivalent of done, so update exclusively uses reward
                q_new[state_act] = q_new[state_act] + alpha * (r_table[state_act] - q_new[state_act])

            else:
                if max_action_q_learning:
                    max_state_act_val = -1e100
                    max_state_act = None

                    for qt_state_act in q_table:
                        if qt_state_act[0] == next_state_act[0] and q_new[qt_state_act] > max_state_act_val\
                                and qt_state_act[1] is not None and qt_state_act in valid_update_states:
                            # print(f"New max: next state act {qt_state_act}, "
                            #       f"val {q_new[qt_state_act]} used for updating {state_act}")
                            max_state_act_val = q_new[qt_state_act]
                            max_state_act = qt_state_act

                    # print(f"Selected max next state act {max_state_act}, "
                    #       f"val {q_new[max_state_act]} used for updating {state_act}")

                    q_new[state_act] = q_new[state_act] + alpha * (r_table[state_act] +
                                                                   q_new[max_state_act] - q_new[state_act])
                else:
                    q_new[state_act] = q_new[state_act] + alpha * (r_table[state_act] +
                                                                   q_new[next_state_act] - q_new[state_act])

        total_diff = 0
        for state_act in q_table.keys():
            total_diff += abs(q_new[state_act] - q_table[state_act])

        q_table = copy.deepcopy(q_new)
        if total_diff < stop_tol:
            break

    print(f"End of ep: {ep_i}, current q(1, 5): {q_table[(1, 5)]}, q(1, 2): {q_table[(1, 2)]}")

for k in q_table:
    q_table[k] = round(q_table[k], 4)

print(f"Final Q Table after {i} iterations: {q_table}")