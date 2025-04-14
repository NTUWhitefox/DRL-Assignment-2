import student_agent

if __name__ == '__main__':
    env =  student_agent.Game2048Env()
    env.reset()
    while True:
        action = student_agent.get_action(env.board, env.score)
        print("1",env.board, end = '\n\n')
        next_next_state, new_score, done, _, next_state_before_spawn, score_before_move = env.step(action)
        if done:
            break
    print(env.score)