def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = BoatGameAI()

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Sauvegarder les points non atteints
            remaining_goals = game.path.copy()

            # Réinitialiser le jeu sans régénérer les objectifs et obstacles
            game.reset(regenerate=False)

            # Restaurer les points non atteints
            game.path = remaining_goals
            game.current_goal = game._find_closest_goal()
            game.previous_distance = game.boat.distance_to_goal(game.current_goal.x, game.current_goal.y)

            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
