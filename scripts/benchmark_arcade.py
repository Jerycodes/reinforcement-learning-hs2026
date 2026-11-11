"""Measure isolated, three-lap rule-bot runs in simulation seconds."""

from kart_rl.game.race import BOT_LEVELS, RaceRoom, advance_field, rule_action


def main():
    for level in BOT_LEVELS:
        room = RaceRoom()
        player = room.join()
        player.level = level
        for step in range(18000):
            action = rule_action(player)
            advance_field([player], [action], room.dt, 3, (step + 1) * room.dt)
            if player.finish_time is not None:
                break
        print(
            level,
            "Club / Standard",
            [round(t, 2) for t in player.laps],
            "Ziel" if player.finish_time is not None else "DNF",
        )
        room.leave(player.id)


if __name__ == "__main__":
    main()
