import random


def guess_number_game():
    number_to_guess = random.randint(1, 100)
    attempts = 0

    while True:
        try:
            user_guess = int(input("1から100までの数を入力してください: "))
            attempts += 1

            if user_guess < number_to_guess:
                print("もっと大きい数です。")
            elif user_guess > number_to_guess:
                print("もっと小さい数です。")
            else:
                print(f"正解です！{attempts}回目で当たりました。")
                break
        except ValueError:
            print("無効な入力です。整数を入力してください。")


guess_number_game()
