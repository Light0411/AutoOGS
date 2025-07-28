import time
from main import AdvancedGoBot

def run_auto_bot(interval_sec=10):
    bot = AdvancedGoBot()
    move_num = 1
    while True:
        print(f"\n🎯 [Move #{move_num}] Thinking...")
        bot.make_move()
        move_num += 1
        time.sleep(interval_sec)

if __name__ == "__main__":
    try:
        run_auto_bot(interval_sec=3)
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped manually.")
