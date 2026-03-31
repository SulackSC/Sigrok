import random
from discord.ext import commands


class IQ(commands.Cog):
    bot: commands.Bot

    def __init__(self, bot):
        self.bot = bot

    def iq_comment(self, iq: int) -> str:
        if iq < 50:
            return random.choice(
                [
                    f"{iq} IQ? You're legally required to have adult supervision online.",
                    f"{iq} IQ? That's not an IQ, that's a body temperature.",
                    f"{iq} IQ? Your neurons are on strike.",
                ]
            )
        elif iq < 70:
            return random.choice(
                [
                    f"{iq} IQ? Impressive—for a garden gnome.",
                    f"{iq} IQ? You're running on potato logic.",
                    f"{iq} IQ? Somehow both underclocked and overheating.",
                ]
            )
        elif iq < 85:
            return random.choice(
                [
                    f"{iq} IQ? Room temperature in Fahrenheit. Ambitious!",
                    f"{iq} IQ? If common sense were currency, you'd be in debt.",
                    f"{iq} IQ? Not dumb, just *retro* thinking.",
                ]
            )
        elif iq < 100:
            return random.choice(
                [
                    f"{iq} IQ? Average—like lukewarm tea and grey wallpaper.",
                    f"{iq} IQ? You're the human equivalent of buffering.",
                    f"{iq} IQ? Mid. In every possible way.",
                ]
            )
        elif iq < 115:
            return random.choice(
                [
                    f"{iq} IQ? Respectable! You'd survive in a cyberpunk dystopia.",
                    f"{iq} IQ? Smart enough to argue online, not smart enough to stop.",
                    f"{iq} IQ? Competent! Just don’t try to rewire the toaster.",
                ]
            )
        elif iq < 130:
            return random.choice(
                [
                    f"{iq} IQ? Pub quiz royalty. Google fears you.",
                    f"{iq} IQ? You're the reason the curve gets curved.",
                    f"{iq} IQ? Sharp. Dangerous if pointed at the wrong topic.",
                ]
            )
        elif iq < 145:
            return random.choice(
                [
                    f"{iq} IQ? Genius-tier. You probably have strong opinions about fonts.",
                    f"{iq} IQ? Smarter than you act online, that's for sure.",
                    f"{iq} IQ? High-functioning sarcasm generator.",
                ]
            )
        elif iq < 160:
            return random.choice(
                [
                    f"{iq} IQ? Borderline superhuman. Do you even experience loading times?",
                    f"{iq} IQ? Ideas per minute: lethal.",
                    f"{iq} IQ? You might be writing this simulation.",
                ]
            )
        else:
            return random.choice(
                [
                    f"{iq} IQ? Honestly, that’s terrifying.",
                    f"{iq} IQ? If you're not an alien, you're at least on contract.",
                    f"{iq} IQ? Go touch grass — for science.",
                ]
            )


def setup(bot):
    bot.add_cog(IQ(bot))
