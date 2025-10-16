# Based on https://python101.readthedocs.io/pl/latest/pygame/pong/#
import pygame
from typing import Type
import skfuzzy as fuzz
import skfuzzy.control as fuzzcontrol

FPS = 30


class Board:
    def __init__(self, width: int, height: int):
        self.surface = pygame.display.set_mode((width, height), 0, 32)
        pygame.display.set_caption("AIFundamentals - PongGame")

    def draw(self, *args):
        background = (0, 0, 0)
        self.surface.fill(background)
        for drawable in args:
            drawable.draw_on(self.surface)

        pygame.display.update()


class Drawable:
    def __init__(self, x: int, y: int, width: int, height: int, color=(255, 255, 255)):
        self.width = width
        self.height = height
        self.color = color
        self.surface = pygame.Surface(
            [width, height], pygame.SRCALPHA, 32
        ).convert_alpha()
        self.rect = self.surface.get_rect(x=x, y=y)

    def draw_on(self, surface):
        surface.blit(self.surface, self.rect)


class Ball(Drawable):
    def __init__(
        self,
        x: int,
        y: int,
        radius: int = 20,
        color=(255, 10, 0),
        speed: int = 3,
    ):
        super(Ball, self).__init__(x, y, radius, radius, color)
        pygame.draw.ellipse(self.surface, self.color, [0, 0, self.width, self.height])
        self.x_speed = speed
        self.y_speed = speed
        self.start_speed = speed
        self.start_x = x
        self.start_y = y
        self.start_color = color
        self.last_collision = 0

    def bounce_y(self):
        self.y_speed *= -1

    def bounce_x(self):
        self.x_speed *= -1

    def bounce_y_power(self):
        self.color = (
            self.color[0],
            self.color[1] + 10 if self.color[1] < 255 else self.color[1],
            self.color[2],
        )
        pygame.draw.ellipse(self.surface, self.color, [0, 0, self.width, self.height])
        self.x_speed *= 1.1
        self.y_speed *= 1.1
        self.bounce_y()

    def reset(self):
        self.rect.x = self.start_x
        self.rect.y = self.start_y
        self.x_speed = self.start_speed
        self.y_speed = self.start_speed
        self.color = self.start_color
        self.bounce_y()

    def move(self, board: Board, *args):
        self.rect.x += round(self.x_speed)
        self.rect.y += round(self.y_speed)

        if self.rect.x < 0 or self.rect.x > (
            board.surface.get_width() - self.rect.width
        ):
            self.bounce_x()

        if self.rect.y < 0 or self.rect.y > (
            board.surface.get_height() - self.rect.height
        ):
            self.reset()

        timestamp = pygame.time.get_ticks()
        if timestamp - self.last_collision < FPS * 4:
            return

        for racket in args:
            if self.rect.colliderect(racket.rect):
                self.last_collision = pygame.time.get_ticks()
                if (self.rect.right < racket.rect.left + racket.rect.width // 4) or (
                    self.rect.left > racket.rect.right - racket.rect.width // 4
                ):
                    self.bounce_y_power()
                else:
                    self.bounce_y()


class Racket(Drawable):
    def __init__(
        self,
        x: int,
        y: int,
        width: int = 80,
        height: int = 20,
        color=(255, 255, 255),
        max_speed: int = 10,
    ):
        super(Racket, self).__init__(x, y, width, height, color)
        self.max_speed = max_speed
        self.surface.fill(color)

    def move(self, x: int, board: Board):
        delta = x - self.rect.x
        delta = self.max_speed if delta > self.max_speed else delta
        delta = -self.max_speed if delta < -self.max_speed else delta
        delta = 0 if (self.rect.x + delta) < 0 else delta
        delta = (
            0
            if (self.rect.x + self.width + delta) > board.surface.get_width()
            else delta
        )
        self.rect.x += delta


class Player:
    def __init__(self, racket: Racket, ball: Ball, board: Board) -> None:
        self.ball = ball
        self.racket = racket
        self.board = board

    def move(self, x: int):
        self.racket.move(x, self.board)

    def move_manual(self, x: int):
        """
        Do nothing, control is defined in derived classes
        """
        pass

    def act(self, x_diff: int, y_diff: int):
        """
        Do nothing, control is defined in derived classes
        """
        pass


class PongGame:
    def __init__(
        self, width: int, height: int, player1: Type[Player], player2: Type[Player]
    ):
        pygame.init()
        self.board = Board(width, height)
        self.fps_clock = pygame.time.Clock()
        self.ball = Ball(width // 2, height // 2)

        self.opponent_paddle = Racket(x=width // 2, y=0)
        self.oponent = player1(self.opponent_paddle, self.ball, self.board)

        self.player_paddle = Racket(x=width // 2, y=height - 20)
        self.player = player2(self.player_paddle, self.ball, self.board)

    def run(self):
        while not self.handle_events():
            self.ball.move(self.board, self.player_paddle, self.opponent_paddle)
            self.board.draw(
                self.ball,
                self.player_paddle,
                self.opponent_paddle,
            )
            self.oponent.act(
                self.oponent.racket.rect.centerx - self.ball.rect.centerx,
                self.oponent.racket.rect.centery - self.ball.rect.centery,
            )
            self.player.act(
                self.player.racket.rect.centerx - self.ball.rect.centerx,
                self.player.racket.rect.centery - self.ball.rect.centery,
            )
            self.fps_clock.tick(FPS)

    def handle_events(self):
        for event in pygame.event.get():
            if (event.type == pygame.QUIT) or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                pygame.quit()
                return True
        keys = pygame.key.get_pressed()
        if keys[pygame.constants.K_LEFT]:
            self.player.move_manual(0)
        elif keys[pygame.constants.K_RIGHT]:
            self.player.move_manual(self.board.surface.get_width())
        return False


class NaiveOponent(Player):
    def __init__(self, racket: Racket, ball: Ball, board: Board):
        super(NaiveOponent, self).__init__(racket, ball, board)

    def act(self, x_diff: int, y_diff: int):
        x_cent = self.ball.rect.centerx
        self.move(x_cent)


class HumanPlayer(Player):
    def __init__(self, racket: Racket, ball: Ball, board: Board):
        super(HumanPlayer, self).__init__(racket, ball, board)

    def move_manual(self, x: int):
        self.move(x)


# ----------------------------------
# DO NOT MODIFY CODE ABOVE THIS LINE
# ----------------------------------

# import numpy as np
# import matplotlib.pyplot as plt


# ---------- Paste below the DO NOT MODIFY line ----------

import numpy as np

import numpy as np


class FuzzyPlayer(Player):
    """
    Minimal Mamdani fuzzy controller with:
      • Intercept prediction (with X-wall reflections),
      • Simple edge-aiming (targets the outer 24.5% of the paddle width),
      • UP/DOWN sprints,
      • Tiny anti-stall.

    Notes (edge hits, required for grade “5”):
      Power-bounce happens when the collision occurs in the outer quarter of the paddle.
      We aim slightly inside that zone (24.5%) and compute the REQUIRED paddle center
      from ball/paddle widths so the inequality is satisfied at impact.
    """

    # ---------- basic knobs ----------
    EDGE_ENABLED = True
    EDGE_FRACTION = 0.245  # aim inside outer quarter (< 0.25)
    EDGE_MARGIN_PX = 2.0  # push deeper into the zone for robustness
    EDGE_GATE_FRAMES = 36.0  # enable edge-aiming if ETA < this (frames @30 FPS)
    EDGE_ALIGN_PX = 40.0  # only edge-aim when almost aligned in X

    UP_SPRINT_ERR = 6.0  # when ball goes up: sprint to its X if |err| > this
    DOWN_SPRINT_ETA = 18.0  # when ball descends: sprint near impact
    DOWN_SPRINT_ERR = 85.0  # or when predicted X error is large
    DEAD_ZONE = 3.0  # do nothing if |error| is tiny
    MIN_STEP = 1.5  # small nudge if fuzzy returns ~0 with non-tiny error

    # fuzzy universes scaling for ETA (frames → 0..400)
    ETA_TO_UNI = 400.0 / 120.0

    def __init__(self, racket: Racket, ball: Ball, board: Board):
        super().__init__(racket, ball, board)

        # universes
        self.x_uni = np.linspace(-400, 400, 801)  # (paddle.centerx - target.centerx)
        self.y_uni = np.linspace(0, 400, 401)  # scaled ETA
        self.v_uni = np.linspace(-10, 10, 401)  # output velocity

        # antecedents / consequent
        self.a_x = fuzzcontrol.Antecedent(self.x_uni, "x_err")
        self.a_y = fuzzcontrol.Antecedent(self.y_uni, "eta")
        self.c_v = fuzzcontrol.Consequent(
            self.v_uni, "vel", defuzzify_method="centroid"
        )

        # memberships (kept small & symmetric)
        self.a_x["far_left"] = fuzz.trapmf(self.x_uni, [-400, -400, -200, -80])
        self.a_x["left"] = fuzz.trimf(self.x_uni, [-200, -80, 0])
        self.a_x["center"] = fuzz.trimf(self.x_uni, [-40, 0, 40])
        self.a_x["right"] = fuzz.trimf(self.x_uni, [0, 80, 200])
        self.a_x["far_right"] = fuzz.trapmf(self.x_uni, [80, 200, 400, 400])

        self.a_y["close"] = fuzz.trapmf(self.y_uni, [0, 0, 60, 140])
        self.a_y["mid"] = fuzz.trimf(self.y_uni, [100, 200, 300])
        self.a_y["far"] = fuzz.trapmf(self.y_uni, [240, 330, 400, 400])

        self.c_v["fast_left"] = fuzz.trapmf(self.v_uni, [-10, -10, -8, -5])
        self.c_v["left"] = fuzz.trimf(self.v_uni, [-7, -4, -1])
        self.c_v["stay"] = fuzz.trimf(self.v_uni, [-1, 0, 1])
        self.c_v["right"] = fuzz.trimf(self.v_uni, [1, 4, 7])
        self.c_v["fast_right"] = fuzz.trapmf(self.v_uni, [5, 8, 10, 10])

        # compact rule base
        rules = [
            fuzzcontrol.Rule(
                self.a_x["far_left"] & self.a_y["close"], self.c_v["fast_right"]
            ),
            fuzzcontrol.Rule(
                self.a_x["left"] & self.a_y["close"], self.c_v["fast_right"]
            ),
            fuzzcontrol.Rule(self.a_x["center"] & self.a_y["close"], self.c_v["stay"]),
            fuzzcontrol.Rule(
                self.a_x["right"] & self.a_y["close"], self.c_v["fast_left"]
            ),
            fuzzcontrol.Rule(
                self.a_x["far_right"] & self.a_y["close"], self.c_v["fast_left"]
            ),
            fuzzcontrol.Rule(self.a_x["far_left"] & self.a_y["mid"], self.c_v["right"]),
            fuzzcontrol.Rule(self.a_x["left"] & self.a_y["mid"], self.c_v["right"]),
            fuzzcontrol.Rule(self.a_x["center"] & self.a_y["mid"], self.c_v["stay"]),
            fuzzcontrol.Rule(self.a_x["right"] & self.a_y["mid"], self.c_v["left"]),
            fuzzcontrol.Rule(self.a_x["far_right"] & self.a_y["mid"], self.c_v["left"]),
            fuzzcontrol.Rule(self.a_x["far_left"] & self.a_y["far"], self.c_v["right"]),
            fuzzcontrol.Rule(self.a_x["left"] & self.a_y["far"], self.c_v["right"]),
            fuzzcontrol.Rule(self.a_x["center"] & self.a_y["far"], self.c_v["stay"]),
            fuzzcontrol.Rule(self.a_x["right"] & self.a_y["far"], self.c_v["left"]),
            fuzzcontrol.Rule(self.a_x["far_right"] & self.a_y["far"], self.c_v["left"]),
        ]
        self.ctrl = fuzzcontrol.ControlSystem(rules)

    # ---------- helpers ----------
    def _reflect_x(
        self, x0: float, vx: float, t: float, xmin: float, xmax: float
    ) -> float:
        """Mirror-like reflection in [xmin, xmax] for constant vx and time t."""
        L = xmax - xmin
        if L <= 0:
            return x0
        d = (x0 - xmin) + vx * t
        period = 2.0 * L
        m = d % period
        if m < 0:
            m += period
        return (xmin + m) if (m <= L) else (xmax - (m - L))

    def _predict_intercept(self):
        """Return (x_intercept, eta_frames) where the ball meets our paddle level; None if ball is going up."""
        vy = float(self.ball.y_speed)
        if vy <= 0.0:
            return None, None
        y_target = float(self.racket.rect.top) - (self.ball.height * 0.5)
        eta = (y_target - float(self.ball.rect.centery)) / vy
        if eta <= 0.0:
            return None, None
        xmin = 0.5 * self.ball.width
        xmax = float(self.board.surface.get_width()) - 0.5 * self.ball.width
        x_hit = self._reflect_x(
            float(self.ball.rect.centerx), float(self.ball.x_speed), eta, xmin, xmax
        )
        return x_hit, eta

    def _target_center_for_edge(self, x_ball: float, edge: str) -> float:
        """
        Compute the REQUIRED paddle center so the collision is inside the outer EDGE_FRACTION of the paddle.
        """
        w_p = float(self.racket.width)
        w_b = float(self.ball.width)
        half_b = 0.5 * w_b
        core = w_p * (0.5 - self.EDGE_FRACTION)

        if edge == "left":
            # ball.right < paddle.left + f*w ⇒ C_paddle > x_ball + w_b/2 + (0.5-f)*w_p
            return (x_ball + half_b + core) + self.EDGE_MARGIN_PX
        else:  # "right"
            # ball.left  > paddle.right - f*w ⇒ C_paddle < x_ball - w_b/2 - (0.5-f)*w_p
            return (x_ball - half_b - core) - self.EDGE_MARGIN_PX

    # ---------- main control ----------
    def act(self, x_diff: int, y_diff: int):
        v = self.make_decision()
        self.move(self.racket.rect.x + v)

    def make_decision(self) -> int:
        x_hit, eta = self._predict_intercept()

        # Ball going up → sprint under current ball.x (simple and fast)
        if x_hit is None:
            err = float(self.racket.rect.centerx) - float(self.ball.rect.centerx)
            if abs(err) > self.UP_SPRINT_ERR:
                v = -np.sign(err) * self.racket.max_speed
            else:
                v = 0.0
            return int(
                max(-self.racket.max_speed, min(self.racket.max_speed, round(v)))
            )

        # Ball descending: choose target (edge or center)
        target = float(x_hit)
        err_now = float(self.racket.rect.centerx) - target

        if (
            self.EDGE_ENABLED
            and (eta < self.EDGE_GATE_FRAMES)
            and (abs(err_now) < self.EDGE_ALIGN_PX)
        ):
            side = "right" if (self.ball.x_speed > 0.0) else "left"
            target = self._target_center_for_edge(x_ball=target, edge=side)

        # Fuzzy compute
        x_err = float(self.racket.rect.centerx) - target
        eta_scaled = max(0.0, min(400.0, eta * self.ETA_TO_UNI))

        sim = fuzzcontrol.ControlSystemSimulation(self.ctrl, cache=False)
        try:
            sim.input["x_err"] = float(np.clip(x_err, -400.0, 400.0))
            sim.input["eta"] = float(eta_scaled)
            sim.compute()
            v = float(sim.output["vel"])
        except Exception:
            v = 0.0

        # Simple sprints / anti-stall
        if (eta < self.DOWN_SPRINT_ETA) or (abs(x_err) > self.DOWN_SPRINT_ERR):
            v = -np.sign(x_err) * self.racket.max_speed
        if abs(v) < 1.0 and abs(x_err) > self.DEAD_ZONE:
            v = -np.sign(x_err) * self.MIN_STEP

        v = max(-self.racket.max_speed, min(self.racket.max_speed, v))
        return int(round(v))


class FuzzyPlayerTSK(Player):
    """
    Minimal zero-order TSK controller with:
      • Intercept prediction (with X-wall reflections),
      • Simple edge-aiming (outer 24.5% zone),
      • UP/DOWN sprints,
      • Tiny anti-stall.

    Uses a small TSK rule base to output a crisp horizontal velocity.
    """

    EDGE_ENABLED = True
    EDGE_FRACTION = 0.245
    EDGE_MARGIN_PX = 2.0
    EDGE_GATE_FRAMES = 36.0
    EDGE_ALIGN_PX = 40.0

    UP_SPRINT_ERR = 6.0
    DOWN_SPRINT_ETA = 18.0
    DOWN_SPRINT_ERR = 85.0
    DEAD_ZONE = 3.0
    MIN_STEP = 1.5

    ETA_TO_UNI = 400.0 / 120.0

    def __init__(self, racket: Racket, ball: Ball, board: Board):
        super().__init__(racket, ball, board)

        self.x_uni = np.linspace(-400, 400, 801)
        self.y_uni = np.linspace(0, 400, 401)

        # compact MFs
        self.x_mf = {
            "far_left": fuzz.trapmf(self.x_uni, [-400, -400, -200, -80]),
            "left": fuzz.trimf(self.x_uni, [-200, -80, 0]),
            "center": fuzz.trimf(self.x_uni, [-40, 0, 40]),
            "right": fuzz.trimf(self.x_uni, [0, 80, 200]),
            "far_right": fuzz.trapmf(self.x_uni, [80, 200, 400, 400]),
        }
        self.y_mf = {
            "close": fuzz.trapmf(self.y_uni, [0, 0, 60, 140]),
            "mid": fuzz.trimf(self.y_uni, [100, 200, 300]),
            "far": fuzz.trapmf(self.y_uni, [240, 330, 400, 400]),
        }

        # crisp consequents (kept small, symmetric)
        self.outputs = {
            ("far_left", "close"): +9,
            ("left", "close"): +7,
            ("center", "close"): 0,
            ("right", "close"): -7,
            ("far_right", "close"): -9,
            ("far_left", "mid"): +6,
            ("left", "mid"): +4,
            ("center", "mid"): 0,
            ("right", "mid"): -4,
            ("far_right", "mid"): -6,
            ("far_left", "far"): +4,
            ("left", "far"): +2,
            ("center", "far"): 0,
            ("right", "far"): -2,
            ("far_right", "far"): -4,
        }

    # ---------- helpers ----------
    def _reflect_x(self, x0, vx, t, xmin, xmax):
        L = xmax - xmin
        if L <= 0:
            return x0
        d = (x0 - xmin) + vx * t
        period = 2.0 * L
        m = d % period
        if m < 0:
            m += period
        return (xmin + m) if (m <= L) else (xmax - (m - L))

    def _predict_intercept(self):
        vy = float(self.ball.y_speed)
        if vy <= 0.0:
            return None, None
        y_target = float(self.racket.rect.top) - (self.ball.height * 0.5)
        eta = (y_target - float(self.ball.rect.centery)) / vy
        if eta <= 0.0:
            return None, None
        xmin = 0.5 * self.ball.width
        xmax = float(self.board.surface.get_width()) - 0.5 * self.ball.width
        x_hit = self._reflect_x(
            float(self.ball.rect.centerx), float(self.ball.x_speed), eta, xmin, xmax
        )
        return x_hit, eta

    def _target_center_for_edge(self, x_ball: float, edge: str) -> float:
        w_p = float(self.racket.width)
        w_b = float(self.ball.width)
        half_b = 0.5 * w_b
        core = w_p * (0.5 - self.EDGE_FRACTION)
        if edge == "left":
            return (x_ball + half_b + core) + self.EDGE_MARGIN_PX
        else:
            return (x_ball - half_b - core) - self.EDGE_MARGIN_PX

    # ---------- main control ----------
    def act(self, x_diff: int, y_diff: int):
        v = self.make_decision()
        self.move(self.racket.rect.x + v)

    def make_decision(self) -> int:
        x_hit, eta = self._predict_intercept()

        # upward: sprint below ball.x
        if x_hit is None:
            err = float(self.racket.rect.centerx) - float(self.ball.rect.centerx)
            v = (
                -np.sign(err) * self.racket.max_speed
                if abs(err) > self.UP_SPRINT_ERR
                else 0.0
            )
            return int(
                max(-self.racket.max_speed, min(self.racket.max_speed, round(v)))
            )

        # descending: pick target (edge if near contact), then TSK average
        target = float(x_hit)
        err_now = float(self.racket.rect.centerx) - target
        if (
            self.EDGE_ENABLED
            and (eta < self.EDGE_GATE_FRAMES)
            and (abs(err_now) < self.EDGE_ALIGN_PX)
        ):
            side = "right" if (self.ball.x_speed > 0.0) else "left"
            target = self._target_center_for_edge(target, side)

        x_err = float(self.racket.rect.centerx) - target
        eta_scaled = max(0.0, min(400.0, eta * self.ETA_TO_UNI))

        # degrees
        x_deg = {
            k: fuzz.interp_membership(self.x_uni, mf, x_err)
            for k, mf in self.x_mf.items()
        }
        y_deg = {
            k: fuzz.interp_membership(self.y_uni, mf, eta_scaled)
            for k, mf in self.y_mf.items()
        }

        num = 0.0
        den = 0.0
        for (xn, yn), out in self.outputs.items():
            w = x_deg[xn] * y_deg[yn]
            num += w * out
            den += w
        v = (num / den) if den > 1e-8 else 0.0

        # simple sprints / anti-stall
        if (eta < self.DOWN_SPRINT_ETA) or (abs(x_err) > self.DOWN_SPRINT_ERR):
            v = -np.sign(x_err) * self.racket.max_speed
        if abs(v) < 1.0 and abs(x_err) > self.DEAD_ZONE:
            v = -np.sign(x_err) * self.MIN_STEP

        v = max(-self.racket.max_speed, min(self.racket.max_speed, v))
        return int(round(v))


if __name__ == "__main__":
    # game = PongGame(800, 400, NaiveOponent, HumanPlayer)
    # game = PongGame(800, 400, NaiveOponent, FuzzyPlayer)
    game = PongGame(800, 400, NaiveOponent, FuzzyPlayerTSK)
    game.run()
