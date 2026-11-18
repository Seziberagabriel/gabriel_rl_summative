"""Pygame top-down village renderer for the NutriBridge CHW Planner."""
from __future__ import annotations

import os

import numpy as np

CELL, MARGIN, PANEL = 84, 20, 290
GREEN, AMBER, RED = (76, 175, 80), (255, 179, 0), (229, 57, 53)
BG, GRIDC = (236, 240, 225), (205, 214, 190)


class Renderer:
    def __init__(self, headless: bool = False):
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame
        self.pg = pygame
        pygame.init()
        from environment.custom_env import GRID
        self.grid = GRID
        self.w = MARGIN * 2 + CELL * GRID + PANEL
        self.h = MARGIN * 2 + CELL * GRID
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("NutriBridge CHW Planner")
        self.font = pygame.font.SysFont("arial", 15)
        self.big = pygame.font.SysFont("arial", 22, bold=True)
        self.small = pygame.font.SysFont("arial", 12)
        self.clock = pygame.time.Clock()

    def _c(self, x, y):
        return MARGIN + x * CELL, MARGIN + y * CELL

    def draw(self, env, return_array=False):
        pg = self.pg
        from environment.custom_env import CLINIC, HOUSES, MAX_STOCK, N_WEEKS, MODERATE, SEVERE
        for e in pg.event.get():
            if e.type == pg.QUIT:
                raise KeyboardInterrupt
        s = self.screen
        s.fill(BG)
        for i in range(self.grid + 1):
            pg.draw.line(s, GRIDC, (MARGIN + i * CELL, MARGIN), (MARGIN + i * CELL, MARGIN + self.grid * CELL))
            pg.draw.line(s, GRIDC, (MARGIN, MARGIN + i * CELL), (MARGIN + self.grid * CELL, MARGIN + i * CELL))
        # clinic
        cx, cy = self._c(*CLINIC)
        pg.draw.rect(s, (255, 255, 255), (cx + 8, cy + 8, CELL - 16, CELL - 16), border_radius=6)
        pg.draw.rect(s, (200, 30, 30), (cx + CELL // 2 - 4, cy + 16, 8, CELL - 32))
        pg.draw.rect(s, (200, 30, 30), (cx + 16, cy + CELL // 2 - 4, CELL - 32, 8))
        s.blit(self.small.render("Health centre", True, (60, 60, 60)), (cx + 6, cy + CELL - 16))
        # households
        for i, (hx, hy) in enumerate(HOUSES):
            x, y = self._c(hx, hy)
            z = env.z[i]
            col = GREEN if z >= MODERATE else AMBER if z >= SEVERE else RED
            pg.draw.polygon(s, (140, 100, 70), [(x + 8, y + 34), (x + CELL // 2, y + 8), (x + CELL - 8, y + 34)])
            pg.draw.rect(s, col, (x + 14, y + 34, CELL - 28, CELL - 48), border_radius=3)
            s.blit(self.font.render(f"H{i}  z={z:+.1f}", True, (20, 20, 20)), (x + 8, y + CELL - 20))
            stale = min(env.since[i], N_WEEKS) / N_WEEKS
            pg.draw.rect(s, (90, 90, 90), (x + 14, y + 4, CELL - 28, 4))
            pg.draw.rect(s, (30, 110, 220), (x + 14, y + 4, int((CELL - 28) * (1 - stale)), 4))
        # CHW
        px, py = self._c(*env.pos)
        pg.draw.circle(s, (30, 80, 200), (px + CELL // 2, py + CELL // 2), 14)
        pg.draw.circle(s, (255, 255, 255), (px + CELL // 2, py + CELL // 2), 14, 2)
        # HUD
        hx0 = MARGIN * 2 + CELL * self.grid
        pg.draw.rect(s, (250, 250, 245), (hx0 - 10, 0, PANEL + 10, self.h))
        y = 18
        s.blit(self.big.render("NutriBridge CHW", True, (30, 70, 30)), (hx0, y)); y += 36
        lines = [f"Week {env._week() + 1}/{N_WEEKS}   (tick {env.tick})",
                 f"Supplements: {env.stock}/{MAX_STOCK}",
                 "LEAN SEASON" if env._is_lean() else "Normal season",
                 f"Children z >= -2: {env._frac_ok():.0%}",
                 f"Severe cases: {int(np.sum(env.z < SEVERE))}",
                 f"Return: {env.episode_return:+.2f}",
                 f"Last reward: {env.last_reward:+.2f}"]
        for t in lines:
            col = (200, 40, 40) if t == "LEAN SEASON" else (30, 30, 30)
            s.blit(self.font.render(t, True, col), (hx0, y)); y += 24
        y += 8
        s.blit(self.font.render("Last action:", True, (90, 90, 90)), (hx0, y)); y += 22
        s.blit(self.font.render(str(env.last_event), True, (20, 20, 120)), (hx0, y)); y += 40
        for col, t in [(GREEN, "z >= -2  adequate"), (AMBER, "-3 <= z < -2  moderate"), (RED, "z < -3  severe")]:
            pg.draw.rect(s, col, (hx0, y, 16, 16)); s.blit(self.small.render(t, True, (30, 30, 30)), (hx0 + 24, y + 1)); y += 22
        s.blit(self.small.render("blue bar = how recently screened", True, (80, 80, 80)), (hx0, y + 6))
        if return_array:
            return np.transpose(np.array(pg.surfarray.pixels3d(s)), (1, 0, 2)).copy()
        pg.display.flip()
        self.clock.tick(env.metadata["render_fps"])
        return None

    def close(self):
        self.pg.quit()
