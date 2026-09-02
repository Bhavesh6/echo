#!/usr/bin/env python3
"""
preview.py - the tool for hand-tuning animations.

Taby's expressions feel authored because a person sat with them frame by frame.
The only way to get that here is to make the edit-and-look loop fast, so this
plays one clip on a loop, reloads it the instant the file changes on disk, and
shows a timeline of every track underneath.

    python preview.py delight

Edit animations/delight.json in any editor, save, and the change is on screen
before you have switched windows back. No restart, no export step.

Keys
    1-9      pick a clip          SPACE   pause / resume
    LEFT/RIGHT  step one frame    R       reload now
    , / .    slower / faster      T       toggle the timeline
    ESC      quit
"""
import os
import sys

import pygame

import anim as anim_mod
from face import Face

W, H = 900, 560
FPS = 60
TRACK_H = 13


def watch_times(directory):
    out = {}
    if os.path.isdir(directory):
        for f in os.listdir(directory):
            if f.lower().endswith(".json"):
                p = os.path.join(directory, f)
                try:
                    out[p] = os.path.getmtime(p)
                except OSError:
                    pass
    return out


def draw_timeline(screen, font, clip, t, y0):
    """Every track as a row, with the playhead. Seeing the curves next to the
    face is most of what makes tuning quick rather than guesswork."""
    if clip is None:
        return
    x0, x1 = 130, W - 20
    span = max(1e-6, clip.length)
    for i, (name, keys) in enumerate(sorted(clip.tracks.items())):
        y = y0 + i * TRACK_H
        screen.blit(font.render(name, True, (120, 140, 155)), (10, y - 5))
        lo = min(v for _, v in keys)
        hi = max(v for _, v in keys)
        rng = (hi - lo) or 1.0
        pts = []
        for k in range(41):
            f = k / 40.0
            v = clip.sample(name, f * span)
            pts.append((x0 + (x1 - x0) * f,
                        y + 4 - (v - lo) / rng * (TRACK_H - 6)))
        pygame.draw.lines(screen, (0, 150, 190), False, pts, 1)
        for kt, kv in keys:
            kx = x0 + (x1 - x0) * (kt / span)
            ky = y + 4 - (kv - lo) / rng * (TRACK_H - 6)
            pygame.draw.circle(screen, (255, 210, 90), (int(kx), int(ky)), 2)
    px = x0 + (x1 - x0) * min(1.0, t / span)
    pygame.draw.line(screen, (230, 90, 90), (px, y0 - 8),
                     (px, y0 + len(clip.tracks) * TRACK_H), 1)


def main():
    pygame.init()
    pygame.display.set_caption("Echo - animation preview")
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas,couriernew,monospace", 12)

    face = Face()
    player = anim_mod.Player()
    face.anim = player
    names = player.names()
    if not names:
        print("no animations found in %s" % player.directory)
        return
    idx = 0
    player.play(names[idx])
    mtimes = watch_times(player.directory)
    speed, paused, show_tl = 1.0, False, True
    check = 0.0

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                k = e.key
                if k == pygame.K_ESCAPE:
                    running = False
                elif pygame.K_1 <= k <= pygame.K_9:
                    i = k - pygame.K_1
                    if i < len(names):
                        idx = i
                        player.play(names[idx])
                elif k == pygame.K_SPACE:
                    paused = not paused
                elif k == pygame.K_r:
                    player.reload()
                    names = player.names()
                    player.play(names[min(idx, len(names) - 1)])
                elif k == pygame.K_t:
                    show_tl = not show_tl
                elif k == pygame.K_COMMA:
                    speed = max(0.1, speed - 0.1)
                elif k == pygame.K_PERIOD:
                    speed = min(3.0, speed + 0.1)
                elif k in (pygame.K_LEFT, pygame.K_RIGHT):
                    paused = True
                    step = 1.0 / FPS * (1 if k == pygame.K_RIGHT else -1)
                    player.t = max(0.0, player.t + step)

        # reload the moment a file changes on disk
        check += dt
        if check > 0.35:
            check = 0.0
            now = watch_times(player.directory)
            if now != mtimes:
                mtimes = now
                keep = player.t
                player.reload()
                names = player.names()
                if names:
                    player.play(names[min(idx, len(names) - 1)])
                    player.t = keep
                print("reloaded")

        clip = player.current
        if clip is not None:
            clip.loop = True                 # always loop while tuning
        if not paused:
            face.update(dt * speed)
        else:
            face.update(0.0)

        face.draw(screen)
        if clip is not None and show_tl:
            draw_timeline(screen, font, clip, player.t, H - 190)
        hud = "%s   %.2f/%.2fs   x%.1f %s" % (
            clip.name if clip else "-", player.t,
            clip.length if clip else 0.0, speed, "PAUSED" if paused else "")
        screen.blit(font.render(hud, True, (255, 235, 0)), (10, 8))
        screen.blit(font.render(
            "  ".join("%d:%s" % (i + 1, n) for i, n in enumerate(names)),
            True, (110, 130, 145)), (10, 24))
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    sys.exit(main())
