from torch import sqrt
def CST3D_StVK_W(x, lame, mu, m_Einv, m_A0, m_h):
    t1 = m_Einv[0, 0] ** 2
    t2 = m_Einv[1, 1] ** 2
    t3 = t2 * t1
    t4 = x[0, 0] ** 2
    t5 = x[1, 1] ** 2
    t6 = t5 * t4
    t9 = x[2, 1] * x[1, 1] * t4
    t12 = x[1, 2] ** 2
    t13 = t12 * t4
    t16 = x[2, 2] * x[1, 2] * t4
    t19 = x[2, 1] ** 2
    t20 = t19 * t4
    t22 = x[2, 2] ** 2
    t23 = t22 * t4
    t25 = x[0, 0] * t3
    t26 = x[1, 0] * x[0, 1]
    t27 = x[1, 1] * t26
    t30 = x[2, 1] * t26
    t33 = x[1, 1] * x[0, 1]
    t34 = x[2, 0] * t33
    t38 = x[0, 1] * x[2, 0] * x[2, 1]
    t41 = x[1, 0] * x[0, 2]
    t42 = x[1, 2] * t41
    t45 = t13 * t3 - 2 * t16 * t3 + t20 * t3 + t23 * t3 - 2 * t27 * t25 + 2 * t30 * t25 + 2 * t34 * t25 - 2 * t38 * t25 - 2 * t42 * t25 + t6 * t3 - 2 * t9 * t3
    t46 = x[2, 2] * t41
    t49 = x[1, 2] * x[0, 2]
    t50 = x[2, 0] * t49
    t54 = x[0, 2] * x[2, 0] * x[2, 2]
    t57 = x[1, 1] * x[1, 0]
    t58 = x[2, 1] * t57
    t61 = x[1, 2] * x[1, 0]
    t62 = x[2, 2] * t61
    t65 = x[1, 0] * x[0, 0]
    t66 = t19 * t65
    t69 = t22 * t65
    t73 = x[2, 0] * t5 * x[0, 0]
    t76 = x[2, 0] * x[1, 1]
    t77 = x[2, 1] * t76
    t81 = x[2, 0] * t12 * x[0, 0]
    t84 = x[2, 0] * x[1, 2]
    t85 = x[2, 2] * t84
    t88 = x[0, 1] ** 2
    t89 = x[1, 0] ** 2
    t90 = t89 * t88
    t92 = 2 * t46 * t25 + 2 * t50 * t25 - 2 * t54 * t25 + 2 * t58 * t25 + 2 * t62 * t25 + 2 * t77 * t25 + 2 * t85 * t25 - 2 * t66 * t3 - 2 * t69 * t3 - 2 * t73 * t3 - 2 * t81 * t3 + t90 * t3
    t95 = x[2, 0] * x[1, 0] * t88
    t98 = t12 * t88
    t101 = x[2, 2] * x[1, 2] * t88
    t104 = x[2, 0] ** 2
    t105 = t104 * t88
    t107 = t22 * t88
    t109 = x[0, 1] * t3
    t110 = x[1, 1] * x[0, 2]
    t111 = x[1, 2] * t110
    t114 = x[2, 2] * t110
    t117 = x[2, 1] * t49
    t121 = x[0, 2] * x[2, 1] * x[2, 2]
    t125 = x[2, 1] * t89 * x[0, 1]
    t128 = x[2, 0] * t57
    t131 = x[2, 0] * x[1, 0]
    t132 = x[2, 1] * t131
    t135 = -2 * t101 * t3 + t105 * t3 + t107 * t3 - 2 * t111 * t109 + 2 * t114 * t109 + 2 * t117 * t109 - 2 * t121 * t109 + 2 * t128 * t109 + 2 * t132 * t109 - 2 * t125 * t3 - 2 * t95 * t3 + t98 * t3
    t136 = x[1, 2] * x[1, 1]
    t137 = x[2, 2] * t136
    t140 = t104 * t33
    t143 = t22 * t33
    t147 = x[2, 1] * t12 * x[0, 1]
    t150 = x[2, 1] * x[1, 2]
    t151 = x[2, 2] * t150
    t154 = x[0, 2] ** 2
    t155 = t89 * t154
    t158 = x[2, 0] * x[1, 0] * t154
    t161 = t5 * t154
    t164 = x[2, 1] * x[1, 1] * t154
    t167 = t104 * t154
    t169 = t19 * t154
    t172 = x[2, 2] * t89 * x[0, 2]
    t175 = 2 * t137 * t109 + 2 * t151 * t109 - 2 * t140 * t3 - 2 * t143 * t3 - 2 * t147 * t3 + t155 * t3 - 2 * t158 * t3 + t161 * t3 - 2 * t164 * t3 + t167 * t3 + t169 * t3 - 2 * t172 * t3
    t178 = x[0, 2] * t3
    t179 = x[2, 0] * t61
    t182 = x[2, 2] * t131
    t186 = x[2, 2] * t5 * x[0, 2]
    t189 = x[2, 1] * t136
    t192 = x[2, 1] * x[1, 1]
    t193 = x[2, 2] * t192
    t196 = t104 * t49
    t199 = t19 * t49
    t202 = t19 * t89
    t204 = t22 * t89
    t206 = x[1, 0] * t3
    t211 = 2 * t179 * t178 + 2 * t182 * t178 + 2 * t189 * t178 + 2 * t193 * t178 - 2 * t186 * t3 - 2 * t196 * t3 - 2 * t199 * t3 + t202 * t3 + t204 * t3 - 2 * t77 * t206 - 2 * t85 * t206
    t212 = t104 * t5
    t214 = t22 * t5
    t219 = t104 * t12
    t221 = t19 * t12
    t223 = m_Einv[0, 0] * m_Einv[0, 1]
    t224 = m_Einv[1, 0] * t223
    t225 = t4 * m_Einv[1, 1]
    t235 = x[1, 2] * x[2, 2]
    t246 = m_Einv[1, 0] * m_Einv[1, 1] * t223
    t247 = x[0, 0] * x[0, 1]
    t251 = -2 * t12 * t225 * t224 - 2 * t151 * x[1, 1] * t3 - 2 * t19 * t225 * t224 + 4 * t192 * t225 * t224 - 2 * t22 * t225 * t224 + 4 * t235 * t225 * t224 - 2 * t5 * t225 * t224 + 4 * t57 * t247 * t246 + t212 * t3 + t214 * t3 + t219 * t3 + t221 * t3
    t258 = x[2, 0] * x[2, 1]
    t261 = x[0, 0] * x[0, 2]
    t269 = x[2, 0] * x[2, 2]
    t276 = m_Einv[1, 1] * x[0, 0]
    t286 = t19 * x[1, 0] * t276 * t224 + t22 * x[1, 0] * t276 * t224 + x[2, 0] * t5 * t276 * t224 - x[1, 0] * x[2, 1] * t247 * t246 - x[1, 0] * x[2, 2] * t261 * t246 - t192 * t65 * t246 - t235 * t65 * t246 + t258 * t247 * t246 - t76 * t247 * t246 + t269 * t261 * t246 + t61 * t261 * t246 - t84 * t261 * t246
    t299 = t88 * m_Einv[1, 1]
    t318 = x[0, 1] * x[0, 2]
    t329 = 4 * x[2, 0] * t12 * t276 * t224 - 4 * t258 * x[0, 0] * x[1, 1] * t246 - 4 * t269 * x[0, 0] * x[1, 2] * t246 - 4 * x[1, 1] * x[2, 2] * t318 * t246 - 2 * t104 * t299 * t224 - 2 * t12 * t299 * t224 + 4 * t131 * t299 * t224 + 4 * t136 * t318 * t246 - 4 * t150 * t318 * t246 - 2 * t22 * t299 * t224 + 4 * t235 * t299 * t224 - 2 * t89 * t299 * t224
    t333 = x[2, 1] * x[2, 2]
    t337 = m_Einv[1, 1] * x[0, 1]
    t367 = t154 * m_Einv[1, 1]
    t374 = 4 * t104 * x[1, 1] * t337 * t224 + 4 * x[2, 1] * t12 * t337 * t224 + 4 * t22 * x[1, 1] * t337 * t224 + 4 * x[2, 1] * t89 * t337 * t224 - 4 * t333 * x[0, 1] * x[1, 2] * t246 + 4 * t131 * t367 * t224 - 2 * t89 * t367 * t224 - 4 * t235 * t33 * t246 - 4 * t258 * t26 * t246 - 4 * t76 * t26 * t246 + 4 * t333 * t318 * t246
    t387 = m_Einv[1, 1] * x[0, 2]
    t416 = 4 * t104 * x[1, 2] * t387 * t224 + 4 * t19 * x[1, 2] * t387 * t224 + 4 * x[2, 2] * t5 * t387 * t224 + 4 * x[2, 2] * t89 * t387 * t224 - 2 * t104 * t367 * t224 - 4 * t150 * t110 * t246 - 4 * t333 * t110 * t246 - 2 * t19 * t367 * t224 + 4 * t192 * t367 * t224 - 2 * t5 * t367 * t224 - 4 * t269 * t41 * t246 - 4 * t84 * t41 * t246
    t418 = t89 * m_Einv[1, 1]
    t431 = t5 * m_Einv[1, 1]
    t441 = t12 * m_Einv[1, 1]
    t448 = m_Einv[0, 1] ** 2
    t449 = m_Einv[1, 0] ** 2
    t450 = t449 * t448
    t455 = -2 * t104 * t431 * t224 - 2 * t104 * t441 * t224 + 4 * t333 * t136 * t246 - 2 * t19 * t418 * t224 - 2 * t19 * t441 * t224 - 2 * t22 * t418 * t224 - 2 * t22 * t431 * t224 + 4 * t258 * t57 * t246 + 4 * t269 * t61 * t246 + t13 * t450 + t6 * t450 - 2 * t9 * t450
    t460 = x[0, 0] * t450
    t479 = -2 * t16 * t450 + t20 * t450 + t23 * t450 - 2 * t27 * t460 + 2 * t30 * t460 + 2 * t34 * t460 - 2 * t38 * t460 - 2 * t42 * t460 + 2 * t46 * t460 + 2 * t50 * t460 - 2 * t54 * t460 + 2 * t58 * t460
    t503 = -2 * t101 * t450 + t105 * t450 - 2 * t66 * t450 - 2 * t69 * t450 - 2 * t73 * t450 - 2 * t81 * t450 + t90 * t450 - 2 * t95 * t450 + t98 * t450 + 2 * t62 * t460 + 2 * t77 * t460 + 2 * t85 * t460
    t505 = x[0, 1] * t450
    t528 = t107 * t450 - 2 * t111 * t505 + 2 * t114 * t505 + 2 * t117 * t505 - 2 * t121 * t505 - 2 * t125 * t450 + 2 * t128 * t505 + 2 * t132 * t505 + 2 * t137 * t505 - 2 * t140 * t450 - 2 * t143 * t450 - 2 * t147 * t450
    t542 = x[0, 2] * t450
    t551 = 2 * t151 * t505 + t155 * t450 - 2 * t158 * t450 + t161 * t450 - 2 * t164 * t450 + t167 * t450 + t169 * t450 - 2 * t172 * t450 + 2 * t179 * t542 + 2 * t182 * t542 - 2 * t186 * t450 + 2 * t189 * t542
    t560 = x[1, 0] * t450
    t572 = -2 * t151 * x[1, 1] * t450 + 2 * t193 * t542 - 2 * t196 * t450 - 2 * t199 * t450 + t202 * t450 + t204 * t450 + t212 * t450 + t214 * t450 + t219 * t450 + t221 * t450 - 2 * t77 * t560 - 2 * t85 * t560
    t577 = sqrt(t45 + t92 + t135 + t175 + t211 + t251 + 4 * t286 + t329 + t374 + t416 + t455 + t479 + t503 + t528 + t551 + t572)
    t579 = (t577 - 1) ** 2
    t582 = x[1, 0] - x[0, 0]
    t584 = x[2, 0] - x[0, 0]
    t586 = m_Einv[0, 0] * t582 + m_Einv[1, 0] * t584
    t587 = t586 ** 2
    t589 = x[1, 1] - x[0, 1]
    t591 = x[2, 1] - x[0, 1]
    t593 = m_Einv[0, 0] * t589 + m_Einv[1, 0] * t591
    t594 = t593 ** 2
    t596 = x[1, 2] - x[0, 2]
    t598 = x[2, 2] - x[0, 2]
    t600 = m_Einv[0, 0] * t596 + m_Einv[1, 0] * t598
    t601 = t600 ** 2
    t604 = (0.5e0 * t587 + 0.5e0 * t594 + 0.5e0 * t601 - 0.5e0) ** 2
    t607 = m_Einv[0, 1] * t582 + m_Einv[1, 1] * t584
    t612 = m_Einv[0, 1] * t589 + m_Einv[1, 1] * t591
    t617 = m_Einv[0, 1] * t596 + m_Einv[1, 1] * t598
    t621 = (0.5e0 * t607 * t586 + 0.5e0 * t612 * t593 + 0.5e0 * t617 * t600) ** 2
    t623 = t607 ** 2
    t625 = t612 ** 2
    t627 = t617 ** 2
    t630 = (0.5e0 * t623 + 0.5e0 * t625 + 0.5e0 * t627 - 0.5e0) ** 2
    return m_h * m_A0 * (0.5e0 * t579 * lame + (t604 + 2 * t621 + t630) * mu)
