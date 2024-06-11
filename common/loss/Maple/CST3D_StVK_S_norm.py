from torch import sqrt
def CST3D_StVK_S_norm(x, lame, mu, m_Einv):
    t1 = x[1, 0] - x[0, 0]
    t3 = x[2, 0] - x[0, 0]
    t5 = m_Einv[0, 0] * t1 + m_Einv[1, 0] * t3
    t6 = t5 ** 2
    t7 = 0.5e0 * t6
    t8 = x[1, 1] - x[0, 1]
    t10 = x[2, 1] - x[0, 1]
    t12 = m_Einv[1, 0] * t10 + m_Einv[0, 0] * t8
    t13 = t12 ** 2
    t14 = 0.5e0 * t13
    t15 = x[1, 2] - x[0, 2]
    t17 = x[2, 2] - x[0, 2]
    t19 = m_Einv[0, 0] * t15 + m_Einv[1, 0] * t17
    t20 = t19 ** 2
    t21 = 0.5e0 * t20
    t24 = m_Einv[0, 1] * t1 + m_Einv[1, 1] * t3
    t25 = t24 ** 2
    t26 = 0.5e0 * t25
    t29 = m_Einv[1, 1] * t10 + m_Einv[0, 1] * t8
    t30 = t29 ** 2
    t31 = 0.5e0 * t30
    t34 = m_Einv[0, 1] * t15 + m_Einv[1, 1] * t17
    t35 = t34 ** 2
    t36 = 0.5e0 * t35
    t38 = (t7 + t14 + t21 - 0.10e1 + t26 + t31 + t36) * lame
    t43 = (t38 + 2 * (t7 + t14 + t21 - 0.5e0) * mu) ** 2
    t44 = mu ** 2
    t52 = (0.5e0 * t24 * t5 + 0.5e0 * t29 * t12 + 0.5e0 * t34 * t19) ** 2
    t59 = (t38 + 2 * (t26 + t31 + t36 - 0.5e0) * mu) ** 2
    t61 = sqrt(8 * t52 * t44 + t43 + t59)
    return t61
