def CST3D_E(x, m_Einv):
    t1 = x[1, 0] - x[0, 0]
    t3 = x[2, 0] - x[0, 0]
    t5 = m_Einv[0, 0] * t1 + m_Einv[1, 0] * t3
    t6 = t5 ** 2
    t8 = x[1, 1] - x[0, 1]
    t10 = x[2, 1] - x[0, 1]
    t12 = m_Einv[1, 0] * t10 + m_Einv[0, 0] * t8
    t13 = t12 ** 2
    t15 = x[1, 2] - x[0, 2]
    t17 = x[2, 2] - x[0, 2]
    t19 = m_Einv[0, 0] * t15 + m_Einv[1, 0] * t17
    t20 = t19 ** 2
    t25 = m_Einv[0, 1] * t1 + m_Einv[1, 1] * t3
    t30 = m_Einv[1, 1] * t10 + m_Einv[0, 1] * t8
    t35 = m_Einv[0, 1] * t15 + m_Einv[1, 1] * t17
    t38 = 0.5e0 * t25 * t5 + 0.5e0 * t30 * t12 + 0.5e0 * t35 * t19
    t39 = t25 ** 2
    t41 = t30 ** 2
    t43 = t35 ** 2
    E = x.new_zeros([2,2] + list(t6.size()))
    E[0, 0] = 0.5e0 * t6 + 0.5e0 * t13 + 0.5e0 * t20 - 0.5e0
    E[0, 1] = t38
    E[1, 0] = t38
    E[1, 1] = 0.5e0 * t39 + 0.5e0 * t41 + 0.5e0 * t43 - 0.5e0
    return E