import models.basicblock as B


def dncnn(nc_in,nc_out,nb,mode,act_mode,bias,nf=64):
    m_head = B.conv(nc_in, nf, mode=mode + act_mode[-1], bias=bias)
    m_body = [B.conv(nf, nf, mode=mode + act_mode, bias=bias) for _ in range(nb - 2)]
    m_tail = B.conv(nf, nc_out, mode=mode, bias=bias)
    model = B.sequential(m_head, *m_body, m_tail)
    return model
