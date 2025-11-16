// Function: sub_8C2C50
// Address: 0x8c2c50
//
char __fastcall sub_8C2C50(_QWORD *src, unsigned __int8 a2)
{
  _BYTE *v2; // rbx
  _BYTE *v3; // r13
  __int64 v4; // rax
  _QWORD *v5; // rdx
  _QWORD *v6; // rcx
  char v7; // dl

  v2 = src;
  if ( (*(_BYTE *)(src - 1) & 1) == 0 )
  {
    v3 = src;
    sub_766570(
      src,
      a2,
      (__int64 (__fastcall *)(_QWORD, _QWORD))sub_8C38E0,
      (__int64 (__fastcall *)(_QWORD, _QWORD))sub_8C3810,
      0);
    v4 = sub_72A270((__int64)src, a2);
    *((_BYTE *)src - 8) &= ~0x80u;
    if ( v4 )
      goto LABEL_3;
LABEL_21:
    if ( a2 == 23 )
    {
      v3[29] &= ~4u;
    }
    else if ( a2 == 40 )
    {
      v3[109] &= ~0x10u;
      *((_QWORD *)v3 + 26) = 0;
    }
    return v4;
  }
  v3 = (_BYTE *)*(src - 3);
  memcpy(v3, src, qword_4B6D500[a2]);
  LOBYTE(v4) = sub_766570(
                 v3,
                 a2,
                 (__int64 (__fastcall *)(_QWORD, _QWORD))sub_8C38E0,
                 (__int64 (__fastcall *)(_QWORD, _QWORD))sub_8C3810,
                 0);
  if ( a2 != 37 )
  {
    v4 = sub_72A270((__int64)v3, a2);
    if ( v4 )
    {
      v6 = *(_QWORD **)(v4 + 32);
      *(_BYTE *)(v4 + 90) |= 4u;
      if ( v6 )
      {
        v7 = *(v3 - 8);
        if ( (v7 & 2) != 0 )
        {
          v2 = v3;
          *(v3 - 8) = v7 & 0x7F;
LABEL_3:
          *(_BYTE *)(v4 + 88) &= ~8u;
          if ( a2 != 6 )
            goto LABEL_4;
LABEL_7:
          LOBYTE(v4) = v2[140] - 9;
          if ( (unsigned __int8)v4 <= 2u )
            v2[178] &= 0x3Fu;
          return v4;
        }
        *v6 = v3;
      }
      *(v3 - 8) &= ~0x80u;
      v2 = v3;
      *(_BYTE *)(v4 + 88) &= ~8u;
      if ( a2 != 6 )
      {
LABEL_4:
        if ( a2 == 11 )
          v2[203] &= 0x73u;
        return v4;
      }
      goto LABEL_7;
    }
    *(v3 - 8) &= ~0x80u;
    goto LABEL_21;
  }
  v5 = (_QWORD *)src[8];
  if ( v5 )
  {
    LOBYTE(v4) = *(v3 - 8);
    if ( (v4 & 2) != 0 )
    {
      LOBYTE(v4) = v4 & 0x7F;
      *(v3 - 8) = v4;
      return v4;
    }
    *v5 = v3;
  }
  *(v3 - 8) &= ~0x80u;
  return v4;
}
