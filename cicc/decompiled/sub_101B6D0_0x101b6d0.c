// Function: sub_101B6D0
// Address: 0x101b6d0
//
__int64 __fastcall sub_101B6D0(unsigned __int8 *a1, unsigned __int8 *a2, __m128i *a3, int a4)
{
  __int64 v6; // rbx
  unsigned __int8 v7; // al
  unsigned __int8 *v8; // r13
  bool v10; // zf
  __int64 v11; // rax
  char v12; // al
  __int64 v13; // rsi
  char v14; // al
  __int64 v15; // rsi
  unsigned __int8 v16; // cl
  int v17; // eax
  unsigned __int8 *v18; // rax
  _QWORD *v19; // [rsp+0h] [rbp-50h] BYREF
  unsigned __int8 *v20; // [rsp+8h] [rbp-48h]
  __int64 *v21; // [rsp+10h] [rbp-40h] BYREF
  __int64 v22; // [rsp+18h] [rbp-38h]

  v6 = (__int64)a1;
  v7 = *a2;
  if ( *a1 > 0x15u )
  {
    v8 = a2;
  }
  else if ( v7 > 0x15u )
  {
    v7 = *a1;
    v8 = a1;
    v6 = (__int64)a2;
  }
  else
  {
    v8 = (unsigned __int8 *)sub_96E6C0(0x1Eu, (__int64)a1, a2, a3->m128i_i64[0]);
    if ( v8 )
      return (__int64)v8;
    v7 = *a2;
    v8 = a2;
  }
  if ( v7 == 13 || (unsigned __int8)sub_1003090((__int64)a3, v8) )
    return (__int64)v8;
  if ( (unsigned __int8)sub_FFFE90((__int64)v8) )
    return v6;
  if ( (unsigned __int8 *)v6 != v8 )
  {
    v10 = *(_BYTE *)v6 == 59;
    v19 = 0;
    v20 = v8;
    if ( v10 )
    {
      v12 = sub_995B10(&v19, *(_QWORD *)(v6 - 64));
      v13 = *(_QWORD *)(v6 - 32);
      if ( v12 )
      {
        if ( (unsigned __int8 *)v13 == v20 )
          return sub_AD62B0(*(_QWORD *)(v6 + 8));
      }
      if ( (unsigned __int8)sub_995B10(&v19, v13) && *(unsigned __int8 **)(v6 - 64) == v20 )
        return sub_AD62B0(*(_QWORD *)(v6 + 8));
    }
    v10 = *v8 == 59;
    v21 = 0;
    v22 = v6;
    if ( v10 )
    {
      v14 = sub_995B10(&v21, *((_QWORD *)v8 - 8));
      v15 = *((_QWORD *)v8 - 4);
      if ( v14 )
      {
        if ( v15 == v22 )
          return sub_AD62B0(*(_QWORD *)(v6 + 8));
      }
      if ( (unsigned __int8)sub_995B10(&v21, v15) && *((_QWORD *)v8 - 8) == v22 )
        return sub_AD62B0(*(_QWORD *)(v6 + 8));
    }
    v11 = sub_1004B70((_BYTE *)v6, v8);
    if ( v11 )
      return v11;
    v11 = sub_1004B70(v8, (_BYTE *)v6);
    if ( v11 )
      return v11;
    v11 = sub_FFEF50((char *)v6, v8, 30);
    if ( v11 )
      return v11;
    v11 = (__int64)sub_101B370(30, (__int64 *)v6, (__int64 *)v8, a3, a4);
    if ( v11 )
      return v11;
    if ( a4 == 3 )
    {
      v11 = sub_FFE7D0(30, v6, v8, a3->m128i_i64[0], a3[2].m128i_i64[1]);
      if ( v11 )
        return v11;
    }
    v16 = *(_BYTE *)v6;
    if ( *(_BYTE *)v6 <= 0x1Cu )
    {
      if ( v16 != 5 )
        return 0;
      v17 = *(unsigned __int16 *)(v6 + 2);
      if ( (*(_WORD *)(v6 + 2) & 0xFFF7) != 0x11 && (v17 & 0xFFFD) != 0xD )
        return 0;
    }
    else
    {
      if ( v16 > 0x36u )
        return 0;
      v17 = v16 - 29;
      if ( ((0x40540000000000uLL >> v16) & 1) == 0 )
        return 0;
    }
    if ( v17 == 15 && (*(_BYTE *)(v6 + 1) & 2) != 0 )
    {
      v18 = *(unsigned __int8 **)(v6 - 64);
      if ( v8 == v18 )
      {
        if ( v18 )
        {
          v6 = *(_QWORD *)(v6 - 32);
          if ( v6 )
          {
            v21 = 0;
            if ( (unsigned __int8)sub_1004E80(&v21, (__int64)v8) )
              return v6;
          }
        }
      }
    }
    return 0;
  }
  return sub_AD6530(*(_QWORD *)(v6 + 8), (__int64)v8);
}
