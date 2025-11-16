// Function: sub_3401740
// Address: 0x3401740
//
unsigned __int8 *__fastcall sub_3401740(
        __int64 a1,
        char a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7)
{
  __int64 v7; // r14
  __int64 v8; // r13
  unsigned int v9; // ebx
  __m128i v10; // xmm0
  _DWORD *v11; // r15
  unsigned __int16 v12; // si
  unsigned int v13; // eax
  __int64 v14; // rsi
  char v16; // [rsp+Fh] [rbp-51h]
  __m128i v17; // [rsp+10h] [rbp-50h] BYREF
  _OWORD v18[4]; // [rsp+20h] [rbp-40h] BYREF

  v7 = a5;
  v8 = a3;
  v9 = a4;
  v10 = _mm_loadu_si128((const __m128i *)&a7);
  v17 = v10;
  if ( !a2 )
  {
    LODWORD(a7) = 0;
    v14 = 0;
    return sub_3400BD0(a1, v14, a3, a4, a5, 0, v10, a7);
  }
  v11 = *(_DWORD **)(a1 + 16);
  v18[0] = _mm_load_si128(&v17);
  if ( !v17.m128i_i16[0] )
  {
    v17.m128i_i64[0] = (__int64)v18;
    v16 = sub_3007030((__int64)v18);
    if ( sub_30070B0(v17.m128i_i64[0]) )
      goto LABEL_17;
    if ( !v16 )
      goto LABEL_6;
LABEL_12:
    v13 = v11[16];
    if ( v13 <= 1 )
      goto LABEL_8;
    goto LABEL_13;
  }
  v12 = v17.m128i_i16[0] - 17;
  if ( (unsigned __int16)(v17.m128i_i16[0] - 10) <= 6u || (unsigned __int16)(v17.m128i_i16[0] - 126) <= 0x31u )
  {
    if ( v12 <= 0xD3u )
      goto LABEL_17;
    goto LABEL_12;
  }
  if ( v12 > 0xD3u )
  {
LABEL_6:
    v13 = v11[15];
    goto LABEL_7;
  }
LABEL_17:
  v13 = v11[17];
LABEL_7:
  if ( v13 <= 1 )
  {
LABEL_8:
    LODWORD(a7) = 0;
    a4 = v9;
    a5 = v7;
    a3 = v8;
    v14 = 1;
    return sub_3400BD0(a1, v14, a3, a4, a5, 0, v10, a7);
  }
LABEL_13:
  if ( v13 != 2 )
    BUG();
  return sub_34015B0(a1, v8, v9, v7, 0, 0, v10);
}
