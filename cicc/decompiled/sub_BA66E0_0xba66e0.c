// Function: sub_BA66E0
// Address: 0xba66e0
//
__m128i *__fastcall sub_BA66E0(__m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v5; // rax
  unsigned __int8 v6; // al
  __int64 v7; // rdx
  __m128i **v8; // rax
  __int64 v9; // rdx

  v5 = a1->m128i_u8[0];
  if ( (unsigned __int8)v5 > 0x24u )
    return (__m128i *)sub_B95B00((unsigned __int8 *)a1, a2, a3, a4, a5);
  a3 = 0x1FBFFDFFE0LL;
  if ( !_bittest64(&a3, v5) )
    return (__m128i *)sub_B95B00((unsigned __int8 *)a1, a2, a3, a4, a5);
  v6 = a1[-1].m128i_u8[0];
  if ( (v6 & 2) != 0 )
  {
    v8 = (__m128i **)a1[-2].m128i_i64[0];
    v7 = a1[-2].m128i_u32[2];
  }
  else
  {
    v7 = ((unsigned __int16)a1[-1].m128i_i16[0] >> 6) & 0xF;
    v8 = (__m128i **)&a1[-1] - ((v6 >> 2) & 0xF);
  }
  v9 = 8 * v7;
  a2 = (__int64)&v8[(unsigned __int64)v9 / 8];
  a4 = v9 >> 3;
  a3 = v9 >> 5;
  if ( a3 )
  {
    a3 = (__int64)&v8[4 * a3];
    while ( a1 != *v8 )
    {
      if ( a1 == v8[1] )
      {
        ++v8;
        goto LABEL_12;
      }
      if ( a1 == v8[2] )
      {
        v8 += 2;
        goto LABEL_12;
      }
      if ( a1 == v8[3] )
      {
        v8 += 3;
        goto LABEL_12;
      }
      v8 += 4;
      if ( v8 == (__m128i **)a3 )
      {
        a4 = (a2 - (__int64)v8) >> 3;
        goto LABEL_17;
      }
    }
    goto LABEL_12;
  }
LABEL_17:
  if ( a4 == 2 )
    goto LABEL_27;
  if ( a4 == 3 )
  {
    if ( a1 == *v8 )
      goto LABEL_12;
    ++v8;
LABEL_27:
    if ( a1 == *v8 )
      goto LABEL_12;
    ++v8;
    goto LABEL_20;
  }
  if ( a4 != 1 )
    return sub_BA6670(a1, a2);
LABEL_20:
  if ( a1 != *v8 )
    return sub_BA6670(a1, a2);
LABEL_12:
  if ( (__m128i **)a2 == v8 )
    return sub_BA6670(a1, a2);
  return (__m128i *)sub_B95B00((unsigned __int8 *)a1, a2, a3, a4, a5);
}
