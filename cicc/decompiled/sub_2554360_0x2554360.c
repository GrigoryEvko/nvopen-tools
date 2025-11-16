// Function: sub_2554360
// Address: 0x2554360
//
__m128i *__fastcall sub_2554360(__m128i *a1, int a2)
{
  __m128i *v2; // rdx
  __int64 v3; // rdx
  __m128i *v5; // [rsp+0h] [rbp-40h] BYREF
  __int64 v6; // [rsp+8h] [rbp-38h]
  __m128i v7[3]; // [rsp+10h] [rbp-30h] BYREF

  if ( !(_BYTE)a2 )
  {
    sub_253C590(a1->m128i_i64, "all memory");
    return a1;
  }
  if ( a2 == 255 )
  {
    sub_253C590(a1->m128i_i64, "no memory");
    return a1;
  }
  sub_253C590((__int64 *)&v5, "memory:");
  if ( (a2 & 1) != 0 )
  {
    if ( (a2 & 2) != 0 )
      goto LABEL_5;
  }
  else
  {
    sub_2241520((unsigned __int64 *)&v5, "stack,");
    if ( (a2 & 2) != 0 )
    {
LABEL_5:
      if ( (a2 & 4) != 0 )
        goto LABEL_6;
      goto LABEL_18;
    }
  }
  sub_2241520((unsigned __int64 *)&v5, "constant,");
  if ( (a2 & 4) != 0 )
  {
LABEL_6:
    if ( (a2 & 8) != 0 )
      goto LABEL_7;
    goto LABEL_19;
  }
LABEL_18:
  sub_2241520((unsigned __int64 *)&v5, "internal global,");
  if ( (a2 & 8) != 0 )
  {
LABEL_7:
    if ( (a2 & 0x10) != 0 )
      goto LABEL_8;
    goto LABEL_20;
  }
LABEL_19:
  sub_2241520((unsigned __int64 *)&v5, "external global,");
  if ( (a2 & 0x10) != 0 )
  {
LABEL_8:
    if ( (a2 & 0x20) != 0 )
      goto LABEL_9;
    goto LABEL_21;
  }
LABEL_20:
  sub_2241520((unsigned __int64 *)&v5, "argument,");
  if ( (a2 & 0x20) != 0 )
  {
LABEL_9:
    if ( (a2 & 0x40) != 0 )
      goto LABEL_10;
LABEL_22:
    sub_2241520((unsigned __int64 *)&v5, "malloced,");
    if ( (a2 & 0x80) != 0 )
      goto LABEL_11;
LABEL_23:
    sub_2241520((unsigned __int64 *)&v5, "unknown,");
    goto LABEL_11;
  }
LABEL_21:
  sub_2241520((unsigned __int64 *)&v5, "inaccessible,");
  if ( (a2 & 0x40) == 0 )
    goto LABEL_22;
LABEL_10:
  if ( (a2 & 0x80) == 0 )
    goto LABEL_23;
LABEL_11:
  sub_2240CE0((__int64 *)&v5, v6 - 1, 1);
  v2 = v5;
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  if ( v2 == v7 )
  {
    a1[1] = _mm_load_si128(v7);
  }
  else
  {
    a1->m128i_i64[0] = (__int64)v2;
    a1[1].m128i_i64[0] = v7[0].m128i_i64[0];
  }
  v3 = v6;
  v5 = v7;
  v6 = 0;
  a1->m128i_i64[1] = v3;
  v7[0].m128i_i8[0] = 0;
  sub_2240A30((unsigned __int64 *)&v5);
  return a1;
}
