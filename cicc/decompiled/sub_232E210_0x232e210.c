// Function: sub_232E210
// Address: 0x232e210
//
unsigned __int64 __fastcall sub_232E210(__int128 a1)
{
  __m128i v2; // kr00_16
  char v3; // [rsp+Eh] [rbp-72h]
  char v4; // [rsp+Fh] [rbp-71h]
  __m128i v5; // [rsp+10h] [rbp-70h] BYREF
  unsigned __int16 v6; // [rsp+2Dh] [rbp-53h] BYREF
  unsigned __int8 v7; // [rsp+2Fh] [rbp-51h]
  __m128i v8; // [rsp+30h] [rbp-50h] BYREF
  __m128i v9[4]; // [rsp+40h] [rbp-40h] BYREF

  v5 = (__m128i)a1;
  if ( !(unsigned __int8)sub_95CB50((const void **)&v5, "function", 8u) )
    goto LABEL_6;
  if ( !v5.m128i_i64[1] )
  {
    v7 = 1;
    v6 = 0;
    return ((unsigned __int64)v7 << 16) | v6;
  }
  if ( !(unsigned __int8)sub_95CB50((const void **)&v5, "<", 1u) || !(unsigned __int8)sub_232E070(&v5, ">", 1u) )
  {
LABEL_6:
    v7 = 0;
    return ((unsigned __int64)v7 << 16) | v6;
  }
  v3 = 0;
  v4 = 0;
  while ( v5.m128i_i64[1] )
  {
    LOBYTE(v6) = 59;
    sub_232E160(&v8, &v5, &v6, 1u);
    v2 = v8;
    v5 = _mm_loadu_si128(v9);
    if ( sub_9691B0((const void *)v8.m128i_i64[0], v8.m128i_u64[1], "eager-inv", 9) )
    {
      v4 = 1;
    }
    else
    {
      if ( !sub_9691B0((const void *)v2.m128i_i64[0], v2.m128i_u64[1], "no-rerun", 8) )
        goto LABEL_6;
      v3 = 1;
    }
  }
  v7 = 1;
  LOBYTE(v6) = v4;
  HIBYTE(v6) = v3;
  return ((unsigned __int64)v7 << 16) | v6;
}
