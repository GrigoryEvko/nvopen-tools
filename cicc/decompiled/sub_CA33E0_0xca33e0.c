// Function: sub_CA33E0
// Address: 0xca33e0
//
__m128i *__fastcall sub_CA33E0(__m128i *a1, __m128i *a2)
{
  __int32 v2; // eax
  __int64 v3; // rdx
  __int64 v5; // rdx
  __int32 v6; // eax
  __int64 v7; // rax
  _BYTE *v8; // rsi
  __m128i *v9; // [rsp+0h] [rbp-E0h] BYREF
  __int64 v10; // [rsp+8h] [rbp-D8h]
  __m128i v11; // [rsp+10h] [rbp-D0h] BYREF
  __m128i *v12; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v13; // [rsp+28h] [rbp-B8h]
  __m128i v14; // [rsp+30h] [rbp-B0h] BYREF

  if ( a2[20].m128i_i8[8] )
  {
    v5 = a2[1].m128i_i64[1];
    if ( (a2[20].m128i_i8[0] & 1) != 0 )
    {
      v6 = a2[1].m128i_i32[0];
      a1[2].m128i_i8[0] |= 1u;
      a1->m128i_i64[1] = v5;
      a1->m128i_i32[0] = v6;
    }
    else
    {
      v8 = (_BYTE *)a2[1].m128i_i64[0];
      v12 = &v14;
      sub_CA1FB0((__int64 *)&v12, v8, (__int64)&v8[v5]);
      a1[2].m128i_i8[0] &= ~1u;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      if ( v12 == &v14 )
      {
        a1[1] = _mm_load_si128(&v14);
      }
      else
      {
        a1->m128i_i64[0] = (__int64)v12;
        a1[1].m128i_i64[0] = v14.m128i_i64[0];
      }
      a1->m128i_i64[1] = v13;
    }
  }
  else
  {
    v13 = 0;
    v12 = (__m128i *)&v14.m128i_u64[1];
    v14.m128i_i64[0] = 128;
    v2 = sub_C82800(&v12);
    if ( v2 )
    {
      a1[2].m128i_i8[0] |= 1u;
      a1->m128i_i32[0] = v2;
      a1->m128i_i64[1] = v3;
    }
    else
    {
      a2 = v12;
      v9 = &v11;
      sub_CA1FB0((__int64 *)&v9, v12, (__int64)v12->m128i_i64 + v13);
      a1[2].m128i_i8[0] &= ~1u;
      a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
      if ( v9 == &v11 )
      {
        a1[1] = _mm_load_si128(&v11);
      }
      else
      {
        a1->m128i_i64[0] = (__int64)v9;
        a1[1].m128i_i64[0] = v11.m128i_i64[0];
      }
      v7 = v10;
      v9 = &v11;
      v10 = 0;
      a1->m128i_i64[1] = v7;
      v11.m128i_i8[0] = 0;
      sub_2240A30(&v9);
    }
    if ( v12 != (__m128i *)&v14.m128i_u64[1] )
      _libc_free(v12, a2);
  }
  return a1;
}
