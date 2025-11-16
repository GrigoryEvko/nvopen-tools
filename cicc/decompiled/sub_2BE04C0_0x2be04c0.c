// Function: sub_2BE04C0
// Address: 0x2be04c0
//
unsigned __int64 __fastcall sub_2BE04C0(unsigned __int64 *a1)
{
  __m128i *v1; // rsi
  __m128i v2; // xmm0
  bool v3; // zf
  __m128i v4; // xmm1
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rax
  unsigned __int64 v8; // rsi
  __int64 v9; // rsi
  __m128i v11; // [rsp+0h] [rbp-40h] BYREF
  __m128i v12; // [rsp+10h] [rbp-30h] BYREF
  __m128i v13; // [rsp+20h] [rbp-20h] BYREF

  v11.m128i_i32[0] = 10;
  v1 = (__m128i *)a1[8];
  v11.m128i_i64[1] = -1;
  if ( v1 == (__m128i *)a1[9] )
  {
    sub_2BE00E0(a1 + 7, v1, &v11);
    v8 = a1[8];
  }
  else
  {
    if ( v1 )
    {
      *v1 = _mm_loadu_si128(&v11);
      v2 = _mm_loadu_si128(&v12);
      v3 = v11.m128i_i32[0] == 11;
      v1[1] = v2;
      v1[2] = _mm_loadu_si128(&v13);
      if ( v3 )
      {
        v4 = _mm_loadu_si128(&v12);
        v12 = v2;
        v5 = v1[2].m128i_i64[1];
        v1[2].m128i_i64[0] = 0;
        v1[1] = v4;
        v6 = v13.m128i_i64[0];
        v13.m128i_i64[0] = 0;
        v1[2].m128i_i64[0] = v6;
        v7 = v13.m128i_i64[1];
        v13.m128i_i64[1] = v5;
        v1[2].m128i_i64[1] = v7;
      }
      v1 = (__m128i *)a1[8];
    }
    v8 = (unsigned __int64)&v1[3];
    a1[8] = v8;
  }
  v9 = v8 - a1[7];
  if ( (unsigned __int64)v9 > 0x493E00 )
    abort();
  if ( v11.m128i_i32[0] != 11 || !v13.m128i_i64[0] )
    return 0xAAAAAAAAAAAAAAABLL * (v9 >> 4) - 1;
  ((void (__fastcall *)(__m128i *, __m128i *, __int64))v13.m128i_i64[0])(&v12, &v12, 3);
  return 0xAAAAAAAAAAAAAAABLL * (v9 >> 4) - 1;
}
