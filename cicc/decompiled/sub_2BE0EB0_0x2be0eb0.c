// Function: sub_2BE0EB0
// Address: 0x2be0eb0
//
unsigned __int64 __fastcall sub_2BE0EB0(unsigned __int64 *a1, __m128i *a2)
{
  __int64 v2; // rdx
  __int64 v3; // rax
  __m128i v4; // xmm0
  __m128i v5; // xmm2
  __m128i v6; // xmm1
  __m128i v7; // xmm3
  __m128i *v8; // rsi
  __m128i v9; // xmm0
  __m128i v10; // xmm1
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  unsigned __int64 v14; // rsi
  __int64 v15; // rsi
  __m128i v17; // [rsp+0h] [rbp-70h] BYREF
  __m128i v18; // [rsp+10h] [rbp-60h] BYREF
  __m128i v19; // [rsp+20h] [rbp-50h] BYREF
  __m128i v20; // [rsp+30h] [rbp-40h] BYREF
  __m128i v21; // [rsp+40h] [rbp-30h] BYREF
  __m128i v22; // [rsp+50h] [rbp-20h] BYREF

  v2 = a2[1].m128i_i64[0];
  v3 = a2[1].m128i_i64[1];
  v17.m128i_i32[0] = 11;
  v4 = _mm_loadu_si128(a2);
  v5 = _mm_loadu_si128(&v20);
  v17.m128i_i64[1] = -1;
  v19.m128i_i64[0] = v2;
  v6 = _mm_loadu_si128(&v17);
  v19.m128i_i64[1] = v3;
  v7 = _mm_loadu_si128(&v19);
  a2[1].m128i_i64[0] = 0;
  v22.m128i_i64[1] = v7.m128i_i64[1];
  v22.m128i_i64[0] = v2;
  a2[1].m128i_i64[1] = 0;
  *a2 = v5;
  v8 = (__m128i *)a1[8];
  v19.m128i_i64[0] = 0;
  v19.m128i_i64[1] = v7.m128i_i64[1];
  v22.m128i_i64[1] = v3;
  v18 = v4;
  v20 = v6;
  v21 = v4;
  if ( v8 == (__m128i *)a1[9] )
  {
    sub_2BE00E0(a1 + 7, v8, &v20);
    v14 = a1[8];
  }
  else
  {
    if ( v8 )
    {
      *v8 = v6;
      v9 = _mm_loadu_si128(&v21);
      v8[1] = v9;
      v8[2] = _mm_loadu_si128(&v22);
      if ( v20.m128i_i32[0] == 11 )
      {
        v8[2].m128i_i64[0] = 0;
        v10 = _mm_loadu_si128(&v21);
        v21 = v9;
        v8[1] = v10;
        v11 = v22.m128i_i64[0];
        v22.m128i_i64[0] = 0;
        v12 = v8[2].m128i_i64[1];
        v8[2].m128i_i64[0] = v11;
        v13 = v22.m128i_i64[1];
        v22.m128i_i64[1] = v12;
        v8[2].m128i_i64[1] = v13;
      }
      v8 = (__m128i *)a1[8];
    }
    v14 = (unsigned __int64)&v8[3];
    a1[8] = v14;
  }
  v15 = v14 - a1[7];
  if ( (unsigned __int64)v15 > 0x493E00 )
    abort();
  if ( v20.m128i_i32[0] == 11 && v22.m128i_i64[0] )
    ((void (__fastcall *)(__m128i *, __m128i *, __int64))v22.m128i_i64[0])(&v21, &v21, 3);
  if ( v17.m128i_i32[0] != 11 || !v19.m128i_i64[0] )
    return 0xAAAAAAAAAAAAAAABLL * (v15 >> 4) - 1;
  ((void (__fastcall *)(__m128i *, __m128i *, __int64))v19.m128i_i64[0])(&v18, &v18, 3);
  return 0xAAAAAAAAAAAAAAABLL * (v15 >> 4) - 1;
}
