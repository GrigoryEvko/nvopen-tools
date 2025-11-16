// Function: sub_2BE0960
// Address: 0x2be0960
//
unsigned __int64 __fastcall sub_2BE0960(unsigned __int64 *a1, __int64 a2, __int64 a3, __int8 a4)
{
  __m128i v4; // xmm3
  __m128i v5; // xmm0
  __m128i *v6; // rsi
  __m128i v7; // xmm2
  __m128i v8; // xmm0
  bool v9; // zf
  __m128i v10; // xmm1
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rax
  unsigned __int64 v14; // rsi
  __int64 v15; // rsi
  __m128i v17; // [rsp+0h] [rbp-70h] BYREF
  __m128i v18; // [rsp+10h] [rbp-60h] BYREF
  __m128i v19; // [rsp+20h] [rbp-50h] BYREF
  __m128i v20; // [rsp+30h] [rbp-40h] BYREF
  __m128i v21; // [rsp+40h] [rbp-30h] BYREF
  __m128i v22; // [rsp+50h] [rbp-20h] BYREF

  v17.m128i_i64[1] = a2;
  v4 = _mm_loadu_si128(&v19);
  v17.m128i_i32[0] = 2;
  v5 = _mm_loadu_si128(&v17);
  v18.m128i_i64[0] = a3;
  v6 = (__m128i *)a1[8];
  v18.m128i_i8[8] = a4;
  v7 = _mm_loadu_si128(&v18);
  v20 = v5;
  v21 = v7;
  v22 = v4;
  if ( v6 == (__m128i *)a1[9] )
  {
    sub_2BE00E0(a1 + 7, v6, &v20);
    v14 = a1[8];
  }
  else
  {
    if ( v6 )
    {
      *v6 = v5;
      v8 = _mm_loadu_si128(&v21);
      v9 = v20.m128i_i32[0] == 11;
      v6[1] = v8;
      v6[2] = _mm_loadu_si128(&v22);
      if ( v9 )
      {
        v10 = _mm_loadu_si128(&v21);
        v21 = v8;
        v11 = v6[2].m128i_i64[1];
        v6[2].m128i_i64[0] = 0;
        v6[1] = v10;
        v12 = v22.m128i_i64[0];
        v22.m128i_i64[0] = 0;
        v6[2].m128i_i64[0] = v12;
        v13 = v22.m128i_i64[1];
        v22.m128i_i64[1] = v11;
        v6[2].m128i_i64[1] = v13;
      }
      v6 = (__m128i *)a1[8];
    }
    v14 = (unsigned __int64)&v6[3];
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
