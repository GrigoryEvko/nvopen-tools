// Function: sub_2BE05D0
// Address: 0x2be05d0
//
unsigned __int64 __fastcall sub_2BE05D0(_QWORD *a1)
{
  __int64 v1; // rax
  _BYTE *v2; // rsi
  __m128i v3; // xmm3
  __m128i *v4; // rsi
  __m128i v5; // xmm0
  __m128i v6; // xmm2
  __m128i v7; // xmm0
  bool v8; // zf
  __m128i v9; // xmm1
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rax
  __m128i *v13; // rsi
  __int64 v14; // rsi
  __int64 v16; // [rsp+8h] [rbp-78h] BYREF
  __m128i v17; // [rsp+10h] [rbp-70h] BYREF
  __m128i v18; // [rsp+20h] [rbp-60h] BYREF
  __m128i v19; // [rsp+30h] [rbp-50h] BYREF
  __m128i v20; // [rsp+40h] [rbp-40h] BYREF
  __m128i v21; // [rsp+50h] [rbp-30h] BYREF
  __m128i v22; // [rsp+60h] [rbp-20h] BYREF

  v2 = (_BYTE *)a1[1];
  v16 = a1[5];
  v1 = v16;
  a1[5] = v16 + 1;
  if ( v2 == (_BYTE *)a1[2] )
  {
    sub_9CA200((__int64)a1, v2, &v16);
    v1 = v16;
  }
  else
  {
    if ( v2 )
    {
      *(_QWORD *)v2 = v1;
      v2 = (_BYTE *)a1[1];
    }
    a1[1] = v2 + 8;
  }
  v3 = _mm_loadu_si128(&v19);
  v4 = (__m128i *)a1[8];
  v17.m128i_i32[0] = 8;
  v17.m128i_i64[1] = -1;
  v5 = _mm_loadu_si128(&v17);
  v18.m128i_i64[0] = v1;
  v6 = _mm_loadu_si128(&v18);
  v20 = v5;
  v21 = v6;
  v22 = v3;
  if ( v4 == (__m128i *)a1[9] )
  {
    sub_2BE00E0(a1 + 7, v4, &v20);
    v13 = (__m128i *)a1[8];
  }
  else
  {
    if ( v4 )
    {
      *v4 = v5;
      v7 = _mm_loadu_si128(&v21);
      v8 = v20.m128i_i32[0] == 11;
      v4[1] = v7;
      v4[2] = _mm_loadu_si128(&v22);
      if ( v8 )
      {
        v9 = _mm_loadu_si128(&v21);
        v21 = v7;
        v10 = v4[2].m128i_i64[1];
        v4[2].m128i_i64[0] = 0;
        v4[1] = v9;
        v11 = v22.m128i_i64[0];
        v22.m128i_i64[0] = 0;
        v4[2].m128i_i64[0] = v11;
        v12 = v22.m128i_i64[1];
        v22.m128i_i64[1] = v10;
        v4[2].m128i_i64[1] = v12;
      }
      v4 = (__m128i *)a1[8];
    }
    v13 = v4 + 3;
    a1[8] = v13;
  }
  v14 = (__int64)v13->m128i_i64 - a1[7];
  if ( (unsigned __int64)v14 > 0x493E00 )
    abort();
  if ( v20.m128i_i32[0] == 11 && v22.m128i_i64[0] )
    ((void (__fastcall *)(__m128i *, __m128i *, __int64))v22.m128i_i64[0])(&v21, &v21, 3);
  if ( v17.m128i_i32[0] != 11 || !v19.m128i_i64[0] )
    return 0xAAAAAAAAAAAAAAABLL * (v14 >> 4) - 1;
  ((void (__fastcall *)(__m128i *, __m128i *, __int64))v19.m128i_i64[0])(&v18, &v18, 3);
  return 0xAAAAAAAAAAAAAAABLL * (v14 >> 4) - 1;
}
