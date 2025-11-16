// Function: sub_6F3BA0
// Address: 0x6f3ba0
//
__int64 __fastcall sub_6F3BA0(__m128i *a1, int a2)
{
  __int8 v2; // al
  bool v3; // zf
  bool v4; // r13
  __m128i v6; // [rsp+0h] [rbp-180h] BYREF
  __m128i v7; // [rsp+10h] [rbp-170h]
  __m128i v8; // [rsp+20h] [rbp-160h]
  __m128i v9; // [rsp+30h] [rbp-150h]
  __m128i v10; // [rsp+40h] [rbp-140h]
  __m128i v11; // [rsp+50h] [rbp-130h]
  __m128i v12; // [rsp+60h] [rbp-120h]
  __m128i v13; // [rsp+70h] [rbp-110h]
  __m128i v14; // [rsp+80h] [rbp-100h]
  __m128i v15; // [rsp+90h] [rbp-F0h]
  __m128i v16; // [rsp+A0h] [rbp-E0h]
  __m128i v17; // [rsp+B0h] [rbp-D0h]
  __m128i v18; // [rsp+C0h] [rbp-C0h]
  __m128i v19; // [rsp+D0h] [rbp-B0h]
  __m128i v20; // [rsp+E0h] [rbp-A0h]
  __m128i v21; // [rsp+F0h] [rbp-90h]
  __m128i v22; // [rsp+100h] [rbp-80h]
  __m128i v23; // [rsp+110h] [rbp-70h]
  __m128i v24; // [rsp+120h] [rbp-60h]
  __m128i v25; // [rsp+130h] [rbp-50h]
  __m128i v26; // [rsp+140h] [rbp-40h]
  __m128i v27; // [rsp+150h] [rbp-30h]

  v2 = a1[1].m128i_i8[0];
  v6 = _mm_loadu_si128(a1);
  v7 = _mm_loadu_si128(a1 + 1);
  v8 = _mm_loadu_si128(a1 + 2);
  v3 = a1[1].m128i_i8[1] == 3;
  v9 = _mm_loadu_si128(a1 + 3);
  v4 = v3;
  v10 = _mm_loadu_si128(a1 + 4);
  v11 = _mm_loadu_si128(a1 + 5);
  v12 = _mm_loadu_si128(a1 + 6);
  v13 = _mm_loadu_si128(a1 + 7);
  v14 = _mm_loadu_si128(a1 + 8);
  if ( v2 == 2 )
  {
    v15 = _mm_loadu_si128(a1 + 9);
    v16 = _mm_loadu_si128(a1 + 10);
    v17 = _mm_loadu_si128(a1 + 11);
    v18 = _mm_loadu_si128(a1 + 12);
    v19 = _mm_loadu_si128(a1 + 13);
    v20 = _mm_loadu_si128(a1 + 14);
    v21 = _mm_loadu_si128(a1 + 15);
    v22 = _mm_loadu_si128(a1 + 16);
    v23 = _mm_loadu_si128(a1 + 17);
    v24 = _mm_loadu_si128(a1 + 18);
    v25 = _mm_loadu_si128(a1 + 19);
    v26 = _mm_loadu_si128(a1 + 20);
    v27 = _mm_loadu_si128(a1 + 21);
  }
  else if ( v2 == 5 || v2 == 1 )
  {
    v15.m128i_i64[0] = a1[9].m128i_i64[0];
  }
  sub_6F3AD0(
    a1[8].m128i_i64[1],
    (a1[1].m128i_i8[3] & 8) != 0,
    a1[6].m128i_i64[1],
    (a1[1].m128i_i8[2] & 0x40) != 0,
    (__int64)a1);
  if ( (v7.m128i_i8[2] & 0x10) != 0 && a1[1].m128i_i8[0] == 2 )
    sub_6E83C0(a1[9].m128i_i64, 1);
  sub_6E4BC0((__int64)a1, (__int64)&v6);
  sub_6E5010(a1, &v6);
  if ( !a2 && v4 )
    sub_6F7FE0(a1, 0);
  return sub_6E5070((__int64)a1, (__int64)&v6);
}
