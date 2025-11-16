// Function: sub_8422F0
// Address: 0x8422f0
//
__int64 __fastcall sub_8422F0(const __m128i *a1, __m128i *a2)
{
  __int8 v3; // al
  __m128i v4; // xmm3
  __m128i v5; // xmm4
  __m128i v6; // xmm5
  __m128i v7; // xmm6
  __m128i v8; // xmm7
  __m128i v9; // xmm0
  __int64 v10; // rdi
  _OWORD v12[9]; // [rsp+0h] [rbp-170h] BYREF
  __m128i v13; // [rsp+90h] [rbp-E0h]
  __m128i v14; // [rsp+A0h] [rbp-D0h]
  __m128i v15; // [rsp+B0h] [rbp-C0h]
  __m128i v16; // [rsp+C0h] [rbp-B0h]
  __m128i v17; // [rsp+D0h] [rbp-A0h]
  __m128i v18; // [rsp+E0h] [rbp-90h]
  __m128i v19; // [rsp+F0h] [rbp-80h]
  __m128i v20; // [rsp+100h] [rbp-70h]
  __m128i v21; // [rsp+110h] [rbp-60h]
  __m128i v22; // [rsp+120h] [rbp-50h]
  __m128i v23; // [rsp+130h] [rbp-40h]
  __m128i v24; // [rsp+140h] [rbp-30h]
  __m128i v25; // [rsp+150h] [rbp-20h]

  v3 = a1[1].m128i_i8[0];
  v4 = _mm_loadu_si128(a1 + 3);
  v5 = _mm_loadu_si128(a1 + 4);
  v12[0] = _mm_loadu_si128(a1);
  v6 = _mm_loadu_si128(a1 + 5);
  v7 = _mm_loadu_si128(a1 + 6);
  v12[1] = _mm_loadu_si128(a1 + 1);
  v8 = _mm_loadu_si128(a1 + 7);
  v9 = _mm_loadu_si128(a1 + 8);
  v12[2] = _mm_loadu_si128(a1 + 2);
  v10 = a1[9].m128i_i64[0];
  v12[3] = v4;
  v12[4] = v5;
  v12[5] = v6;
  v12[6] = v7;
  v12[7] = v8;
  v12[8] = v9;
  if ( v3 == 2 )
  {
    v13 = _mm_loadu_si128(a1 + 9);
    v14 = _mm_loadu_si128(a1 + 10);
    v15 = _mm_loadu_si128(a1 + 11);
    v16 = _mm_loadu_si128(a1 + 12);
    v17 = _mm_loadu_si128(a1 + 13);
    v18 = _mm_loadu_si128(a1 + 14);
    v19 = _mm_loadu_si128(a1 + 15);
    v20 = _mm_loadu_si128(a1 + 16);
    v21 = _mm_loadu_si128(a1 + 17);
    v22 = _mm_loadu_si128(a1 + 18);
    v23 = _mm_loadu_si128(a1 + 19);
    v24 = _mm_loadu_si128(a1 + 20);
    v25 = _mm_loadu_si128(a1 + 21);
  }
  else if ( v3 == 5 || v3 == 1 )
  {
    v13.m128i_i64[0] = v10;
  }
  sub_839D30(v10, a2, 0, 0, 0, 0, 1, 0, 0, (__int64)a1, 0, 0);
  return sub_6E4BC0((__int64)a1, (__int64)v12);
}
