// Function: sub_6F4B70
// Address: 0x6f4b70
//
__int64 __fastcall sub_6F4B70(__m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int8 v6; // al
  __int64 v8; // [rsp+8h] [rbp-178h] BYREF
  __m128i v9; // [rsp+10h] [rbp-170h] BYREF
  __m128i v10; // [rsp+20h] [rbp-160h]
  __m128i v11; // [rsp+30h] [rbp-150h]
  __m128i v12; // [rsp+40h] [rbp-140h]
  __m128i v13; // [rsp+50h] [rbp-130h]
  __m128i v14; // [rsp+60h] [rbp-120h]
  __m128i v15; // [rsp+70h] [rbp-110h]
  __m128i v16; // [rsp+80h] [rbp-100h]
  __m128i v17; // [rsp+90h] [rbp-F0h]
  __m128i v18; // [rsp+A0h] [rbp-E0h]
  __m128i v19; // [rsp+B0h] [rbp-D0h]
  __m128i v20; // [rsp+C0h] [rbp-C0h]
  __m128i v21; // [rsp+D0h] [rbp-B0h]
  __m128i v22; // [rsp+E0h] [rbp-A0h]
  __m128i v23; // [rsp+F0h] [rbp-90h]
  __m128i v24; // [rsp+100h] [rbp-80h]
  __m128i v25; // [rsp+110h] [rbp-70h]
  __m128i v26; // [rsp+120h] [rbp-60h]
  __m128i v27; // [rsp+130h] [rbp-50h]
  __m128i v28; // [rsp+140h] [rbp-40h]
  __m128i v29; // [rsp+150h] [rbp-30h]
  __m128i v30; // [rsp+160h] [rbp-20h]

  v8 = sub_724DC0(a1, a2, a3, a4, a5, a6);
  v9 = _mm_loadu_si128(a1);
  v10 = _mm_loadu_si128(a1 + 1);
  v11 = _mm_loadu_si128(a1 + 2);
  v6 = a1[1].m128i_i8[0];
  v12 = _mm_loadu_si128(a1 + 3);
  v13 = _mm_loadu_si128(a1 + 4);
  v14 = _mm_loadu_si128(a1 + 5);
  v15 = _mm_loadu_si128(a1 + 6);
  v16 = _mm_loadu_si128(a1 + 7);
  v17 = _mm_loadu_si128(a1 + 8);
  if ( v6 == 2 )
  {
    v18 = _mm_loadu_si128(a1 + 9);
    v19 = _mm_loadu_si128(a1 + 10);
    v20 = _mm_loadu_si128(a1 + 11);
    v21 = _mm_loadu_si128(a1 + 12);
    v22 = _mm_loadu_si128(a1 + 13);
    v23 = _mm_loadu_si128(a1 + 14);
    v24 = _mm_loadu_si128(a1 + 15);
    v25 = _mm_loadu_si128(a1 + 16);
    v26 = _mm_loadu_si128(a1 + 17);
    v27 = _mm_loadu_si128(a1 + 18);
    v28 = _mm_loadu_si128(a1 + 19);
    v29 = _mm_loadu_si128(a1 + 20);
    v30 = _mm_loadu_si128(a1 + 21);
  }
  else if ( v6 == 5 || v6 == 1 )
  {
    v18.m128i_i64[0] = a1[9].m128i_i64[0];
  }
  sub_6F4910(a1, v8, 0);
  sub_6E6A50(v8, (__int64)a1);
  sub_6E4BC0((__int64)a1, (__int64)&v9);
  a1[1].m128i_i8[1] = v10.m128i_i8[1];
  return sub_724E30(&v8);
}
