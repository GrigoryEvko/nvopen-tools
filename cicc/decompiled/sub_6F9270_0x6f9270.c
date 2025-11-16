// Function: sub_6F9270
// Address: 0x6f9270
//
__int64 __fastcall sub_6F9270(const __m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rcx
  __int64 v7; // rax
  __int64 i; // rdx
  __int64 v10; // rax
  __int64 *v11; // rax
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

  v6 = a1[1].m128i_u8[0];
  if ( !(_BYTE)v6 )
    return sub_6E6870((__int64)a1);
  v7 = a1->m128i_i64[0];
  for ( i = *(unsigned __int8 *)(a1->m128i_i64[0] + 140); (_BYTE)i == 12; i = *(unsigned __int8 *)(v7 + 140) )
    v7 = *(_QWORD *)(v7 + 160);
  if ( !(_BYTE)i )
    return sub_6E6870((__int64)a1);
  v12[0] = _mm_loadu_si128(a1);
  v12[1] = _mm_loadu_si128(a1 + 1);
  v12[2] = _mm_loadu_si128(a1 + 2);
  v12[3] = _mm_loadu_si128(a1 + 3);
  v12[4] = _mm_loadu_si128(a1 + 4);
  v12[5] = _mm_loadu_si128(a1 + 5);
  v12[6] = _mm_loadu_si128(a1 + 6);
  v12[7] = _mm_loadu_si128(a1 + 7);
  v12[8] = _mm_loadu_si128(a1 + 8);
  if ( (_BYTE)v6 == 2 )
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
  else if ( (_BYTE)v6 == 5 || (_BYTE)v6 == 1 )
  {
    v13.m128i_i64[0] = a1[9].m128i_i64[0];
  }
  v10 = sub_6F6F40(a1, 0, i, v6, a5, a6);
  v11 = (__int64 *)sub_73DCD0(v10);
  sub_6E7150(v11, (__int64)a1);
  return sub_6E4EE0((__int64)a1, (__int64)v12);
}
