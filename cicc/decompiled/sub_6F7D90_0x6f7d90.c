// Function: sub_6F7D90
// Address: 0x6f7d90
//
__int64 __fastcall sub_6F7D90(const __m128i *a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int8 v6; // al
  __int64 *v7; // rax
  __int64 *v8; // r12
  __int64 v9; // rsi
  __int64 v10; // rax
  int v11; // edx
  __int64 result; // rax
  int v13; // [rsp+Ch] [rbp-184h] BYREF
  __m128i v14; // [rsp+10h] [rbp-180h] BYREF
  __m128i v15; // [rsp+20h] [rbp-170h]
  __m128i v16; // [rsp+30h] [rbp-160h]
  __m128i v17; // [rsp+40h] [rbp-150h]
  __m128i v18; // [rsp+50h] [rbp-140h]
  __m128i v19; // [rsp+60h] [rbp-130h]
  __m128i v20; // [rsp+70h] [rbp-120h]
  __m128i v21; // [rsp+80h] [rbp-110h]
  __m128i v22; // [rsp+90h] [rbp-100h]
  _OWORD v23[10]; // [rsp+A0h] [rbp-F0h] BYREF
  __m128i v24; // [rsp+140h] [rbp-50h]
  __m128i v25; // [rsp+150h] [rbp-40h]
  __m128i v26; // [rsp+160h] [rbp-30h]

  v6 = a1[1].m128i_i8[0];
  v14 = _mm_loadu_si128(a1);
  v15 = _mm_loadu_si128(a1 + 1);
  v16 = _mm_loadu_si128(a1 + 2);
  v17 = _mm_loadu_si128(a1 + 3);
  v18 = _mm_loadu_si128(a1 + 4);
  v19 = _mm_loadu_si128(a1 + 5);
  v20 = _mm_loadu_si128(a1 + 6);
  v21 = _mm_loadu_si128(a1 + 7);
  v22 = _mm_loadu_si128(a1 + 8);
  if ( v6 == 2 )
  {
    v23[0] = _mm_loadu_si128(a1 + 9);
    v23[1] = _mm_loadu_si128(a1 + 10);
    v23[2] = _mm_loadu_si128(a1 + 11);
    v23[3] = _mm_loadu_si128(a1 + 12);
    v23[4] = _mm_loadu_si128(a1 + 13);
    v23[5] = _mm_loadu_si128(a1 + 14);
    v23[6] = _mm_loadu_si128(a1 + 15);
    v23[7] = _mm_loadu_si128(a1 + 16);
    v23[8] = _mm_loadu_si128(a1 + 17);
    v23[9] = _mm_loadu_si128(a1 + 18);
    v24 = _mm_loadu_si128(a1 + 19);
    v25 = _mm_loadu_si128(a1 + 20);
    v26 = _mm_loadu_si128(a1 + 21);
  }
  else if ( v6 == 5 || v6 == 1 )
  {
    *(_QWORD *)&v23[0] = a1[9].m128i_i64[0];
  }
  v7 = (__int64 *)sub_6F6F40(a1, 0, a3, a4, a5, a6);
  v8 = v7;
  if ( a2 )
    v9 = sub_73C570(*v7, 1, -1);
  else
    v9 = *v7;
  v10 = sub_73DC30(116, v9, v8);
  *(_BYTE *)(v10 + 27) |= 2u;
  *(_QWORD *)(v10 + 28) = *(__int64 *)((char *)v8 + 28);
  sub_6E7150((__int64 *)v10, (__int64)a1);
  v13 = 0;
  if ( v24.m128i_i8[13] == 12 && (unsigned int)sub_6DF6A0((__int64)v23, &v13) && v13 )
    a1[1].m128i_i8[1] = 3;
  sub_6E4BC0((__int64)a1, (__int64)&v14);
  v11 = v15.m128i_i8[2] & 0x28;
  result = v11 | a1[1].m128i_i8[2] & 0xD7u;
  a1[1].m128i_i8[2] = v11 | a1[1].m128i_i8[2] & 0xD7;
  return result;
}
