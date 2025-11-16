// Function: sub_6F7690
// Address: 0x6f7690
//
__int64 __fastcall sub_6F7690(const __m128i *a1, __int64 a2)
{
  __int64 v2; // r13
  _DWORD *v3; // r14
  char v4; // dl
  __int64 v5; // rax
  __int64 result; // rax
  int v7; // eax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 *v15; // rax
  _OWORD v16[9]; // [rsp+0h] [rbp-180h] BYREF
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

  if ( !a1[1].m128i_i8[0] )
    return sub_6E6870((__int64)a1);
  v2 = a1->m128i_i64[0];
  v3 = (_DWORD *)a2;
  v4 = *(_BYTE *)(a1->m128i_i64[0] + 140);
  if ( v4 == 12 )
  {
    v5 = a1->m128i_i64[0];
    do
    {
      v5 = *(_QWORD *)(v5 + 160);
      v4 = *(_BYTE *)(v5 + 140);
    }
    while ( v4 == 12 );
  }
  if ( !v4 )
    return sub_6E6870((__int64)a1);
  result = sub_8DAAE0(a1->m128i_i64[0], a2);
  if ( !(_DWORD)result )
  {
    v16[0] = _mm_loadu_si128(a1);
    v7 = a1[1].m128i_u8[0];
    v16[1] = _mm_loadu_si128(a1 + 1);
    v16[2] = _mm_loadu_si128(a1 + 2);
    v16[3] = _mm_loadu_si128(a1 + 3);
    v16[4] = _mm_loadu_si128(a1 + 4);
    v16[5] = _mm_loadu_si128(a1 + 5);
    v16[6] = _mm_loadu_si128(a1 + 6);
    v16[7] = _mm_loadu_si128(a1 + 7);
    v16[8] = _mm_loadu_si128(a1 + 8);
    if ( (_BYTE)v7 == 2 )
    {
      v17 = _mm_loadu_si128(a1 + 9);
      v18 = _mm_loadu_si128(a1 + 10);
      v19 = _mm_loadu_si128(a1 + 11);
      v20 = _mm_loadu_si128(a1 + 12);
      v21 = _mm_loadu_si128(a1 + 13);
      v22 = _mm_loadu_si128(a1 + 14);
      v23 = _mm_loadu_si128(a1 + 15);
      v24 = _mm_loadu_si128(a1 + 16);
      v25 = _mm_loadu_si128(a1 + 17);
      v26 = _mm_loadu_si128(a1 + 18);
      v27 = _mm_loadu_si128(a1 + 19);
      v28 = _mm_loadu_si128(a1 + 20);
      v29 = _mm_loadu_si128(a1 + 21);
    }
    else if ( (_BYTE)v7 == 5 || (_BYTE)v7 == 1 )
    {
      v17.m128i_i64[0] = a1[9].m128i_i64[0];
    }
    if ( !(unsigned int)sub_8D3A70(v2) || !(unsigned int)sub_8D3A70(a2) )
      goto LABEL_23;
    for ( ; *(_BYTE *)(v2 + 140) == 12; v2 = *(_QWORD *)(v2 + 160) )
      ;
    for ( ; *(_BYTE *)(a2 + 140) == 12; a2 = *(_QWORD *)(a2 + 160) )
      ;
    if ( a2 == v2
      || (v9 = dword_4F07588) != 0 && (v12 = *(_QWORD *)(v2 + 32), *(_QWORD *)(a2 + 32) == v12) && v12
      || (v13 = sub_8D5CE0(v2, a2)) == 0 )
    {
LABEL_23:
      v14 = sub_6F6F40(a1, 0, v8, v9, v10, v11);
      v15 = (__int64 *)sub_73DC50(v14, v3);
      sub_6E7150(v15, (__int64)a1);
    }
    else
    {
      sub_6F7270(a1, v13, v3, 1, 0, 1, 0, 0);
    }
    return sub_6E4EE0((__int64)a1, (__int64)v16);
  }
  return result;
}
