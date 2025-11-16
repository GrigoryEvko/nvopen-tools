// Function: sub_6FC8A0
// Address: 0x6fc8a0
//
__int64 __fastcall sub_6FC8A0(__m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int8 v7; // al
  int v8; // ebx
  __int64 v9; // rdi
  unsigned int v10; // r13d
  __int64 v12; // rax
  __int64 v13; // r13
  _BOOL4 v14; // eax
  __int64 v15; // rax
  __int8 v16; // al
  _BYTE v17[32]; // [rsp+0h] [rbp-1A0h] BYREF
  _OWORD v18[4]; // [rsp+20h] [rbp-180h] BYREF
  _OWORD v19[5]; // [rsp+60h] [rbp-140h] BYREF
  __m128i v20; // [rsp+B0h] [rbp-F0h]
  __m128i v21; // [rsp+C0h] [rbp-E0h]
  __m128i v22; // [rsp+D0h] [rbp-D0h]
  __m128i v23; // [rsp+E0h] [rbp-C0h]
  __m128i v24; // [rsp+F0h] [rbp-B0h]
  __m128i v25; // [rsp+100h] [rbp-A0h]
  __m128i v26; // [rsp+110h] [rbp-90h]
  __m128i v27; // [rsp+120h] [rbp-80h]
  __m128i v28; // [rsp+130h] [rbp-70h]
  __m128i v29; // [rsp+140h] [rbp-60h]
  __m128i v30; // [rsp+150h] [rbp-50h]
  __m128i v31; // [rsp+160h] [rbp-40h]
  __m128i v32; // [rsp+170h] [rbp-30h]

  if ( a1[1].m128i_i8[0] == 1 )
  {
    v12 = a1[9].m128i_i64[0];
    if ( *(_BYTE *)(v12 + 24) == 1
      && *(_BYTE *)(v12 + 56) == 73
      && *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v12 + 72) + 16LL) + 24LL) == 2
      && sub_6E53E0(5, 0xBBu, &a1[4].m128i_i32[1]) )
    {
      sub_684B30(0xBBu, &a1[4].m128i_i32[1]);
    }
  }
  if ( *(_BYTE *)(qword_4D03C50 + 16LL) > 3u )
    sub_6E6B60(a1, 0, a3, a4, a5, a6);
  v18[0] = _mm_loadu_si128(a1);
  v7 = a1[1].m128i_i8[0];
  v18[1] = _mm_loadu_si128(a1 + 1);
  v18[2] = _mm_loadu_si128(a1 + 2);
  v18[3] = _mm_loadu_si128(a1 + 3);
  v19[0] = _mm_loadu_si128(a1 + 4);
  v19[1] = _mm_loadu_si128(a1 + 5);
  v19[2] = _mm_loadu_si128(a1 + 6);
  v19[3] = _mm_loadu_si128(a1 + 7);
  v19[4] = _mm_loadu_si128(a1 + 8);
  if ( v7 == 2 )
  {
    v20 = _mm_loadu_si128(a1 + 9);
    v21 = _mm_loadu_si128(a1 + 10);
    v22 = _mm_loadu_si128(a1 + 11);
    v23 = _mm_loadu_si128(a1 + 12);
    v24 = _mm_loadu_si128(a1 + 13);
    v25 = _mm_loadu_si128(a1 + 14);
    v26 = _mm_loadu_si128(a1 + 15);
    v27 = _mm_loadu_si128(a1 + 16);
    v28 = _mm_loadu_si128(a1 + 17);
    v29 = _mm_loadu_si128(a1 + 18);
    v30 = _mm_loadu_si128(a1 + 19);
    v31 = _mm_loadu_si128(a1 + 20);
    v32 = _mm_loadu_si128(a1 + 21);
  }
  else if ( v7 == 5 || v7 == 1 )
  {
    v20.m128i_i64[0] = a1[9].m128i_i64[0];
  }
  if ( word_4D04898
    || a1[1].m128i_i16[0] != 514
    || !(unsigned int)sub_8D2E30(a1->m128i_i64[0]) && !(unsigned int)sub_8D3D10(a1->m128i_i64[0]) )
  {
    goto LABEL_9;
  }
  if ( a1[19].m128i_i8[13] != 6 )
    goto LABEL_32;
  v16 = a1[20].m128i_i8[0];
  if ( !v16 )
  {
    if ( (*(_BYTE *)(a1[20].m128i_i64[1] + 200) & 0x20) == 0 )
    {
LABEL_32:
      v8 = 1;
      goto LABEL_10;
    }
LABEL_9:
    v8 = 0;
    goto LABEL_10;
  }
  v8 = 1;
  if ( v16 == 1 && (*(_BYTE *)(a1[20].m128i_i64[1] + 168) & 8) != 0 )
    goto LABEL_9;
LABEL_10:
  v9 = a1->m128i_i64[0];
  if ( unk_4D0439C )
  {
    if ( !(unsigned int)sub_8D29A0(v9) )
    {
      v13 = sub_72C390();
      v14 = sub_6EB660((__int64)a1);
      v10 = sub_8E1010(
              a1->m128i_i64[0],
              a1[1].m128i_i8[0] == 2,
              (a1[1].m128i_i8[3] & 0x10) != 0,
              v14,
              0,
              (int)a1 + 144,
              v13,
              0,
              0,
              0,
              711,
              (__int64)v17,
              0);
      if ( !v10 )
      {
        sub_6E68E0(0x2C7u, (__int64)a1);
        goto LABEL_14;
      }
      v15 = sub_72C390();
      sub_6FC3F0(v15, a1, 1u);
    }
  }
  else if ( !(unsigned int)sub_8D3D10(v9) && !(unsigned int)sub_8D3D40(a1->m128i_i64[0]) )
  {
    v10 = sub_6E96B0((__int64)a1);
    if ( !v10 )
      goto LABEL_14;
    goto LABEL_13;
  }
  v10 = 1;
LABEL_13:
  if ( v8 && sub_6E53E0(5, 0xECu, (_DWORD *)v19 + 1) )
    sub_684B30(0xECu, (_DWORD *)v19 + 1);
LABEL_14:
  sub_6E4BC0((__int64)a1, (__int64)v18);
  return v10;
}
