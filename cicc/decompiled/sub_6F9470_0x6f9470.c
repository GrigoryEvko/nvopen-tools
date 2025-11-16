// Function: sub_6F9470
// Address: 0x6f9470
//
__int64 __fastcall sub_6F9470(const __m128i *a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int8 v7; // al
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  int v16; // [rsp+Ch] [rbp-194h] BYREF
  __int64 v17; // [rsp+10h] [rbp-190h] BYREF
  __int64 v18; // [rsp+18h] [rbp-188h] BYREF
  _OWORD v19[9]; // [rsp+20h] [rbp-180h] BYREF
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

  v7 = a1[1].m128i_i8[0];
  v19[0] = _mm_loadu_si128(a1);
  v19[1] = _mm_loadu_si128(a1 + 1);
  v19[2] = _mm_loadu_si128(a1 + 2);
  v19[3] = _mm_loadu_si128(a1 + 3);
  v19[4] = _mm_loadu_si128(a1 + 4);
  v19[5] = _mm_loadu_si128(a1 + 5);
  v19[6] = _mm_loadu_si128(a1 + 6);
  v19[7] = _mm_loadu_si128(a1 + 7);
  v19[8] = _mm_loadu_si128(a1 + 8);
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
    goto LABEL_4;
  }
  if ( v7 == 5 )
  {
    v20.m128i_i64[0] = a1[9].m128i_i64[0];
    goto LABEL_4;
  }
  if ( v7 != 1 )
  {
LABEL_4:
    v17 = 0;
    v18 = 0;
    v8 = sub_6F6F40(a1, 0, a3, a4, a5, a6);
    goto LABEL_5;
  }
  v14 = a1[9].m128i_i64[0];
  v17 = 0;
  v18 = 0;
  v20.m128i_i64[0] = v14;
  v15 = sub_6F6F40(a1, 0, a3, a4, a5, a6);
  v16 = 0;
  v11 = sub_6EED10(v15, &v16, 0, 0, 0, 0);
  if ( v16 )
  {
    if ( a2 )
      *(_BYTE *)(v11 + 25) = *(_BYTE *)(v11 + 25) & 0xFC | 2;
    goto LABEL_12;
  }
  v8 = sub_6ECFC0(v11, &v17, &v18);
LABEL_5:
  v9 = sub_731410(v8, a2);
  v10 = v17;
  v11 = v9;
  if ( v17 )
  {
    v12 = v18;
    *(_QWORD *)(v18 + 72) = v11;
    v11 = v10;
    if ( *(_BYTE *)(v10 + 56) == 9 )
    {
      sub_73D8E0(v10, 8, *(_QWORD *)v10, 1, *(_QWORD *)(v10 + 72));
      if ( a2 )
        *(_BYTE *)(v10 + 25) = *(_BYTE *)(v10 + 25) & 0xFC | 2;
      v10 = *(_QWORD *)(v10 + 72);
      v12 = v18;
      v11 = v17;
    }
    while ( 1 )
    {
      if ( a2 )
      {
        *(_BYTE *)(v10 + 25) |= 2u;
        if ( v10 == v12 )
          break;
      }
      else
      {
        *(_BYTE *)(v10 + 25) |= 1u;
        if ( v10 == v12 )
          break;
      }
      v10 = *(_QWORD *)(v10 + 72);
    }
  }
LABEL_12:
  sub_6E7150((__int64 *)v11, (__int64)a1);
  return sub_6E4BC0((__int64)a1, (__int64)v19);
}
