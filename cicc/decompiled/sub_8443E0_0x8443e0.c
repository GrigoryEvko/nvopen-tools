// Function: sub_8443E0
// Address: 0x8443e0
//
__int64 __fastcall sub_8443E0(__m128i *a1, __int64 a2, int a3)
{
  __int64 v4; // r14
  __int8 v6; // bl
  __int64 i; // r12
  int v9; // ecx
  _BOOL4 v10; // edx
  _BOOL4 v11; // eax
  __int64 v12; // rdi
  int v13; // esi
  __int64 v14; // r12
  int v15; // eax
  _BOOL4 v16; // [rsp+4h] [rbp-1ACh]
  int v17; // [rsp+8h] [rbp-1A8h]
  int v18; // [rsp+14h] [rbp-19Ch] BYREF
  __int64 v19; // [rsp+18h] [rbp-198h] BYREF
  _OWORD v20[4]; // [rsp+20h] [rbp-190h] BYREF
  _OWORD v21[5]; // [rsp+60h] [rbp-150h] BYREF
  __m128i v22; // [rsp+B0h] [rbp-100h]
  __m128i v23; // [rsp+C0h] [rbp-F0h]
  __m128i v24; // [rsp+D0h] [rbp-E0h]
  __m128i v25; // [rsp+E0h] [rbp-D0h]
  __m128i v26; // [rsp+F0h] [rbp-C0h]
  __m128i v27; // [rsp+100h] [rbp-B0h]
  __m128i v28; // [rsp+110h] [rbp-A0h]
  __m128i v29; // [rsp+120h] [rbp-90h]
  __m128i v30; // [rsp+130h] [rbp-80h]
  __m128i v31; // [rsp+140h] [rbp-70h]
  __m128i v32; // [rsp+150h] [rbp-60h]
  __m128i v33; // [rsp+160h] [rbp-50h]
  __m128i v34; // [rsp+170h] [rbp-40h]

  v4 = a2;
  v6 = a1[1].m128i_i8[0];
  v20[0] = _mm_loadu_si128(a1);
  v20[1] = _mm_loadu_si128(a1 + 1);
  v20[2] = _mm_loadu_si128(a1 + 2);
  v20[3] = _mm_loadu_si128(a1 + 3);
  v21[0] = _mm_loadu_si128(a1 + 4);
  v21[1] = _mm_loadu_si128(a1 + 5);
  v21[2] = _mm_loadu_si128(a1 + 6);
  v21[3] = _mm_loadu_si128(a1 + 7);
  v21[4] = _mm_loadu_si128(a1 + 8);
  if ( v6 == 2 )
  {
    v22 = _mm_loadu_si128(a1 + 9);
    v23 = _mm_loadu_si128(a1 + 10);
    v24 = _mm_loadu_si128(a1 + 11);
    v25 = _mm_loadu_si128(a1 + 12);
    v26 = _mm_loadu_si128(a1 + 13);
    v27 = _mm_loadu_si128(a1 + 14);
    v28 = _mm_loadu_si128(a1 + 15);
    v29 = _mm_loadu_si128(a1 + 16);
    v30 = _mm_loadu_si128(a1 + 17);
    v31 = _mm_loadu_si128(a1 + 18);
    v32 = _mm_loadu_si128(a1 + 19);
    v33 = _mm_loadu_si128(a1 + 20);
    v34 = _mm_loadu_si128(a1 + 21);
    goto LABEL_4;
  }
  if ( v6 != 5 && v6 != 1 )
  {
LABEL_4:
    if ( a2 )
      goto LABEL_5;
    goto LABEL_13;
  }
  v22.m128i_i64[0] = a1[9].m128i_i64[0];
  if ( a2 )
  {
LABEL_5:
    for ( i = a2; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    if ( dword_4F077C4 != 2 )
      goto LABEL_8;
LABEL_16:
    if ( (unsigned int)sub_8D23B0(v4) )
      sub_8AE000(v4);
    goto LABEL_8;
  }
LABEL_13:
  for ( i = a1->m128i_i64[0]; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v4 = i;
  if ( dword_4F077C4 == 2 )
    goto LABEL_16;
LABEL_8:
  if ( v6 == 2 || !(unsigned int)sub_8D3A70(i) || dword_4F077C4 != 2 )
    goto LABEL_10;
  v9 = (_DWORD)a1 + 68;
  v10 = 1;
  if ( a1[1].m128i_i8[1] != 2 )
  {
    v11 = sub_6ED0A0((__int64)a1);
    v9 = (_DWORD)a1 + 68;
    v10 = v11;
  }
  v12 = a1->m128i_i64[0];
  v13 = 0;
  if ( (*(_BYTE *)(a1->m128i_i64[0] + 140) & 0xFB) == 8 )
  {
    v16 = v10;
    v17 = v9;
    v15 = sub_8D4C10(v12, dword_4F077C4 != 2);
    v10 = v16;
    v9 = v17;
    v13 = v15;
  }
  v14 = sub_6EB190(i, v13, v10, v9, (int)&v18, 0);
  if ( v18 )
  {
LABEL_10:
    sub_8283A0((__int64)a1, v4, a3, 0);
  }
  else if ( v14 )
  {
    sub_8441D0(a1, v14, 0, 0, &v19, &v18);
    sub_8284D0(v14, v19, v4, a3, v18, 0, 0, (__int64 *)((char *)v21 + 4), (__int64)a1);
  }
  else
  {
    sub_6E6840((__int64)a1);
  }
  return sub_6E4BC0((__int64)a1, (__int64)v20);
}
