// Function: sub_371D8F0
// Address: 0x371d8f0
//
__int64 __fastcall sub_371D8F0(__int64 a1, _QWORD *a2, __int64 *a3, __int64 a4, __int64 a5)
{
  unsigned int v6; // r15d
  unsigned int v8; // ebx
  __m128i *v9; // rdi
  __int64 v10; // rax
  unsigned __int64 v11; // r11
  const __m128i *v12; // rdx
  __m128i *v13; // rdi
  __m128i v14; // xmm2
  __int64 v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // rcx
  const __m128i *v18; // rax
  __m128i v19; // xmm0
  __int64 m128i_i64; // r11
  unsigned int v21; // ecx
  __int64 v22; // rdx
  bool v23; // cc
  const __m128i *v24; // rsi
  __int64 v25; // rcx
  unsigned int v26; // edx
  __m128i v27; // xmm0
  __int64 v28; // rax
  __int64 result; // rax
  char *v30; // r12
  _QWORD *v32; // [rsp+18h] [rbp-488h]
  _QWORD *v33; // [rsp+18h] [rbp-488h]
  unsigned __int8 v34; // [rsp+18h] [rbp-488h]
  __m128i v35; // [rsp+20h] [rbp-480h] BYREF
  __int128 v36; // [rsp+30h] [rbp-470h]
  __m128i v37; // [rsp+40h] [rbp-460h]
  __int64 v38; // [rsp+50h] [rbp-450h]
  __m128i *v39; // [rsp+60h] [rbp-440h] BYREF
  __int64 v40; // [rsp+68h] [rbp-438h]
  _BYTE v41[1072]; // [rsp+70h] [rbp-430h] BYREF

  v6 = 0;
  v8 = 0;
  v9 = (__m128i *)v41;
  v39 = (__m128i *)v41;
  v40 = 0x2000000000LL;
  while ( 1 )
  {
    v15 = *a2;
    v16 = a3[1];
    v17 = *a3;
    if ( v6 >= -1431655765 * (unsigned int)((__int64)(a2[1] - *a2) >> 3) )
    {
      if ( v8 >= -1431655765 * (unsigned int)((v16 - v17) >> 3) )
      {
        result = 0;
        goto LABEL_31;
      }
      v35 = 0;
      v36 = 0;
      v18 = (const __m128i *)(v17 + 24LL * v8);
LABEL_10:
      v19 = _mm_loadu_si128(v18);
      ++v8;
      v38 = v18[1].m128i_i64[0];
      *(_QWORD *)&v36 = v38;
      v37 = v19;
      v35 = v19;
      goto LABEL_11;
    }
    v35 = 0;
    v36 = 0;
    v24 = (const __m128i *)(v15 + 24LL * v6);
    if ( v8 < -1431655765 * (unsigned int)((v16 - v17) >> 3) )
    {
      v18 = (const __m128i *)(v17 + 24LL * v8);
      v25 = v24->m128i_i64[1];
      v26 = *(_DWORD *)(v18->m128i_i64[1] + 72);
      if ( *(_DWORD *)(v25 + 72) == v26 )
      {
        if ( v24[1].m128i_i32[0] >= (unsigned __int32)v18[1].m128i_i32[0] )
          goto LABEL_10;
      }
      else if ( *(_DWORD *)(v25 + 72) >= v26 )
      {
        goto LABEL_10;
      }
    }
    v27 = _mm_loadu_si128(v24);
    v28 = v24[1].m128i_i64[0];
    ++v6;
    BYTE8(v36) = 1;
    v38 = v28;
    *(_QWORD *)&v36 = v28;
    v37 = v27;
    v35 = v27;
LABEL_11:
    v10 = (unsigned int)v40;
    if ( !(_DWORD)v40 )
      goto LABEL_5;
    m128i_i64 = (__int64)v9[2 * (unsigned int)v40 - 2].m128i_i64;
    v21 = *(_DWORD *)(v35.m128i_i64[1] + 72);
    v22 = *(_QWORD *)(m128i_i64 + 8);
    v23 = *(_DWORD *)(v22 + 72) <= v21;
    if ( *(_DWORD *)(v22 + 72) == v21 )
    {
LABEL_16:
      if ( *(_DWORD *)(m128i_i64 + 16) <= (unsigned int)v36 )
        goto LABEL_3;
      goto LABEL_14;
    }
    while ( !v23 || *(_DWORD *)(v35.m128i_i64[1] + 76) > *(_DWORD *)(v22 + 76) )
    {
LABEL_14:
      v10 = (unsigned int)(v10 - 1);
      m128i_i64 -= 32;
      LODWORD(v40) = v10;
      if ( !(_DWORD)v10 )
        goto LABEL_5;
      v22 = *(_QWORD *)(m128i_i64 + 8);
      v23 = *(_DWORD *)(v22 + 72) <= v21;
      if ( *(_DWORD *)(v22 + 72) == v21 )
        goto LABEL_16;
    }
LABEL_3:
    if ( *(_BYTE *)(m128i_i64 + 24) == BYTE8(v36) )
    {
      v10 = (unsigned int)v40;
      goto LABEL_5;
    }
    v32 = a2;
    result = sub_371D350(a1, v35.m128i_i64, (_QWORD *)m128i_i64);
    if ( (_BYTE)result )
      break;
    v10 = (unsigned int)v40;
    v9 = v39;
    a2 = v32;
LABEL_5:
    v11 = v10 + 1;
    v12 = &v35;
    if ( v10 + 1 > (unsigned __int64)HIDWORD(v40) )
    {
      v33 = a2;
      if ( v9 > &v35 || &v35 >= &v9[2 * v10] )
      {
        sub_C8D5F0((__int64)&v39, v41, v11, 0x20u, a5, (__int64)a2);
        v9 = v39;
        v10 = (unsigned int)v40;
        v12 = &v35;
        a2 = v33;
      }
      else
      {
        v30 = (char *)((char *)&v35 - (char *)v9);
        sub_C8D5F0((__int64)&v39, v41, v11, 0x20u, a5, (__int64)a2);
        v9 = v39;
        v10 = (unsigned int)v40;
        a2 = v33;
        v12 = (const __m128i *)&v30[(_QWORD)v39];
      }
    }
    v13 = &v9[2 * v10];
    *v13 = _mm_loadu_si128(v12);
    v14 = _mm_loadu_si128(v12 + 1);
    LODWORD(v40) = v40 + 1;
    v13[1] = v14;
    v9 = v39;
  }
  v9 = v39;
LABEL_31:
  if ( v9 != (__m128i *)v41 )
  {
    v34 = result;
    _libc_free((unsigned __int64)v9);
    return v34;
  }
  return result;
}
