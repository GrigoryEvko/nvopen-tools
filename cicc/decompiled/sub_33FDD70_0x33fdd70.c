// Function: sub_33FDD70
// Address: 0x33fdd70
//
_QWORD *__fastcall sub_33FDD70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int16 *v7; // rdx
  __int64 v8; // r14
  unsigned int v9; // r15d
  unsigned __int64 v10; // r8
  const void *v11; // r11
  size_t v12; // r13
  int *v13; // rax
  int v14; // r8d
  __int64 v15; // rcx
  __int64 v16; // r9
  int v17; // edx
  int v18; // edi
  const __m128i *v19; // rdx
  __int64 v20; // rsi
  int *v21; // r8
  __int64 v22; // r9
  __m128i v23; // xmm0
  __int64 v24; // r10
  __int64 v25; // r11
  _QWORD *v26; // r14
  int *v28; // rdi
  __int64 v29; // [rsp+0h] [rbp-B0h]
  __int64 v30; // [rsp+8h] [rbp-A8h]
  int *v31; // [rsp+10h] [rbp-A0h]
  const void *v32; // [rsp+10h] [rbp-A0h]
  __int64 v33; // [rsp+18h] [rbp-98h]
  int v35; // [rsp+30h] [rbp-80h]
  int v36; // [rsp+30h] [rbp-80h]
  __int64 v37; // [rsp+40h] [rbp-70h] BYREF
  int v38; // [rsp+48h] [rbp-68h]
  int *v39; // [rsp+50h] [rbp-60h] BYREF
  __int64 v40; // [rsp+58h] [rbp-58h]
  _BYTE dest[80]; // [rsp+60h] [rbp-50h] BYREF

  v7 = *(unsigned __int16 **)(a2 + 48);
  v8 = *((_QWORD *)v7 + 1);
  v9 = *v7;
  LOWORD(v39) = *v7;
  v40 = v8;
  if ( (_WORD)v39 )
  {
    if ( (unsigned __int16)((_WORD)v39 - 176) > 0x34u )
    {
LABEL_3:
      v10 = word_4456340[(unsigned __int16)v39 - 1];
      goto LABEL_6;
    }
  }
  else if ( !sub_3007100((__int64)&v39) )
  {
    goto LABEL_5;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( (_WORD)v39 )
  {
    if ( (unsigned __int16)((_WORD)v39 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    goto LABEL_3;
  }
LABEL_5:
  v10 = (unsigned int)sub_3007130((__int64)&v39, a2);
LABEL_6:
  v11 = *(const void **)(a2 + 96);
  v12 = 4 * v10;
  v39 = (int *)dest;
  v40 = 0x800000000LL;
  if ( v10 > 8 )
  {
    v32 = v11;
    v36 = v10;
    sub_C8D5F0((__int64)&v39, dest, v10, 4u, v10, a6);
    LODWORD(v10) = v36;
    v11 = v32;
    v28 = &v39[(unsigned int)v40];
  }
  else
  {
    v13 = (int *)dest;
    if ( !v12 )
      goto LABEL_8;
    v28 = (int *)dest;
  }
  v35 = v10;
  memcpy(v28, v11, v12);
  v13 = v39;
  LODWORD(v12) = v40;
  LODWORD(v10) = v35;
LABEL_8:
  v14 = v12 + v10;
  v15 = 0;
  LODWORD(v40) = v14;
  if ( v14 )
  {
    v16 = (__int64)&v13[v14 - 1 + 1];
    do
    {
      v17 = *v13;
      if ( *v13 >= 0 )
      {
        v18 = v17 - v14;
        if ( v17 < v14 )
          v18 = v14 + v17;
        *v13 = v18;
      }
      ++v13;
    }
    while ( v13 != (int *)v16 );
    v13 = v39;
    v15 = (unsigned int)v40;
  }
  v19 = *(const __m128i **)(a2 + 40);
  v20 = *(_QWORD *)(a2 + 80);
  v21 = v13;
  v22 = v15;
  v23 = _mm_loadu_si128(v19);
  v24 = v19[2].m128i_i64[1];
  v37 = v20;
  v25 = v19[3].m128i_i64[0];
  if ( v20 )
  {
    v29 = v24;
    v30 = v19[3].m128i_i64[0];
    v31 = v13;
    v33 = v15;
    sub_B96E90((__int64)&v37, v20, 1);
    v24 = v29;
    v25 = v30;
    v21 = v31;
    v22 = v33;
  }
  v38 = *(_DWORD *)(a2 + 72);
  v26 = sub_33FCE10(a1, v9, v8, (__int64)&v37, v24, v25, v23, v23.m128i_i64[0], v23.m128i_i64[1], v21, v22);
  if ( v37 )
    sub_B91220((__int64)&v37, v37);
  if ( v39 != (int *)dest )
    _libc_free((unsigned __int64)v39);
  return v26;
}
