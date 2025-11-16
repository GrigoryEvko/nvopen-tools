// Function: sub_378E9D0
// Address: 0x378e9d0
//
unsigned __int8 *__fastcall sub_378E9D0(__int64 *a1, __int64 a2)
{
  __int64 v4; // rsi
  __int16 *v5; // rax
  __int16 v6; // dx
  unsigned int *v7; // rax
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // rdx
  __int64 (__fastcall *v11)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v12; // rsi
  int v13; // eax
  unsigned __int64 v14; // rdx
  const __m128i *v15; // rax
  int v16; // ecx
  unsigned __int64 v17; // rsi
  unsigned __int64 v18; // r9
  __int64 v19; // r8
  __m128i *v20; // rdx
  _QWORD *v21; // rdi
  _QWORD *v22; // rax
  int v23; // edx
  int v24; // r15d
  __int64 v25; // rdx
  __int64 v26; // r8
  __int64 v27; // r9
  int v28; // ecx
  unsigned __int64 v29; // rsi
  _QWORD *v30; // rdx
  _QWORD *v31; // rdi
  unsigned __int8 *v32; // r14
  __int64 v34; // rdx
  __int128 v35; // [rsp-10h] [rbp-1C0h]
  const __m128i *v36; // [rsp+0h] [rbp-1B0h]
  __int64 v37; // [rsp+8h] [rbp-1A8h]
  _QWORD *v38; // [rsp+10h] [rbp-1A0h]
  int v39; // [rsp+10h] [rbp-1A0h]
  int v40; // [rsp+18h] [rbp-198h]
  _QWORD *v41; // [rsp+18h] [rbp-198h]
  int v42; // [rsp+20h] [rbp-190h]
  __int64 v43; // [rsp+20h] [rbp-190h]
  unsigned __int16 v44; // [rsp+28h] [rbp-188h]
  int v45; // [rsp+28h] [rbp-188h]
  __int64 v46; // [rsp+30h] [rbp-180h] BYREF
  int v47; // [rsp+38h] [rbp-178h]
  __int64 v48; // [rsp+40h] [rbp-170h] BYREF
  __int64 v49; // [rsp+48h] [rbp-168h]
  __int64 v50; // [rsp+50h] [rbp-160h] BYREF
  __int64 v51; // [rsp+58h] [rbp-158h]
  __int64 v52; // [rsp+60h] [rbp-150h] BYREF
  int v53; // [rsp+68h] [rbp-148h]
  _QWORD *v54; // [rsp+70h] [rbp-140h] BYREF
  __int64 v55; // [rsp+78h] [rbp-138h]
  _QWORD v56[38]; // [rsp+80h] [rbp-130h] BYREF

  v4 = *(_QWORD *)(a2 + 80);
  v46 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v46, v4, 1);
  v47 = *(_DWORD *)(a2 + 72);
  v5 = *(__int16 **)(a2 + 48);
  v6 = *v5;
  v49 = *((_QWORD *)v5 + 1);
  v7 = *(unsigned int **)(a2 + 40);
  LOWORD(v48) = v6;
  v8 = *(_QWORD *)(*(_QWORD *)v7 + 48LL) + 16LL * v7[2];
  v9 = *(_QWORD *)(v8 + 8);
  v44 = *(_WORD *)v8;
  if ( v6 )
  {
    if ( (unsigned __int16)(v6 - 176) > 0x34u )
    {
LABEL_5:
      v42 = word_4456340[(unsigned __int16)v48 - 1];
      goto LABEL_8;
    }
  }
  else if ( !sub_3007100((__int64)&v48) )
  {
    goto LABEL_7;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( (_WORD)v48 )
  {
    if ( (unsigned __int16)(v48 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    goto LABEL_5;
  }
LABEL_7:
  v42 = sub_3007130((__int64)&v48, v4);
LABEL_8:
  v10 = a1[1];
  v11 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  if ( v11 == sub_2D56A50 )
  {
    v12 = *a1;
    sub_2FE6CC0((__int64)&v54, *a1, *(_QWORD *)(v10 + 64), v48, v49);
    LOWORD(v13) = v55;
    LOWORD(v50) = v55;
    v51 = v56[0];
  }
  else
  {
    v12 = *(_QWORD *)(v10 + 64);
    v13 = v11(*a1, v12, v48, v49);
    LODWORD(v50) = v13;
    v51 = v34;
  }
  if ( (_WORD)v13 )
  {
    if ( (unsigned __int16)(v13 - 176) > 0x34u )
    {
LABEL_12:
      v40 = word_4456340[(unsigned __int16)v50 - 1];
      goto LABEL_15;
    }
  }
  else if ( !sub_3007100((__int64)&v50) )
  {
    goto LABEL_14;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( (_WORD)v50 )
  {
    if ( (unsigned __int16)(v50 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    goto LABEL_12;
  }
LABEL_14:
  v40 = sub_3007130((__int64)&v50, v12);
LABEL_15:
  v14 = *(unsigned int *)(a2 + 64);
  v15 = *(const __m128i **)(a2 + 40);
  v16 = 0;
  v54 = v56;
  v55 = 0x1000000000LL;
  v17 = 40 * v14;
  v18 = v14;
  v19 = (__int64)&v15->m128i_i64[5 * v14];
  v20 = (__m128i *)v56;
  if ( v17 > 0x280 )
  {
    v36 = v15;
    v37 = v19;
    v39 = v18;
    sub_C8D5F0((__int64)&v54, v56, v18, 0x10u, v19, v18);
    v16 = v55;
    v15 = v36;
    v19 = v37;
    LODWORD(v18) = v39;
    v20 = (__m128i *)&v54[2 * (unsigned int)v55];
  }
  if ( v15 != (const __m128i *)v19 )
  {
    do
    {
      if ( v20 )
        *v20 = _mm_loadu_si128(v15);
      v15 = (const __m128i *)((char *)v15 + 40);
      ++v20;
    }
    while ( (const __m128i *)v19 != v15 );
    v16 = v55;
  }
  v21 = (_QWORD *)a1[1];
  LODWORD(v55) = v16 + v18;
  v52 = 0;
  v53 = 0;
  v22 = sub_33F17F0(v21, 51, (__int64)&v52, v44, v9);
  v24 = v23;
  if ( v52 )
  {
    v38 = v22;
    sub_B91220((__int64)&v52, v52);
    v22 = v38;
  }
  v25 = (unsigned int)v55;
  v26 = (unsigned int)(v40 - v42);
  v27 = v26;
  v28 = v55;
  if ( v26 + (unsigned __int64)(unsigned int)v55 > HIDWORD(v55) )
  {
    v41 = v22;
    v43 = v26;
    v45 = v26;
    sub_C8D5F0((__int64)&v54, v56, v26 + (unsigned int)v55, 0x10u, v26, v26);
    v25 = (unsigned int)v55;
    v22 = v41;
    v27 = v43;
    LODWORD(v26) = v45;
    v28 = v55;
  }
  v29 = (unsigned __int64)v54;
  v30 = &v54[2 * v25];
  if ( v27 )
  {
    do
    {
      if ( v30 )
      {
        *v30 = v22;
        *((_DWORD *)v30 + 2) = v24;
      }
      v30 += 2;
      --v27;
    }
    while ( v27 );
    v29 = (unsigned __int64)v54;
    v28 = v55;
  }
  v31 = (_QWORD *)a1[1];
  LODWORD(v55) = v28 + v26;
  *((_QWORD *)&v35 + 1) = (unsigned int)(v28 + v26);
  *(_QWORD *)&v35 = v29;
  v32 = sub_33FC220(v31, 156, (__int64)&v46, v50, v51, v27, v35);
  if ( v54 != v56 )
    _libc_free((unsigned __int64)v54);
  if ( v46 )
    sub_B91220((__int64)&v46, v46);
  return v32;
}
