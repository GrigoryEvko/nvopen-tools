// Function: sub_3779320
// Address: 0x3779320
//
void __fastcall sub_3779320(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rsi
  __int64 v7; // rsi
  __int16 *v8; // rax
  __int16 v9; // dx
  __int64 v10; // rdx
  const __m128i *v11; // rax
  __int64 v12; // r9
  __int64 v13; // r8
  unsigned __int64 v14; // r11
  __m128i *v15; // rdx
  int v16; // esi
  _BYTE *v17; // rcx
  unsigned int v18; // r11d
  _QWORD *v19; // rdi
  unsigned __int8 *v20; // rax
  __int64 v21; // r9
  int v22; // edx
  __int64 v23; // rax
  __int64 v24; // r14
  __int64 v25; // r8
  __int64 v26; // rax
  const __m128i *v27; // r14
  unsigned __int64 v28; // rdx
  __m128i *v29; // rax
  __int32 v30; // ecx
  unsigned __int16 *v31; // rsi
  _QWORD *v32; // rdi
  unsigned __int8 *v33; // rax
  unsigned __int16 *v34; // rdi
  int v35; // edx
  __int128 v36; // [rsp-10h] [rbp-1F0h]
  __int128 v37; // [rsp-10h] [rbp-1F0h]
  __int64 v38; // [rsp+8h] [rbp-1D8h]
  const __m128i *v39; // [rsp+10h] [rbp-1D0h]
  __int64 v40; // [rsp+18h] [rbp-1C8h]
  __int8 *v41; // [rsp+18h] [rbp-1C8h]
  __int64 v42; // [rsp+28h] [rbp-1B8h]
  unsigned __int64 v43; // [rsp+28h] [rbp-1B8h]
  unsigned __int64 v44; // [rsp+28h] [rbp-1B8h]
  unsigned __int16 v47; // [rsp+46h] [rbp-19Ah]
  __int64 v48; // [rsp+48h] [rbp-198h]
  __m128i v49; // [rsp+70h] [rbp-170h] BYREF
  __int64 v50; // [rsp+80h] [rbp-160h] BYREF
  int v51; // [rsp+88h] [rbp-158h]
  _BYTE *v52; // [rsp+90h] [rbp-150h] BYREF
  __int64 v53; // [rsp+98h] [rbp-148h]
  _BYTE v54[128]; // [rsp+A0h] [rbp-140h] BYREF
  __m128i v55; // [rsp+120h] [rbp-C0h] BYREF
  unsigned __int16 v56; // [rsp+130h] [rbp-B0h] BYREF
  __int64 v57; // [rsp+138h] [rbp-A8h]

  v6 = *(_QWORD *)(a2 + 80);
  v49.m128i_i16[0] = 0;
  v49.m128i_i64[1] = 0;
  v50 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v50, v6, 1);
  v7 = *(_QWORD *)(a1 + 8);
  v51 = *(_DWORD *)(a2 + 72);
  v8 = *(__int16 **)(a2 + 48);
  v9 = *v8;
  v53 = *((_QWORD *)v8 + 1);
  LOWORD(v52) = v9;
  sub_33D0340((__int64)&v55, v7, (__int64 *)&v52);
  v47 = v56;
  v49 = _mm_loadu_si128(&v55);
  v48 = v57;
  if ( v55.m128i_i16[0] )
  {
    if ( (unsigned __int16)(v55.m128i_i16[0] - 176) > 0x34u )
    {
LABEL_5:
      v10 = word_4456340[v49.m128i_u16[0] - 1];
      goto LABEL_8;
    }
  }
  else if ( !sub_3007100((__int64)&v49) )
  {
    goto LABEL_7;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( v49.m128i_i16[0] )
  {
    if ( (unsigned __int16)(v49.m128i_i16[0] - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    goto LABEL_5;
  }
LABEL_7:
  v10 = (unsigned int)sub_3007130((__int64)&v49, v7);
LABEL_8:
  v11 = *(const __m128i **)(a2 + 40);
  v12 = 40 * v10;
  v52 = v54;
  v53 = 0x800000000LL;
  v13 = (__int64)&v11->m128i_i64[5 * v10];
  v14 = 0xCCCCCCCCCCCCCCCDLL * ((40 * v10) >> 3);
  if ( (unsigned __int64)(40 * v10) > 0x140 )
  {
    v38 = 40 * v10;
    v39 = v11;
    v41 = &v11->m128i_i8[v12];
    v44 = 0xCCCCCCCCCCCCCCCDLL * (v12 >> 3);
    sub_C8D5F0((__int64)&v52, v54, v44, 0x10u, v13, v12);
    v16 = v53;
    v17 = v52;
    LODWORD(v14) = v44;
    v13 = (__int64)v41;
    v11 = v39;
    v12 = v38;
    v15 = (__m128i *)&v52[16 * (unsigned int)v53];
  }
  else
  {
    v15 = (__m128i *)v54;
    v16 = 0;
    v17 = v54;
  }
  if ( (const __m128i *)v13 != v11 )
  {
    do
    {
      if ( v15 )
        *v15 = _mm_loadu_si128(v11);
      v11 = (const __m128i *)((char *)v11 + 40);
      ++v15;
    }
    while ( (const __m128i *)v13 != v11 );
    v17 = v52;
    v16 = v53;
  }
  v18 = v16 + v14;
  v19 = *(_QWORD **)(a1 + 8);
  v42 = v12;
  *((_QWORD *)&v36 + 1) = v18;
  *(_QWORD *)&v36 = v17;
  LODWORD(v53) = v18;
  v20 = sub_33FC220(v19, 156, (__int64)&v50, v49.m128i_i64[0], v49.m128i_i64[1], v12, v36);
  v21 = v42;
  *(_QWORD *)a3 = v20;
  v55.m128i_i64[0] = (__int64)&v56;
  *(_DWORD *)(a3 + 8) = v22;
  v23 = *(unsigned int *)(a2 + 64);
  v24 = *(_QWORD *)(a2 + 40);
  v55.m128i_i64[1] = 0x800000000LL;
  v23 *= 40;
  v25 = v24 + v23;
  v26 = v23 - v42;
  v27 = (const __m128i *)(v42 + v24);
  v28 = 0xCCCCCCCCCCCCCCCDLL * (v26 >> 3);
  if ( (unsigned __int64)v26 > 0x140 )
  {
    v40 = v25;
    v43 = 0xCCCCCCCCCCCCCCCDLL * (v26 >> 3);
    sub_C8D5F0((__int64)&v55, &v56, v28, 0x10u, v25, v21);
    v30 = v55.m128i_i32[2];
    v31 = (unsigned __int16 *)v55.m128i_i64[0];
    LODWORD(v28) = v43;
    v25 = v40;
    v29 = (__m128i *)(v55.m128i_i64[0] + 16LL * v55.m128i_u32[2]);
  }
  else
  {
    v29 = (__m128i *)&v56;
    v30 = 0;
    v31 = &v56;
  }
  if ( v27 != (const __m128i *)v25 )
  {
    do
    {
      if ( v29 )
        *v29 = _mm_loadu_si128(v27);
      v27 = (const __m128i *)((char *)v27 + 40);
      ++v29;
    }
    while ( (const __m128i *)v25 != v27 );
    v31 = (unsigned __int16 *)v55.m128i_i64[0];
    v30 = v55.m128i_i32[2];
  }
  v32 = *(_QWORD **)(a1 + 8);
  v55.m128i_i32[2] = v28 + v30;
  *((_QWORD *)&v37 + 1) = (unsigned int)(v28 + v30);
  *(_QWORD *)&v37 = v31;
  v33 = sub_33FC220(v32, 156, (__int64)&v50, v47, v48, v21, v37);
  v34 = (unsigned __int16 *)v55.m128i_i64[0];
  *(_QWORD *)a4 = v33;
  *(_DWORD *)(a4 + 8) = v35;
  if ( v34 != &v56 )
    _libc_free((unsigned __int64)v34);
  if ( v52 != v54 )
    _libc_free((unsigned __int64)v52);
  if ( v50 )
    sub_B91220((__int64)&v50, v50);
}
