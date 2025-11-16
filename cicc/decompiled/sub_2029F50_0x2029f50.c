// Function: sub_2029F50
// Address: 0x2029f50
//
__int64 *__fastcall sub_2029F50(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // rsi
  __m128i v5; // xmm1
  __m128i v6; // xmm0
  __int64 v7; // rcx
  __m128i v8; // xmm2
  __int64 v9; // rax
  const void **v10; // rax
  int v11; // eax
  __int64 *v12; // r13
  __int64 v13; // rax
  char v14; // dl
  __int64 v15; // rax
  __m128i v16; // xmm4
  __int64 *v17; // r15
  __int64 v18; // rax
  char v19; // dl
  __int64 v20; // rax
  __m128i v21; // xmm6
  __int64 *v22; // r15
  __int64 v23; // rax
  char v24; // dl
  __int64 v25; // rax
  __m128 v26; // xmm0
  __int64 *v27; // rax
  unsigned __int64 v28; // rdx
  unsigned __int64 v29; // r15
  __int64 v30; // r14
  __int128 v31; // rax
  __int64 *v32; // r14
  unsigned __int64 v34; // [rsp+10h] [rbp-160h]
  __int16 *v35; // [rsp+18h] [rbp-158h]
  const void **v36; // [rsp+20h] [rbp-150h]
  unsigned __int64 v37; // [rsp+28h] [rbp-148h]
  const void **v38; // [rsp+30h] [rbp-140h]
  unsigned __int64 v39; // [rsp+38h] [rbp-138h]
  __int64 v40; // [rsp+40h] [rbp-130h]
  __int64 v41; // [rsp+48h] [rbp-128h]
  __int64 v42; // [rsp+50h] [rbp-120h]
  __int64 v43; // [rsp+58h] [rbp-118h]
  __int128 v44; // [rsp+60h] [rbp-110h]
  __int128 v45; // [rsp+70h] [rbp-100h]
  __m128i v46; // [rsp+80h] [rbp-F0h] BYREF
  __m128i v47; // [rsp+90h] [rbp-E0h] BYREF
  __m128i v48; // [rsp+A0h] [rbp-D0h] BYREF
  __int64 v49; // [rsp+B0h] [rbp-C0h] BYREF
  const void **v50; // [rsp+B8h] [rbp-B8h]
  __int64 v51; // [rsp+C0h] [rbp-B0h] BYREF
  int v52; // [rsp+C8h] [rbp-A8h]
  __int64 v53; // [rsp+D0h] [rbp-A0h] BYREF
  int v54; // [rsp+D8h] [rbp-98h]
  __int64 v55; // [rsp+E0h] [rbp-90h] BYREF
  int v56; // [rsp+E8h] [rbp-88h]
  __m128i v57; // [rsp+F0h] [rbp-80h] BYREF
  __m128i v58; // [rsp+100h] [rbp-70h] BYREF
  __int64 v59; // [rsp+110h] [rbp-60h] BYREF
  __int64 v60; // [rsp+118h] [rbp-58h]
  __m128i v61; // [rsp+120h] [rbp-50h] BYREF
  __m128i v62; // [rsp+130h] [rbp-40h] BYREF

  v3 = *(_QWORD *)(a2 + 32);
  v4 = *(_QWORD *)(a2 + 72);
  v5 = _mm_loadu_si128((const __m128i *)(v3 + 40));
  v6 = _mm_loadu_si128((const __m128i *)v3);
  v7 = *(_QWORD *)(v3 + 40);
  v8 = _mm_loadu_si128((const __m128i *)(v3 + 80));
  v47 = v5;
  v46 = v6;
  v9 = *(_QWORD *)(v7 + 40) + 16LL * v5.m128i_u32[2];
  v48 = v8;
  LOBYTE(v7) = *(_BYTE *)v9;
  v10 = *(const void ***)(v9 + 8);
  v51 = v4;
  LOBYTE(v49) = v7;
  v50 = v10;
  if ( v4 )
  {
    sub_1623A60((__int64)&v51, v4, 2);
    v3 = *(_QWORD *)(a2 + 32);
  }
  v11 = *(_DWORD *)(a2 + 64);
  v54 = 0;
  v56 = 0;
  v52 = v11;
  v53 = 0;
  v55 = 0;
  sub_2017DE0(a1, *(_QWORD *)v3, *(_QWORD *)(v3 + 8), &v53, &v55);
  sub_1D19A30((__int64)&v61, *(_QWORD *)(a1 + 8), &v49);
  v57.m128i_i8[0] = 0;
  v12 = *(__int64 **)(a1 + 8);
  v58.m128i_i8[0] = 0;
  v39 = v61.m128i_i64[0];
  v36 = (const void **)v62.m128i_i64[1];
  v57.m128i_i64[1] = 0;
  v58.m128i_i64[1] = 0;
  v13 = *(_QWORD *)(v47.m128i_i64[0] + 40) + 16LL * v47.m128i_u32[2];
  v14 = *(_BYTE *)v13;
  v15 = *(_QWORD *)(v13 + 8);
  v38 = (const void **)v61.m128i_i64[1];
  v37 = v62.m128i_i64[0];
  LOBYTE(v59) = v14;
  v60 = v15;
  sub_1D19A30((__int64)&v61, (__int64)v12, &v59);
  v16 = _mm_loadu_si128(&v62);
  v57 = _mm_loadu_si128(&v61);
  v58 = v16;
  sub_1D40600(
    (__int64)&v61,
    v12,
    (__int64)&v47,
    (__int64)&v51,
    (const void ***)&v57,
    (const void ***)&v58,
    v6,
    *(double *)v5.m128i_i64,
    v8);
  v57.m128i_i8[0] = 0;
  v58.m128i_i8[0] = 0;
  v57.m128i_i64[1] = 0;
  *((_QWORD *)&v45 + 1) = v61.m128i_u32[2];
  v58.m128i_i64[1] = 0;
  v17 = *(__int64 **)(a1 + 8);
  *(_QWORD *)&v45 = v61.m128i_i64[0];
  *((_QWORD *)&v44 + 1) = v62.m128i_u32[2];
  *(_QWORD *)&v44 = v62.m128i_i64[0];
  v18 = *(_QWORD *)(v48.m128i_i64[0] + 40) + 16LL * v48.m128i_u32[2];
  v19 = *(_BYTE *)v18;
  v20 = *(_QWORD *)(v18 + 8);
  LOBYTE(v59) = v19;
  v60 = v20;
  sub_1D19A30((__int64)&v61, (__int64)v17, &v59);
  v21 = _mm_loadu_si128(&v62);
  v57 = _mm_loadu_si128(&v61);
  v58 = v21;
  sub_1D40600(
    (__int64)&v61,
    v17,
    (__int64)&v48,
    (__int64)&v51,
    (const void ***)&v57,
    (const void ***)&v58,
    v6,
    *(double *)v5.m128i_i64,
    v8);
  v57.m128i_i8[0] = 0;
  v58.m128i_i8[0] = 0;
  v22 = *(__int64 **)(a1 + 8);
  v57.m128i_i64[1] = 0;
  v43 = v61.m128i_u32[2];
  v58.m128i_i64[1] = 0;
  v42 = v61.m128i_i64[0];
  v41 = v62.m128i_u32[2];
  v40 = v62.m128i_i64[0];
  v23 = *(_QWORD *)(v46.m128i_i64[0] + 40) + 16LL * v46.m128i_u32[2];
  v24 = *(_BYTE *)v23;
  v25 = *(_QWORD *)(v23 + 8);
  LOBYTE(v59) = v24;
  v60 = v25;
  sub_1D19A30((__int64)&v61, (__int64)v22, &v59);
  v26 = (__m128)_mm_loadu_si128(&v62);
  v57 = _mm_loadu_si128(&v61);
  v58 = (__m128i)v26;
  sub_1D40600(
    (__int64)&v61,
    v22,
    (__int64)&v46,
    (__int64)&v51,
    (const void ***)&v57,
    (const void ***)&v58,
    (__m128i)v26,
    *(double *)v5.m128i_i64,
    v8);
  v34 = v62.m128i_i64[0];
  v35 = (__int16 *)v62.m128i_u32[2];
  v27 = sub_1D3A900(
          *(__int64 **)(a1 + 8),
          0x87u,
          (__int64)&v51,
          v39,
          v38,
          0,
          v26,
          *(double *)v5.m128i_i64,
          v8,
          v61.m128i_u64[0],
          (__int16 *)v61.m128i_u32[2],
          v45,
          v42,
          v43);
  v29 = v28;
  v30 = (__int64)v27;
  *(_QWORD *)&v31 = sub_1D3A900(
                      *(__int64 **)(a1 + 8),
                      0x87u,
                      (__int64)&v51,
                      v37,
                      v36,
                      0,
                      v26,
                      *(double *)v5.m128i_i64,
                      v8,
                      v34,
                      v35,
                      v44,
                      v40,
                      v41);
  v32 = sub_1D332F0(
          *(__int64 **)(a1 + 8),
          107,
          (__int64)&v51,
          (unsigned int)v49,
          v50,
          0,
          *(double *)v26.m128_u64,
          *(double *)v5.m128i_i64,
          v8,
          v30,
          v29,
          v31);
  if ( v51 )
    sub_161E7C0((__int64)&v51, v51);
  return v32;
}
