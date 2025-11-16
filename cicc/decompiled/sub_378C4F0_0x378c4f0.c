// Function: sub_378C4F0
// Address: 0x378c4f0
//
unsigned __int8 *__fastcall sub_378C4F0(__int64 *a1, __int64 a2)
{
  __int64 v4; // rsi
  int v5; // eax
  __int16 *v6; // rax
  __int64 v7; // rsi
  __int16 v8; // dx
  __int64 v9; // rax
  __m128i v10; // xmm0
  __int64 v11; // rax
  __m128i v12; // xmm1
  __int64 v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rdx
  _QWORD *v19; // r13
  __int16 v20; // ax
  __int64 v21; // rdx
  unsigned int v22; // edx
  unsigned __int8 *v23; // r12
  _QWORD *v25; // r15
  __int64 v26; // rcx
  __int64 v27; // rdi
  __int64 v28; // rdx
  __int64 v29; // rdx
  __m128i v30; // xmm3
  unsigned __int8 *v31; // rax
  __int64 v32; // rdx
  __int64 v33; // r15
  unsigned __int8 *v34; // r14
  __int64 v35; // r9
  __int128 v36; // rax
  __int128 v37; // [rsp-20h] [rbp-110h]
  __int128 v38; // [rsp-10h] [rbp-100h]
  __int64 v39; // [rsp+8h] [rbp-E8h]
  __int128 v40; // [rsp+10h] [rbp-E0h]
  __int128 v41; // [rsp+20h] [rbp-D0h]
  __int128 v42; // [rsp+30h] [rbp-C0h]
  __int64 v43; // [rsp+40h] [rbp-B0h] BYREF
  int v44; // [rsp+48h] [rbp-A8h]
  __m128i v45; // [rsp+50h] [rbp-A0h] BYREF
  __m128i v46; // [rsp+60h] [rbp-90h] BYREF
  __m128i v47; // [rsp+70h] [rbp-80h] BYREF
  __m128i v48; // [rsp+80h] [rbp-70h] BYREF
  __int64 v49; // [rsp+90h] [rbp-60h] BYREF
  __int64 v50; // [rsp+98h] [rbp-58h]
  __m128i v51; // [rsp+A0h] [rbp-50h] BYREF
  __m128i v52; // [rsp+B0h] [rbp-40h] BYREF

  v4 = *(_QWORD *)(a2 + 80);
  v43 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v43, v4, 1);
  v5 = *(_DWORD *)(a2 + 72);
  v45.m128i_i16[0] = 0;
  v44 = v5;
  v6 = *(__int16 **)(a2 + 48);
  v45.m128i_i64[1] = 0;
  v46.m128i_i16[0] = 0;
  v7 = a1[1];
  v46.m128i_i64[1] = 0;
  v8 = *v6;
  v9 = *((_QWORD *)v6 + 1);
  LOWORD(v49) = v8;
  v50 = v9;
  sub_33D0340((__int64)&v51, v7, &v49);
  v10 = _mm_loadu_si128(&v51);
  v11 = a1[1];
  v12 = _mm_loadu_si128(&v52);
  v13 = *a1;
  v45 = v10;
  v14 = *(_QWORD *)(v11 + 64);
  v46 = v12;
  sub_2FE6CC0((__int64)&v51, v13, v14, v10.m128i_u16[0], v10.m128i_i64[1]);
  if ( v51.m128i_i8[0]
    || (v13 = *a1,
        sub_2FE6CC0((__int64)&v51, *a1, *(_QWORD *)(a1[1] + 64), v46.m128i_u16[0], v46.m128i_i64[1]),
        v51.m128i_i8[0]) )
  {
    v18 = *(_QWORD *)(a2 + 48);
    v19 = (_QWORD *)a1[1];
    v20 = *(_WORD *)v18;
    v21 = *(_QWORD *)(v18 + 8);
    v51.m128i_i16[0] = v20;
    v51.m128i_i64[1] = v21;
    if ( v20 )
    {
      if ( (unsigned __int16)(v20 - 176) > 0x34u )
        goto LABEL_13;
    }
    else if ( !sub_3007100((__int64)&v51) )
    {
LABEL_7:
      v22 = sub_3007130((__int64)&v51, v13);
LABEL_8:
      v23 = sub_3412A00(v19, a2, v22, v15, v16, v17, v10);
      goto LABEL_9;
    }
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    if ( !v51.m128i_i16[0] )
      goto LABEL_7;
    if ( (unsigned __int16)(v51.m128i_i16[0] - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
LABEL_13:
    v22 = word_4456340[v51.m128i_u16[0] - 1];
    goto LABEL_8;
  }
  sub_3408290(
    (__int64)&v51,
    (_QWORD *)a1[1],
    *(__int128 **)(a2 + 40),
    (__int64)&v43,
    (unsigned int *)&v45,
    (unsigned int *)&v46,
    v10);
  v25 = (_QWORD *)a1[1];
  *(_QWORD *)&v42 = v51.m128i_i64[0];
  *(_QWORD *)&v41 = v52.m128i_i64[0];
  v47.m128i_i64[1] = 0;
  v48.m128i_i64[1] = 0;
  *((_QWORD *)&v42 + 1) = v51.m128i_u32[2];
  v26 = *(_QWORD *)(a2 + 40);
  v47.m128i_i16[0] = 0;
  *((_QWORD *)&v41 + 1) = v52.m128i_u32[2];
  v27 = *(_QWORD *)(v26 + 40);
  v48.m128i_i16[0] = 0;
  v39 = v26;
  v28 = *(_QWORD *)(v27 + 48) + 16LL * *(unsigned int *)(v26 + 48);
  LOWORD(v27) = *(_WORD *)v28;
  v29 = *(_QWORD *)(v28 + 8);
  LOWORD(v49) = v27;
  v50 = v29;
  sub_33D0340((__int64)&v51, (__int64)v25, &v49);
  v30 = _mm_loadu_si128(&v52);
  v47 = _mm_loadu_si128(&v51);
  v48 = v30;
  sub_3408290(
    (__int64)&v51,
    v25,
    (__int128 *)(v39 + 40),
    (__int64)&v43,
    (unsigned int *)&v47,
    (unsigned int *)&v48,
    v10);
  *(_QWORD *)&v40 = v52.m128i_i64[0];
  *((_QWORD *)&v38 + 1) = v51.m128i_u32[2];
  *(_QWORD *)&v38 = v51.m128i_i64[0];
  *((_QWORD *)&v40 + 1) = v52.m128i_u32[2];
  v31 = sub_3406EB0(
          (_QWORD *)a1[1],
          *(_DWORD *)(a2 + 24),
          (__int64)&v43,
          v45.m128i_u32[0],
          v45.m128i_i64[1],
          0,
          v42,
          v38);
  v33 = v32;
  v34 = v31;
  *(_QWORD *)&v36 = sub_3406EB0(
                      (_QWORD *)a1[1],
                      *(_DWORD *)(a2 + 24),
                      (__int64)&v43,
                      v46.m128i_u32[0],
                      v46.m128i_i64[1],
                      v35,
                      v41,
                      v40);
  *((_QWORD *)&v37 + 1) = v33;
  *(_QWORD *)&v37 = v34;
  v23 = sub_3406EB0(
          (_QWORD *)a1[1],
          0x9Fu,
          (__int64)&v43,
          **(unsigned __int16 **)(a2 + 48),
          *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
          a1[1],
          v37,
          v36);
LABEL_9:
  if ( v43 )
    sub_B91220((__int64)&v43, v43);
  return v23;
}
