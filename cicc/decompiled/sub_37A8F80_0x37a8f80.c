// Function: sub_37A8F80
// Address: 0x37a8f80
//
__m128i *__fastcall sub_37A8F80(__int64 *a1, __int64 a2, int a3)
{
  unsigned int v3; // r14d
  const __m128i *v6; // rax
  __m128i v7; // xmm0
  __int64 v8; // rdx
  unsigned __int64 v9; // r13
  __int16 v10; // si
  __int64 v11; // rdx
  __int64 v12; // r12
  __int64 v13; // rsi
  __int64 v14; // rdx
  __int64 (__fastcall *v15)(__int64, __int64, unsigned int, __int64); // rax
  unsigned __int8 *v16; // rax
  __int64 v17; // r8
  unsigned __int16 *v18; // r12
  unsigned int v19; // edx
  unsigned int v20; // edx
  bool v21; // al
  unsigned int v22; // edx
  unsigned int v23; // eax
  __int64 v24; // rdx
  __int64 v25; // rcx
  unsigned __int16 v26; // ax
  unsigned int v27; // r12d
  int v28; // eax
  __int64 v29; // r8
  __int64 v30; // r13
  unsigned int v31; // edx
  __int64 v32; // r12
  __int64 v33; // r8
  __m128i *v34; // r12
  __int64 v36; // rdx
  __int64 v37; // rdx
  __int64 v38; // rsi
  __int64 v39; // r8
  unsigned int v40; // edx
  __int64 v41; // rdx
  unsigned __int16 v42; // ax
  __int64 v43; // rdx
  unsigned __int16 *v44; // rdx
  __int64 v45; // rcx
  unsigned __int16 v46; // ax
  unsigned int v47; // r14d
  __int64 v48; // rax
  __int64 v49; // r8
  __int64 v50; // rcx
  unsigned int v51; // edx
  __int64 v52; // rdx
  __int64 v53; // rdx
  __int64 v54; // rdx
  __int64 v55; // [rsp+8h] [rbp-C8h]
  __int64 *v56; // [rsp+10h] [rbp-C0h]
  __int64 v57; // [rsp+10h] [rbp-C0h]
  __int64 v58; // [rsp+18h] [rbp-B8h]
  unsigned int v59; // [rsp+18h] [rbp-B8h]
  unsigned int v60; // [rsp+18h] [rbp-B8h]
  unsigned int v61; // [rsp+18h] [rbp-B8h]
  int v62; // [rsp+18h] [rbp-B8h]
  unsigned __int8 *v63; // [rsp+20h] [rbp-B0h]
  __int64 *v64; // [rsp+20h] [rbp-B0h]
  int v65; // [rsp+2Ch] [rbp-A4h]
  unsigned int v66; // [rsp+2Ch] [rbp-A4h]
  unsigned int v67; // [rsp+2Ch] [rbp-A4h]
  int v68; // [rsp+2Ch] [rbp-A4h]
  __int64 v69; // [rsp+38h] [rbp-98h]
  __int64 v70; // [rsp+50h] [rbp-80h] BYREF
  __int64 v71; // [rsp+58h] [rbp-78h]
  __int64 v72; // [rsp+60h] [rbp-70h] BYREF
  int v73; // [rsp+68h] [rbp-68h]
  unsigned int v74; // [rsp+70h] [rbp-60h] BYREF
  __int64 v75; // [rsp+78h] [rbp-58h]
  unsigned __int16 v76; // [rsp+80h] [rbp-50h] BYREF
  __int64 v77; // [rsp+88h] [rbp-48h]
  __int64 v78; // [rsp+90h] [rbp-40h]

  v6 = *(const __m128i **)(a2 + 40);
  v7 = _mm_loadu_si128(v6 + 10);
  v8 = *(_QWORD *)(v6[10].m128i_i64[0] + 48) + 16LL * v6[10].m128i_u32[2];
  v9 = v6[2].m128i_u64[1];
  v10 = *(_WORD *)v8;
  v11 = *(_QWORD *)(v8 + 8);
  v69 = v6[3].m128i_i64[0];
  v12 = v6[3].m128i_u32[0];
  LOWORD(v70) = v10;
  v13 = *(_QWORD *)(a2 + 80);
  v71 = v11;
  v72 = v13;
  if ( v13 )
  {
    v65 = a3;
    sub_B96E90((__int64)&v72, v13, 1);
    a3 = v65;
  }
  v73 = *(_DWORD *)(a2 + 72);
  if ( a3 != 1 )
  {
    v14 = a1[1];
    v15 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
    if ( v15 == sub_2D56A50 )
    {
      sub_2FE6CC0((__int64)&v76, *a1, *(_QWORD *)(v14 + 64), v70, v71);
      LOWORD(v74) = v77;
      v75 = v78;
    }
    else
    {
      v74 = v15(*a1, *(_QWORD *)(v14 + 64), v70, v71);
      v75 = v52;
    }
    v16 = sub_3790540((__int64)a1, v7.m128i_i64[0], v7.m128i_i64[1], v74, v75, 1, v7);
    v18 = (unsigned __int16 *)(*(_QWORD *)(v9 + 48) + 16 * v12);
    v63 = v16;
    v66 = v19;
    v20 = *v18;
    v77 = *((_QWORD *)v18 + 1);
    v76 = v20;
    if ( (_WORD)v74 )
    {
      if ( (unsigned __int16)(v74 - 176) > 0x34u )
        goto LABEL_17;
    }
    else
    {
      v59 = v20;
      v21 = sub_3007100((__int64)&v74);
      v22 = v59;
      if ( !v21 )
        goto LABEL_8;
    }
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    if ( !(_WORD)v74 )
    {
      v22 = v76;
LABEL_8:
      v60 = v22;
      v23 = sub_3007130((__int64)&v74, v7.m128i_i64[0]);
      v24 = v60;
      v25 = v23;
      if ( (_WORD)v60 )
      {
LABEL_9:
        v55 = 0;
        v26 = word_4456580[(unsigned __int16)v24 - 1];
LABEL_10:
        v61 = v25;
        v27 = v26;
        v56 = *(__int64 **)(a1[1] + 64);
        LOWORD(v28) = sub_2D43050(v26, v25);
        v29 = 0;
        if ( !(_WORD)v28 )
        {
          v28 = sub_3009400(v56, v27, v55, v61, 0);
          HIWORD(v3) = HIWORD(v28);
          v29 = v37;
        }
        LOWORD(v3) = v28;
        v30 = (__int64)sub_3790540((__int64)a1, v9, v69, v3, v29, 0, v7);
        v32 = v31;
        goto LABEL_13;
      }
LABEL_18:
      v62 = v25;
      v26 = sub_3009970((__int64)&v76, v7.m128i_i64[0], v24, v25, v17);
      LODWORD(v25) = v62;
      v55 = v36;
      goto LABEL_10;
    }
    if ( (unsigned __int16)(v74 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
LABEL_17:
    v25 = word_4456340[(unsigned __int16)v74 - 1];
    v24 = v76;
    if ( v76 )
      goto LABEL_9;
    goto LABEL_18;
  }
  v38 = v9;
  v30 = sub_379AB60((__int64)a1, v9, v69);
  v32 = v40;
  v41 = *(_QWORD *)(v30 + 48) + 16LL * v40;
  v42 = *(_WORD *)v41;
  v43 = *(_QWORD *)(v41 + 8);
  v76 = v42;
  v77 = v43;
  if ( v42 )
  {
    if ( (unsigned __int16)(v42 - 176) > 0x34u )
      goto LABEL_30;
    goto LABEL_34;
  }
  if ( sub_3007100((__int64)&v76) )
  {
LABEL_34:
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    if ( !v76 )
      goto LABEL_23;
    if ( (unsigned __int16)(v76 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
LABEL_30:
    v44 = word_4456340;
    v45 = word_4456340[v76 - 1];
    goto LABEL_24;
  }
LABEL_23:
  v45 = (unsigned int)sub_3007130((__int64)&v76, v38);
LABEL_24:
  if ( (_WORD)v70 )
  {
    v57 = 0;
    v46 = word_4456580[(unsigned __int16)v70 - 1];
  }
  else
  {
    v68 = v45;
    v46 = sub_3009970((__int64)&v70, v38, (__int64)v44, v45, v39);
    LODWORD(v45) = v68;
    v57 = v54;
  }
  v67 = v45;
  v47 = v46;
  v64 = *(__int64 **)(a1[1] + 64);
  LOWORD(v48) = sub_2D43050(v46, v45);
  v49 = 0;
  if ( !(_WORD)v48 )
  {
    v48 = sub_3009400(v64, v47, v57, v67, 0);
    v58 = v48;
    v49 = v53;
  }
  v50 = v58;
  LOWORD(v50) = v48;
  v63 = sub_3790540((__int64)a1, v7.m128i_i64[0], v7.m128i_i64[1], v50, v49, 1, v7);
  v66 = v51;
LABEL_13:
  v33 = *(_QWORD *)(a2 + 40);
  v34 = sub_33F65D0(
          (__int64 *)a1[1],
          *(_QWORD *)v33,
          *(_QWORD *)(v33 + 8),
          (__int64)&v72,
          v30,
          v32 | v69 & 0xFFFFFFFF00000000LL,
          *(_QWORD *)(v33 + 80),
          *(_QWORD *)(v33 + 88),
          *(_OWORD *)(v33 + 120),
          __PAIR128__(v66 | v7.m128i_i64[1] & 0xFFFFFFFF00000000LL, (unsigned __int64)v63),
          *(unsigned __int16 *)(a2 + 96),
          *(_QWORD *)(a2 + 104),
          *(const __m128i **)(a2 + 112),
          (*(_WORD *)(a2 + 32) >> 7) & 7,
          0,
          (*(_BYTE *)(a2 + 33) & 8) != 0);
  if ( v72 )
    sub_B91220((__int64)&v72, v72);
  return v34;
}
