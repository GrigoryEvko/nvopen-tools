// Function: sub_3791380
// Address: 0x3791380
//
unsigned __int8 *__fastcall sub_3791380(
        __int64 a1,
        unsigned __int64 a2,
        __m128i si128,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        unsigned int a8,
        __int64 a9)
{
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 v13; // rax
  __int64 v14; // r12
  __int64 v15; // r13
  _BYTE *v16; // rdx
  _BYTE *v17; // rdx
  __int64 v18; // rsi
  _QWORD *v19; // r12
  __int64 v20; // r10
  __int128 v21; // kr00_16
  unsigned __int8 *v22; // rax
  __int64 v23; // rsi
  unsigned __int8 *v24; // r12
  unsigned int v25; // edx
  unsigned __int16 v26; // r14
  __int64 *v27; // r15
  __int64 v28; // rdx
  __int64 v29; // r14
  __int64 v30; // rdx
  unsigned __int16 v31; // dx
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rax
  unsigned __int16 v36; // r14
  __int64 v37; // rax
  unsigned int v38; // r14d
  __int16 v39; // ax
  __int64 v40; // rsi
  __int64 v41; // r14
  __int64 v42; // r9
  unsigned __int8 *v43; // r14
  __int64 v44; // rdx
  __int64 v45; // r15
  __int64 v46; // rsi
  _QWORD *v47; // rbx
  __m128i v49; // xmm1
  __int128 v50; // kr10_16
  unsigned __int8 *v51; // rax
  unsigned __int64 v52; // rcx
  unsigned int v53; // edx
  bool v54; // al
  __int64 v55; // rcx
  __int64 v56; // r8
  unsigned __int16 v57; // ax
  __int64 v58; // rdx
  __int64 v59; // r8
  __int64 v60; // rdx
  __int64 v61; // rcx
  __int64 v62; // r8
  unsigned int v63; // r15d
  __int64 v64; // rax
  __int64 v65; // rdx
  __int64 v66; // r8
  __int64 v67; // r9
  __int64 v68; // rax
  __int64 *v69; // r9
  __int64 v70; // rsi
  _QWORD *v71; // r13
  __int128 v72; // rcx
  __int64 v73; // rdx
  __int64 v74; // rcx
  __int64 v75; // r8
  unsigned int v76; // r14d
  int v77; // eax
  __int64 v78; // rdx
  unsigned int v79; // edi
  unsigned int v80; // eax
  __int64 v81; // rsi
  unsigned int v82; // r15d
  __int64 v83; // rdx
  __int64 v84; // r14
  __int64 v85; // r9
  __int64 v86; // rcx
  __int64 v87; // r8
  __int64 v88; // rsi
  __int64 v89; // rdi
  __int64 v90; // rdx
  __int64 v91; // rcx
  __int64 v92; // r8
  unsigned __int16 v93; // ax
  unsigned int v94; // r14d
  int v95; // eax
  __int64 v96; // rsi
  __int64 v97; // r14
  unsigned int v98; // edi
  unsigned __int8 *v99; // rax
  unsigned int v100; // edx
  __int64 v101; // rdx
  __int64 v102; // rdx
  __int128 v103; // [rsp-10h] [rbp-220h]
  __int64 v104; // [rsp+0h] [rbp-210h]
  __int16 v105; // [rsp+2h] [rbp-20Eh]
  unsigned __int8 *v106; // [rsp+8h] [rbp-208h]
  unsigned int v107; // [rsp+8h] [rbp-208h]
  int v108; // [rsp+8h] [rbp-208h]
  unsigned int v109; // [rsp+10h] [rbp-200h]
  __int16 v110; // [rsp+12h] [rbp-1FEh]
  unsigned int v111; // [rsp+20h] [rbp-1F0h]
  __int128 v112; // [rsp+20h] [rbp-1F0h]
  __int64 v113; // [rsp+20h] [rbp-1F0h]
  __int64 v114; // [rsp+20h] [rbp-1F0h]
  __int128 v115; // [rsp+30h] [rbp-1E0h] BYREF
  __m128i v116; // [rsp+40h] [rbp-1D0h] BYREF
  __int64 v117; // [rsp+50h] [rbp-1C0h] BYREF
  __int64 v118; // [rsp+58h] [rbp-1B8h]
  __int64 v119; // [rsp+60h] [rbp-1B0h]
  __int64 v120; // [rsp+68h] [rbp-1A8h]
  __int64 v121; // [rsp+70h] [rbp-1A0h] BYREF
  __int64 v122; // [rsp+78h] [rbp-198h]
  _BYTE *v123; // [rsp+80h] [rbp-190h] BYREF
  __int64 v124; // [rsp+88h] [rbp-188h]
  _BYTE v125[64]; // [rsp+90h] [rbp-180h] BYREF
  __m128i v126; // [rsp+D0h] [rbp-140h] BYREF
  __int16 v127; // [rsp+E0h] [rbp-130h] BYREF
  __int64 v128; // [rsp+E8h] [rbp-128h]

  v123 = v125;
  v124 = 0x400000000LL;
  v11 = *(unsigned int *)(a2 + 64);
  v116.m128i_i64[0] = a5;
  v116.m128i_i64[1] = a6;
  if ( (_DWORD)v11 )
  {
    v12 = 5 * v11;
    v13 = 0;
    v14 = 40;
    v15 = 8 * v12;
    si128 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
    v16 = v125;
    while ( 1 )
    {
      *(__m128i *)&v16[16 * v13] = si128;
      v13 = (unsigned int)(v124 + 1);
      LODWORD(v124) = v124 + 1;
      if ( v15 == v14 )
        break;
      si128 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + v14));
      if ( v13 + 1 > (unsigned __int64)HIDWORD(v124) )
      {
        v115 = (__int128)si128;
        sub_C8D5F0((__int64)&v123, v125, v13 + 1, 0x10u, a6, a7);
        v13 = (unsigned int)v124;
        si128 = _mm_load_si128((const __m128i *)&v115);
      }
      v16 = v123;
      v14 += 40;
    }
    v17 = v123;
  }
  else
  {
    v17 = v125;
    v13 = 0;
  }
  v18 = *(unsigned int *)(a2 + 24);
  v19 = *(_QWORD **)(a1 + 8);
  v20 = *(_QWORD *)(a2 + 80);
  if ( (int)v18 > 239 )
  {
    if ( (unsigned int)(v18 - 242) > 1 )
    {
LABEL_11:
      v126.m128i_i64[0] = *(_QWORD *)(a2 + 80);
      v21 = __PAIR128__(v13, (unsigned __int64)v17);
      if ( v20 )
      {
        *(_QWORD *)&v115 = v17;
        *((_QWORD *)&v115 + 1) = v13;
        sub_B96E90((__int64)&v126, v20, 1);
        v18 = *(unsigned int *)(a2 + 24);
        v21 = v115;
      }
      v126.m128i_i32[2] = *(_DWORD *)(a2 + 72);
      v22 = sub_33FC220(v19, v18, (__int64)&v126, v116.m128i_u32[0], v116.m128i_i64[1], *((__int64 *)&v21 + 1), v21);
      v23 = v126.m128i_i64[0];
      *(_QWORD *)&v115 = v22;
      v24 = v22;
      v111 = v25;
      *((_QWORD *)&v115 + 1) = v25;
      if ( v126.m128i_i64[0] )
        sub_B91220((__int64)&v126, v126.m128i_i64[0]);
      goto LABEL_15;
    }
  }
  else if ( (int)v18 <= 237 && (unsigned int)(v18 - 101) > 0x2F )
  {
    goto LABEL_11;
  }
  v49 = _mm_load_si128(&v116);
  v127 = 1;
  v128 = 0;
  v121 = v20;
  v126 = v49;
  v50 = __PAIR128__(v13, (unsigned __int64)v17);
  if ( v20 )
  {
    *(_QWORD *)&v115 = v17;
    *((_QWORD *)&v115 + 1) = v13;
    sub_B96E90((__int64)&v121, v20, 1);
    LODWORD(v18) = *(_DWORD *)(a2 + 24);
    v50 = v115;
  }
  LODWORD(v122) = *(_DWORD *)(a2 + 72);
  v51 = sub_3411BE0(v19, v18, (__int64)&v121, (unsigned __int16 *)&v126, 2, *((__int64 *)&v50 + 1), v50);
  *(_QWORD *)&v115 = v51;
  v52 = (unsigned __int64)v51;
  v24 = v51;
  v111 = v53;
  *((_QWORD *)&v115 + 1) = v53;
  if ( v121 )
  {
    v106 = v51;
    sub_B91220((__int64)&v121, v121);
    v52 = (unsigned __int64)v106;
  }
  v23 = a2;
  sub_3760E70(a1, a2, 1, v52, 1);
LABEL_15:
  v26 = v116.m128i_i16[0];
  v27 = *(__int64 **)(*(_QWORD *)(a1 + 8) + 64LL);
  if ( v116.m128i_i16[0] )
  {
    if ( (unsigned __int16)(v116.m128i_i16[0] - 17) > 0xD3u )
    {
LABEL_17:
      v28 = v116.m128i_i64[1];
      goto LABEL_18;
    }
    v26 = word_4456580[v116.m128i_u16[0] - 1];
    v28 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v116) )
      goto LABEL_17;
    v26 = sub_3009970((__int64)&v116, v23, v60, v61, v62);
  }
LABEL_18:
  v126.m128i_i16[0] = v26;
  v126.m128i_i64[1] = v28;
  if ( v26 )
  {
    if ( v26 == 1 || (unsigned __int16)(v26 - 504) <= 7u )
      goto LABEL_104;
    v29 = *(_QWORD *)&byte_444C4A0[16 * v26 - 16];
  }
  else
  {
    v119 = sub_3007260((__int64)&v126);
    LODWORD(v29) = v119;
    v120 = v30;
  }
  v31 = a8;
  if ( (_WORD)a8 )
  {
    if ( (unsigned __int16)(a8 - 17) > 0xD3u )
    {
LABEL_22:
      v32 = a9;
      goto LABEL_23;
    }
    v31 = word_4456580[(unsigned __int16)a8 - 1];
    v32 = 0;
  }
  else
  {
    v54 = sub_30070B0((__int64)&a8);
    v31 = 0;
    if ( !v54 )
      goto LABEL_22;
    v57 = sub_3009970((__int64)&a8, v23, 0, v55, v56);
    v59 = v58;
    v31 = v57;
    v32 = v59;
  }
LABEL_23:
  LOWORD(v117) = v31;
  v118 = v32;
  if ( !v31 )
  {
    v33 = sub_3007260((__int64)&v117);
    v121 = v33;
    v122 = v34;
    goto LABEL_25;
  }
  if ( v31 == 1 || (unsigned __int16)(v31 - 504) <= 7u )
LABEL_104:
    BUG();
  v33 = *(_QWORD *)&byte_444C4A0[16 * v31 - 16];
LABEL_25:
  if ( (unsigned int)v29 < (unsigned int)v33 )
  {
    v91 = (unsigned int)sub_3281500(&v116, v23);
    if ( (_WORD)a8 )
    {
      v104 = 0;
      v93 = word_4456580[(unsigned __int16)a8 - 1];
    }
    else
    {
      v108 = v91;
      v93 = sub_3009970((__int64)&a8, v23, v90, v91, v92);
      LODWORD(v91) = v108;
      v104 = v102;
    }
    v107 = v91;
    v94 = v93;
    LOWORD(v95) = sub_2D43050(v93, v91);
    v87 = 0;
    if ( !(_WORD)v95 )
    {
      v95 = sub_3009400(v27, v94, v104, v107, 0);
      v110 = HIWORD(v95);
      v87 = v101;
    }
    v96 = *((_QWORD *)v24 + 10);
    v97 = *(_QWORD *)(a1 + 8);
    *(_QWORD *)&v115 = v24;
    HIWORD(v98) = v110;
    v126.m128i_i64[0] = v96;
    LOWORD(v98) = v95;
    v109 = v98;
    *((_QWORD *)&v115 + 1) = v111 | *((_QWORD *)&v115 + 1) & 0xFFFFFFFF00000000LL;
    if ( v96 )
    {
      v114 = v87;
      sub_B96E90((__int64)&v126, v96, 1);
      v87 = v114;
    }
    v88 = 213;
    v89 = v97;
    v86 = v109;
    v126.m128i_i32[2] = *((_DWORD *)v24 + 18);
  }
  else
  {
    if ( (unsigned int)v29 <= (unsigned int)v33 )
      goto LABEL_27;
    v76 = sub_3281500(&v116, v23);
    if ( (_WORD)a8 )
    {
      LOWORD(v77) = word_4456580[(unsigned __int16)a8 - 1];
      v78 = 0;
    }
    else
    {
      v77 = sub_3009970((__int64)&a8, v23, v73, v74, v75);
      v105 = HIWORD(v77);
    }
    HIWORD(v79) = v105;
    LOWORD(v79) = v77;
    v80 = sub_327FCF0(v27, v79, v78, v76, 0);
    v81 = *((_QWORD *)v24 + 10);
    *(_QWORD *)&v115 = v24;
    v82 = v80;
    v84 = v83;
    v126.m128i_i64[0] = v81;
    v85 = *(_QWORD *)(a1 + 8);
    *((_QWORD *)&v115 + 1) = v111 | *((_QWORD *)&v115 + 1) & 0xFFFFFFFF00000000LL;
    if ( v81 )
    {
      v113 = v85;
      sub_B96E90((__int64)&v126, v81, 1);
      v85 = v113;
    }
    v86 = v82;
    v87 = v84;
    v88 = 216;
    v89 = v85;
    v126.m128i_i32[2] = *((_DWORD *)v24 + 18);
  }
  v99 = sub_33FAF80(v89, v88, (__int64)&v126, v86, v87, v85, si128);
  v23 = v126.m128i_i64[0];
  *(_QWORD *)&v115 = v99;
  v24 = v99;
  v111 = v100;
  *((_QWORD *)&v115 + 1) = v100 | *((_QWORD *)&v115 + 1) & 0xFFFFFFFF00000000LL;
  if ( v126.m128i_i64[0] )
    sub_B91220((__int64)&v126, v126.m128i_i64[0]);
LABEL_27:
  v35 = *((_QWORD *)v24 + 6);
  v36 = *(_WORD *)v35;
  v37 = *(_QWORD *)(v35 + 8);
  v126.m128i_i16[0] = v36;
  v126.m128i_i64[1] = v37;
  if ( v36 )
  {
    if ( (unsigned __int16)(v36 - 176) <= 0x34u )
    {
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    }
    v38 = word_4456340[v36 - 1];
    v39 = a8;
    if ( !(_WORD)a8 )
      goto LABEL_31;
LABEL_57:
    if ( (unsigned __int16)(v39 - 176) > 0x34u )
      goto LABEL_58;
    goto LABEL_75;
  }
  if ( sub_3007100((__int64)&v126) )
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
  v38 = sub_3007130((__int64)&v126, v23);
  v39 = a8;
  if ( (_WORD)a8 )
    goto LABEL_57;
LABEL_31:
  if ( !sub_3007100((__int64)&a8) )
  {
    if ( v38 > (unsigned int)sub_3007130((__int64)&a8, v23) )
    {
LABEL_33:
      v40 = *((_QWORD *)v24 + 10);
      v41 = *(_QWORD *)(a1 + 8);
      *(_QWORD *)&v115 = v24;
      v126.m128i_i64[0] = v40;
      *((_QWORD *)&v115 + 1) = v111 | *((_QWORD *)&v115 + 1) & 0xFFFFFFFF00000000LL;
      if ( v40 )
        sub_B96E90((__int64)&v126, v40, 1);
      v126.m128i_i32[2] = *((_DWORD *)v24 + 18);
      v43 = sub_3400EE0(v41, 0, (__int64)&v126, 0, si128);
      v45 = v44;
      if ( v126.m128i_i64[0] )
        sub_B91220((__int64)&v126, v126.m128i_i64[0]);
      v46 = *((_QWORD *)v24 + 10);
      v47 = *(_QWORD **)(a1 + 8);
      v126.m128i_i64[0] = v46;
      if ( v46 )
        sub_B96E90((__int64)&v126, v46, 1);
      *((_QWORD *)&v103 + 1) = v45;
      *(_QWORD *)&v103 = v43;
      v126.m128i_i32[2] = *((_DWORD *)v24 + 18);
      v24 = sub_3406EB0(v47, 0xA1u, (__int64)&v126, a8, a9, v42, v115, v103);
      if ( v126.m128i_i64[0] )
        sub_B91220((__int64)&v126, v126.m128i_i64[0]);
      goto LABEL_41;
    }
    goto LABEL_79;
  }
LABEL_75:
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( !(_WORD)a8 )
  {
    if ( (unsigned int)sub_3007130((__int64)&a8, v23) < v38 )
      goto LABEL_33;
    goto LABEL_77;
  }
  if ( (unsigned __int16)(a8 - 176) <= 0x34u )
    sub_CA17B0(
      "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT::"
      "getVectorElementCount() instead");
LABEL_58:
  if ( v38 > word_4456340[(unsigned __int16)a8 - 1] )
    goto LABEL_33;
  if ( (_WORD)a8 )
  {
    if ( (unsigned __int16)(a8 - 176) > 0x34u )
      goto LABEL_61;
    goto LABEL_78;
  }
LABEL_77:
  if ( !sub_3007100((__int64)&a8) )
    goto LABEL_79;
LABEL_78:
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( !(_WORD)a8 )
  {
LABEL_79:
    if ( v38 >= (unsigned int)sub_3007130((__int64)&a8, v23) )
      goto LABEL_41;
    goto LABEL_62;
  }
  if ( (unsigned __int16)(a8 - 176) <= 0x34u )
    sub_CA17B0(
      "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT::"
      "getVectorElementCount() instead");
LABEL_61:
  if ( v38 >= word_4456340[(unsigned __int16)a8 - 1] )
    goto LABEL_41;
LABEL_62:
  v63 = (unsigned int)sub_3281500(&a8, v23) / v38;
  v64 = sub_3288990(*(_QWORD *)(a1 + 8), **((unsigned __int16 **)v24 + 6), *(_QWORD *)(*((_QWORD *)v24 + 6) + 8LL));
  v126.m128i_i64[0] = (__int64)&v127;
  v126.m128i_i64[1] = 0x1000000000LL;
  sub_32982C0((__int64)&v126, v63, v64, v65, v66, v67);
  v68 = v126.m128i_i64[0];
  v69 = &v117;
  *(_QWORD *)v126.m128i_i64[0] = v24;
  *(_DWORD *)(v68 + 8) = v111;
  v70 = *((_QWORD *)v24 + 10);
  v71 = *(_QWORD **)(a1 + 8);
  *(_QWORD *)&v72 = v126.m128i_i64[0];
  v117 = v70;
  *((_QWORD *)&v72 + 1) = v126.m128i_u32[2];
  if ( v70 )
  {
    *(_QWORD *)&v112 = v126.m128i_i64[0];
    *((_QWORD *)&v112 + 1) = v126.m128i_u32[2];
    *(_QWORD *)&v115 = &v117;
    sub_B96E90((__int64)&v117, v70, 1);
    v72 = v112;
    v69 = (__int64 *)v115;
  }
  LODWORD(v118) = *((_DWORD *)v24 + 18);
  *(_QWORD *)&v115 = v69;
  v24 = sub_33FC220(v71, 159, (__int64)v69, a8, a9, (__int64)v69, v72);
  if ( v117 )
    sub_B91220(v115, v117);
  if ( (__int16 *)v126.m128i_i64[0] != &v127 )
    _libc_free(v126.m128i_u64[0]);
LABEL_41:
  if ( v123 != v125 )
    _libc_free((unsigned __int64)v123);
  return v24;
}
