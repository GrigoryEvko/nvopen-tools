// Function: sub_345B810
// Address: 0x345b810
//
unsigned __int8 *__fastcall sub_345B810(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rsi
  __int16 *v6; // rax
  __int16 v7; // dx
  __int64 v8; // rax
  int v9; // ebx
  __m128i v10; // xmm0
  __int64 v11; // r12
  __int64 v12; // r13
  __int64 v13; // rsi
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  unsigned __int16 v17; // r15
  __int64 v18; // rdx
  unsigned __int64 v19; // r13
  __int64 v20; // rcx
  __int64 v21; // r8
  int v22; // r9d
  __int64 v23; // rdx
  __int64 v24; // r9
  __int128 v25; // rax
  __int64 v26; // r9
  unsigned __int8 *v27; // rax
  __int64 v28; // rdx
  __int64 v29; // r13
  unsigned __int8 *v30; // r12
  __int128 v31; // rax
  __int64 v32; // r9
  __int128 v33; // rax
  __int64 v34; // r9
  unsigned __int8 *v35; // rax
  unsigned __int8 *v36; // r12
  bool v38; // r15
  unsigned __int8 *v39; // r12
  unsigned __int64 v40; // rdx
  unsigned __int64 v41; // r13
  __int128 v42; // rax
  __int64 v43; // r9
  unsigned __int64 v44; // rax
  int v45; // eax
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // rdx
  __int64 v49; // rsi
  __int64 v50; // rdx
  unsigned int v51; // esi
  __int64 v52; // rax
  unsigned int v53; // r10d
  __int64 v54; // rdx
  __int64 v55; // r11
  __int128 v56; // rax
  __int64 v57; // r9
  __int64 v58; // rdx
  unsigned int v59; // edx
  unsigned __int64 v60; // rax
  int v61; // eax
  __int64 (*v62)(); // rax
  __m128i v63; // xmm1
  unsigned int *v64; // rax
  __int64 v65; // rdx
  __int64 v66; // r9
  __int128 v67; // rax
  __int64 v68; // r9
  __m128i v69; // rax
  int v70; // r9d
  unsigned __int8 *v71; // rax
  unsigned __int16 v72; // bx
  unsigned __int8 *v73; // r12
  __int64 v74; // rdx
  __int64 v75; // r13
  __int64 v76; // rax
  __int64 v77; // rax
  __int64 v78; // rdx
  __int64 v79; // rsi
  __int64 v80; // rdx
  __int64 v81; // rax
  __int64 v82; // rsi
  __int128 v83; // rax
  __int64 v84; // r9
  __int128 v85; // rax
  __int64 v86; // r9
  __int64 v87; // rdx
  __int64 v88; // rcx
  __int64 v89; // r8
  __int64 v90; // rdx
  __int64 v91; // rdx
  unsigned __int64 v92; // r13
  int v93; // r9d
  __int64 v94; // rdx
  __int64 v95; // r9
  unsigned __int8 *v96; // rax
  __int64 v97; // r11
  unsigned int v98; // r10d
  unsigned __int8 *v99; // r12
  unsigned __int64 v100; // rdx
  unsigned __int64 v101; // r13
  __int128 v102; // rax
  __int64 v103; // r9
  unsigned __int8 *v104; // rax
  __int64 v105; // rdx
  __int128 v106; // rax
  __int64 v107; // r9
  __int64 v108; // rdx
  int v109; // r9d
  __int128 v110; // [rsp-30h] [rbp-190h]
  __int128 v111; // [rsp-30h] [rbp-190h]
  __int128 v112; // [rsp-20h] [rbp-180h]
  __int128 v113; // [rsp-20h] [rbp-180h]
  __int128 v114; // [rsp-20h] [rbp-180h]
  __int128 v115; // [rsp-20h] [rbp-180h]
  __int128 v116; // [rsp-20h] [rbp-180h]
  __int128 v117; // [rsp-20h] [rbp-180h]
  __int128 v118; // [rsp-20h] [rbp-180h]
  __int128 v119; // [rsp-20h] [rbp-180h]
  __int128 v120; // [rsp-10h] [rbp-170h]
  __int128 v121; // [rsp-10h] [rbp-170h]
  unsigned int v122; // [rsp+0h] [rbp-160h]
  __int64 v123; // [rsp+8h] [rbp-158h]
  unsigned int v124; // [rsp+14h] [rbp-14Ch]
  __int128 v125; // [rsp+20h] [rbp-140h]
  unsigned int v126; // [rsp+30h] [rbp-130h]
  unsigned int v127; // [rsp+38h] [rbp-128h]
  unsigned int v128; // [rsp+38h] [rbp-128h]
  unsigned int v129; // [rsp+38h] [rbp-128h]
  __m128i v130; // [rsp+40h] [rbp-120h] BYREF
  unsigned __int8 *v131; // [rsp+50h] [rbp-110h]
  __int64 v132; // [rsp+58h] [rbp-108h]
  unsigned __int8 *v133; // [rsp+60h] [rbp-100h]
  __int64 v134; // [rsp+68h] [rbp-F8h]
  unsigned __int8 *v135; // [rsp+70h] [rbp-F0h]
  __int64 v136; // [rsp+78h] [rbp-E8h]
  unsigned __int8 *v137; // [rsp+80h] [rbp-E0h]
  __int64 v138; // [rsp+88h] [rbp-D8h]
  __int64 v139; // [rsp+90h] [rbp-D0h]
  __int64 v140; // [rsp+98h] [rbp-C8h]
  unsigned __int8 *v141; // [rsp+A0h] [rbp-C0h]
  __int64 v142; // [rsp+A8h] [rbp-B8h]
  unsigned __int8 *v143; // [rsp+B0h] [rbp-B0h]
  __int64 v144; // [rsp+B8h] [rbp-A8h]
  __int64 v145; // [rsp+C0h] [rbp-A0h] BYREF
  int v146; // [rsp+C8h] [rbp-98h]
  unsigned int v147; // [rsp+D0h] [rbp-90h] BYREF
  __int64 v148; // [rsp+D8h] [rbp-88h]
  __int64 v149; // [rsp+E0h] [rbp-80h]
  __int64 v150; // [rsp+E8h] [rbp-78h]
  unsigned __int64 v151; // [rsp+F0h] [rbp-70h] BYREF
  __int64 v152; // [rsp+F8h] [rbp-68h]
  unsigned __int64 v153; // [rsp+100h] [rbp-60h]
  unsigned int v154; // [rsp+108h] [rbp-58h]
  __m128i v155; // [rsp+110h] [rbp-50h] BYREF
  unsigned __int64 v156; // [rsp+120h] [rbp-40h]
  __int64 v157; // [rsp+128h] [rbp-38h]

  v5 = *(_QWORD *)(a2 + 80);
  v145 = v5;
  if ( v5 )
    sub_B96E90((__int64)&v145, v5, 1);
  v146 = *(_DWORD *)(a2 + 72);
  v6 = *(__int16 **)(a2 + 48);
  v7 = *v6;
  v148 = *((_QWORD *)v6 + 1);
  v8 = *(_QWORD *)(a2 + 40);
  v9 = *(_DWORD *)(a2 + 24);
  LOWORD(v147) = v7;
  v10 = _mm_loadu_si128((const __m128i *)(v8 + 40));
  v11 = *(_QWORD *)v8;
  v12 = *(_QWORD *)(v8 + 8);
  v126 = v9 - 174;
  v130 = v10;
  if ( ((v9 - 174) & 0xFFFFFFFD) != 0 )
  {
    v13 = a3;
    sub_33DD090((__int64)&v151, a3, v11, v12, 0);
    if ( (unsigned int)v152 > 0x40 )
    {
      v45 = sub_C44500((__int64)&v151);
    }
    else
    {
      if ( !(_DWORD)v152 )
      {
        v38 = 0;
        goto LABEL_23;
      }
      v14 = (unsigned int)(64 - v152);
      if ( v151 << (64 - (unsigned __int8)v152) == -1 )
      {
LABEL_45:
        v13 = a3;
        sub_33DD090((__int64)&v155, a3, v130.m128i_i64[0], v130.m128i_i64[1], 0);
        v59 = v155.m128i_u32[2];
        if ( v155.m128i_i32[2] > 0x40u )
        {
          v128 = v155.m128i_u32[2];
          v61 = sub_C44500((__int64)&v155);
          v59 = v128;
LABEL_72:
          v38 = v61 != 0;
          if ( (unsigned int)v157 <= 0x40 )
          {
LABEL_75:
            if ( v59 > 0x40 && v155.m128i_i64[0] )
              j_j___libc_free_0_0(v155.m128i_u64[0]);
            goto LABEL_23;
          }
LABEL_73:
          if ( v156 )
          {
            j_j___libc_free_0_0(v156);
            v59 = v155.m128i_u32[2];
          }
          goto LABEL_75;
        }
        if ( v155.m128i_i32[2] )
        {
          v14 = (unsigned int)(64 - v155.m128i_i32[2]);
          if ( v155.m128i_i64[0] << (64 - v155.m128i_i8[8]) != -1 )
          {
            _BitScanReverse64(&v60, ~(v155.m128i_i64[0] << (64 - v155.m128i_i8[8])));
            v61 = v60 ^ 0x3F;
            goto LABEL_72;
          }
          v38 = 1;
        }
        else
        {
          v38 = 0;
        }
        if ( (unsigned int)v157 > 0x40 )
          goto LABEL_73;
LABEL_23:
        if ( v154 > 0x40 && v153 )
          j_j___libc_free_0_0(v153);
        if ( (unsigned int)v152 > 0x40 && v151 )
          j_j___libc_free_0_0(v151);
        v124 = 214;
        v127 = 192;
        goto LABEL_14;
      }
      _BitScanReverse64(&v44, ~(v151 << (64 - (unsigned __int8)v152)));
      v45 = v44 ^ 0x3F;
    }
    v38 = 0;
    if ( !v45 )
      goto LABEL_23;
    goto LABEL_45;
  }
  v13 = v11;
  if ( (unsigned int)sub_33D4D80(a3, v11, v12, 0) <= 1 )
  {
    v124 = 213;
    v127 = 191;
    goto LABEL_6;
  }
  v13 = v130.m128i_i64[0];
  v124 = 213;
  v127 = 191;
  v38 = (unsigned int)sub_33D4D80(a3, v130.m128i_i64[0], v130.m128i_i64[1], 0) > 1;
LABEL_14:
  if ( v38 )
  {
    *((_QWORD *)&v115 + 1) = v12;
    *(_QWORD *)&v115 = v11;
    v39 = sub_3406EB0((_QWORD *)a3, 0x38u, (__int64)&v145, v147, v148, v16, v115, *(_OWORD *)&v130);
    v41 = v40;
    if ( v126 > 1 )
    {
      *(_QWORD *)&v56 = sub_3400BD0(a3, 1, (__int64)&v145, v147, v148, 0, v10, 0);
      *((_QWORD *)&v110 + 1) = v41;
      *(_QWORD *)&v110 = v39;
      v143 = sub_3406EB0((_QWORD *)a3, 0x38u, (__int64)&v145, v147, v148, v57, v110, v56);
      v39 = v143;
      v144 = v58;
      v41 = (unsigned int)v58 | v41 & 0xFFFFFFFF00000000LL;
    }
    *(_QWORD *)&v42 = sub_3400E40(a3, 1, v147, v148, (__int64)&v145, v10);
    *((_QWORD *)&v116 + 1) = v41;
    *(_QWORD *)&v116 = v39;
    v35 = sub_3406EB0((_QWORD *)a3, v127, (__int64)&v145, v147, v148, v43, v116, v42);
    goto LABEL_9;
  }
LABEL_6:
  v17 = v147;
  if ( (_WORD)v147 )
  {
    if ( (unsigned __int16)(v147 - 2) > 7u )
    {
LABEL_8:
      v141 = sub_33FB960(a3, v11, v12, v10, v14, v15, v16);
      v142 = v18;
      v19 = (unsigned int)v18 | v12 & 0xFFFFFFFF00000000LL;
      v130.m128i_i64[0] = (__int64)sub_33FB960(a3, v130.m128i_i64[0], v130.m128i_u32[2], v10, v20, v21, v22);
      v139 = v130.m128i_i64[0];
      v140 = v23;
      v130.m128i_i64[1] = (unsigned int)v23 | v130.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      *((_QWORD *)&v112 + 1) = v19;
      *(_QWORD *)&v112 = v141;
      *(_QWORD *)&v25 = sub_3406EB0(
                          (_QWORD *)a3,
                          187 - (unsigned int)((unsigned int)(v9 - 174) < 2),
                          (__int64)&v145,
                          v147,
                          v148,
                          v24,
                          v112,
                          *(_OWORD *)&v130);
      *((_QWORD *)&v113 + 1) = v19;
      *(_QWORD *)&v113 = v141;
      v125 = v25;
      v27 = sub_3406EB0((_QWORD *)a3, 0xBCu, (__int64)&v145, v147, v148, v26, v113, *(_OWORD *)&v130);
      v29 = v28;
      v30 = v27;
      *(_QWORD *)&v31 = sub_3400E40(a3, 1, v147, v148, (__int64)&v145, v10);
      *((_QWORD *)&v114 + 1) = v29;
      *(_QWORD *)&v114 = v30;
      *(_QWORD *)&v33 = sub_3406EB0((_QWORD *)a3, v127, (__int64)&v145, v147, v148, v32, v114, v31);
      v35 = sub_3406EB0(
              (_QWORD *)a3,
              57 - (unsigned int)((unsigned int)(v9 - 174) < 2),
              (__int64)&v145,
              v147,
              v148,
              v34,
              v125,
              v33);
LABEL_9:
      v36 = v35;
      goto LABEL_10;
    }
LABEL_32:
    v48 = v148;
    goto LABEL_33;
  }
  if ( !sub_30070A0((__int64)&v147) )
    goto LABEL_8;
  if ( !sub_30070B0((__int64)&v147) )
    goto LABEL_32;
  v17 = sub_3009970((__int64)&v147, v13, v46, v47, v15);
LABEL_33:
  v155.m128i_i16[0] = v17;
  v155.m128i_i64[1] = v48;
  if ( v17 )
  {
    if ( v17 == 1 || (unsigned __int16)(v17 - 504) <= 7u )
      goto LABEL_97;
    v49 = *(_QWORD *)&byte_444C4A0[16 * v17 - 16];
  }
  else
  {
    v149 = sub_3007260((__int64)&v155);
    LODWORD(v49) = v149;
    v150 = v50;
  }
  v51 = 2 * v49;
  switch ( v51 )
  {
    case 2u:
      v52 = 3;
      v14 = 3;
LABEL_50:
      v53 = (unsigned __int16)v14;
      v55 = 0;
      goto LABEL_51;
    case 4u:
      v52 = 4;
      v14 = 4;
      goto LABEL_50;
    case 8u:
      v52 = 5;
      v14 = 5;
      goto LABEL_50;
    case 0x10u:
      v52 = 6;
      v14 = 6;
      goto LABEL_50;
    case 0x20u:
      v52 = 7;
      v14 = 7;
      goto LABEL_50;
    case 0x40u:
      v52 = 8;
      v14 = 8;
      goto LABEL_50;
    case 0x80u:
      v52 = 9;
      v14 = 9;
      goto LABEL_50;
  }
  LODWORD(v52) = sub_3007020(*(_QWORD **)(a3 + 64), v51);
  HIWORD(v53) = WORD1(v52);
  v14 = (unsigned int)v52;
  v55 = v54;
  if ( !(_WORD)v52 )
    goto LABEL_53;
  v52 = (unsigned __int16)v52;
LABEL_51:
  if ( !*(_QWORD *)(a1 + 8 * v52 + 112)
    || (v62 = *(__int64 (**)())(*(_QWORD *)a1 + 1392LL), v62 == sub_2FE3480)
    || (LOWORD(v53) = v14,
        v123 = v55,
        v122 = v53,
        !((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64, _QWORD, __int64))v62)(a1, v53, v55, v147, v148)) )
  {
LABEL_53:
    if ( v9 != 175 )
      goto LABEL_8;
    if ( (_WORD)v147 )
    {
      if ( (unsigned __int16)(v147 - 2) > 7u || *(_QWORD *)(a1 + 8LL * (unsigned __int16)v147 + 112) )
        goto LABEL_8;
    }
    else if ( !sub_30070A0((__int64)&v147) )
    {
      goto LABEL_8;
    }
    v156 = v11;
    v63 = _mm_load_si128(&v130);
    v157 = v12;
    v155 = v63;
    v64 = (unsigned int *)sub_33E5110((__int64 *)a3, v147, v148, 2, 0);
    *((_QWORD *)&v120 + 1) = 2;
    *(_QWORD *)&v120 = &v155;
    v130 = (__m128i)(unsigned __int64)sub_3411630((_QWORD *)a3, 77, (__int64)&v145, v64, v65, v66, v120);
    *(_QWORD *)&v67 = sub_3400E40(a3, 1, v147, v148, (__int64)&v145, v10);
    v69.m128i_i64[0] = (__int64)sub_3406EB0((_QWORD *)a3, 0xC0u, (__int64)&v145, v147, v148, v68, *(_OWORD *)&v130, v67);
    v130 = v69;
    v71 = sub_33FAF80(a3, 215, (__int64)&v145, v147, v148, v70, v10);
    v72 = v147;
    v73 = v71;
    v75 = v74;
    if ( (_WORD)v147 )
    {
      if ( (unsigned __int16)(v147 - 17) <= 0xD3u )
      {
        v72 = word_4456580[(unsigned __int16)v147 - 1];
        v76 = 0;
        goto LABEL_59;
      }
    }
    else if ( sub_30070B0((__int64)&v147) )
    {
      v72 = sub_3009970((__int64)&v147, 215, v87, v88, v89);
      v76 = v90;
      goto LABEL_59;
    }
    v76 = v148;
LABEL_59:
    LOWORD(v151) = v72;
    v152 = v76;
    if ( !v72 )
    {
      v77 = sub_3007260((__int64)&v151);
      v79 = v78;
      v80 = v77;
      v81 = v79;
      v155.m128i_i64[0] = v80;
      v82 = v80;
      v155.m128i_i64[1] = v81;
LABEL_61:
      *(_QWORD *)&v83 = sub_3400E40(a3, v82 - 1, v147, v148, (__int64)&v145, v10);
      *((_QWORD *)&v117 + 1) = v75;
      *(_QWORD *)&v117 = v73;
      *(_QWORD *)&v85 = sub_3406EB0((_QWORD *)a3, 0xBEu, (__int64)&v145, v147, v148, v84, v117, v83);
      v35 = sub_3406EB0((_QWORD *)a3, 0xBBu, (__int64)&v145, v147, v148, v86, *(_OWORD *)&v130, v85);
      goto LABEL_9;
    }
    if ( v72 != 1 && (unsigned __int16)(v72 - 504) > 7u )
    {
      v82 = *(_QWORD *)&byte_444C4A0[16 * v72 - 16];
      goto LABEL_61;
    }
LABEL_97:
    BUG();
  }
  v137 = sub_33FAF80(a3, v124, (__int64)&v145, v122, v123, v16, v10);
  v138 = v91;
  v92 = (unsigned int)v91 | v12 & 0xFFFFFFFF00000000LL;
  v135 = sub_33FAF80(a3, v124, (__int64)&v145, v122, v123, v93, v10);
  v136 = v94;
  v130.m128i_i64[1] = (unsigned int)v94 | v130.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  *((_QWORD *)&v121 + 1) = v130.m128i_i64[1];
  *(_QWORD *)&v121 = v135;
  *((_QWORD *)&v118 + 1) = v92;
  *(_QWORD *)&v118 = v137;
  v130.m128i_i64[0] = v123;
  v96 = sub_3406EB0((_QWORD *)a3, 0x38u, (__int64)&v145, v122, v123, v95, v118, v121);
  v97 = v123;
  v98 = v122;
  v99 = v96;
  v101 = v100;
  if ( v126 > 1 )
  {
    *(_QWORD *)&v102 = sub_3400BD0(a3, 1, (__int64)&v145, v122, v130.m128i_i64[0], 0, v10, 0);
    *((_QWORD *)&v111 + 1) = v101;
    *(_QWORD *)&v111 = v99;
    v104 = sub_3406EB0((_QWORD *)a3, 0x38u, (__int64)&v145, v122, v130.m128i_i64[0], v103, v111, v102);
    v98 = v122;
    v97 = v130.m128i_i64[0];
    v133 = v104;
    v99 = v104;
    v134 = v105;
    v101 = (unsigned int)v105 | v101 & 0xFFFFFFFF00000000LL;
  }
  v129 = v98;
  v130.m128i_i64[0] = v97;
  *(_QWORD *)&v106 = sub_3400E40(a3, 1, v98, v97, (__int64)&v145, v10);
  *((_QWORD *)&v119 + 1) = v101;
  *(_QWORD *)&v119 = v99;
  v131 = sub_3406EB0((_QWORD *)a3, 0xC0u, (__int64)&v145, v129, v130.m128i_i64[0], v107, v119, v106);
  v132 = v108;
  v36 = sub_33FAF80(a3, 216, (__int64)&v145, v147, v148, v109, v10);
LABEL_10:
  if ( v145 )
    sub_B91220((__int64)&v145, v145);
  return v36;
}
