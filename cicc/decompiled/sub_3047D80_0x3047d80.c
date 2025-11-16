// Function: sub_3047D80
// Address: 0x3047d80
//
void __fastcall sub_3047D80(__int64 a1, __int64 a2, __int64 a3)
{
  int v4; // eax
  unsigned __int16 *v5; // rdx
  int v6; // r13d
  __int64 v7; // rbx
  int v8; // eax
  __int64 v9; // rax
  __int64 v10; // rdx
  __int8 v11; // al
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // rdx
  __int64 v16; // r8
  __int64 v17; // r9
  unsigned int v18; // ebx
  unsigned __int64 v19; // r14
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // rdx
  __int64 v22; // rdx
  unsigned __int16 *v23; // rax
  __int64 v24; // rsi
  __m128i v25; // xmm1
  __m128i v26; // xmm0
  __int64 v27; // rdi
  int v28; // eax
  __int64 v29; // rax
  __m128i v30; // xmm3
  __int64 v31; // rax
  __int32 v32; // edx
  __m128i v33; // xmm2
  __int64 v34; // r8
  __int64 v35; // r9
  __m128i *v36; // rax
  __int64 v37; // rax
  __int64 v38; // rbx
  int v39; // eax
  int v40; // edx
  __int64 v41; // r10
  __int64 v42; // r11
  int v43; // ecx
  int v44; // r8d
  __int64 v45; // rax
  __int64 v46; // r8
  unsigned __int64 v47; // r9
  __int64 v48; // rsi
  __int64 v49; // r13
  __int64 v50; // rdx
  unsigned int v51; // ebx
  __int64 v52; // rax
  unsigned int v53; // edx
  __int64 v54; // r8
  __int64 v55; // rax
  unsigned __int64 v56; // rdx
  __int64 *v57; // rax
  __int64 v58; // rax
  __int64 v59; // r10
  __int64 v60; // r11
  bool v61; // al
  __int64 v62; // rcx
  __int64 v63; // r8
  __int16 v64; // ax
  __int64 v65; // rdx
  __int64 v66; // r8
  __int16 v67; // dx
  int v68; // r8d
  __int64 v69; // rsi
  __int16 v70; // r15
  __int64 v71; // rax
  bool v72; // al
  __int64 v73; // rcx
  __int64 v74; // r8
  __int16 v75; // ax
  int v76; // edx
  __m128i *v77; // rdx
  __int64 v78; // rsi
  __int128 v79; // rcx
  __int64 v80; // rax
  __int64 v81; // r8
  __int64 v82; // rdx
  __int64 v83; // r9
  __int64 v84; // rdx
  __int64 *v85; // rdx
  __int64 v86; // rsi
  __int64 v87; // rax
  __int64 v88; // rbx
  __int64 *v89; // rax
  bool v90; // al
  bool v91; // al
  __int64 v92; // rcx
  __int64 v93; // rcx
  __int64 v94; // rbx
  unsigned int v95; // eax
  __int64 v96; // r12
  __int64 v97; // rax
  __int64 *v98; // rax
  unsigned __int64 v99; // rcx
  __int64 v100; // rax
  __int64 *v101; // rax
  bool v102; // al
  __int64 v103; // rdx
  unsigned __int64 v104; // rax
  unsigned __int64 v105; // rdx
  unsigned __int16 *v106; // rax
  unsigned __int16 *v107; // rdx
  unsigned __int16 *v108; // rdx
  __int64 v109; // rcx
  __int16 v110; // bx
  int v111; // r8d
  __int64 v112; // rsi
  __int16 v113; // cx
  unsigned int v114; // edx
  __int16 v115; // ax
  __int64 v116; // rdx
  __int64 v117; // rcx
  __int64 v118; // r8
  int v119; // edx
  __int128 v120; // [rsp-10h] [rbp-280h]
  int v121; // [rsp+10h] [rbp-260h]
  __int64 v122; // [rsp+18h] [rbp-258h]
  int v124; // [rsp+30h] [rbp-240h]
  unsigned int v125; // [rsp+30h] [rbp-240h]
  __int64 v126; // [rsp+30h] [rbp-240h]
  __int64 v127; // [rsp+38h] [rbp-238h]
  __int64 v128; // [rsp+40h] [rbp-230h]
  __int64 v129; // [rsp+40h] [rbp-230h]
  unsigned int v130; // [rsp+40h] [rbp-230h]
  unsigned __int64 v131; // [rsp+40h] [rbp-230h]
  __int64 v132; // [rsp+48h] [rbp-228h]
  __int64 v133; // [rsp+48h] [rbp-228h]
  __int64 v134; // [rsp+50h] [rbp-220h]
  int v135; // [rsp+50h] [rbp-220h]
  __int64 v136; // [rsp+50h] [rbp-220h]
  __int64 v137; // [rsp+50h] [rbp-220h]
  unsigned __int64 v138; // [rsp+50h] [rbp-220h]
  __int64 v139; // [rsp+58h] [rbp-218h]
  unsigned __int16 v140; // [rsp+60h] [rbp-210h]
  unsigned __int64 v142; // [rsp+70h] [rbp-200h]
  __int128 v143; // [rsp+70h] [rbp-200h]
  int v144; // [rsp+70h] [rbp-200h]
  unsigned int v145; // [rsp+70h] [rbp-200h]
  __int64 v146; // [rsp+70h] [rbp-200h]
  __int64 v147; // [rsp+78h] [rbp-1F8h]
  int v148; // [rsp+80h] [rbp-1F0h] BYREF
  __int64 v149; // [rsp+88h] [rbp-1E8h]
  __int64 v150; // [rsp+90h] [rbp-1E0h]
  __int64 v151; // [rsp+98h] [rbp-1D8h]
  __int64 v152; // [rsp+A0h] [rbp-1D0h] BYREF
  int v153; // [rsp+A8h] [rbp-1C8h]
  __m128i v154; // [rsp+B0h] [rbp-1C0h] BYREF
  __m128i v155; // [rsp+C0h] [rbp-1B0h] BYREF
  _BYTE *v156; // [rsp+D0h] [rbp-1A0h] BYREF
  __int64 v157; // [rsp+D8h] [rbp-198h]
  _BYTE v158[64]; // [rsp+E0h] [rbp-190h] BYREF
  unsigned __int16 *v159; // [rsp+120h] [rbp-150h] BYREF
  __int64 v160; // [rsp+128h] [rbp-148h]
  _QWORD v161[16]; // [rsp+130h] [rbp-140h] BYREF
  __m128i v162; // [rsp+1B0h] [rbp-C0h] BYREF
  __m128i v163; // [rsp+1C0h] [rbp-B0h] BYREF
  __m128i v164; // [rsp+1D0h] [rbp-A0h] BYREF
  __m128i v165; // [rsp+1E0h] [rbp-90h] BYREF
  __m128i v166[8]; // [rsp+1F0h] [rbp-80h] BYREF

  v4 = sub_3032270(a1, a2);
  v5 = *(unsigned __int16 **)(a1 + 48);
  v6 = v4;
  v7 = *((_QWORD *)v5 + 1);
  v8 = *v5;
  v149 = v7;
  LOWORD(v148) = v8;
  if ( (_WORD)v8 )
  {
    if ( (unsigned __int16)(v8 - 17) > 0xD3u )
    {
      LOWORD(v159) = v8;
      v160 = v7;
      goto LABEL_4;
    }
    LOWORD(v8) = word_4456580[v8 - 1];
    v22 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v148) )
    {
      v160 = v7;
      LOWORD(v159) = 0;
LABEL_9:
      v150 = sub_3007260((__int64)&v159);
      v151 = v15;
      v10 = v150;
      v11 = v151;
      goto LABEL_10;
    }
    LOWORD(v8) = sub_3009970((__int64)&v148, a2, v12, v13, v14);
  }
  LOWORD(v159) = v8;
  v160 = v22;
  if ( !(_WORD)v8 )
    goto LABEL_9;
LABEL_4:
  if ( (_WORD)v8 == 1 || (unsigned __int16)(v8 - 504) <= 7u )
    BUG();
  v9 = 16LL * ((unsigned __int16)v8 - 1);
  v10 = *(_QWORD *)&byte_444C4A0[v9];
  v11 = byte_444C4A0[v9 + 8];
LABEL_10:
  v162.m128i_i8[8] = v11;
  v162.m128i_i64[0] = v10;
  if ( (unsigned __int64)sub_CA1930(&v162) <= 0xF )
  {
    v122 = 0;
    v140 = 6;
  }
  else
  {
    v140 = (unsigned __int16)v159;
    v122 = v160;
  }
  v159 = (unsigned __int16 *)v161;
  v160 = 0x800000000LL;
  if ( !(_WORD)v148 )
  {
    if ( sub_30070B0((__int64)&v148) )
    {
      v102 = sub_3007100((__int64)&v148);
      v103 = 8;
      if ( !v102 )
        goto LABEL_90;
      goto LABEL_105;
    }
LABEL_25:
    v19 = 1;
    LODWORD(v160) = 1;
    v161[0] = v140;
    v161[1] = v122;
    goto LABEL_26;
  }
  if ( (unsigned __int16)(v148 - 17) > 0xD3u )
    goto LABEL_25;
  if ( (unsigned __int16)(v148 - 176) > 0x34u )
    goto LABEL_15;
LABEL_105:
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( (_WORD)v148 )
  {
    if ( (unsigned __int16)(v148 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
LABEL_15:
    v18 = word_4456340[(unsigned __int16)v148 - 1];
    v19 = word_4456340[(unsigned __int16)v148 - 1];
    if ( v19 > HIDWORD(v160) )
      goto LABEL_16;
LABEL_91:
    v104 = (unsigned int)v160;
    v105 = (unsigned int)v160;
    if ( v19 <= (unsigned int)v160 )
      v105 = v19;
    if ( v105 )
    {
      v106 = v159;
      v107 = &v159[8 * v105];
      do
      {
        *v106 = v140;
        v106 += 8;
        *((_QWORD *)v106 - 1) = v122;
      }
      while ( v107 != v106 );
      v104 = (unsigned int)v160;
    }
    if ( v19 > v104 )
    {
      v108 = &v159[8 * v104];
      v109 = v19 - v104;
      if ( v19 != v104 )
      {
        do
        {
          if ( v108 )
          {
            *v108 = v140;
            *((_QWORD *)v108 + 1) = v122;
          }
          v108 += 8;
          --v109;
        }
        while ( v109 );
      }
    }
    goto LABEL_20;
  }
  v103 = HIDWORD(v160);
LABEL_90:
  v138 = v103;
  v18 = sub_3007130((__int64)&v148, a2);
  v19 = v18;
  if ( v18 <= v138 )
    goto LABEL_91;
LABEL_16:
  LODWORD(v160) = 0;
  sub_C8D5F0((__int64)&v159, v161, v19, 0x10u, v16, v17);
  v20 = (unsigned __int64)v159;
  v21 = v19;
  do
  {
    if ( v20 )
    {
      *(_WORD *)v20 = v140;
      *(_QWORD *)(v20 + 8) = v122;
    }
    v20 += 16LL;
    --v21;
  }
  while ( v21 );
LABEL_20:
  LODWORD(v160) = v18;
  if ( HIDWORD(v160) < v19 + 1 )
  {
    sub_C8D5F0((__int64)&v159, v161, v19 + 1, 0x10u, 1, v17);
    v19 = (unsigned int)v160;
  }
LABEL_26:
  v23 = &v159[8 * v19];
  *(_QWORD *)v23 = 1;
  *((_QWORD *)v23 + 1) = 0;
  LODWORD(v160) = v160 + 1;
  sub_3030880((__int64)&v154, a1, a2);
  v24 = *(_QWORD *)(a1 + 80);
  v25 = _mm_load_si128(&v154);
  v26 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a1 + 40));
  v152 = v24;
  v163 = v25;
  v162 = v26;
  if ( v24 )
    sub_B96E90((__int64)&v152, v24, 1);
  v27 = *(_QWORD *)(a1 + 112);
  v153 = *(_DWORD *)(a1 + 72);
  v28 = sub_2EAC1E0(v27);
  v29 = sub_3400BD0(a2, v28, (unsigned int)&v152, 5, 0, 1, 0);
  v30 = _mm_load_si128(&v155);
  v164.m128i_i64[0] = v29;
  v31 = *(_QWORD *)(a1 + 40);
  v164.m128i_i32[2] = v32;
  v33 = _mm_loadu_si128((const __m128i *)(v31 + 120));
  v156 = v158;
  v157 = 0x400000000LL;
  v165 = v33;
  v166[0] = v30;
  sub_C8D5F0((__int64)&v156, v158, 5u, 0x10u, v34, v35);
  v36 = (__m128i *)&v156[16 * (unsigned int)v157];
  *v36 = _mm_load_si128(&v162);
  v36[1] = _mm_load_si128(&v163);
  v36[2] = _mm_load_si128(&v164);
  v36[3] = _mm_load_si128(&v165);
  v36[4] = _mm_load_si128(v166);
  v37 = (unsigned int)(v157 + 5);
  LODWORD(v157) = v157 + 5;
  if ( v152 )
  {
    sub_B91220((__int64)&v152, v152);
    v37 = (unsigned int)v157;
  }
  v139 = v37;
  v134 = (__int64)v156;
  v128 = *(unsigned __int16 *)(a1 + 96);
  v132 = *(_QWORD *)(a1 + 104);
  v38 = *(_QWORD *)(a1 + 112);
  v39 = sub_33E5830(a2, v159);
  v41 = v128;
  v42 = v132;
  v43 = v39;
  v44 = v40;
  v162.m128i_i64[0] = *(_QWORD *)(a1 + 80);
  if ( v162.m128i_i64[0] )
  {
    v121 = v40;
    v124 = v39;
    sub_B96E90((__int64)&v162, v162.m128i_i64[0], 1);
    v44 = v121;
    v43 = v124;
    v41 = v128;
    v42 = v132;
  }
  v162.m128i_i32[2] = *(_DWORD *)(a1 + 72);
  v45 = sub_33EA9D0(a2, v6, (unsigned int)&v162, v43, v44, v38, v134, v139, v41, v42);
  v48 = v162.m128i_i64[0];
  v49 = v45;
  if ( v162.m128i_i64[0] )
    sub_B91220((__int64)&v162, v162.m128i_i64[0]);
  LODWORD(v50) = (unsigned __int16)v148;
  if ( !(_WORD)v148 )
  {
    v90 = sub_30070B0((__int64)&v148);
    LODWORD(v50) = 0;
    if ( v90 )
      goto LABEL_36;
    v91 = sub_30070B0((__int64)&v148);
    LOWORD(v50) = 0;
    if ( v91 )
    {
      v115 = sub_3009970((__int64)&v148, v48, 0, v92, v46);
      v93 = v50;
      LOWORD(v50) = v115;
      goto LABEL_81;
    }
LABEL_80:
    v93 = v149;
LABEL_81:
    if ( (_WORD)v50 == v140 )
    {
      v94 = v49;
      v95 = 0;
      if ( (_WORD)v50 || v93 == v122 )
      {
LABEL_83:
        v96 = v95;
        v97 = *(unsigned int *)(a3 + 8);
        if ( v97 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
        {
          sub_C8D5F0(a3, (const void *)(a3 + 16), v97 + 1, 0x10u, v46, v47);
          v97 = *(unsigned int *)(a3 + 8);
        }
        v98 = (__int64 *)(*(_QWORD *)a3 + 16 * v97);
        *v98 = v94;
        v98[1] = v96;
        v99 = *(unsigned int *)(a3 + 12);
        v100 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
        *(_DWORD *)(a3 + 8) = v100;
        if ( v100 + 1 > v99 )
        {
          sub_C8D5F0(a3, (const void *)(a3 + 16), v100 + 1, 0x10u, v46, v47);
          v100 = *(unsigned int *)(a3 + 8);
        }
        v101 = (__int64 *)(*(_QWORD *)a3 + 16 * v100);
        *v101 = v49;
        v101[1] = 1;
        ++*(_DWORD *)(a3 + 8);
        goto LABEL_73;
      }
    }
    v110 = v148;
    if ( (_WORD)v148 )
    {
      if ( (unsigned __int16)(v148 - 17) <= 0xD3u )
      {
        v111 = 0;
        v110 = word_4456580[(unsigned __int16)v148 - 1];
        goto LABEL_111;
      }
    }
    else if ( sub_30070B0((__int64)&v148) )
    {
      v110 = sub_3009970((__int64)&v148, v48, v116, v117, v118);
      v111 = v119;
      goto LABEL_111;
    }
    v111 = v149;
LABEL_111:
    v112 = *(_QWORD *)(a1 + 80);
    v113 = v110;
    v162.m128i_i64[0] = v112;
    if ( v112 )
    {
      v144 = v111;
      sub_B96E90((__int64)&v162, v112, 1);
      v113 = v110;
      v111 = v144;
    }
    v162.m128i_i32[2] = *(_DWORD *)(a1 + 72);
    v94 = sub_33FAF80(a2, 216, (unsigned int)&v162, v113, v111, v47, (unsigned __int64)v49);
    v95 = v114;
    if ( v162.m128i_i64[0] )
    {
      v145 = v114;
      sub_B91220((__int64)&v162, v162.m128i_i64[0]);
      v95 = v145;
    }
    goto LABEL_83;
  }
  if ( (unsigned __int16)(v148 - 17) > 0xD3u )
    goto LABEL_80;
LABEL_36:
  v162.m128i_i64[0] = (__int64)&v163;
  v162.m128i_i64[1] = 0x800000000LL;
  if ( *(_DWORD *)(v49 + 68) != 1 )
  {
    v51 = 0;
    while ( 1 )
    {
      v59 = v49;
      v60 = v51;
      if ( !(_WORD)v50 )
        break;
      if ( (unsigned __int16)(v50 - 17) > 0xD3u )
        goto LABEL_39;
      LOWORD(v50) = word_4456580[(unsigned __int16)v50 - 1];
      v52 = 0;
LABEL_40:
      v48 = v140;
      if ( (_WORD)v50 != v140 )
        goto LABEL_49;
LABEL_41:
      v53 = v51;
      v54 = v49;
      if ( (_WORD)v48 || v52 == v122 )
        goto LABEL_42;
LABEL_49:
      v67 = v148;
      if ( (_WORD)v148 )
      {
        if ( (unsigned __int16)(v148 - 17) <= 0xD3u )
        {
          v68 = 0;
          v67 = word_4456580[(unsigned __int16)v148 - 1];
          goto LABEL_52;
        }
      }
      else
      {
        v126 = v59;
        v127 = v60;
        v72 = sub_30070B0((__int64)&v148);
        v67 = 0;
        v59 = v126;
        v60 = v127;
        if ( v72 )
        {
          v75 = sub_3009970((__int64)&v148, v48, 0, v73, v74);
          v59 = v126;
          v60 = v127;
          v68 = v76;
          v67 = v75;
          goto LABEL_52;
        }
      }
      v68 = v149;
LABEL_52:
      v69 = *(_QWORD *)(a1 + 80);
      v70 = v67;
      v152 = v69;
      if ( v69 )
      {
        v129 = v59;
        v133 = v60;
        v135 = v68;
        sub_B96E90((__int64)&v152, v69, 1);
        v59 = v129;
        v60 = v133;
        v68 = v135;
      }
      *((_QWORD *)&v120 + 1) = v60;
      *(_QWORD *)&v120 = v59;
      v153 = *(_DWORD *)(a1 + 72);
      v71 = sub_33FAF80(a2, 216, (unsigned int)&v152, v70, v68, v47, v120);
      v48 = v152;
      v54 = v71;
      if ( v152 )
      {
        v130 = v53;
        v136 = v71;
        sub_B91220((__int64)&v152, v152);
        v53 = v130;
        v54 = v136;
      }
LABEL_42:
      v55 = v162.m128i_u32[2];
      v47 = v53 | v142 & 0xFFFFFFFF00000000LL;
      v56 = v162.m128i_u32[2] + 1LL;
      v142 = v47;
      if ( v56 > v162.m128i_u32[3] )
      {
        v48 = (__int64)&v163;
        v131 = v47;
        v137 = v54;
        sub_C8D5F0((__int64)&v162, &v163, v56, 0x10u, v54, v47);
        v55 = v162.m128i_u32[2];
        v47 = v131;
        v54 = v137;
      }
      v57 = (__int64 *)(v162.m128i_i64[0] + 16 * v55);
      ++v51;
      *v57 = v54;
      v57[1] = v47;
      v58 = (unsigned int)++v162.m128i_i32[2];
      if ( *(_DWORD *)(v49 + 68) - 1 <= v51 )
      {
        v77 = (__m128i *)v162.m128i_i64[0];
        goto LABEL_63;
      }
      LODWORD(v50) = (unsigned __int16)v148;
    }
    v125 = v50;
    v61 = sub_30070B0((__int64)&v148);
    v59 = v49;
    v60 = v51;
    LOWORD(v50) = v125;
    if ( v61 )
    {
      v64 = sub_3009970((__int64)&v148, v48, v125, v62, v63);
      v48 = v140;
      v59 = v49;
      v66 = v65;
      LOWORD(v65) = v64;
      v60 = v51;
      v52 = v66;
      if ( (_WORD)v65 != v140 )
        goto LABEL_49;
      goto LABEL_41;
    }
LABEL_39:
    v52 = v149;
    goto LABEL_40;
  }
  v77 = &v163;
  v58 = 0;
LABEL_63:
  v78 = *(_QWORD *)(a1 + 80);
  *(_QWORD *)&v79 = v77;
  *((_QWORD *)&v79 + 1) = v58;
  v152 = v78;
  if ( v78 )
  {
    *(_QWORD *)&v143 = v77;
    *((_QWORD *)&v143 + 1) = v58;
    sub_B96E90((__int64)&v152, v78, 1);
    v79 = v143;
  }
  v153 = *(_DWORD *)(a1 + 72);
  v80 = sub_33FC220(a2, 156, (unsigned int)&v152, v148, v149, v47, v79);
  v81 = v80;
  v83 = v82;
  v84 = *(unsigned int *)(a3 + 8);
  if ( v84 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    v146 = v80;
    v147 = v83;
    sub_C8D5F0(a3, (const void *)(a3 + 16), v84 + 1, 0x10u, v80, v83);
    v84 = *(unsigned int *)(a3 + 8);
    v81 = v146;
    v83 = v147;
  }
  v85 = (__int64 *)(*(_QWORD *)a3 + 16 * v84);
  *v85 = v81;
  v85[1] = v83;
  v86 = v152;
  v87 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v87;
  if ( v86 )
  {
    sub_B91220((__int64)&v152, v86);
    v87 = *(unsigned int *)(a3 + 8);
  }
  v88 = (unsigned int)(*(_DWORD *)(v49 + 68) - 1);
  if ( v87 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    sub_C8D5F0(a3, (const void *)(a3 + 16), v87 + 1, 0x10u, v81, v83);
    v87 = *(unsigned int *)(a3 + 8);
  }
  v89 = (__int64 *)(*(_QWORD *)a3 + 16 * v87);
  *v89 = v49;
  v89[1] = v88;
  ++*(_DWORD *)(a3 + 8);
  if ( (__m128i *)v162.m128i_i64[0] != &v163 )
    _libc_free(v162.m128i_u64[0]);
LABEL_73:
  if ( v156 != v158 )
    _libc_free((unsigned __int64)v156);
  if ( v159 != (unsigned __int16 *)v161 )
    _libc_free((unsigned __int64)v159);
}
