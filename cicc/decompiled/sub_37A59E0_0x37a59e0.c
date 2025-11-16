// Function: sub_37A59E0
// Address: 0x37a59e0
//
unsigned __int8 *__fastcall sub_37A59E0(
        __int64 a1,
        unsigned __int64 a2,
        __m128i a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  __int64 v6; // r15
  _QWORD *v7; // r14
  unsigned __int16 *v8; // rdx
  int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // rbx
  int v13; // eax
  __int64 v14; // rdx
  unsigned int v15; // edx
  __int64 v16; // r13
  unsigned __int16 *v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  char v20; // dl
  unsigned __int64 v21; // rsi
  __int64 *v22; // rdi
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rax
  unsigned __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // rdx
  int v29; // eax
  _QWORD *v30; // rdi
  const __m128i *v31; // rax
  __m128i v32; // xmm2
  __m128i v33; // xmm3
  __int64 v34; // rdx
  unsigned __int64 v35; // r13
  __int64 v36; // r10
  __int64 v37; // r11
  int v38; // eax
  unsigned int v39; // edi
  unsigned int v40; // r15d
  __int16 *v41; // rax
  __int64 v42; // rcx
  __int16 *v43; // rdx
  __int16 *i; // rdx
  int v45; // eax
  __int64 v46; // r14
  _QWORD *v47; // r13
  __int128 v48; // rax
  __int64 v49; // r9
  int v50; // r9d
  unsigned __int8 *v51; // rax
  __int64 v52; // rdx
  __int64 v53; // rdi
  unsigned __int8 *v54; // rdx
  __int64 v55; // rax
  __int16 *v56; // rax
  __int16 *v57; // rsi
  unsigned __int8 *v58; // r14
  int v60; // ecx
  const __m128i *v61; // r13
  unsigned __int64 v62; // rsi
  unsigned __int64 v63; // rdx
  __m128 *v64; // rax
  __int64 v65; // r8
  unsigned __int64 v66; // r13
  __int64 v67; // rax
  __int64 v68; // rbx
  __int64 v69; // r15
  __int128 v70; // rax
  unsigned __int8 *v71; // rax
  __int64 v72; // rdx
  __int64 v73; // rdi
  unsigned __int8 *v74; // rdx
  unsigned __int64 v75; // rax
  _QWORD *v76; // rdi
  __int64 v77; // r9
  unsigned __int8 *v78; // rcx
  __int64 v79; // rdx
  __int64 v80; // rdi
  unsigned __int64 v81; // rdx
  __int64 v82; // rax
  unsigned __int64 v83; // r8
  __int64 v84; // r11
  __int64 v85; // rax
  unsigned __int64 v86; // rdx
  _QWORD *v87; // rax
  __int64 v88; // rax
  _OWORD *v89; // rdx
  unsigned __int8 *v90; // rax
  __int64 v91; // rdx
  unsigned __int8 *v92; // rax
  __int64 v93; // rdx
  _QWORD *v94; // r14
  __int128 v95; // rax
  __int64 v96; // r9
  __int64 v97; // rax
  __int64 v98; // rdx
  __int64 v99; // rdx
  int v100; // eax
  __m128i v101; // xmm4
  __m128i v102; // xmm6
  __int64 v103; // rdx
  unsigned __int8 *v104; // rax
  __int128 v105; // [rsp-20h] [rbp-520h]
  __int128 v106; // [rsp-10h] [rbp-510h]
  __int128 v107; // [rsp-10h] [rbp-510h]
  __int128 v108; // [rsp-10h] [rbp-510h]
  __int128 v109; // [rsp-10h] [rbp-510h]
  __int128 v110; // [rsp-10h] [rbp-510h]
  __int128 v111; // [rsp-10h] [rbp-510h]
  __int64 v112; // [rsp-8h] [rbp-508h]
  __int64 v113; // [rsp+0h] [rbp-500h]
  int v114; // [rsp+20h] [rbp-4E0h]
  __int64 v115; // [rsp+28h] [rbp-4D8h]
  unsigned __int64 v116; // [rsp+30h] [rbp-4D0h]
  _QWORD *v117; // [rsp+30h] [rbp-4D0h]
  unsigned __int64 v118; // [rsp+30h] [rbp-4D0h]
  unsigned int v120; // [rsp+40h] [rbp-4C0h]
  __int64 v121; // [rsp+48h] [rbp-4B8h]
  __int64 v122; // [rsp+50h] [rbp-4B0h]
  __int64 v123; // [rsp+58h] [rbp-4A8h]
  unsigned int v124; // [rsp+60h] [rbp-4A0h]
  __int16 v125; // [rsp+62h] [rbp-49Eh]
  unsigned int v126; // [rsp+68h] [rbp-498h]
  __int16 v127; // [rsp+6Eh] [rbp-492h]
  _QWORD *v128; // [rsp+70h] [rbp-490h]
  __m128i v129; // [rsp+80h] [rbp-480h] BYREF
  unsigned __int8 *v130; // [rsp+90h] [rbp-470h]
  __int64 v131; // [rsp+98h] [rbp-468h]
  unsigned __int8 *v132; // [rsp+A0h] [rbp-460h]
  __int64 v133; // [rsp+A8h] [rbp-458h]
  unsigned __int8 *v134; // [rsp+B0h] [rbp-450h]
  __int64 v135; // [rsp+B8h] [rbp-448h]
  unsigned __int8 *v136; // [rsp+C0h] [rbp-440h]
  __int64 v137; // [rsp+C8h] [rbp-438h]
  unsigned __int8 *v138; // [rsp+D0h] [rbp-430h]
  __int64 v139; // [rsp+D8h] [rbp-428h]
  unsigned __int8 *v140; // [rsp+E0h] [rbp-420h]
  __int64 v141; // [rsp+E8h] [rbp-418h]
  unsigned __int8 *v142; // [rsp+F0h] [rbp-410h]
  __int64 v143; // [rsp+F8h] [rbp-408h]
  unsigned __int64 v144; // [rsp+108h] [rbp-3F8h]
  __int64 v145; // [rsp+110h] [rbp-3F0h] BYREF
  __int64 v146; // [rsp+118h] [rbp-3E8h]
  __int64 v147; // [rsp+120h] [rbp-3E0h] BYREF
  int v148; // [rsp+128h] [rbp-3D8h]
  unsigned __int16 v149; // [rsp+130h] [rbp-3D0h] BYREF
  __int64 v150; // [rsp+138h] [rbp-3C8h]
  _QWORD v151[2]; // [rsp+140h] [rbp-3C0h] BYREF
  __int16 v152; // [rsp+150h] [rbp-3B0h]
  __int64 v153; // [rsp+158h] [rbp-3A8h]
  _BYTE *v154; // [rsp+160h] [rbp-3A0h] BYREF
  __int64 v155; // [rsp+168h] [rbp-398h]
  _BYTE v156[64]; // [rsp+170h] [rbp-390h] BYREF
  __int16 *v157; // [rsp+1B0h] [rbp-350h] BYREF
  __int64 v158; // [rsp+1B8h] [rbp-348h]
  __int16 v159; // [rsp+1C0h] [rbp-340h] BYREF
  __int64 v160; // [rsp+1C8h] [rbp-338h]
  __m128i v161; // [rsp+2C0h] [rbp-240h] BYREF
  _OWORD v162[35]; // [rsp+2D0h] [rbp-230h] BYREF

  v7 = (_QWORD *)a1;
  v8 = *(unsigned __int16 **)(a2 + 48);
  v9 = *v8;
  v10 = *((_QWORD *)v8 + 1);
  LOWORD(v145) = v9;
  v146 = v10;
  if ( (_WORD)v9 )
  {
    v123 = 0;
    v127 = word_4456580[v9 - 1];
  }
  else
  {
    v97 = sub_3009970((__int64)&v145, a2, v10, a5, a6);
    v127 = v97;
    v6 = v97;
    v123 = v98;
  }
  LOWORD(v6) = v127;
  v11 = *(_QWORD *)(a2 + 80);
  v12 = v6;
  v147 = v11;
  if ( v11 )
    sub_B96E90((__int64)&v147, v11, 1);
  v148 = *(_DWORD *)(a2 + 72);
  v13 = *(_DWORD *)(a2 + 24);
  if ( v13 > 239 )
  {
    v14 = (unsigned int)(v13 - 242) < 2 ? 0x28 : 0;
  }
  else
  {
    v14 = 40;
    if ( v13 <= 237 )
      v14 = (unsigned int)(v13 - 101) < 0x30 ? 0x28 : 0;
  }
  v129 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + v14));
  v18 = sub_379AB60(a1, v129.m128i_u64[0], v129.m128i_i64[1]);
  v16 = v15;
  v120 = v15;
  v121 = v18;
  v17 = (unsigned __int16 *)(*(_QWORD *)(v18 + 48) + 16LL * v15);
  LODWORD(v18) = *v17;
  v19 = *((_QWORD *)v17 + 1);
  v126 = *(_DWORD *)(a2 + 24);
  v149 = v18;
  v150 = v19;
  if ( (_WORD)v18 )
  {
    v20 = (unsigned __int16)(v18 - 176) <= 0x34u;
    v21 = word_4456340[(int)v18 - 1];
    LOBYTE(v18) = v20;
  }
  else
  {
    v21 = sub_3007240((__int64)&v149);
    v18 = HIDWORD(v21);
    v144 = v21;
    v20 = BYTE4(v21);
  }
  v22 = *(__int64 **)(*(_QWORD *)(a1 + 8) + 64LL);
  v161.m128i_i32[0] = v21;
  v161.m128i_i8[4] = v18;
  if ( v20 )
    v24 = (unsigned int)sub_2D43AD0(v127, v21);
  else
    v24 = (unsigned int)sub_2D43050(v127, v21);
  if ( (_WORD)v24 )
  {
    v25 = *v7;
    v26 = (unsigned __int16)v24;
    v27 = 0;
  }
  else
  {
    v21 = (unsigned int)v6;
    v26 = sub_3009450(v22, (unsigned int)v6, v123, v161.m128i_i64[0], v23, v24);
    v24 = (unsigned int)v26;
    v27 = v28;
    if ( !(_WORD)v26 )
      goto LABEL_24;
    v25 = *v7;
  }
  v28 = (unsigned __int16)v24;
  if ( *(_QWORD *)(v25 + 8LL * (unsigned __int16)v24 + 112) )
  {
    v29 = *(_DWORD *)(a2 + 24);
    if ( v29 > 239 )
    {
      if ( (unsigned int)(v29 - 242) > 1 )
      {
        v30 = (_QWORD *)v7[1];
        goto LABEL_73;
      }
    }
    else if ( v29 <= 237 )
    {
      v28 = (unsigned int)(v29 - 101);
      if ( (unsigned int)v28 > 0x2F )
      {
        v30 = (_QWORD *)v7[1];
        if ( v29 <= 148 && v29 > 100 )
        {
          v31 = *(const __m128i **)(a2 + 40);
          if ( v126 == 145 )
          {
            LOWORD(v26) = v24;
            v101 = _mm_loadu_si128(v31);
            v129.m128i_i64[0] = v121;
            v161 = v101;
            v129.m128i_i64[1] = v16 | v129.m128i_i64[1] & 0xFFFFFFFF00000000LL;
            v162[0] = _mm_load_si128(&v129);
            v102 = _mm_loadu_si128(v31 + 5);
            *((_QWORD *)&v110 + 1) = 3;
            *(_QWORD *)&v110 = &v161;
            v157 = (__int16 *)v26;
            v158 = v27;
            v159 = 1;
            v160 = 0;
            v162[1] = v102;
            v142 = sub_3411BE0(v30, 0x91u, (__int64)&v147, (unsigned __int16 *)&v157, 2, v24, v110);
            v35 = (unsigned __int64)v142;
            v143 = v34;
          }
          else
          {
            v32 = _mm_loadu_si128(v31);
            LOWORD(v26) = v24;
            *((_QWORD *)&v106 + 1) = 2;
            v129.m128i_i64[0] = v121;
            v157 = (__int16 *)v26;
            v158 = v27;
            v159 = 1;
            v129.m128i_i64[1] = v16 | v129.m128i_i64[1] & 0xFFFFFFFF00000000LL;
            v33 = _mm_load_si128(&v129);
            *(_QWORD *)&v106 = &v161;
            v160 = 0;
            v161 = v32;
            v162[0] = v33;
            v140 = sub_3411BE0(v30, v126, (__int64)&v147, (unsigned __int16 *)&v157, 2, v24, v106);
            v35 = (unsigned __int64)v140;
            v141 = v34;
          }
          v129.m128i_i64[0] = v35;
          v129.m128i_i64[1] = (unsigned int)v34;
          sub_3760E70((__int64)v7, a2, 1, v35, 1);
          v37 = v129.m128i_i64[1];
          v36 = v129.m128i_i64[0];
LABEL_76:
          v94 = (_QWORD *)v7[1];
          v129.m128i_i64[0] = v36;
          v129.m128i_i64[1] = v37;
          *(_QWORD *)&v95 = sub_3400EE0((__int64)v94, 0, (__int64)&v147, 0, a3);
          *((_QWORD *)&v105 + 1) = v129.m128i_i64[1];
          *(_QWORD *)&v105 = v35;
          v58 = sub_3406EB0(v94, 0xA1u, (__int64)&v147, (unsigned int)v145, v146, v96, v105, v95);
          goto LABEL_45;
        }
LABEL_73:
        LOWORD(v26) = v24;
        if ( v126 == 230 )
        {
          v103 = *(_QWORD *)(a2 + 40);
          v129.m128i_i64[0] = v121;
          v111 = *(_OWORD *)(v103 + 40);
          v129.m128i_i64[1] = v16 | v129.m128i_i64[1] & 0xFFFFFFFF00000000LL;
          v104 = sub_3406EB0(v30, 0xE6u, (__int64)&v147, v26, v27, v24, __PAIR128__(v129.m128i_u64[1], v121), v111);
          v139 = v93;
          v35 = (unsigned __int64)v104;
          v36 = (__int64)v104;
          v138 = v104;
        }
        else
        {
          v129.m128i_i64[0] = v121;
          v129.m128i_i64[1] = v16 | v129.m128i_i64[1] & 0xFFFFFFFF00000000LL;
          v92 = sub_33FAF80((__int64)v30, v126, (__int64)&v147, v26, v27, v24, a3);
          v137 = v93;
          v35 = (unsigned __int64)v92;
          v36 = (__int64)v92;
          v136 = v92;
        }
        v37 = (unsigned int)v93;
        goto LABEL_76;
      }
    }
  }
LABEL_24:
  if ( v149 )
  {
    v115 = 0;
    LOWORD(v38) = word_4456580[v149 - 1];
  }
  else
  {
    v38 = sub_3009970((__int64)&v149, v21, v28, v26, v27);
    v125 = HIWORD(v38);
    v115 = v99;
  }
  HIWORD(v39) = v125;
  LOWORD(v39) = v38;
  v124 = v39;
  if ( (_WORD)v145 )
  {
    if ( (unsigned __int16)(v145 - 176) > 0x34u )
      goto LABEL_52;
  }
  else if ( !sub_3007100((__int64)&v145) )
  {
LABEL_28:
    v40 = sub_3007130((__int64)&v145, v21);
    goto LABEL_29;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( !(_WORD)v145 )
    goto LABEL_28;
  if ( (unsigned __int16)(v145 - 176) <= 0x34u )
    sub_CA17B0(
      "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT::"
      "getVectorElementCount() instead");
LABEL_52:
  v40 = word_4456340[(unsigned __int16)v145 - 1];
LABEL_29:
  v41 = &v159;
  v122 = v40;
  v42 = v40;
  v157 = &v159;
  v158 = 0x1000000000LL;
  if ( v40 )
  {
    v43 = &v159;
    if ( v40 > 0x10uLL )
    {
      sub_C8D5F0((__int64)&v157, &v159, v40, 0x10u, v27, v24);
      v43 = v157;
      v41 = &v157[8 * (unsigned int)v158];
    }
    for ( i = &v43[8 * v40]; i != v41; v41 += 8 )
    {
      if ( v41 )
      {
        *(_QWORD *)v41 = 0;
        *((_DWORD *)v41 + 2) = 0;
      }
    }
    LODWORD(v158) = v40;
    v45 = *(_DWORD *)(a2 + 24);
    if ( v45 > 239 )
    {
      if ( (unsigned int)(v45 - 242) > 1 )
      {
LABEL_39:
        v128 = v7;
        v46 = 0;
        do
        {
          v47 = (_QWORD *)v128[1];
          *(_QWORD *)&v48 = sub_3400EE0((__int64)v47, v46, (__int64)&v147, 0, a3);
          v129.m128i_i64[0] = v121;
          v129.m128i_i64[1] = v120 | v129.m128i_i64[1] & 0xFFFFFFFF00000000LL;
          sub_3406EB0(v47, 0x9Eu, (__int64)&v147, v124, v115, v49, __PAIR128__(v129.m128i_u64[1], v121), v48);
          LOWORD(v12) = v127;
          v51 = sub_33FAF80((__int64)v47, v126, (__int64)&v147, (unsigned int)v12, v123, v50, a3);
          v53 = v52;
          v54 = v51;
          v55 = v46++;
          v56 = &v157[8 * v55];
          v130 = v54;
          v131 = v53;
          *(_QWORD *)v56 = v54;
          *((_DWORD *)v56 + 2) = v131;
        }
        while ( v40 != v46 );
        v7 = v128;
LABEL_42:
        v57 = v157;
        v42 = (unsigned int)v158;
        goto LABEL_43;
      }
    }
    else if ( v45 <= 237 && (unsigned int)(v45 - 101) > 0x2F )
    {
      goto LABEL_39;
    }
LABEL_54:
    v155 = 0x400000000LL;
    v60 = 0;
    v61 = *(const __m128i **)(a2 + 40);
    v62 = 5LL * *(unsigned int *)(a2 + 64);
    v63 = *(unsigned int *)(a2 + 64);
    v64 = (__m128 *)v156;
    v65 = (__int64)&v61->m128i_i64[v62];
    v154 = v156;
    if ( v62 > 20 )
    {
      v114 = v63;
      sub_C8D5F0((__int64)&v154, v156, v63, 0x10u, v65, v24);
      v60 = v155;
      v65 = (__int64)&v61->m128i_i64[v62];
      LODWORD(v63) = v114;
      v64 = (__m128 *)&v154[16 * (unsigned int)v155];
    }
    if ( v61 != (const __m128i *)v65 )
    {
      do
      {
        if ( v64 )
        {
          a3 = _mm_loadu_si128(v61);
          *v64 = (__m128)a3;
        }
        v61 = (const __m128i *)((char *)v61 + 40);
        ++v64;
      }
      while ( (const __m128i *)v65 != v61 );
      v60 = v155;
    }
    v161.m128i_i64[0] = (__int64)v162;
    LODWORD(v155) = v60 + v63;
    v161.m128i_i64[1] = 0x2000000000LL;
    if ( v40 )
    {
      v66 = v116;
      v67 = v12;
      v68 = 0;
      v69 = v67;
      do
      {
        v117 = (_QWORD *)v7[1];
        *(_QWORD *)&v70 = sub_3400EE0((__int64)v117, v68, (__int64)&v147, 0, a3);
        v129.m128i_i64[0] = v121;
        v129.m128i_i64[1] = v120 | v129.m128i_i64[1] & 0xFFFFFFFF00000000LL;
        v71 = sub_3406EB0(
                v117,
                0x9Eu,
                (__int64)&v147,
                v124,
                v115,
                *((__int64 *)&v70 + 1),
                __PAIR128__(v129.m128i_u64[1], v121),
                v70);
        LOWORD(v69) = v127;
        v73 = v72;
        v74 = v71;
        v75 = (unsigned __int64)v154;
        v134 = v74;
        v135 = v73;
        *((_QWORD *)v154 + 2) = v74;
        *(_DWORD *)(v75 + 24) = v135;
        v76 = (_QWORD *)v7[1];
        v151[0] = v69;
        v152 = 1;
        *((_QWORD *)&v108 + 1) = (unsigned int)v155;
        *(_QWORD *)&v108 = v75;
        v151[1] = v123;
        v153 = 0;
        v78 = sub_3411BE0(v76, v126, (__int64)&v147, (unsigned __int16 *)v151, 2, v77, v108);
        v80 = v79;
        v81 = (unsigned __int64)v157;
        v82 = 8 * v68;
        v133 = v80;
        v132 = v78;
        v83 = v66 & 0xFFFFFFFF00000000LL | 1;
        *(_QWORD *)&v157[v82] = v78;
        v66 = v83;
        *(_DWORD *)(v81 + v82 * 2 + 8) = v133;
        v84 = *(_QWORD *)&v157[8 * v68];
        v85 = v161.m128i_u32[2];
        v86 = v161.m128i_u32[2] + 1LL;
        if ( v86 > v161.m128i_u32[3] )
        {
          v113 = *(_QWORD *)&v157[8 * v68];
          v118 = v83;
          sub_C8D5F0((__int64)&v161, v162, v86, 0x10u, v83, v24);
          v85 = v161.m128i_u32[2];
          v84 = v113;
          v83 = v118;
        }
        v87 = (_QWORD *)(v161.m128i_i64[0] + 16 * v85);
        ++v68;
        *v87 = v84;
        v87[1] = v83;
        v88 = (unsigned int)++v161.m128i_i32[2];
      }
      while ( v68 != v122 );
      v89 = (_OWORD *)v161.m128i_i64[0];
    }
    else
    {
      v89 = v162;
      v88 = 0;
    }
    *((_QWORD *)&v109 + 1) = v88;
    *(_QWORD *)&v109 = v89;
    v90 = sub_33FC220((_QWORD *)v7[1], 2, (__int64)&v147, 1, 0, v24, v109);
    sub_3760E70((__int64)v7, a2, 1, (unsigned __int64)v90, v91);
    v24 = v112;
    if ( (_OWORD *)v161.m128i_i64[0] != v162 )
      _libc_free(v161.m128i_u64[0]);
    if ( v154 != v156 )
      _libc_free((unsigned __int64)v154);
    goto LABEL_42;
  }
  v100 = *(_DWORD *)(a2 + 24);
  if ( v100 > 239 )
  {
    v57 = &v159;
    if ( (unsigned int)(v100 - 242) <= 1 )
      goto LABEL_54;
  }
  else
  {
    if ( v100 > 237 )
      goto LABEL_54;
    v57 = &v159;
    if ( (unsigned int)(v100 - 101) <= 0x2F )
      goto LABEL_54;
  }
LABEL_43:
  *((_QWORD *)&v107 + 1) = v42;
  *(_QWORD *)&v107 = v57;
  v58 = sub_33FC220((_QWORD *)v7[1], 156, (__int64)&v147, v145, v146, v24, v107);
  if ( v157 != &v159 )
    _libc_free((unsigned __int64)v157);
LABEL_45:
  if ( v147 )
    sub_B91220((__int64)&v147, v147);
  return v58;
}
