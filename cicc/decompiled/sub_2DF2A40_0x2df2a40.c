// Function: sub_2DF2A40
// Address: 0x2df2a40
//
__int64 __fastcall sub_2DF2A40(__int64 a1)
{
  _BYTE *v1; // rsi
  __int64 v2; // rdx
  __int64 v3; // rax
  bool v4; // zf
  __int64 v5; // rax
  const char *v6; // rax
  _QWORD *v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rsi
  _QWORD *i; // rdx
  unsigned int v11; // r12d
  __int64 v12; // r15
  char *v13; // rax
  char *v14; // rax
  int v15; // r11d
  __int64 *v16; // rdi
  unsigned int v17; // ecx
  _QWORD *v18; // rdx
  __int64 v19; // r8
  __int64 v20; // rdi
  __int16 v21; // dx
  __int64 v22; // r8
  char v23; // r12
  __int64 *v24; // r13
  unsigned __int64 v25; // rbx
  _QWORD *v26; // r14
  unsigned __int16 v27; // ax
  __int64 v28; // r12
  __int64 *v29; // rax
  __int64 v30; // rbx
  __int64 v31; // rax
  int v32; // edx
  unsigned int v33; // ecx
  _BYTE *v34; // r12
  _BYTE *v35; // rax
  unsigned __int8 v36; // al
  _QWORD *v37; // rdx
  size_t v38; // rdx
  unsigned __int8 *v39; // rdi
  __int64 *v41; // rax
  __int64 *v42; // r13
  __int64 *v43; // rdi
  __int64 *v44; // r12
  unsigned __int64 v45; // r14
  unsigned __int8 *v46; // r12
  __int64 v47; // rax
  __int64 v48; // r14
  _QWORD *v49; // rdi
  __int64 *v50; // r13
  unsigned __int64 v51; // rax
  __int64 v52; // rdx
  const char *v53; // rax
  size_t v54; // rdx
  size_t v55; // r13
  char *v56; // r14
  __int64 v57; // rax
  __int64 v58; // rdx
  __int64 v59; // rcx
  __m128i *v60; // rax
  unsigned __int64 *v61; // rax
  unsigned __int64 v62; // rax
  unsigned __int64 v63; // rdi
  unsigned __int64 *v64; // rax
  __m128i *v65; // rcx
  unsigned __int64 *v66; // rdx
  __int64 v67; // r13
  __int64 v68; // rax
  unsigned __int8 v69; // dl
  __int64 v70; // rax
  _BYTE *v71; // rax
  unsigned __int8 v72; // al
  _QWORD *v73; // rdx
  unsigned __int8 v74; // dl
  __int64 v75; // rax
  __int64 v76; // rdi
  __int64 v77; // rax
  unsigned __int64 v78; // rdx
  const void *v79; // r9
  size_t v80; // r14
  unsigned __int64 v81; // r8
  unsigned __int8 *v82; // rdi
  unsigned __int8 v83; // al
  _BYTE **v84; // r12
  size_t v85; // rdx
  char *v86; // rdi
  unsigned __int8 v87; // al
  char **v88; // r12
  __int64 v89; // rax
  __int64 v90; // rdx
  char *v91; // rbx
  char *v92; // r13
  char v93; // r12
  size_t v94; // r14
  unsigned __int64 v95; // rdx
  char *v96; // rcx
  size_t v97; // r9
  unsigned __int8 *v98; // rax
  unsigned int v99; // ecx
  int v100; // esi
  unsigned __int64 v101; // rdx
  char v102; // cl
  __m128i *v103; // rax
  __m128i *v104; // rsi
  size_t v105; // rax
  unsigned __int64 *v106; // rax
  unsigned __int64 *v107; // rax
  __m128i *v108; // rax
  __int64 *v109; // r10
  int v110; // r12d
  unsigned int v111; // ecx
  _BYTE *v112; // r8
  unsigned __int8 v113; // al
  _BYTE *v114; // rax
  unsigned __int8 v115; // al
  _QWORD *v116; // rdx
  unsigned __int8 v117; // dl
  _BYTE *v118; // rax
  char *v119; // rdi
  size_t v120; // rdx
  _BYTE *v121; // rax
  unsigned __int8 v122; // al
  _QWORD *v123; // rdx
  unsigned __int8 v124; // dl
  const char **v125; // rax
  const char *v126; // rdi
  size_t v127; // rdx
  size_t v128; // rax
  int v129; // r12d
  __int64 *v130; // r10
  __int64 v131; // [rsp+10h] [rbp-360h]
  unsigned __int8 *v132; // [rsp+20h] [rbp-350h]
  char *s; // [rsp+28h] [rbp-348h]
  unsigned int v134; // [rsp+38h] [rbp-338h]
  int v135; // [rsp+3Ch] [rbp-334h]
  void *src; // [rsp+48h] [rbp-328h]
  void *srca; // [rsp+48h] [rbp-328h]
  __int64 *v139; // [rsp+58h] [rbp-318h]
  __int64 v140; // [rsp+60h] [rbp-310h]
  unsigned __int64 v141; // [rsp+60h] [rbp-310h]
  unsigned __int64 v142; // [rsp+60h] [rbp-310h]
  bool v143; // [rsp+6Bh] [rbp-305h]
  unsigned int v144; // [rsp+6Ch] [rbp-304h]
  char v145; // [rsp+70h] [rbp-300h]
  __m128i **v146; // [rsp+78h] [rbp-2F8h]
  __int64 v147; // [rsp+90h] [rbp-2E0h]
  const char *v148; // [rsp+B0h] [rbp-2C0h] BYREF
  _BYTE *v149; // [rsp+B8h] [rbp-2B8h] BYREF
  __int64 v150; // [rsp+C0h] [rbp-2B0h] BYREF
  _QWORD *v151; // [rsp+C8h] [rbp-2A8h]
  __int64 v152; // [rsp+D0h] [rbp-2A0h]
  unsigned int v153; // [rsp+D8h] [rbp-298h]
  __int64 v154[2]; // [rsp+E0h] [rbp-290h] BYREF
  _QWORD v155[2]; // [rsp+F0h] [rbp-280h] BYREF
  __m128i *v156; // [rsp+100h] [rbp-270h] BYREF
  unsigned __int64 v157; // [rsp+108h] [rbp-268h]
  __m128i v158; // [rsp+110h] [rbp-260h] BYREF
  char *v159; // [rsp+120h] [rbp-250h] BYREF
  size_t v160; // [rsp+128h] [rbp-248h]
  _QWORD v161[2]; // [rsp+130h] [rbp-240h] BYREF
  __int16 v162; // [rsp+140h] [rbp-230h]
  __int64 v163[2]; // [rsp+150h] [rbp-220h] BYREF
  _QWORD v164[2]; // [rsp+160h] [rbp-210h] BYREF
  __int16 v165; // [rsp+170h] [rbp-200h]
  __m128i *v166; // [rsp+180h] [rbp-1F0h] BYREF
  size_t v167; // [rsp+188h] [rbp-1E8h]
  __m128i v168; // [rsp+190h] [rbp-1E0h] BYREF
  __int16 v169; // [rsp+1A0h] [rbp-1D0h]
  __m128i *v170; // [rsp+1B0h] [rbp-1C0h] BYREF
  size_t v171; // [rsp+1B8h] [rbp-1B8h]
  __m128i v172; // [rsp+1C0h] [rbp-1B0h] BYREF
  __int16 v173; // [rsp+1D0h] [rbp-1A0h]
  _QWORD *v174; // [rsp+1E0h] [rbp-190h] BYREF
  _QWORD v175[6]; // [rsp+1F0h] [rbp-180h] BYREF
  __m128i *v176; // [rsp+220h] [rbp-150h] BYREF
  size_t v177; // [rsp+228h] [rbp-148h]
  __m128i v178; // [rsp+230h] [rbp-140h] BYREF
  _BYTE **v179; // [rsp+240h] [rbp-130h]

  v1 = *(_BYTE **)(a1 + 232);
  v2 = *(_QWORD *)(a1 + 240);
  v139 = *(__int64 **)a1;
  v174 = v175;
  sub_2DF2050((__int64 *)&v174, v1, (__int64)&v1[v2]);
  LODWORD(v1) = *(_DWORD *)(a1 + 284);
  v3 = *(_QWORD *)(a1 + 264);
  v4 = *(_DWORD *)(a1 + 276) == 14;
  v150 = 0;
  v175[2] = v3;
  v135 = (int)v1;
  v175[3] = *(_QWORD *)(a1 + 272);
  v5 = *(_QWORD *)(a1 + 280);
  v153 = 16;
  v175[4] = v5;
  v143 = *(_DWORD *)(a1 + 264) == 38 && *(_DWORD *)(a1 + 280) == 27 && v4;
  v6 = ".data.just.my.code";
  if ( (_DWORD)v1 != 3 )
    v6 = ".msvcjmc";
  v148 = v6;
  v7 = (_QWORD *)sub_C7D670(256, 8);
  v152 = 0;
  v151 = v7;
  v8 = (__int64)v7;
  v9 = 2LL * v153;
  for ( i = &v7[v9]; i != v7; v7 += 2 )
  {
    if ( v7 )
      *v7 = -4096;
  }
  v11 = 0;
  v12 = *(_QWORD *)(a1 + 32);
  if ( v12 != a1 + 24 )
  {
    v13 = "__";
    v147 = 0;
    if ( v143 )
      v13 = "_";
    s = v13;
    v14 = "__JustMyCode_Default";
    if ( v143 )
      v14 = "_JustMyCode_Default";
    v11 = 0;
    v132 = (unsigned __int8 *)v14;
    while ( 1 )
    {
      v30 = v12 - 56;
      if ( !v12 )
        v30 = 0;
      if ( sub_B2FC80(v30) )
        goto LABEL_25;
      v31 = sub_B92180(v30);
      v149 = (_BYTE *)v31;
      if ( !v31 )
        goto LABEL_25;
      if ( !v153 )
        break;
      v15 = 1;
      v16 = 0;
      v17 = (v153 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
      v18 = &v151[2 * v17];
      v19 = *v18;
      if ( v31 != *v18 )
      {
        while ( v19 != -4096 )
        {
          if ( v19 == -8192 && !v16 )
            v16 = v18;
          v17 = (v153 - 1) & (v15 + v17);
          v18 = &v151[2 * v17];
          v19 = *v18;
          if ( v31 == *v18 )
            goto LABEL_14;
          ++v15;
        }
        if ( !v16 )
          v16 = v18;
        ++v150;
        v32 = v152 + 1;
        if ( 4 * ((int)v152 + 1) < 3 * v153 )
        {
          if ( v153 - HIDWORD(v152) - v32 <= v153 >> 3 )
          {
            sub_2DF2860((__int64)&v150, v153);
            if ( !v153 )
            {
LABEL_230:
              LODWORD(v152) = v152 + 1;
              BUG();
            }
            v31 = (__int64)v149;
            v109 = 0;
            v110 = 1;
            v111 = (v153 - 1) & (((unsigned int)v149 >> 9) ^ ((unsigned int)v149 >> 4));
            v32 = v152 + 1;
            v16 = &v151[2 * v111];
            v112 = (_BYTE *)*v16;
            if ( (_BYTE *)*v16 != v149 )
            {
              while ( v112 != (_BYTE *)-4096LL )
              {
                if ( !v109 && v112 == (_BYTE *)-8192LL )
                  v109 = v16;
                v111 = (v153 - 1) & (v110 + v111);
                v16 = &v151[2 * v111];
                v112 = (_BYTE *)*v16;
                if ( v149 == (_BYTE *)*v16 )
                  goto LABEL_34;
                ++v110;
              }
              if ( v109 )
                v16 = v109;
            }
          }
          goto LABEL_34;
        }
LABEL_32:
        sub_2DF2860((__int64)&v150, 2 * v153);
        if ( !v153 )
          goto LABEL_230;
        v32 = v152 + 1;
        v33 = (v153 - 1) & (((unsigned int)v149 >> 9) ^ ((unsigned int)v149 >> 4));
        v16 = &v151[2 * v33];
        v31 = *v16;
        if ( v149 != (_BYTE *)*v16 )
        {
          v129 = 1;
          v130 = 0;
          while ( v31 != -4096 )
          {
            if ( !v130 && v31 == -8192 )
              v130 = v16;
            v33 = (v153 - 1) & (v129 + v33);
            v16 = &v151[2 * v33];
            v31 = *v16;
            if ( v149 == (_BYTE *)*v16 )
              goto LABEL_34;
            ++v129;
          }
          v31 = (__int64)v149;
          if ( v130 )
            v16 = v130;
        }
LABEL_34:
        LODWORD(v152) = v32;
        if ( *v16 != -4096 )
          --HIDWORD(v152);
        *v16 = v31;
        v16[1] = 0;
        v146 = (__m128i **)(v16 + 1);
        goto LABEL_37;
      }
LABEL_14:
      v146 = (__m128i **)(v18 + 1);
      if ( v18[1] )
        goto LABEL_15;
LABEL_37:
      v34 = v149;
      v35 = v149;
      if ( *v149 == 16
        || ((v36 = *(v149 - 16), (v36 & 2) != 0)
          ? (v37 = (_QWORD *)*((_QWORD *)v149 - 4))
          : (v37 = &v149[-8 * ((v36 >> 2) & 0xF) - 16]),
            (v35 = (_BYTE *)*v37) != 0) )
      {
        v69 = *(v35 - 16);
        if ( (v69 & 2) != 0 )
          v70 = *((_QWORD *)v35 - 4);
        else
          v70 = (__int64)&v35[-8 * ((v69 >> 2) & 0xF) - 16];
        v39 = *(unsigned __int8 **)(v70 + 8);
        if ( v39 )
          v39 = (unsigned __int8 *)sub_B91420((__int64)v39);
        else
          v38 = 0;
      }
      else
      {
        v38 = 0;
        v39 = (unsigned __int8 *)byte_3F871B3;
      }
      v176 = (__m128i *)v39;
      LOWORD(v179) = 261;
      v177 = v38;
      if ( (unsigned __int8)sub_C81280((__int64)&v176, 3u) )
        goto LABEL_89;
      v114 = v34;
      if ( *v34 == 16
        || ((v115 = *(v34 - 16), (v115 & 2) == 0)
          ? (v116 = &v34[-8 * ((v115 >> 2) & 0xF) - 16])
          : (v116 = (_QWORD *)*((_QWORD *)v34 - 4)),
            (v114 = (_BYTE *)*v116) != 0) )
      {
        v117 = *(v114 - 16);
        v118 = (v117 & 2) != 0 ? (_BYTE *)*((_QWORD *)v114 - 4) : &v114[-8 * ((v117 >> 2) & 0xF) - 16];
        v119 = (char *)*((_QWORD *)v118 + 1);
        if ( v119 )
          v119 = (char *)sub_B91420((__int64)v119);
        else
          v120 = 0;
      }
      else
      {
        v120 = 0;
        v119 = (char *)byte_3F871B3;
      }
      v170 = (__m128i *)v119;
      v171 = v120;
      if ( sub_C931B0((__int64 *)&v170, "\\", 1u, 0) != -1 )
        goto LABEL_89;
      v121 = v34;
      if ( *v34 == 16
        || ((v122 = *(v34 - 16), (v122 & 2) == 0)
          ? (v123 = &v34[-8 * ((v122 >> 2) & 0xF) - 16])
          : (v123 = (_QWORD *)*((_QWORD *)v34 - 4)),
            (v121 = (_BYTE *)*v123) != 0) )
      {
        v124 = *(v121 - 16);
        v125 = (v124 & 2) != 0
             ? (const char **)*((_QWORD *)v121 - 4)
             : (const char **)&v121[-8 * ((v124 >> 2) & 0xF) - 16];
        v126 = *v125;
        if ( *v125 )
        {
          v126 = (const char *)sub_B91420((__int64)v126);
          v128 = v127;
        }
        else
        {
          v128 = 0;
        }
      }
      else
      {
        v128 = 0;
        v126 = byte_3F871B3;
      }
      v166 = (__m128i *)v126;
      v167 = v128;
      if ( sub_C931B0((__int64 *)&v166, "\\", 1u, 0) == -1 )
        v134 = 1;
      else
LABEL_89:
        v134 = 3;
      v71 = v34;
      if ( (*v34 == 16
         || ((v72 = *(v34 - 16), (v72 & 2) != 0)
           ? (v73 = (_QWORD *)*((_QWORD *)v34 - 4))
           : (v73 = &v34[-8 * ((v72 >> 2) & 0xF) - 16]),
             (v71 = (_BYTE *)*v73) != 0))
        && ((v74 = *(v71 - 16), (v74 & 2) == 0)
          ? (v75 = (__int64)&v71[-8 * ((v74 >> 2) & 0xF) - 16])
          : (v75 = *((_QWORD *)v71 - 4)),
            (v76 = *(_QWORD *)(v75 + 8)) != 0) )
      {
        v77 = sub_B91420(v76);
        v177 = 0;
        v79 = (const void *)v77;
        v80 = v78;
        v81 = v78;
        v176 = (__m128i *)&v178.m128i_u64[1];
        v178.m128i_i64[0] = 256;
        if ( v78 <= 0x100 )
        {
          if ( !v78 )
            goto LABEL_101;
          v82 = &v178.m128i_u8[8];
        }
        else
        {
          src = (void *)v77;
          v141 = v78;
          sub_C8D290((__int64)&v176, &v178.m128i_u64[1], v78, 1u, v78, v77);
          v81 = v141;
          v79 = src;
          v82 = &v176->m128i_u8[v177];
        }
        v142 = v81;
        memcpy(v82, v79, v80);
        v81 = v142;
      }
      else
      {
        v81 = 0;
        v177 = 0;
        v176 = (__m128i *)&v178.m128i_u64[1];
        v178.m128i_i64[0] = 256;
      }
LABEL_101:
      v177 += v81;
      v173 = 257;
      v169 = 257;
      v165 = 257;
      if ( *v34 == 16
        || ((v83 = *(v34 - 16), (v83 & 2) != 0)
          ? (v84 = (_BYTE **)*((_QWORD *)v34 - 4))
          : (v84 = (_BYTE **)&v34[-8 * ((v83 >> 2) & 0xF) - 16]),
            (v34 = *v84) != 0) )
      {
        v87 = *(v34 - 16);
        if ( (v87 & 2) != 0 )
          v88 = (char **)*((_QWORD *)v34 - 4);
        else
          v88 = (char **)&v34[-8 * ((v87 >> 2) & 0xF) - 16];
        v86 = *v88;
        if ( *v88 )
          v86 = (char *)sub_B91420((__int64)v86);
        else
          v85 = 0;
      }
      else
      {
        v85 = 0;
        v86 = (char *)byte_3F871B3;
      }
      v159 = v86;
      v162 = 261;
      v160 = v85;
      sub_C81390(&v176, (char *)v134, (__int64)&v159, (__int64)v163, (__int64)&v166, (__int64)&v170);
      sub_C83970((char **)&v176, v134);
      sub_C84CF0((unsigned __int8 **)&v176, 1u, v134);
      v159 = (char *)v161;
      v160 = 0;
      LOBYTE(v161[0]) = 0;
      v89 = sub_C80C60((__int64)v176, v177, v134);
      if ( v89 != v89 + v90 )
      {
        v131 = v30;
        v91 = (char *)(v89 + v90);
        v92 = (char *)v89;
        do
        {
          v93 = *v92;
          v94 = v160;
          v95 = 15;
          v96 = v159;
          v97 = v160 + 1;
          if ( *v92 == 46 )
            v93 = 64;
          if ( v159 != (char *)v161 )
            v95 = v161[0];
          if ( v97 > v95 )
          {
            srca = (void *)(v160 + 1);
            sub_2240BB0((unsigned __int64 *)&v159, v160, 0, 0, 1u);
            v96 = v159;
            v97 = (size_t)srca;
          }
          v96[v94] = v93;
          ++v92;
          v160 = v97;
          v159[v94 + 1] = 0;
        }
        while ( v91 != v92 );
        v30 = v131;
      }
      sub_C80DE0((__int64)&v176, v134);
      v98 = (unsigned __int8 *)v176;
      v99 = 5381;
      if ( v176 == (__m128i *)&v176->m128i_i8[v177] )
      {
        v102 = 53;
        v101 = 5381;
      }
      else
      {
        do
        {
          v100 = *v98++;
          v99 += v100 + 32 * v99;
        }
        while ( &v176->m128i_i8[v177] != (__int8 *)v98 );
        v101 = v99;
        if ( !v99 )
        {
          v172.m128i_i8[0] = 48;
          v102 = 48;
          v103 = &v172;
          goto LABEL_124;
        }
        v102 = a0123456789abcd_10[v99 & 0xF];
      }
      v103 = (__m128i *)&v172.m128i_i8[1];
LABEL_124:
      v104 = (__m128i *)((char *)v103 - 8);
      while ( 1 )
      {
        v103 = (__m128i *)((char *)v103 - 1);
        v101 >>= 4;
        v103->m128i_i8[0] = v102;
        if ( v104 == v103 )
          break;
        v102 = a0123456789abcd_10[v101 & 0xF];
      }
      v163[0] = (__int64)v164;
      sub_2DF2050(v163, v104, (__int64)v172.m128i_i64 + 1);
      v105 = strlen(s);
      v106 = sub_2241130((unsigned __int64 *)v163, 0, 0, s, v105);
      v166 = &v168;
      if ( (unsigned __int64 *)*v106 == v106 + 2 )
      {
        v168 = _mm_loadu_si128((const __m128i *)v106 + 1);
      }
      else
      {
        v166 = (__m128i *)*v106;
        v168.m128i_i64[0] = v106[2];
      }
      v167 = v106[1];
      *v106 = (unsigned __int64)(v106 + 2);
      v106[1] = 0;
      *((_BYTE *)v106 + 16) = 0;
      if ( v167 == 0x3FFFFFFFFFFFFFFFLL )
        goto LABEL_228;
      v107 = sub_2241490((unsigned __int64 *)&v166, "_", 1u);
      v170 = &v172;
      if ( (unsigned __int64 *)*v107 == v107 + 2 )
      {
        v172 = _mm_loadu_si128((const __m128i *)v107 + 1);
      }
      else
      {
        v170 = (__m128i *)*v107;
        v172.m128i_i64[0] = v107[2];
      }
      v171 = v107[1];
      *v107 = (unsigned __int64)(v107 + 2);
      v107[1] = 0;
      *((_BYTE *)v107 + 16) = 0;
      v108 = (__m128i *)sub_2241490((unsigned __int64 *)&v170, v159, v160);
      v156 = &v158;
      if ( (__m128i *)v108->m128i_i64[0] == &v108[1] )
      {
        v158 = _mm_loadu_si128(v108 + 1);
      }
      else
      {
        v156 = (__m128i *)v108->m128i_i64[0];
        v158.m128i_i64[0] = v108[1].m128i_i64[0];
      }
      v157 = v108->m128i_u64[1];
      v108->m128i_i64[0] = (__int64)v108[1].m128i_i64;
      v108->m128i_i64[1] = 0;
      v108[1].m128i_i8[0] = 0;
      if ( v170 != &v172 )
        j_j___libc_free_0((unsigned __int64)v170);
      if ( v166 != &v168 )
        j_j___libc_free_0((unsigned __int64)v166);
      if ( (_QWORD *)v163[0] != v164 )
        j_j___libc_free_0(v163[0]);
      if ( v159 != (char *)v161 )
        j_j___libc_free_0((unsigned __int64)v159);
      if ( v176 != (__m128i *)&v178.m128i_u64[1] )
        _libc_free((unsigned __int64)v176);
      v170 = (__m128i *)sub_BCB2B0(v139);
      v176 = (__m128i *)a1;
      v177 = (size_t)&v170;
      v178.m128i_i64[0] = (__int64)&v156;
      v178.m128i_i64[1] = (__int64)&v148;
      v179 = &v149;
      *v146 = (__m128i *)sub_BA8D20(
                           a1,
                           (__int64)v156,
                           v157,
                           (__int64)v170,
                           (__int64 (__fastcall *)(__int64))sub_2DF2100,
                           (__int64)&v176);
      if ( v156 != &v158 )
        j_j___libc_free_0((unsigned __int64)v156);
LABEL_15:
      if ( !v147 )
      {
        LOWORD(v179) = 259;
        v42 = *(__int64 **)a1;
        v43 = *(__int64 **)a1;
        v176 = (__m128i *)v132;
        v44 = (__int64 *)sub_BCB120(v43);
        v170 = (__m128i *)sub_BCE3C0(v42, 0);
        v45 = sub_BCF480(v44, &v170, 1, 0);
        v46 = (unsigned __int8 *)sub_BD2DA0(136);
        if ( v46 )
          sub_B2C3B0((__int64)v46, v45, 0, 0xFFFFFFFF, (__int64)&v176, a1);
        v46[32] = v46[32] & 0x3F | 0x80;
        sub_B2D3C0((__int64)v46, 0, 40);
        if ( v143 )
          sub_B2D3C0((__int64)v46, 0, 15);
        LOWORD(v179) = 257;
        v47 = sub_22077B0(0x50u);
        v48 = v47;
        if ( v47 )
          sub_AA4D50(v47, (__int64)v42, (__int64)&v176, (__int64)v46, 0);
        sub_B43C20((__int64)&v176, v48);
        v49 = sub_BD2C40(72, 0);
        if ( v49 )
          sub_B4BB80((__int64)v49, (__int64)v42, 0, 0, (__int64)v176, v177);
        if ( v135 == 3 )
        {
          v176 = (__m128i *)"__CheckForDebuggerJustMyCode";
          LOWORD(v179) = 259;
          sub_BD6B50(v46, (const char **)&v176);
          v147 = (__int64)v46;
          v113 = v46[32] & 0xF0 | 4;
          v46[32] = v113;
          if ( (v113 & 0x30) != 0 )
            v46[33] |= 0x40u;
          goto LABEL_16;
        }
        v50 = (__int64 *)sub_BCB120(v139);
        v176 = (__m128i *)sub_BCE3C0(v139, 0);
        v51 = sub_BCF480(v50, &v176, 1, 0);
        sub_BA8CA0(a1, (__int64)"__CheckForDebuggerJustMyCode", 0x1Cu, v51);
        v147 = v52;
        *(_BYTE *)(v52 + 32) = *(_BYTE *)(v52 + 32) & 0x3F | 0x80;
        sub_B2D3C0(v52, 0, 40);
        if ( v143 )
        {
          *(_WORD *)(v147 + 2) = *(_WORD *)(v147 + 2) & 0xC00F | 0x410;
          sub_B2D3C0(v147, 0, 15);
        }
        v53 = sub_BD5D20((__int64)v46);
        v176 = (__m128i *)v46;
        v55 = v54;
        v56 = (char *)v53;
        sub_2A413E0((__int64 **)a1, (unsigned __int64 *)&v176, 1);
        v57 = sub_BAA410(a1, v56, v55);
        *(_DWORD *)(v57 + 8) = 0;
        sub_B2F990((__int64)v46, v57, v58, v59);
        v170 = &v172;
        if ( v56 )
        {
          sub_2DF1FA0((__int64 *)&v170, v56, (__int64)&v56[v55]);
        }
        else
        {
          v171 = 0;
          v172.m128i_i8[0] = 0;
        }
        v154[0] = (__int64)v155;
        sub_2DF1FA0(v154, "/alternatename:", (__int64)"");
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v154[1]) <= 0x1B )
          goto LABEL_228;
        v60 = (__m128i *)sub_2241490((unsigned __int64 *)v154, "__CheckForDebuggerJustMyCode", 0x1Cu);
        v156 = &v158;
        if ( (__m128i *)v60->m128i_i64[0] == &v60[1] )
        {
          v158 = _mm_loadu_si128(v60 + 1);
        }
        else
        {
          v156 = (__m128i *)v60->m128i_i64[0];
          v158.m128i_i64[0] = v60[1].m128i_i64[0];
        }
        v157 = v60->m128i_u64[1];
        v60->m128i_i64[0] = (__int64)v60[1].m128i_i64;
        v60->m128i_i64[1] = 0;
        v60[1].m128i_i8[0] = 0;
        if ( v157 == 0x3FFFFFFFFFFFFFFFLL )
LABEL_228:
          sub_4262D8((__int64)"basic_string::append");
        v61 = sub_2241490((unsigned __int64 *)&v156, "=", 1u);
        v176 = &v178;
        if ( (unsigned __int64 *)*v61 == v61 + 2 )
        {
          v178 = _mm_loadu_si128((const __m128i *)v61 + 1);
        }
        else
        {
          v176 = (__m128i *)*v61;
          v178.m128i_i64[0] = v61[2];
        }
        v177 = v61[1];
        *v61 = (unsigned __int64)(v61 + 2);
        v61[1] = 0;
        *((_BYTE *)v61 + 16) = 0;
        v62 = 15;
        v63 = 15;
        if ( v176 != &v178 )
          v63 = v178.m128i_i64[0];
        if ( v177 + v171 <= v63 )
          goto LABEL_72;
        if ( v170 != &v172 )
          v62 = v172.m128i_i64[0];
        if ( v177 + v171 <= v62 )
        {
          v64 = sub_2241130((unsigned __int64 *)&v170, 0, 0, v176, v177);
          v166 = &v168;
          v65 = (__m128i *)*v64;
          v66 = v64 + 2;
          if ( (unsigned __int64 *)*v64 == v64 + 2 )
            goto LABEL_201;
LABEL_73:
          v166 = v65;
          v168.m128i_i64[0] = v64[2];
        }
        else
        {
LABEL_72:
          v64 = sub_2241490((unsigned __int64 *)&v176, v170->m128i_i8, v171);
          v166 = &v168;
          v65 = (__m128i *)*v64;
          v66 = v64 + 2;
          if ( (unsigned __int64 *)*v64 != v64 + 2 )
            goto LABEL_73;
LABEL_201:
          v168 = _mm_loadu_si128((const __m128i *)v64 + 1);
        }
        v167 = v64[1];
        *v64 = (unsigned __int64)v66;
        v64[1] = 0;
        *((_BYTE *)v64 + 16) = 0;
        if ( v176 != &v178 )
          j_j___libc_free_0((unsigned __int64)v176);
        if ( v156 != &v158 )
          j_j___libc_free_0((unsigned __int64)v156);
        if ( (_QWORD *)v154[0] != v155 )
          j_j___libc_free_0(v154[0]);
        if ( v170 != &v172 )
          j_j___libc_free_0((unsigned __int64)v170);
        v176 = (__m128i *)sub_B9B140(v139, v166, v167);
        v67 = sub_B9C770(v139, (__int64 *)&v176, (__int64 *)1, 0, 1);
        v68 = sub_BA8E40(a1, "llvm.linker.options", 0x13u);
        sub_B979A0(v68, v67);
        if ( v166 != &v168 )
          j_j___libc_free_0((unsigned __int64)v166);
      }
LABEL_16:
      v20 = *(_QWORD *)(v30 + 80);
      if ( v20 )
        v20 -= 24;
      v22 = sub_AA5190(v20);
      if ( v22 )
      {
        v23 = v21;
        v145 = HIBYTE(v21);
      }
      else
      {
        v145 = 0;
        v23 = 0;
      }
      v140 = v22;
      LOWORD(v179) = 257;
      v166 = *v146;
      v24 = (__int64 *)sub_BCB120(v139);
      v170 = (__m128i *)sub_BCE3C0(v139, 0);
      v25 = sub_BCF480(v24, &v170, 1, 0);
      v26 = sub_BD2C40(88, 2u);
      if ( v26 )
      {
        v144 = v144 & 0xE0000000 | 2;
        LOBYTE(v27) = v23;
        v28 = (__int64)v26;
        HIBYTE(v27) = v145;
        sub_B44260((__int64)v26, **(_QWORD **)(v25 + 16), 56, v144, v140, v27);
        v26[9] = 0;
        sub_B4A290((__int64)v26, v25, v147, (__int64 *)&v166, 1, (__int64)&v176, 0, 0);
      }
      else
      {
        v28 = 0;
      }
      v29 = (__int64 *)sub_BD5C60(v28);
      v26[9] = sub_A7A090(v26 + 9, v29, 1, 40);
      if ( v143 )
      {
        *((_WORD *)v26 + 1) = *((_WORD *)v26 + 1) & 0xF003 | 0x104;
        v41 = (__int64 *)sub_BD5C60(v28);
        v26[9] = sub_A7A090(v26 + 9, v41, 1, 15);
      }
      v11 = 1;
LABEL_25:
      v12 = *(_QWORD *)(v12 + 8);
      if ( a1 + 24 == v12 )
      {
        v8 = (__int64)v151;
        v9 = 2LL * v153;
        goto LABEL_43;
      }
    }
    ++v150;
    goto LABEL_32;
  }
LABEL_43:
  sub_C7D6A0(v8, v9 * 8, 8);
  if ( v174 != v175 )
    j_j___libc_free_0((unsigned __int64)v174);
  return v11;
}
