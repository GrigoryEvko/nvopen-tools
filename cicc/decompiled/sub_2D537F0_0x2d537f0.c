// Function: sub_2D537F0
// Address: 0x2d537f0
//
__int64 *__fastcall sub_2D537F0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v7; // rax
  bool v8; // zf
  __int64 *v9; // r13
  __int64 *v10; // r12
  __m128i v11; // rax
  unsigned __int64 v12; // rdi
  char *v13; // r13
  size_t v14; // r14
  int v15; // eax
  __int64 v16; // r9
  unsigned int v17; // r15d
  _QWORD *v18; // r8
  __int64 v19; // rax
  _QWORD *v20; // r8
  _QWORD *v21; // rbx
  __int64 v22; // rax
  unsigned __int64 v23; // rax
  __int64 v24; // r9
  __int64 v25; // rcx
  unsigned __int64 v26; // rdx
  unsigned __int64 v27; // rsi
  unsigned __int64 v28; // rdx
  __int64 v29; // r8
  unsigned __int64 *v30; // rax
  char *v31; // rax
  char *v32; // r14
  size_t v33; // rbx
  char *v34; // rdi
  size_t v35; // rax
  unsigned __int64 *v36; // r8
  __int64 v37; // rbx
  __int64 *v38; // r12
  __int64 v39; // r14
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // r8
  __int64 v43; // r9
  unsigned int v44; // r10d
  __int64 v45; // rdx
  __int64 v46; // rcx
  unsigned int *v47; // r15
  unsigned int *v48; // rax
  __int64 *v49; // r12
  void *v50; // rdi
  __int64 v52; // rax
  _QWORD *v53; // r15
  char v54; // r14
  __int64 v55; // rax
  unsigned int v56; // r15d
  __int64 v57; // r9
  unsigned __int64 v58; // rax
  unsigned __int64 v59; // rcx
  int v60; // edx
  _DWORD *v61; // rax
  unsigned int *v62; // rbx
  char v63; // r12
  __int64 v64; // rax
  __int64 v65; // rax
  __int64 v66; // rdx
  __int64 v67; // r11
  unsigned __int64 v68; // rbx
  _QWORD *v69; // rax
  __int64 v70; // rax
  __int64 v71; // rdx
  char v72; // r15
  __int64 v73; // rax
  __m128i v74; // xmm1
  unsigned __int64 v75; // r15
  __int64 v76; // r13
  __int64 v77; // rax
  __int64 v78; // r13
  size_t v79; // r14
  const void *v80; // r13
  int v81; // eax
  int v82; // eax
  __int64 v83; // rdx
  __int64 *v84; // rax
  __int64 v85; // rbx
  const __m128i *v86; // r14
  const char *v87; // rax
  const void *v88; // r13
  size_t v89; // r15
  int v90; // eax
  unsigned int v91; // r8d
  __int64 *v92; // r9
  __int64 v93; // rax
  unsigned int v94; // r8d
  __int64 *v95; // r9
  __int64 v96; // rcx
  __int64 v97; // rax
  size_t v98; // r13
  const void *v99; // r14
  int v100; // eax
  int v101; // eax
  __int64 v102; // rdx
  __int64 *v103; // rax
  __int64 v104; // rax
  size_t v105; // r13
  const void *v106; // r14
  int v107; // eax
  int v108; // eax
  __int64 v109; // rdx
  __int64 *v110; // rax
  __int64 v111; // rax
  size_t v112; // r14
  const void *v113; // r13
  int v114; // eax
  int v115; // eax
  __int64 v116; // rdx
  __int64 *v117; // rax
  __int64 v118; // rax
  __int64 *v119; // r13
  size_t v120; // r14
  const void *v121; // rbx
  int v122; // eax
  int v123; // eax
  __int64 v124; // rdx
  __int64 *v125; // rax
  __int64 v126; // rax
  __int64 v127; // rax
  size_t v128; // r14
  const void *v129; // rbx
  int v130; // eax
  int v131; // eax
  __int64 v132; // rdx
  __int64 *v133; // rax
  __int64 v134; // rax
  size_t v135; // r14
  const void *v136; // rbx
  int v137; // eax
  int v138; // eax
  __int64 v139; // rdx
  __int64 *v140; // rax
  __int64 v141; // rax
  __int128 v142; // [rsp-30h] [rbp-2B0h]
  __int128 v143; // [rsp-30h] [rbp-2B0h]
  __int128 v144; // [rsp-30h] [rbp-2B0h]
  __int128 v145; // [rsp-30h] [rbp-2B0h]
  __int128 v146; // [rsp-20h] [rbp-2A0h]
  __int128 v147; // [rsp-20h] [rbp-2A0h]
  __int128 v148; // [rsp-20h] [rbp-2A0h]
  __int64 *v149; // [rsp+10h] [rbp-270h]
  unsigned int v151; // [rsp+28h] [rbp-258h]
  __int64 *v152; // [rsp+28h] [rbp-258h]
  unsigned int v153; // [rsp+38h] [rbp-248h]
  __int64 v154; // [rsp+38h] [rbp-248h]
  __int64 *v155; // [rsp+40h] [rbp-240h]
  unsigned int v156; // [rsp+48h] [rbp-238h]
  __int64 v157; // [rsp+48h] [rbp-238h]
  char *v158; // [rsp+50h] [rbp-230h]
  const char *v159; // [rsp+58h] [rbp-228h]
  __int64 *v160; // [rsp+58h] [rbp-228h]
  __int64 v161; // [rsp+60h] [rbp-220h]
  _QWORD *v162; // [rsp+60h] [rbp-220h]
  _QWORD *v163; // [rsp+60h] [rbp-220h]
  const char *v164; // [rsp+60h] [rbp-220h]
  unsigned int v165; // [rsp+60h] [rbp-220h]
  unsigned int v166; // [rsp+60h] [rbp-220h]
  __int64 v167; // [rsp+60h] [rbp-220h]
  _QWORD *v168; // [rsp+68h] [rbp-218h]
  __int64 *v169; // [rsp+68h] [rbp-218h]
  unsigned __int64 v170; // [rsp+68h] [rbp-218h]
  __m128i v171; // [rsp+70h] [rbp-210h] BYREF
  __m128i v172; // [rsp+80h] [rbp-200h] BYREF
  char *v173; // [rsp+90h] [rbp-1F0h]
  unsigned __int64 v174; // [rsp+98h] [rbp-1E8h]
  __int64 v175; // [rsp+A0h] [rbp-1E0h]
  _QWORD v176[2]; // [rsp+B0h] [rbp-1D0h] BYREF
  char *v177; // [rsp+C0h] [rbp-1C0h]
  __int64 v178; // [rsp+C8h] [rbp-1B8h]
  __int64 v179; // [rsp+D0h] [rbp-1B0h]
  unsigned __int64 v180[2]; // [rsp+E0h] [rbp-1A0h] BYREF
  char *v181; // [rsp+F0h] [rbp-190h]
  unsigned __int64 v182; // [rsp+F8h] [rbp-188h]
  __int64 v183; // [rsp+100h] [rbp-180h]
  unsigned int *v184; // [rsp+110h] [rbp-170h] BYREF
  __int64 v185; // [rsp+118h] [rbp-168h]
  _BYTE v186[16]; // [rsp+120h] [rbp-160h] BYREF
  __int64 v187; // [rsp+130h] [rbp-150h] BYREF
  __int64 v188; // [rsp+138h] [rbp-148h] BYREF
  unsigned __int64 v189; // [rsp+140h] [rbp-140h]
  __int64 *v190; // [rsp+148h] [rbp-138h]
  __int64 *v191; // [rsp+150h] [rbp-130h]
  __int64 v192; // [rsp+158h] [rbp-128h]
  const char *v193; // [rsp+160h] [rbp-120h] BYREF
  __int64 v194; // [rsp+168h] [rbp-118h]
  __int128 v195; // [rsp+170h] [rbp-110h] BYREF
  __int64 v196; // [rsp+180h] [rbp-100h]
  void *dest; // [rsp+1B0h] [rbp-D0h] BYREF
  size_t n; // [rsp+1B8h] [rbp-C8h]
  unsigned __int64 v199; // [rsp+1C0h] [rbp-C0h] BYREF
  _BYTE v200[184]; // [rsp+1C8h] [rbp-B8h] BYREF

  v6 = *(unsigned int *)(a2 + 104);
  v149 = (__int64 *)(a2 + 96);
  v7 = *(_QWORD *)(a2 + 96);
  v8 = *(_BYTE *)(a2 + 40) == 0;
  LODWORD(v188) = 0;
  v189 = 0;
  v9 = (__int64 *)(v7 + 8 * v6);
  v184 = (unsigned int *)v186;
  v185 = 0x400000000LL;
  v190 = &v188;
  v191 = &v188;
  v192 = 0;
  if ( v8 )
  {
    v12 = 0;
    goto LABEL_6;
  }
  v156 = 0;
  v10 = (__int64 *)a2;
  while ( 1 )
  {
    v11 = *(__m128i *)(v10 + 7);
    v171 = v11;
    if ( *(_BYTE *)v11.m128i_i64[0] == 64 )
      goto LABEL_21;
    if ( !v11.m128i_i64[1]
      || *(_BYTE *)v11.m128i_i64[0] != 33
      || (v171.m128i_i64[0] = v11.m128i_i64[0] + 1, v171.m128i_i64[1] = v11.m128i_i64[1] - 1, v11.m128i_i64[1] == 1) )
    {
LABEL_5:
      v12 = v189;
LABEL_6:
      *a1 = 1;
      goto LABEL_53;
    }
    if ( *(_BYTE *)(v11.m128i_i64[0] + 1) == 33 )
      break;
    LOBYTE(dest) = 32;
    v23 = sub_C931B0(v171.m128i_i64, &dest, 1u, 0);
    if ( v23 == -1 )
    {
      v74 = _mm_load_si128(&v171);
      v173 = 0;
      v172 = v74;
LABEL_92:
      v174 = 0;
      dest = v200;
      n = 0;
      v199 = 128;
      goto LABEL_93;
    }
    v25 = v171.m128i_i64[1];
    v26 = v23 + 1;
    if ( v23 + 1 > v171.m128i_i64[1] )
    {
      v172.m128i_i64[0] = v171.m128i_i64[0];
      if ( v23 > v171.m128i_i64[1] )
        v23 = v171.m128i_u64[1];
      v172.m128i_i64[1] = v23;
      v173 = (char *)(v171.m128i_i64[0] + v171.m128i_i64[1]);
      goto LABEL_92;
    }
    v172.m128i_i64[0] = v171.m128i_i64[0];
    n = 0;
    v27 = v171.m128i_i64[1] - v26;
    v28 = v171.m128i_i64[0] + v26;
    if ( v23 > v171.m128i_i64[1] )
      v23 = v171.m128i_u64[1];
    v173 = (char *)v28;
    v174 = v27;
    v172.m128i_i64[1] = v23;
    dest = v200;
    v199 = 128;
    if ( v27 > 1 )
    {
      if ( *(_WORD *)v28 != 15693 )
        goto LABEL_32;
      v31 = sub_C81F40((char *)(v28 + 2), v27 - 2, 0);
      n = 0;
      v32 = v31;
      v33 = v28;
      if ( v28 > v199 )
      {
        sub_C8D290((__int64)&dest, v200, v28, 1u, v29, v24);
        v35 = n;
        if ( v33 )
        {
          v34 = (char *)dest + n;
          goto LABEL_36;
        }
      }
      else
      {
        v34 = (char *)dest;
        if ( !v28 )
          goto LABEL_38;
LABEL_36:
        memcpy(v34, v32, v33);
        v35 = n;
      }
      n = v35 + v33;
      if ( !(v35 + v33) )
      {
LABEL_38:
        v30 = (unsigned __int64 *)"empty module name specifier";
        v193 = "empty module name specifier";
        LOWORD(v196) = 259;
LABEL_33:
        *((_QWORD *)&v142 + 1) = v194;
        *(_QWORD *)&v142 = v30;
        sub_2D507F0(a1, v10, v28, v25, v29, v24, v142, v195, v196);
        goto LABEL_146;
      }
      goto LABEL_93;
    }
    if ( v27 )
    {
      v27 = 1;
LABEL_32:
      v29 = 770;
      v180[0] = (unsigned __int64)"unknown string found: '";
      v30 = v180;
      LOWORD(v183) = 1283;
      v181 = (char *)v28;
      v182 = v27;
      v193 = (const char *)v180;
      *(_QWORD *)&v195 = "'";
      LOWORD(v196) = 770;
      goto LABEL_33;
    }
LABEL_93:
    v193 = (const char *)&v195;
    v194 = 0x400000000LL;
    sub_C93960((char **)&v172, (__int64)&v193, 47, -1, 1, v24);
    v75 = (unsigned __int64)v193;
    v76 = 16LL * (unsigned int)v194;
    v159 = &v193[v76];
    v77 = v76 >> 4;
    v78 = v76 >> 6;
    if ( !v78 )
      goto LABEL_132;
    v164 = &v193[64 * v78];
    while ( 1 )
    {
      v79 = *(_QWORD *)(v75 + 8);
      v80 = *(const void **)v75;
      v81 = sub_C92610();
      v82 = sub_C92860(v10 + 9, v80, v79, v81);
      if ( v82 != -1 )
      {
        v83 = v10[9];
        v84 = (__int64 *)(v83 + 8LL * v82);
        if ( v84 != (__int64 *)(v83 + 8LL * *((unsigned int *)v10 + 20)) )
        {
          if ( !n )
            goto LABEL_98;
          v97 = *v84;
          if ( n == *(_QWORD *)(v97 + 16) && !memcmp(*(const void **)(v97 + 8), dest, n) )
            goto LABEL_98;
        }
      }
      v98 = *(_QWORD *)(v75 + 24);
      v99 = *(const void **)(v75 + 16);
      v170 = v75 + 16;
      v100 = sub_C92610();
      v101 = sub_C92860(v10 + 9, v99, v98, v100);
      if ( v101 != -1 )
      {
        v102 = v10[9];
        v103 = (__int64 *)(v102 + 8LL * v101);
        if ( v103 != (__int64 *)(v102 + 8LL * *((unsigned int *)v10 + 20)) )
        {
          if ( !n )
            break;
          v104 = *v103;
          if ( n == *(_QWORD *)(v104 + 16) && !memcmp(*(const void **)(v104 + 8), dest, n) )
            break;
        }
      }
      v105 = *(_QWORD *)(v75 + 40);
      v106 = *(const void **)(v75 + 32);
      v170 = v75 + 32;
      v107 = sub_C92610();
      v108 = sub_C92860(v10 + 9, v106, v105, v107);
      if ( v108 != -1 )
      {
        v109 = v10[9];
        v110 = (__int64 *)(v109 + 8LL * v108);
        if ( v110 != (__int64 *)(v109 + 8LL * *((unsigned int *)v10 + 20)) )
        {
          if ( !n )
            break;
          v111 = *v110;
          if ( n == *(_QWORD *)(v111 + 16) && !memcmp(*(const void **)(v111 + 8), dest, n) )
            break;
        }
      }
      v112 = *(_QWORD *)(v75 + 56);
      v113 = *(const void **)(v75 + 48);
      v170 = v75 + 48;
      v114 = sub_C92610();
      v115 = sub_C92860(v10 + 9, v113, v112, v114);
      if ( v115 != -1 )
      {
        v116 = v10[9];
        v117 = (__int64 *)(v116 + 8LL * v115);
        if ( v117 != (__int64 *)(v116 + 8LL * *((unsigned int *)v10 + 20)) )
        {
          if ( !n )
            break;
          v118 = *v117;
          if ( n == *(_QWORD *)(v118 + 16) && !memcmp(*(const void **)(v118 + 8), dest, n) )
            break;
        }
      }
      v75 += 64LL;
      if ( v164 == (const char *)v75 )
      {
        v77 = (__int64)&v159[-v75] >> 4;
LABEL_132:
        if ( v77 != 2 )
        {
          if ( v77 != 3 )
          {
            if ( v77 != 1 )
              goto LABEL_116;
            v119 = v10 + 9;
LABEL_136:
            v120 = *(_QWORD *)(v75 + 8);
            v121 = *(const void **)v75;
            v122 = sub_C92610();
            v123 = sub_C92860(v119, v121, v120, v122);
            if ( v123 == -1
              || (v124 = v10[9],
                  v125 = (__int64 *)(v124 + 8LL * v123),
                  v125 == (__int64 *)(v124 + 8LL * *((unsigned int *)v10 + 20)))
              || n && ((v126 = *v125, n != *(_QWORD *)(v126 + 16)) || memcmp(*(const void **)(v126 + 8), dest, n)) )
            {
LABEL_116:
              v9 = (__int64 *)(v10[12] + 8LL * *((unsigned int *)v10 + 26));
              if ( v193 != (const char *)&v195 )
                _libc_free((unsigned __int64)v193);
              if ( dest != v200 )
                _libc_free((unsigned __int64)dest);
              goto LABEL_21;
            }
            goto LABEL_98;
          }
          v135 = *(_QWORD *)(v75 + 8);
          v136 = *(const void **)v75;
          v119 = v10 + 9;
          v137 = sub_C92610();
          v138 = sub_C92860(v10 + 9, v136, v135, v137);
          if ( v138 == -1
            || (v139 = v10[9],
                v140 = (__int64 *)(v139 + 8LL * v138),
                v140 == (__int64 *)(v139 + 8LL * *((unsigned int *)v10 + 20)))
            || n && ((v141 = *v140, n != *(_QWORD *)(v141 + 16)) || memcmp(*(const void **)(v141 + 8), dest, n)) )
          {
            v75 += 16LL;
LABEL_158:
            v128 = *(_QWORD *)(v75 + 8);
            v129 = *(const void **)v75;
            v130 = sub_C92610();
            v131 = sub_C92860(v119, v129, v128, v130);
            if ( v131 == -1
              || (v132 = v10[9],
                  v133 = (__int64 *)(v132 + 8LL * v131),
                  v133 == (__int64 *)(v132 + 8LL * *((unsigned int *)v10 + 20)))
              || n && ((v134 = *v133, n != *(_QWORD *)(v134 + 16)) || memcmp(*(const void **)(v134 + 8), dest, n)) )
            {
              v75 += 16LL;
              goto LABEL_136;
            }
          }
LABEL_98:
          if ( v159 == (const char *)v75 )
            goto LABEL_116;
          goto LABEL_99;
        }
        v119 = v10 + 9;
        goto LABEL_158;
      }
    }
    if ( v159 == (const char *)v170 )
      goto LABEL_116;
LABEL_99:
    v85 = 1;
    v169 = v10 + 15;
    if ( (unsigned int)v194 > 1 )
    {
      while ( 1 )
      {
        v86 = (const __m128i *)v193;
        v87 = &v193[16 * v85];
        v88 = *(const void **)v87;
        v89 = *((_QWORD *)v87 + 1);
        v90 = sub_C92610();
        v91 = sub_C92740((__int64)v169, v88, v89, v90);
        v92 = (__int64 *)(v10[15] + 8LL * v91);
        if ( !*v92 )
          goto LABEL_105;
        if ( *v92 == -8 )
          break;
LABEL_101:
        if ( ++v85 >= (unsigned __int64)(unsigned int)v194 )
          goto LABEL_7;
      }
      --*((_DWORD *)v10 + 34);
LABEL_105:
      v160 = v92;
      v165 = v91;
      v93 = sub_C7D670(v89 + 25, 8);
      v94 = v165;
      v95 = v160;
      v96 = v93;
      if ( v89 )
      {
        v157 = v93;
        memcpy((void *)(v93 + 24), v88, v89);
        v94 = v165;
        v95 = v160;
        v96 = v157;
      }
      *(_BYTE *)(v96 + v89 + 24) = 0;
      *(_QWORD *)v96 = v89;
      *(__m128i *)(v96 + 8) = _mm_loadu_si128(v86);
      *v95 = v96;
      ++*((_DWORD *)v10 + 33);
      sub_C929D0(v169, v94);
      goto LABEL_101;
    }
LABEL_7:
    v13 = *(char **)v193;
    v14 = *((_QWORD *)v193 + 1);
    v15 = sub_C92610();
    v17 = sub_C92740((__int64)v149, v13, v14, v15);
    v18 = (_QWORD *)(v10[12] + 8LL * v17);
    if ( *v18 )
    {
      if ( *v18 != -8 )
      {
        LOWORD(v179) = 1283;
        v176[0] = "duplicate profile for function '";
        v177 = *(char **)v193;
        v127 = *((_QWORD *)v193 + 1);
        v181 = "'";
        v178 = v127;
        LOWORD(v183) = 770;
        *((_QWORD *)&v147 + 1) = v182;
        *(_QWORD *)&v147 = "'";
        *((_QWORD *)&v144 + 1) = v180[1];
        *(_QWORD *)&v144 = v176;
        v180[0] = (unsigned __int64)v176;
        sub_2D507F0(a1, v10, (__int64)v177, 770, (__int64)v18, v16, v144, v147, v183);
        if ( v193 != (const char *)&v195 )
          _libc_free((unsigned __int64)v193);
LABEL_146:
        v50 = dest;
        if ( dest != v200 )
          goto LABEL_51;
        goto LABEL_52;
      }
      --*((_DWORD *)v10 + 28);
    }
    v168 = v18;
    v19 = sub_C7D670(v14 + 153, 8);
    v20 = v168;
    v21 = (_QWORD *)v19;
    if ( v14 )
    {
      memcpy((void *)(v19 + 152), v13, v14);
      v20 = v168;
    }
    *((_BYTE *)v21 + v14 + 152) = 0;
    *v21 = v14;
    memset(v21 + 1, 0, 0x90u);
    v21[1] = v21 + 3;
    v21[2] = 0x300000000LL;
    v21[9] = v21 + 11;
    v21[10] = 0x100000000LL;
    *v20 = v21;
    ++*((_DWORD *)v10 + 27);
    v9 = (__int64 *)(v10[12] + 8LL * (unsigned int)sub_C929D0(v149, v17));
    if ( !*v9 || *v9 == -8 )
    {
      do
      {
        do
        {
          v22 = v9[1];
          ++v9;
        }
        while ( v22 == -8 );
      }
      while ( !v22 );
    }
    LODWORD(v185) = 0;
    sub_2D50620(v189);
    v189 = 0;
    v192 = 0;
    v190 = &v188;
    v191 = &v188;
    if ( v193 != (const char *)&v195 )
      _libc_free((unsigned __int64)v193);
    if ( dest != v200 )
      _libc_free((unsigned __int64)dest);
    v156 = 0;
LABEL_21:
    sub_C7C5C0((__int64)(v10 + 1));
    if ( !*((_BYTE *)v10 + 40) )
      goto LABEL_5;
  }
  v171.m128i_i64[0] = v11.m128i_i64[0] + 2;
  v11.m128i_i64[0] = v10[12];
  v171.m128i_i64[1] = v11.m128i_i64[1] - 2;
  if ( v9 == (__int64 *)(v11.m128i_i64[0] + 8LL * *((unsigned int *)v10 + 26)) )
    goto LABEL_21;
  dest = &v199;
  n = 0x400000000LL;
  sub_C93960((char **)&v171, (__int64)&dest, 32, -1, 1, a6);
  v36 = (unsigned __int64 *)dest;
  v158 = (char *)dest + 16 * (unsigned int)n;
  if ( v158 == dest )
  {
LABEL_83:
    ++v156;
    if ( v36 != &v199 )
      _libc_free((unsigned __int64)v36);
    goto LABEL_21;
  }
  v37 = 0;
  v155 = v10;
  v38 = (__int64 *)dest;
  while ( 2 )
  {
    v161 = *v38;
    v39 = v38[1];
    if ( sub_C93C90(*v38, v39, 0xAu, v180) )
    {
      v193 = "unsigned integer expected: '";
      v173 = "'";
      *(_QWORD *)&v195 = v161;
      LOWORD(v175) = 770;
      *((_QWORD *)&v148 + 1) = v174;
      *(_QWORD *)&v148 = "'";
      *((_QWORD *)&v145 + 1) = v172.m128i_i64[1];
      *(_QWORD *)&v145 = &v193;
      *((_QWORD *)&v195 + 1) = v39;
      LOWORD(v196) = 1283;
      v172.m128i_i64[0] = (__int64)&v193;
      sub_2D507F0(a1, v155, v40, v41, v42, v43, v145, v148, v175);
      v50 = dest;
      if ( dest != &v199 )
        goto LABEL_51;
      goto LABEL_52;
    }
    v44 = v180[0];
    LODWORD(v193) = v180[0];
    v45 = LODWORD(v180[0]);
    if ( !v192 )
    {
      v46 = (__int64)v184;
      v47 = &v184[(unsigned int)v185];
      if ( v184 == v47 )
      {
        if ( (unsigned int)v185 <= 3uLL )
          goto LABEL_67;
      }
      else
      {
        v48 = v184;
        while ( LODWORD(v180[0]) != *v48 )
        {
          if ( v47 == ++v48 )
            goto LABEL_66;
        }
        if ( v47 != v48 )
        {
          v49 = v155;
          goto LABEL_50;
        }
LABEL_66:
        if ( (unsigned int)v185 <= 3uLL )
        {
LABEL_67:
          if ( (unsigned __int64)(unsigned int)v185 + 1 > HIDWORD(v185) )
          {
            v166 = v180[0];
            sub_C8D5F0((__int64)&v184, v186, (unsigned int)v185 + 1LL, 4u, v42, v43);
            v44 = v166;
            v47 = &v184[(unsigned int)v185];
          }
          *v47 = v44;
          LODWORD(v185) = v185 + 1;
          goto LABEL_61;
        }
        v153 = v37;
        v62 = v184;
        v152 = v38;
        do
        {
          v65 = sub_B9AB10(&v187, (__int64)&v188, v62);
          if ( v66 )
          {
            v63 = v65 || (__int64 *)v66 == &v188 || *v62 < *(_DWORD *)(v66 + 32);
            v162 = (_QWORD *)v66;
            v64 = sub_22077B0(0x28u);
            *(_DWORD *)(v64 + 32) = *v62;
            sub_220F040(v63, v64, v162, &v188);
            ++v192;
          }
          ++v62;
        }
        while ( v47 != v62 );
        v37 = v153;
        v38 = v152;
      }
      LODWORD(v185) = 0;
      v70 = sub_B996D0((__int64)&v187, (unsigned int *)&v193);
      if ( v71 )
      {
        v72 = v70 || (__int64 *)v71 == &v188 || (unsigned int)v193 < *(_DWORD *)(v71 + 32);
        v163 = (_QWORD *)v71;
        v73 = sub_22077B0(0x28u);
        *(_DWORD *)(v73 + 32) = (_DWORD)v193;
        sub_220F040(v72, v73, v163, &v188);
        ++v192;
      }
      goto LABEL_61;
    }
    v151 = v180[0];
    v52 = sub_B996D0((__int64)&v187, (unsigned int *)&v193);
    v53 = (_QWORD *)v45;
    if ( v45 )
    {
      v54 = v52 || (__int64 *)v45 == &v188 || v151 < *(_DWORD *)(v45 + 32);
      v55 = sub_22077B0(0x28u);
      *(_DWORD *)(v55 + 32) = (_DWORD)v193;
      sub_220F040(v54, v55, v53, &v188);
      ++v192;
LABEL_61:
      v56 = v37 + 1;
      v57 = *v9;
      v58 = *(unsigned int *)(*v9 + 16);
      v59 = *(unsigned int *)(*v9 + 20);
      v60 = *(_DWORD *)(*v9 + 16);
      if ( v58 >= v59 )
      {
        v67 = LODWORD(v180[0]);
        v68 = v156 | (unsigned __int64)(v37 << 32);
        if ( v59 < v58 + 1 )
        {
          v154 = LODWORD(v180[0]);
          v167 = *v9;
          sub_C8D5F0(v57 + 8, (const void *)(v57 + 24), v58 + 1, 0x10u, v42, v57);
          v57 = v167;
          v67 = v154;
          v58 = *(unsigned int *)(v167 + 16);
        }
        v69 = (_QWORD *)(*(_QWORD *)(v57 + 8) + 16 * v58);
        v38 += 2;
        *v69 = v67;
        v69[1] = v68;
        ++*(_DWORD *)(v57 + 16);
        if ( v158 == (char *)v38 )
        {
LABEL_82:
          v10 = v155;
          v36 = (unsigned __int64 *)dest;
          goto LABEL_83;
        }
      }
      else
      {
        v61 = (_DWORD *)(*(_QWORD *)(v57 + 8) + 16 * v58);
        if ( v61 )
        {
          *v61 = v180[0];
          v61[1] = 0;
          v61[2] = v156;
          v61[3] = v37;
          v60 = *(_DWORD *)(v57 + 16);
        }
        v38 += 2;
        *(_DWORD *)(v57 + 16) = v60 + 1;
        if ( v158 == (char *)v38 )
          goto LABEL_82;
      }
      v37 = v56;
      continue;
    }
    break;
  }
  v49 = v155;
LABEL_50:
  v193 = "duplicate basic block id found '";
  LOWORD(v179) = 770;
  *(_QWORD *)&v195 = v161;
  v177 = "'";
  *((_QWORD *)&v146 + 1) = v178;
  *(_QWORD *)&v146 = "'";
  *((_QWORD *)&v143 + 1) = v176[1];
  *(_QWORD *)&v143 = &v193;
  v176[0] = &v193;
  *((_QWORD *)&v195 + 1) = v39;
  LOWORD(v196) = 1283;
  sub_2D507F0(a1, v49, v45, v46, v42, 1283, v143, v146, v179);
  v50 = dest;
  if ( dest == &v199 )
    goto LABEL_52;
LABEL_51:
  _libc_free((unsigned __int64)v50);
LABEL_52:
  v12 = v189;
LABEL_53:
  sub_2D50620(v12);
  if ( v184 != (unsigned int *)v186 )
    _libc_free((unsigned __int64)v184);
  return a1;
}
