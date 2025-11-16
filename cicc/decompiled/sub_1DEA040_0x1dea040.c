// Function: sub_1DEA040
// Address: 0x1dea040
//
__int64 __fastcall sub_1DEA040(__int64 a1, __int64 a2, void ***a3, __int64 a4)
{
  __int64 v5; // r14
  __int64 v6; // r12
  __int64 v8; // rax
  int v9; // r8d
  __int64 v10; // rcx
  unsigned int v11; // edx
  __int64 *v12; // r13
  __int64 v13; // rsi
  void ***v14; // rbx
  __int64 v15; // r12
  __int64 *v16; // rdi
  __int64 *v17; // r13
  __int64 *v18; // rcx
  __int64 v19; // rax
  __int64 *v20; // rbx
  __int64 *v21; // rax
  __int64 v22; // rdi
  unsigned __int32 v23; // r13d
  unsigned int v24; // eax
  int v25; // r15d
  unsigned __int8 v26; // al
  unsigned int v27; // edx
  __int64 v28; // r15
  char *v29; // rax
  __int64 v30; // rcx
  char *v31; // rax
  __int64 *v32; // rbx
  __int64 *v34; // r8
  __int64 *v35; // r9
  __int64 *v36; // r15
  __int64 v37; // rsi
  __int64 *v38; // rdi
  __int64 *v39; // rax
  __int64 *v40; // rcx
  _BYTE *v41; // rdx
  __int64 v42; // r14
  __int64 *v43; // rbx
  __int64 *v44; // rdx
  __int64 v45; // r8
  int v46; // esi
  int v47; // r9d
  unsigned __int32 v48; // eax
  __int64 v49; // rdi
  void ***v50; // r12
  _QWORD *v51; // rax
  char v52; // dl
  __int64 *v53; // r12
  __int64 *v54; // rdx
  __int64 *v55; // r13
  char v56; // bl
  _BYTE *v57; // rdi
  __int64 *v58; // rbx
  __int64 *v59; // r13
  __int64 *v60; // r8
  __int64 *v61; // r9
  __int64 v62; // rsi
  __int64 *v63; // rdi
  __int64 *v64; // rax
  __int64 *v65; // rcx
  __int64 v66; // rax
  _QWORD *v67; // r14
  char *v68; // r12
  __int64 v69; // rbx
  __int64 v70; // r13
  __int64 v71; // rax
  __int64 v72; // r15
  __int64 v73; // rcx
  _QWORD *v74; // rbx
  __int64 v75; // rsi
  __int64 v76; // r8
  __int64 v77; // rax
  __int64 v78; // r12
  __int64 v79; // r14
  __int64 v80; // rax
  __int64 v81; // rdx
  __int64 v82; // rax
  __int64 v83; // rsi
  __int64 v84; // r15
  int v85; // esi
  _QWORD *v86; // rdi
  _QWORD *v87; // rcx
  void **p_src; // r14
  __int64 v89; // rbx
  __int64 *v90; // rcx
  __int64 *v91; // r12
  void **v92; // rax
  __int64 *v93; // r14
  __int64 v94; // r12
  __int64 v95; // rcx
  int v96; // edx
  int v97; // r10d
  unsigned int v98; // eax
  __int64 v99; // rdi
  __int64 v100; // rsi
  int v101; // r8d
  int v102; // r9d
  __int64 v103; // rax
  __m128i *v104; // rax
  int v105; // edx
  __m128i *v106; // r15
  __int64 v107; // r12
  __int64 v108; // rax
  __m128i *v109; // rbx
  __m128i *v110; // r15
  __int64 v111; // rax
  __m128i *v112; // rbx
  _QWORD *v113; // rdx
  __int64 *v114; // r15
  __int64 v115; // r13
  __int64 v116; // rbx
  __int64 v117; // r8
  __int64 v118; // rdi
  unsigned int v119; // esi
  __int64 v120; // r11
  unsigned int v121; // r8d
  __int64 v122; // rax
  __int64 v123; // rdi
  unsigned __int64 v124; // rax
  unsigned __int64 v125; // rax
  unsigned __int64 v126; // rcx
  __int64 v127; // rax
  _QWORD *v128; // r8
  char *v129; // rsi
  _DWORD *v130; // rdi
  unsigned __int64 v131; // rbx
  unsigned __int64 v132; // rax
  __int64 v133; // rsi
  __int64 v134; // rsi
  int v135; // eax
  int v136; // esi
  int v137; // esi
  int v138; // edi
  __int64 v139; // rcx
  unsigned int j; // edx
  __int64 v141; // r9
  int v142; // edx
  int v143; // ecx
  __int64 v144; // rdx
  int v145; // ecx
  int v146; // esi
  int v147; // esi
  __int64 v148; // rdi
  int v149; // r8d
  unsigned int i; // r15d
  __int64 *v151; // rcx
  __int64 v152; // rdx
  unsigned int v153; // edx
  unsigned int v154; // r15d
  __int64 *v155; // [rsp+0h] [rbp-2F0h]
  unsigned int v156; // [rsp+0h] [rbp-2F0h]
  char *v157; // [rsp+0h] [rbp-2F0h]
  __int64 v158; // [rsp+8h] [rbp-2E8h]
  __int64 v159; // [rsp+8h] [rbp-2E8h]
  const void *v160; // [rsp+8h] [rbp-2E8h]
  int v161; // [rsp+8h] [rbp-2E8h]
  __int64 *v162; // [rsp+20h] [rbp-2D0h]
  int v163; // [rsp+28h] [rbp-2C8h]
  __m128i **v164; // [rsp+28h] [rbp-2C8h]
  __int64 *v165; // [rsp+30h] [rbp-2C0h]
  __int64 v166; // [rsp+30h] [rbp-2C0h]
  __int64 v167; // [rsp+30h] [rbp-2C0h]
  int v168; // [rsp+30h] [rbp-2C0h]
  unsigned int v169; // [rsp+38h] [rbp-2B8h]
  __int64 v170; // [rsp+38h] [rbp-2B8h]
  __int64 v171; // [rsp+38h] [rbp-2B8h]
  __int64 v172; // [rsp+40h] [rbp-2B0h]
  __int64 *v173; // [rsp+40h] [rbp-2B0h]
  __int64 *v174; // [rsp+40h] [rbp-2B0h]
  __int64 v175; // [rsp+40h] [rbp-2B0h]
  _QWORD *v176; // [rsp+40h] [rbp-2B0h]
  unsigned int v177; // [rsp+48h] [rbp-2A8h]
  __int64 v178; // [rsp+48h] [rbp-2A8h]
  __int64 *v179; // [rsp+48h] [rbp-2A8h]
  __int64 v180; // [rsp+48h] [rbp-2A8h]
  __int64 v181; // [rsp+48h] [rbp-2A8h]
  char *v182; // [rsp+48h] [rbp-2A8h]
  int v185; // [rsp+6Ch] [rbp-284h] BYREF
  __m128i *v186; // [rsp+70h] [rbp-280h] BYREF
  __int64 v187; // [rsp+78h] [rbp-278h] BYREF
  __m128i v188; // [rsp+80h] [rbp-270h] BYREF
  __m128i *v189; // [rsp+90h] [rbp-260h]
  __int64 *v190; // [rsp+A0h] [rbp-250h] BYREF
  __int64 v191; // [rsp+A8h] [rbp-248h]
  _BYTE v192[32]; // [rsp+B0h] [rbp-240h] BYREF
  __int64 v193; // [rsp+D0h] [rbp-220h] BYREF
  __int64 *v194; // [rsp+D8h] [rbp-218h]
  __int64 *v195; // [rsp+E0h] [rbp-210h]
  __int64 v196; // [rsp+E8h] [rbp-208h]
  int v197; // [rsp+F0h] [rbp-200h]
  _BYTE v198[40]; // [rsp+F8h] [rbp-1F8h] BYREF
  void *src; // [rsp+120h] [rbp-1D0h] BYREF
  __int64 v200; // [rsp+128h] [rbp-1C8h]
  _BYTE *v201; // [rsp+130h] [rbp-1C0h] BYREF
  __int64 v202; // [rsp+138h] [rbp-1B8h]
  int v203; // [rsp+140h] [rbp-1B0h]
  _BYTE v204[168]; // [rsp+148h] [rbp-1A8h] BYREF
  void *v205; // [rsp+1F0h] [rbp-100h]
  __int64 v206; // [rsp+1F8h] [rbp-F8h]
  _BYTE v207[192]; // [rsp+200h] [rbp-F0h] BYREF
  char v208; // [rsp+2C0h] [rbp-30h] BYREF

  v5 = a2;
  v6 = a1;
  sub_16AF710(&v185, qword_4FC5920[20], 0x64u);
  v190 = (__int64 *)v192;
  v191 = 0x400000000LL;
  v177 = sub_1DE7010(a1, a2, a3, a4, (__int64)&v190);
  v8 = *(unsigned int *)(a1 + 544);
  if ( (_DWORD)v8 )
  {
    v9 = 1;
    v10 = *(_QWORD *)(a1 + 528);
    v11 = (v8 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v12 = (__int64 *)(v10 + 24LL * v11);
    v13 = *v12;
    if ( v5 == *v12 )
    {
LABEL_3:
      if ( v12 != (__int64 *)(v10 + 24 * v8) )
      {
        src = (void *)v12[1];
        *v12 = -16;
        --*(_DWORD *)(a1 + 536);
        ++*(_DWORD *)(a1 + 540);
        v14 = (void ***)sub_1DE4FA0(a1 + 888, (__int64 *)&src)[1];
        if ( sub_1DD6970(v5, (__int64)src)
          && (!a4 || (unsigned int)sub_1DE8F90(a4, (__int64)src))
          && a3 != v14
          && **v14 == src )
        {
          v15 = v12[1];
          v16 = v190;
          goto LABEL_36;
        }
      }
    }
    else
    {
      while ( v13 != -8 )
      {
        v11 = (v8 - 1) & (v9 + v11);
        v12 = (__int64 *)(v10 + 24LL * v11);
        v13 = *v12;
        if ( v5 == *v12 )
          goto LABEL_3;
        ++v9;
      }
    }
  }
  v17 = *(__int64 **)(v5 + 96);
  v18 = *(__int64 **)(v5 + 88);
  v19 = (unsigned int)v191;
  if ( (unsigned int)(v17 - v18) != 2 || (_DWORD)v191 != 2 )
  {
LABEL_14:
    v16 = v190;
    src = &v201;
    v200 = 0x400000000LL;
    v165 = &v190[v19];
    if ( v190 == v165 )
    {
      v15 = 0;
      goto LABEL_36;
    }
    v20 = v190;
    v169 = 0;
    v172 = 0;
    do
    {
      v22 = *(_QWORD *)(v6 + 560);
      v23 = 0x80000000;
      v193 = *v20;
      v24 = sub_1DF1780(v22, v5, v193);
      v25 = v24;
      if ( v177 > v24 )
      {
        sub_16AF710(&v188, v24, v177);
        v23 = v188.m128i_i32[0];
      }
      v21 = sub_1DE4FA0(v6 + 888, &v193);
      if ( (unsigned __int8)sub_1DE73C0(v6, v5, v193, v21[1], v25, (__int64)a3, a4) )
      {
        if ( byte_4FC5020 )
        {
          if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v6 + 552) + 8LL) + 640LL) & 1) == 0 )
          {
            v158 = v193;
            v26 = sub_1F34100(v193);
            if ( (unsigned int)((__int64)(*(_QWORD *)(v158 + 96) - *(_QWORD *)(v158 + 88)) >> 3) != 1 )
            {
              if ( (unsigned __int8)sub_1F34340(v6 + 616, v26, v158) )
              {
                v27 = v200;
                v28 = v193;
                if ( (unsigned int)v200 >= HIDWORD(v200) )
                {
                  v156 = v200;
                  v124 = (((HIDWORD(v200) + 2LL) | (((unsigned __int64)HIDWORD(v200) + 2) >> 1)) >> 2)
                       | (HIDWORD(v200) + 2LL)
                       | (((unsigned __int64)HIDWORD(v200) + 2) >> 1);
                  v125 = (((v124 >> 4) | v124) >> 8) | (v124 >> 4) | v124;
                  v126 = (v125 | (v125 >> 16) | HIDWORD(v125)) + 1;
                  v127 = 0xFFFFFFFFLL;
                  if ( v126 <= 0xFFFFFFFF )
                    v127 = v126;
                  v161 = v127;
                  v29 = (char *)malloc(16 * v127);
                  v27 = v156;
                  if ( !v29 )
                  {
                    sub_16BD1C0("Allocation failed", 1u);
                    v27 = v200;
                    v29 = 0;
                  }
                  v128 = src;
                  v129 = v29;
                  v30 = 16LL * v27;
                  v130 = src;
                  if ( v30 )
                  {
                    do
                    {
                      if ( v129 )
                      {
                        *(_QWORD *)v129 = *(_QWORD *)v130;
                        *((_DWORD *)v129 + 2) = v130[2];
                      }
                      v129 += 16;
                      v130 += 4;
                    }
                    while ( &v29[v30] != v129 );
                  }
                  if ( v128 != &v201 )
                  {
                    v157 = v29;
                    _libc_free((unsigned __int64)v128);
                    v29 = v157;
                    v27 = v200;
                    v30 = 16LL * (unsigned int)v200;
                  }
                  src = v29;
                  HIDWORD(v200) = v161;
                }
                else
                {
                  v29 = (char *)src;
                  v30 = 16LL * (unsigned int)v200;
                }
                v31 = &v29[v30];
                if ( v31 )
                {
                  *(_QWORD *)v31 = v28;
                  *((_DWORD *)v31 + 2) = v23;
                  v27 = v200;
                }
                LODWORD(v200) = v27 + 1;
              }
            }
          }
        }
      }
      else if ( !v172 || v23 > v169 )
      {
        v169 = v23;
        v172 = v193;
      }
      ++v20;
    }
    while ( v165 != v20 );
    v32 = (__int64 *)src;
    if ( !(_DWORD)v200 )
      goto LABEL_33;
    v66 = 16LL * (unsigned int)v200;
    v178 = v6;
    v166 = v5;
    v67 = src;
    v68 = (char *)src + v66;
    v69 = v66 >> 4;
    while ( 1 )
    {
      v70 = 16 * v69;
      v71 = sub_2207800(16 * v69, &unk_435FF63);
      v72 = v71;
      if ( v71 )
        break;
      v69 >>= 1;
      if ( !v69 )
      {
        v117 = (__int64)v68;
        v78 = v178;
        v70 = 0;
        v118 = (__int64)v67;
        v79 = v166;
        sub_1DE4D70(v118, v117);
LABEL_103:
        j_j___libc_free_0(v72, v70);
        v32 = (__int64 *)src;
        if ( (char *)src + 16 * (unsigned int)v200 == src )
          goto LABEL_33;
        v179 = (__int64 *)((char *)src + 16 * (unsigned int)v200);
        while ( 1 )
        {
          v84 = *v32;
          if ( v169 > *((_DWORD *)v32 + 2) )
            goto LABEL_111;
          v83 = (unsigned __int8)sub_1F34100(*v32);
          if ( (unsigned int)((__int64)(*(_QWORD *)(v84 + 96) - *(_QWORD *)(v84 + 88)) >> 3) != 1
            && (unsigned __int8)sub_1F34340(v78 + 616, v83, v84)
            && (unsigned __int8)sub_1DE8B00(v78, v79, v84, (__int64)a3, a4)
            && (unsigned __int8)sub_1DE7930(v78, v79, v84, v169, a3, a4) )
          {
            break;
          }
          v32 += 2;
          if ( v179 == v32 )
          {
            v32 = (__int64 *)src;
            goto LABEL_33;
          }
        }
        v172 = v84;
LABEL_111:
        v32 = (__int64 *)src;
LABEL_33:
        v15 = v172;
        if ( v32 != (__int64 *)&v201 )
          _libc_free((unsigned __int64)v32);
        v16 = v190;
        goto LABEL_36;
      }
    }
    v73 = v69;
    v74 = v67;
    v75 = v71 + v70;
    v76 = (__int64)v68;
    v77 = *v67;
    v78 = v178;
    v79 = v166;
    *(_QWORD *)v72 = v77;
    *(_DWORD *)(v72 + 8) = *((_DWORD *)v74 + 2);
    v80 = v72 + 16;
    if ( v75 == v72 + 16 )
    {
      v82 = v72;
    }
    else
    {
      do
      {
        v81 = *(_QWORD *)(v80 - 16);
        v80 += 16;
        *(_QWORD *)(v80 - 16) = v81;
        *(_DWORD *)(v80 - 8) = *(_DWORD *)(v80 - 24);
      }
      while ( v75 != v80 );
      v82 = v72 + v70 - 16;
    }
    *((_DWORD *)v74 + 2) = *(_DWORD *)(v82 + 8);
    *v74 = *(_QWORD *)v82;
    sub_1DE9F70(v74, v76, (__int64 *)v72, v73);
    goto LABEL_103;
  }
  v193 = 0;
  v34 = (__int64 *)v198;
  v194 = (__int64 *)v198;
  v195 = (__int64 *)v198;
  v196 = 2;
  v197 = 0;
  if ( v17 == v18 )
    goto LABEL_54;
  v35 = (__int64 *)v198;
  v36 = v18;
  do
  {
LABEL_44:
    v37 = *v36;
    if ( v34 != v35 )
    {
LABEL_42:
      sub_16CCBA0((__int64)&v193, v37);
      v34 = v195;
      v35 = v194;
      goto LABEL_43;
    }
    v38 = &v34[HIDWORD(v196)];
    if ( v38 == v34 )
    {
LABEL_147:
      if ( HIDWORD(v196) >= (unsigned int)v196 )
        goto LABEL_42;
      ++HIDWORD(v196);
      *v38 = v37;
      v35 = v194;
      ++v193;
      v34 = v195;
    }
    else
    {
      v39 = v34;
      v40 = 0;
      while ( v37 != *v39 )
      {
        if ( *v39 == -2 )
          v40 = v39;
        if ( v38 == ++v39 )
        {
          if ( !v40 )
            goto LABEL_147;
          ++v36;
          *v40 = v37;
          v34 = v195;
          --v197;
          v35 = v194;
          ++v193;
          if ( v17 != v36 )
            goto LABEL_44;
          goto LABEL_53;
        }
      }
    }
LABEL_43:
    ++v36;
  }
  while ( v17 != v36 );
LABEL_53:
  v19 = (unsigned int)v191;
LABEL_54:
  v41 = v204;
  src = 0;
  v200 = (__int64)v204;
  v201 = v204;
  v202 = 8;
  v203 = 0;
  v155 = &v190[v19];
  if ( v190 != v155 )
  {
    v162 = v190;
    v159 = v6;
    v170 = v5;
    v42 = v6 + 888;
    while ( 1 )
    {
      v187 = *v162;
      v43 = *(__int64 **)(v187 + 64);
      v173 = *(__int64 **)(v187 + 72);
      if ( v173 == v43 )
      {
LABEL_74:
        v6 = v159;
        v5 = v170;
        v56 = 0;
        v57 = v201;
        v41 = (_BYTE *)v200;
        goto LABEL_75;
      }
      v163 = 0;
      do
      {
        v188.m128i_i64[0] = *v43;
        if ( sub_1DA1810((__int64)&v193, v188.m128i_i64[0]) )
        {
          v53 = *(__int64 **)(v188.m128i_i64[0] + 96);
          v54 = *(__int64 **)(v188.m128i_i64[0] + 88);
          if ( v53 != v54 )
          {
            v55 = *(__int64 **)(v188.m128i_i64[0] + 88);
            if ( !sub_1DA1810((__int64)&v193, *v54) )
              goto LABEL_74;
            while ( v53 != ++v55 )
            {
              if ( !sub_1DA1810((__int64)&v193, *v55) )
                goto LABEL_74;
            }
          }
        }
        else
        {
          v44 = sub_1DE4FA0(v42, v188.m128i_i64);
          if ( v170 == v188.m128i_i64[0] )
            goto LABEL_68;
          if ( !a4 )
            goto LABEL_63;
          if ( (*(_BYTE *)(a4 + 8) & 1) != 0 )
          {
            v45 = a4 + 16;
            v46 = 15;
          }
          else
          {
            v85 = *(_DWORD *)(a4 + 24);
            v45 = *(_QWORD *)(a4 + 16);
            if ( !v85 )
              goto LABEL_68;
            v46 = v85 - 1;
          }
          v47 = 1;
          v48 = v46 & (((unsigned __int32)v188.m128i_i32[0] >> 9) ^ ((unsigned __int32)v188.m128i_i32[0] >> 4));
          v49 = *(_QWORD *)(v45 + 8LL * v48);
          if ( v188.m128i_i64[0] != v49 )
          {
            while ( v49 != -8 )
            {
              v48 = v46 & (v47 + v48);
              v49 = *(_QWORD *)(v45 + 8LL * v48);
              if ( v188.m128i_i64[0] == v49 )
                goto LABEL_63;
              ++v47;
            }
          }
          else
          {
LABEL_63:
            v50 = (void ***)v44[1];
            if ( a3 == v50 || v50 == (void ***)sub_1DE4FA0(v42, &v187)[1] )
              goto LABEL_68;
            ++v163;
            v51 = (_QWORD *)v200;
            if ( v201 != (_BYTE *)v200 )
              goto LABEL_66;
            v86 = (_QWORD *)(v200 + 8LL * HIDWORD(v202));
            if ( (_QWORD *)v200 != v86 )
            {
              v87 = 0;
              while ( v188.m128i_i64[0] != *v51 )
              {
                if ( *v51 == -2 )
                  v87 = v51;
                if ( v86 == ++v51 )
                {
                  if ( !v87 )
                    goto LABEL_169;
                  *v87 = v188.m128i_i64[0];
                  --v203;
                  src = (char *)src + 1;
                  goto LABEL_67;
                }
              }
              goto LABEL_68;
            }
LABEL_169:
            if ( HIDWORD(v202) < (unsigned int)v202 )
            {
              ++HIDWORD(v202);
              *v86 = v188.m128i_i64[0];
              src = (char *)src + 1;
            }
            else
            {
LABEL_66:
              sub_16CCBA0((__int64)&src, v188.m128i_i64[0]);
              if ( !v52 )
                goto LABEL_68;
            }
LABEL_67:
            if ( !(unsigned __int8)sub_1DE89B0(v188.m128i_i64[0], (__int64)&v193) )
              goto LABEL_74;
          }
        }
LABEL_68:
        ++v43;
      }
      while ( v173 != v43 );
      if ( !v163 )
        goto LABEL_74;
      if ( v155 == ++v162 )
      {
        v6 = v159;
        v5 = v170;
        v57 = v201;
        v41 = (_BYTE *)v200;
        goto LABEL_146;
      }
    }
  }
  v57 = v204;
LABEL_146:
  v56 = 1;
LABEL_75:
  if ( v41 != v57 )
    _libc_free((unsigned __int64)v57);
  if ( v195 != v194 )
    _libc_free((unsigned __int64)v195);
  if ( !v56 )
  {
    v19 = (unsigned int)v191;
    goto LABEL_14;
  }
  v58 = *(__int64 **)(v5 + 96);
  v59 = *(__int64 **)(v5 + 88);
  v193 = 0;
  v60 = (__int64 *)v198;
  v196 = 4;
  v197 = 0;
  v194 = (__int64 *)v198;
  v61 = (__int64 *)v198;
  v195 = (__int64 *)v198;
  if ( v58 == v59 )
  {
    v116 = 0;
    goto LABEL_96;
  }
LABEL_84:
  while ( 2 )
  {
    v62 = *v59;
    if ( v60 != v61 )
      goto LABEL_82;
    v63 = &v60[HIDWORD(v196)];
    if ( v63 == v60 )
    {
LABEL_123:
      if ( HIDWORD(v196) < (unsigned int)v196 )
      {
        ++HIDWORD(v196);
        *v63 = v62;
        v60 = v194;
        ++v193;
        v61 = v195;
        goto LABEL_83;
      }
LABEL_82:
      sub_16CCBA0((__int64)&v193, v62);
      v61 = v195;
      v60 = v194;
      goto LABEL_83;
    }
    v64 = v60;
    v65 = 0;
    while ( v62 != *v64 )
    {
      if ( *v64 == -2 )
        v65 = v64;
      if ( v63 == ++v64 )
      {
        if ( !v65 )
          goto LABEL_123;
        ++v59;
        *v65 = v62;
        v61 = v195;
        --v197;
        v60 = v194;
        ++v193;
        if ( v58 != v59 )
          goto LABEL_84;
        goto LABEL_93;
      }
    }
LABEL_83:
    if ( v58 != ++v59 )
      continue;
    break;
  }
LABEL_93:
  v116 = 0;
  if ( HIDWORD(v196) - v197 != 2 || (_DWORD)v191 != 2 )
    goto LABEL_94;
  v180 = v6;
  src = &v201;
  v200 = 0x800000000LL;
  v206 = 0x800000000LL;
  v164 = (__m128i **)v190;
  v205 = v207;
  v171 = v5;
  p_src = &src;
  v89 = v6 + 888;
  while ( 2 )
  {
    v160 = p_src + 2;
    v90 = (__int64 *)(*v164)[4].m128i_i64[1];
    v91 = (__int64 *)(*v164)[4].m128i_i64[0];
    v186 = *v164;
    v174 = v90;
    if ( v90 == v91 )
      goto LABEL_156;
    v92 = p_src;
    v93 = v91;
    v94 = (__int64)v92;
    while ( 2 )
    {
      v100 = *v93;
      v187 = v100;
      if ( v171 == v100 )
      {
LABEL_134:
        v168 = sub_1DF1780(*(_QWORD *)(v180 + 560), v100, v186);
        v188.m128i_i64[0] = sub_20D7490(*(_QWORD *)(v180 + 568), v187);
        v188.m128i_i64[0] = sub_16AF500(v188.m128i_i64, v168);
        v188.m128i_i64[1] = v187;
        v189 = v186;
        v103 = *(unsigned int *)(v94 + 8);
        if ( (unsigned int)v103 >= *(_DWORD *)(v94 + 12) )
        {
          sub_16CD150(v94, v160, 0, 24, v101, v102);
          v103 = *(unsigned int *)(v94 + 8);
        }
        v104 = (__m128i *)(*(_QWORD *)v94 + 24 * v103);
        *v104 = _mm_loadu_si128(&v188);
        v104[1].m128i_i64[0] = (__int64)v189;
        ++*(_DWORD *)(v94 + 8);
      }
      else
      {
        if ( !a4 )
          goto LABEL_131;
        if ( (*(_BYTE *)(a4 + 8) & 1) != 0 )
        {
          v95 = a4 + 16;
          v96 = 15;
          goto LABEL_130;
        }
        v105 = *(_DWORD *)(a4 + 24);
        v95 = *(_QWORD *)(a4 + 16);
        if ( v105 )
        {
          v96 = v105 - 1;
LABEL_130:
          v97 = 1;
          v98 = v96 & (((unsigned int)v100 >> 9) ^ ((unsigned int)v100 >> 4));
          v99 = *(_QWORD *)(v95 + 8LL * v98);
          if ( v100 == v99 )
          {
LABEL_131:
            if ( a3 != (void ***)sub_1DE4FA0(v89, &v187)[1] )
            {
              v167 = sub_1DE4FA0(v89, &v187)[1];
              if ( v167 != sub_1DE4FA0(v89, (__int64 *)&v186)[1] )
              {
                v100 = v187;
                goto LABEL_134;
              }
            }
          }
          else
          {
            while ( v99 != -8 )
            {
              v98 = v96 & (v97 + v98);
              v99 = *(_QWORD *)(v95 + 8LL * v98);
              if ( v100 == v99 )
                goto LABEL_131;
              ++v97;
            }
          }
        }
      }
      if ( v174 != ++v93 )
        continue;
      break;
    }
    p_src = (void **)v94;
LABEL_156:
    ++v164;
    p_src += 26;
    if ( p_src != (void **)&v208 )
      continue;
    break;
  }
  v106 = (__m128i *)src;
  v107 = v180;
  v108 = 24LL * (unsigned int)v200;
  v109 = (__m128i *)((char *)src + v108);
  sub_1DE31A0(v188.m128i_i64, (__m128i *)src, 0xAAAAAAAAAAAAAAABLL * (v108 >> 3));
  if ( v189 )
    sub_1DE5580(v106, v109, v189, (const __m128i *)v188.m128i_i64[1]);
  else
    sub_1DE4940((unsigned __int64 *)v106, (unsigned __int64 *)v109);
  j_j___libc_free_0(v189, 24 * v188.m128i_i64[1]);
  v110 = (__m128i *)v205;
  v111 = 24LL * (unsigned int)v206;
  v112 = (__m128i *)((char *)v205 + v111);
  sub_1DE31A0(v188.m128i_i64, (__m128i *)v205, 0xAAAAAAAAAAAAAAABLL * (v111 >> 3));
  if ( v189 )
    sub_1DE5580(v110, v112, v189, (const __m128i *)v188.m128i_i64[1]);
  else
    sub_1DE4940((unsigned __int64 *)v110, (unsigned __int64 *)v112);
  j_j___libc_free_0(v189, 24 * v188.m128i_i64[1]);
  v113 = src;
  v114 = (__int64 *)v205;
  v115 = *((_QWORD *)src + 1);
  v181 = *((_QWORD *)v205 + 1);
  if ( v115 == v181 )
  {
    v176 = src;
    v182 = (char *)src + 24;
    v131 = sub_16AF590((__int64 *)src, *((_QWORD *)v205 + 3));
    v132 = sub_16AF590(v114, v176[3]);
    v113 = v176;
    if ( v132 <= v131 )
      v114 += 3;
    else
      v113 = v182;
    v115 = v113[1];
    v181 = v114[1];
  }
  if ( v171 == v181 )
  {
    v116 = v114[2];
    v175 = v113[2];
  }
  else
  {
    v116 = v113[2];
    v175 = v114[2];
    if ( v171 != v115 )
    {
      v116 = 0;
      goto LABEL_165;
    }
    v115 = v181;
    v181 = v171;
  }
  if ( v116 != v115
    || !byte_4FC5020
    || (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v107 + 552) + 8LL) + 640LL) & 1) != 0
    || (v133 = (unsigned __int8)sub_1F34100(v175),
        (unsigned int)((__int64)(*(_QWORD *)(v175 + 96) - *(_QWORD *)(v175 + 88)) >> 3) == 1)
    || !(unsigned __int8)sub_1F34340(v107 + 616, v133, v175)
    || (v134 = (unsigned __int8)sub_1F34100(v175),
        (unsigned int)((__int64)(*(_QWORD *)(v175 + 96) - *(_QWORD *)(v175 + 88)) >> 3) == 1)
    || !(unsigned __int8)sub_1F34340(v107 + 616, v134, v175)
    || !(unsigned __int8)sub_1DE8B00(v107, v181, v175, (__int64)a3, a4)
    || (v135 = sub_1DF1780(*(_QWORD *)(v107 + 560), v181, v116),
        !(unsigned __int8)sub_1DE7930(v107, v181, v175, v135, a3, a4)) )
  {
    v119 = *(_DWORD *)(v107 + 544);
    if ( v119 )
    {
      v120 = *(_QWORD *)(v107 + 528);
      v121 = (v119 - 1) & (((unsigned int)v115 >> 4) ^ ((unsigned int)v115 >> 9));
      v122 = v120 + 24LL * v121;
      v123 = *(_QWORD *)v122;
      if ( *(_QWORD *)v122 == v115 )
      {
LABEL_183:
        *(_BYTE *)(v122 + 16) = 0;
        *(_QWORD *)(v122 + 8) = v175;
        goto LABEL_165;
      }
      v143 = 1;
      v144 = 0;
      while ( v123 != -8 )
      {
        if ( !v144 && v123 == -16 )
          v144 = v122;
        v121 = (v119 - 1) & (v143 + v121);
        v122 = v120 + 24LL * v121;
        v123 = *(_QWORD *)v122;
        if ( *(_QWORD *)v122 == v115 )
          goto LABEL_183;
        ++v143;
      }
      v145 = *(_DWORD *)(v107 + 536);
      if ( v144 )
        v122 = v144;
      ++*(_QWORD *)(v107 + 520);
      v142 = v145 + 1;
      if ( 4 * (v145 + 1) < 3 * v119 )
      {
        if ( v119 - *(_DWORD *)(v107 + 540) - v142 > v119 >> 3 )
        {
LABEL_215:
          *(_DWORD *)(v107 + 536) = v142;
          if ( *(_QWORD *)v122 != -8 )
            --*(_DWORD *)(v107 + 540);
          *(_QWORD *)v122 = v115;
          *(_QWORD *)(v122 + 8) = 0;
          *(_BYTE *)(v122 + 16) = 0;
          goto LABEL_183;
        }
        sub_1DE4600(v107 + 520, v119);
        v146 = *(_DWORD *)(v107 + 544);
        if ( v146 )
        {
          v147 = v146 - 1;
          v149 = 1;
          v122 = 0;
          for ( i = v147 & (((unsigned int)v115 >> 4) ^ ((unsigned int)v115 >> 9)); ; i = v147 & v154 )
          {
            v148 = *(_QWORD *)(v107 + 528);
            v151 = (__int64 *)(v148 + 24LL * i);
            v152 = *v151;
            if ( *v151 == v115 )
            {
              v142 = *(_DWORD *)(v107 + 536) + 1;
              v122 = v148 + 24LL * i;
              goto LABEL_215;
            }
            if ( v152 == -8 )
              break;
            if ( v152 != -16 || v122 )
              v151 = (__int64 *)v122;
            v154 = v149 + i;
            v122 = (__int64)v151;
            ++v149;
          }
          if ( !v122 )
            v122 = v148 + 24LL * i;
          v142 = *(_DWORD *)(v107 + 536) + 1;
          goto LABEL_215;
        }
LABEL_251:
        ++*(_DWORD *)(v107 + 536);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(v107 + 520);
    }
    sub_1DE4600(v107 + 520, 2 * v119);
    v136 = *(_DWORD *)(v107 + 544);
    if ( v136 )
    {
      v137 = v136 - 1;
      v138 = 1;
      v139 = 0;
      for ( j = v137 & (((unsigned int)v115 >> 9) ^ ((unsigned int)v115 >> 4)); ; j = v137 & v153 )
      {
        v122 = *(_QWORD *)(v107 + 528) + 24LL * j;
        v141 = *(_QWORD *)v122;
        if ( *(_QWORD *)v122 == v115 )
        {
          v142 = *(_DWORD *)(v107 + 536) + 1;
          goto LABEL_215;
        }
        if ( v141 == -8 )
          break;
        if ( v139 || v141 != -16 )
          v122 = v139;
        v153 = v138 + j;
        v139 = v122;
        ++v138;
      }
      if ( v139 )
        v122 = v139;
      v142 = *(_DWORD *)(v107 + 536) + 1;
      goto LABEL_215;
    }
    goto LABEL_251;
  }
  v116 = v175;
LABEL_165:
  if ( v205 != v207 )
    _libc_free((unsigned __int64)v205);
  if ( src != &v201 )
    _libc_free((unsigned __int64)src);
LABEL_94:
  if ( v195 != v194 )
    _libc_free((unsigned __int64)v195);
LABEL_96:
  v16 = v190;
  v15 = v116;
LABEL_36:
  if ( v16 != (__int64 *)v192 )
    _libc_free((unsigned __int64)v16);
  return v15;
}
