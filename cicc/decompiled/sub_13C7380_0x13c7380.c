// Function: sub_13C7380
// Address: 0x13c7380
//
__int64 __fastcall sub_13C7380(__int64 a1, _QWORD *a2)
{
  __int64 v2; // rsi
  __int64 v3; // rsi
  __int64 **v4; // rax
  unsigned int v5; // esi
  __int64 v6; // rcx
  unsigned int v7; // ebx
  unsigned int v8; // edx
  _QWORD *v9; // r13
  __int64 v10; // rax
  __int64 v11; // r14
  _QWORD *v12; // rax
  _QWORD *v13; // r12
  __int64 v14; // rax
  __int64 *v15; // rcx
  __int64 **v16; // rbx
  _QWORD *v17; // r15
  char v18; // al
  __int64 v19; // rax
  __int64 v20; // r8
  __int64 v21; // r14
  __int64 v22; // r15
  __int64 v23; // r12
  __int64 v24; // r14
  __int64 v25; // rax
  __int64 v26; // r14
  __int64 v27; // r15
  __int64 v28; // rax
  _QWORD *v29; // r15
  __int64 i; // r13
  __int64 v31; // rdi
  unsigned __int8 v32; // al
  __int64 v33; // rax
  __int64 *v34; // rdx
  __int64 v35; // rax
  unsigned __int64 v36; // r12
  __int64 v37; // r10
  _QWORD *v38; // r11
  __int64 *v39; // rax
  __int64 v40; // r10
  __int64 *v41; // rdx
  unsigned __int64 v42; // rax
  __int64 v43; // r12
  unsigned __int64 v44; // r12
  unsigned __int64 *v45; // rax
  __int64 **v46; // r15
  int v47; // eax
  __int64 v48; // rcx
  int v49; // r8d
  __int64 v50; // rdx
  unsigned int v51; // eax
  __int64 *v52; // r12
  __int64 v53; // rdi
  unsigned __int64 v54; // r13
  _QWORD *v55; // rdi
  _QWORD *v56; // rax
  _QWORD *v57; // rsi
  __int64 v58; // rcx
  __int64 v59; // rdx
  __int64 v60; // rax
  __int64 v61; // rdx
  __int64 v62; // rdi
  __int64 *v63; // rdx
  char v64; // al
  unsigned __int64 v65; // rbx
  unsigned __int64 v66; // r12
  _QWORD *v67; // r14
  char v68; // r8
  __int64 v69; // r15
  unsigned __int64 v70; // r13
  char v71; // cl
  unsigned __int64 v72; // rdi
  unsigned int v73; // edx
  unsigned __int64 v74; // rax
  __int64 v75; // r9
  _QWORD *v76; // rax
  _QWORD *v77; // rdx
  _QWORD *v78; // rax
  unsigned int v79; // edx
  int v80; // edi
  unsigned int v81; // r9d
  _QWORD *v82; // rax
  __int64 v83; // r8
  unsigned __int64 v84; // rbx
  void *v85; // r14
  _QWORD *v86; // rdx
  _QWORD *v87; // rax
  char v88; // al
  __int64 v89; // r15
  __int64 v90; // rdi
  __int64 v91; // rax
  size_t v92; // rdx
  __int64 **v93; // rax
  unsigned __int64 v94; // r13
  int v96; // r11d
  unsigned __int64 v97; // r10
  __int64 **v98; // r14
  __int64 **v99; // r13
  int v100; // eax
  __int64 v101; // rcx
  int v102; // r8d
  __int64 v103; // rdx
  unsigned int v104; // eax
  __int64 *v105; // rbx
  __int64 v106; // rdi
  unsigned __int64 v107; // r12
  unsigned __int64 v108; // rdi
  int v109; // ecx
  unsigned int v110; // edx
  unsigned __int64 v111; // rdi
  int v112; // ecx
  unsigned int v113; // edx
  int v114; // r11d
  unsigned __int64 v115; // r9
  unsigned __int64 v116; // rbx
  __int64 v117; // rax
  __int64 v118; // rax
  char *v119; // rax
  __int64 v120; // rax
  __int64 v121; // rax
  unsigned __int64 v122; // r13
  unsigned int v123; // r14d
  unsigned __int64 v124; // r12
  __int64 v125; // r9
  unsigned int v126; // edx
  _QWORD *v127; // rbx
  __int64 v128; // rdi
  unsigned __int64 v129; // r13
  __int64 v130; // r13
  int v131; // eax
  int v132; // ecx
  __int64 v133; // r10
  __int64 v134; // rax
  int v135; // edx
  __int64 v136; // r9
  _QWORD *v137; // rax
  unsigned __int64 v138; // r9
  void *v139; // r10
  _QWORD *v140; // rdx
  _QWORD *v141; // rax
  char v142; // al
  __int64 v143; // rdx
  __int64 v144; // rdi
  __int64 v145; // rax
  size_t v146; // rdx
  int v147; // ecx
  int v148; // ecx
  int v149; // r8d
  _QWORD *v150; // rcx
  int v151; // edi
  int v152; // edx
  int v153; // ecx
  __int64 v154; // r10
  int v155; // edi
  __int64 v156; // rax
  __int64 v157; // r9
  int v158; // r10d
  unsigned __int64 v159; // r11
  int v160; // r11d
  int v161; // r9d
  _QWORD *v162; // r8
  int v163; // eax
  int v164; // edi
  int v165; // edi
  __int64 v166; // r8
  unsigned int v167; // edx
  __int64 v168; // r10
  int v169; // esi
  _QWORD *v170; // rcx
  int v171; // edi
  int v172; // edi
  __int64 v173; // r8
  _QWORD *v174; // rdx
  unsigned int v175; // ebx
  int v176; // ecx
  __int64 v177; // rsi
  int v178; // edi
  __int64 v179; // [rsp+8h] [rbp-F8h]
  int v181; // [rsp+24h] [rbp-DCh]
  __int64 v182; // [rsp+28h] [rbp-D8h]
  void *v183; // [rsp+28h] [rbp-D8h]
  void *src; // [rsp+30h] [rbp-D0h]
  char *srca; // [rsp+30h] [rbp-D0h]
  void *srcb; // [rsp+30h] [rbp-D0h]
  __int64 v187; // [rsp+38h] [rbp-C8h]
  void *v188; // [rsp+38h] [rbp-C8h]
  int v189; // [rsp+38h] [rbp-C8h]
  __int64 *v190; // [rsp+40h] [rbp-C0h]
  __int64 v191; // [rsp+40h] [rbp-C0h]
  char v192; // [rsp+40h] [rbp-C0h]
  char v193; // [rsp+40h] [rbp-C0h]
  char v194; // [rsp+40h] [rbp-C0h]
  _QWORD *v195; // [rsp+40h] [rbp-C0h]
  unsigned __int64 v196; // [rsp+40h] [rbp-C0h]
  _QWORD *v197; // [rsp+40h] [rbp-C0h]
  unsigned int v198; // [rsp+40h] [rbp-C0h]
  _QWORD *v199; // [rsp+48h] [rbp-B8h]
  __int64 v200; // [rsp+48h] [rbp-B8h]
  unsigned __int64 v201; // [rsp+48h] [rbp-B8h]
  __int64 v202; // [rsp+48h] [rbp-B8h]
  int v204; // [rsp+58h] [rbp-A8h]
  __int64 *v205; // [rsp+58h] [rbp-A8h]
  __int64 v206; // [rsp+58h] [rbp-A8h]
  __int64 v207; // [rsp+58h] [rbp-A8h]
  __int64 v208; // [rsp+58h] [rbp-A8h]
  int v209; // [rsp+58h] [rbp-A8h]
  _QWORD v210[2]; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v211; // [rsp+70h] [rbp-90h]
  __int64 v212; // [rsp+78h] [rbp-88h]
  __int64 v213; // [rsp+80h] [rbp-80h]
  __int64 v214; // [rsp+88h] [rbp-78h]
  __int64 v215; // [rsp+90h] [rbp-70h]
  __int64 v216; // [rsp+98h] [rbp-68h]
  __int64 **v217; // [rsp+A0h] [rbp-60h]
  __int64 **v218; // [rsp+A8h] [rbp-58h]
  __int64 v219; // [rsp+B0h] [rbp-50h]
  __int64 v220; // [rsp+B8h] [rbp-48h]
  __int64 v221; // [rsp+C0h] [rbp-40h]
  __int64 v222; // [rsp+C8h] [rbp-38h]

  v2 = a2[7];
  v210[0] = 0;
  v210[1] = 0;
  v211 = 0;
  v212 = 0;
  v213 = 0;
  v214 = 0;
  v215 = 0;
  v216 = 0;
  v217 = 0;
  v218 = 0;
  v219 = 0;
  v220 = 0;
  v221 = 0;
  v222 = 0;
  sub_13C69A0((__int64)v210, v2);
  sub_13C6E30((__int64)v210);
  v3 = a1 + 264;
  v4 = v217;
  v179 = a1 + 264;
  if ( v218 == v217 )
    goto LABEL_149;
  do
  {
    v187 = **v4;
    if ( !v187 || (unsigned __int8)sub_15E4B50(v187, v3) )
    {
      v98 = v218;
      v99 = v217;
      if ( v217 == v218 )
        goto LABEL_148;
      while ( 1 )
      {
        v100 = *(_DWORD *)(a1 + 288);
        if ( v100 )
        {
          v3 = (unsigned int)(v100 - 1);
          v101 = *(_QWORD *)(a1 + 272);
          v102 = 1;
          v103 = **v99;
          v104 = v3 & (((unsigned int)v103 >> 9) ^ ((unsigned int)v103 >> 4));
          v105 = (__int64 *)(v101 + 16LL * v104);
          v106 = *v105;
          if ( v103 == *v105 )
          {
LABEL_168:
            v107 = v105[1] & 0xFFFFFFFFFFFFFFF8LL;
            if ( v107 )
            {
              if ( (*(_BYTE *)(v107 + 8) & 1) == 0 )
                j___libc_free_0(*(_QWORD *)(v107 + 16));
              v3 = 272;
              j_j___libc_free_0(v107, 272);
            }
            *v105 = -16;
            --*(_DWORD *)(a1 + 280);
            ++*(_DWORD *)(a1 + 284);
          }
          else
          {
            while ( v106 != -8 )
            {
              v104 = v3 & (v102 + v104);
              v105 = (__int64 *)(v101 + 16LL * v104);
              v106 = *v105;
              if ( v103 == *v105 )
                goto LABEL_168;
              ++v102;
            }
          }
        }
        if ( v98 == ++v99 )
          goto LABEL_148;
      }
    }
    v5 = *(_DWORD *)(a1 + 288);
    if ( v5 )
    {
      v6 = *(_QWORD *)(a1 + 272);
      v7 = ((unsigned int)v187 >> 9) ^ ((unsigned int)v187 >> 4);
      v8 = (v5 - 1) & v7;
      v9 = (_QWORD *)(v6 + 16LL * v8);
      v10 = *v9;
      if ( v187 == *v9 )
        goto LABEL_6;
      v161 = 1;
      v162 = 0;
      while ( v10 != -8 )
      {
        if ( v10 == -16 && !v162 )
          v162 = v9;
        v8 = (v5 - 1) & (v161 + v8);
        v9 = (_QWORD *)(v6 + 16LL * v8);
        v10 = *v9;
        if ( v187 == *v9 )
          goto LABEL_6;
        ++v161;
      }
      if ( v162 )
        v9 = v162;
      ++*(_QWORD *)(a1 + 264);
      v163 = *(_DWORD *)(a1 + 280) + 1;
      if ( 4 * v163 < 3 * v5 )
      {
        if ( v5 - *(_DWORD *)(a1 + 284) - v163 <= v5 >> 3 )
        {
          sub_13C4B40(v179, v5);
          v171 = *(_DWORD *)(a1 + 288);
          if ( !v171 )
          {
LABEL_373:
            ++*(_DWORD *)(a1 + 280);
            BUG();
          }
          v172 = v171 - 1;
          v173 = *(_QWORD *)(a1 + 272);
          v174 = 0;
          v175 = v172 & v7;
          v9 = (_QWORD *)(v173 + 16LL * v175);
          v176 = 1;
          v177 = *v9;
          v163 = *(_DWORD *)(a1 + 280) + 1;
          if ( v187 != *v9 )
          {
            while ( v177 != -8 )
            {
              if ( !v174 && v177 == -16 )
                v174 = v9;
              v175 = v172 & (v176 + v175);
              v9 = (_QWORD *)(v173 + 16LL * v175);
              v177 = *v9;
              if ( v187 == *v9 )
                goto LABEL_316;
              ++v176;
            }
            if ( v174 )
              v9 = v174;
          }
        }
        goto LABEL_316;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 264);
    }
    sub_13C4B40(v179, 2 * v5);
    v164 = *(_DWORD *)(a1 + 288);
    if ( !v164 )
      goto LABEL_373;
    v165 = v164 - 1;
    v166 = *(_QWORD *)(a1 + 272);
    v167 = v165 & (((unsigned int)v187 >> 9) ^ ((unsigned int)v187 >> 4));
    v9 = (_QWORD *)(v166 + 16LL * v167);
    v168 = *v9;
    v163 = *(_DWORD *)(a1 + 280) + 1;
    if ( v187 != *v9 )
    {
      v169 = 1;
      v170 = 0;
      while ( v168 != -8 )
      {
        if ( !v170 && v168 == -16 )
          v170 = v9;
        v167 = v165 & (v169 + v167);
        v9 = (_QWORD *)(v166 + 16LL * v167);
        v168 = *v9;
        if ( v187 == *v9 )
          goto LABEL_316;
        ++v169;
      }
      if ( v170 )
        v9 = v170;
    }
LABEL_316:
    *(_DWORD *)(a1 + 280) = v163;
    if ( *v9 != -8 )
      --*(_DWORD *)(a1 + 284);
    v9[1] = 0;
    *v9 = v187;
LABEL_6:
    v11 = *(_QWORD *)(a1 + 328);
    v12 = (_QWORD *)sub_22077B0(64);
    v12[3] = 2;
    v13 = v12;
    v12[4] = 0;
    v12[5] = v187;
    if ( v187 != -8 && v187 != -16 )
      sub_164C220(v12 + 3);
    v3 = v11;
    v13[7] = 0;
    v13[6] = a1;
    v13[2] = &unk_49EA488;
    sub_2208C80(v13, v11);
    v14 = *(_QWORD *)(a1 + 328);
    ++*(_QWORD *)(a1 + 344);
    *(_QWORD *)(v14 + 56) = v14;
    v15 = (__int64 *)v218;
    v16 = v217;
    v181 = v218 - v217;
    if ( !v181 )
      goto LABEL_18;
    v17 = v9;
    v204 = 0;
    src = (void *)(v187 + 112);
    while ( (unsigned __int8)sub_15E4F60(v187) || (unsigned __int8)sub_1560180(src, 35) )
    {
      if ( (unsigned __int8)sub_1560180(src, 36) )
        goto LABEL_14;
      if ( (unsigned __int8)sub_1560180(src, 36) || (unsigned __int8)sub_1560180(src, 37) )
      {
        v17[1] |= 1uLL;
        if ( (*(_BYTE *)(v187 + 33) & 0x20) != 0 || (v18 = sub_1560180(src, 4)) != 0 )
LABEL_14:
          v18 = 0;
        else
          v17[1] |= 4uLL;
      }
      else
      {
        v17[1] |= 3uLL;
        v18 = ((*(_BYTE *)(v187 + 33) >> 5) ^ 1) & 1;
      }
      v3 = (unsigned int)++v204;
      if ( v181 == v204 || v18 )
      {
        v15 = (__int64 *)v218;
        v16 = v217;
        v9 = v17;
        if ( v18 )
          goto LABEL_64;
LABEL_18:
        v19 = v9[1];
        if ( v16 != (__int64 **)v15 )
        {
          v205 = (__int64 *)v16;
          v190 = v15;
          while ( 2 )
          {
            v3 = (__int64)v205;
            v20 = v19 & 7;
            v21 = *v205;
            if ( (v19 & 3) == 3 )
              goto LABEL_212;
            if ( (unsigned __int8)sub_1560180(*(_QWORD *)v21 + 112LL, 35) )
              goto LABEL_20;
            v22 = *(_QWORD *)(*(_QWORD *)v21 + 80LL);
            v23 = *(_QWORD *)v21 + 72LL;
            if ( v23 == v22 )
            {
              v26 = *(_QWORD *)(*(_QWORD *)v21 + 80LL);
              v27 = 0;
            }
            else
            {
              if ( !v22 )
                BUG();
              while ( 1 )
              {
                v24 = *(_QWORD *)(v22 + 24);
                if ( v24 != v22 + 16 )
                  break;
                v22 = *(_QWORD *)(v22 + 8);
                if ( v23 == v22 )
                  break;
                if ( !v22 )
                  BUG();
              }
              v25 = v24;
              v26 = v22;
              v27 = v25;
            }
LABEL_31:
            if ( v26 == v23 )
              goto LABEL_20;
            v28 = v27;
            v29 = v9;
            i = v28;
            while ( 2 )
            {
              v19 = v29[1];
              v31 = i - 24;
              if ( !i )
                v31 = 0;
              v20 = v29[1] & 7LL;
              if ( (v19 & 3) == 3 )
              {
                v9 = v29;
                goto LABEL_21;
              }
              v32 = *(_BYTE *)(v31 + 16);
              if ( v32 <= 0x17u )
                goto LABEL_39;
              if ( v32 == 78 )
              {
                v201 = v31 | 4;
              }
              else
              {
                if ( v32 != 29 )
                  goto LABEL_39;
                v201 = v31 & 0xFFFFFFFFFFFFFFFBLL;
              }
              srca = (char *)(v201 & 0xFFFFFFFFFFFFFFF8LL);
              if ( (v201 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
              {
                if ( (unsigned __int8)sub_140AF60(v31, *(_QWORD *)(a1 + 16), 0)
                  || sub_140B650(v31, *(_QWORD *)(a1 + 16)) )
                {
                  goto LABEL_200;
                }
                v119 = srca - 72;
                if ( (v201 & 4) != 0 )
                  v119 = srca - 24;
                v120 = *(_QWORD *)v119;
                if ( !*(_BYTE *)(v120 + 16) && (*(_BYTE *)(v120 + 33) & 0x20) != 0 )
                {
                  if ( *(_BYTE *)(v31 + 16) != 78
                    || (v121 = *(_QWORD *)(v31 - 24), *(_BYTE *)(v121 + 16))
                    || (*(_BYTE *)(v121 + 33) & 0x20) == 0
                    || (unsigned int)(*(_DWORD *)(v121 + 36) - 35) > 3 )
                  {
LABEL_200:
                    v29[1] |= 3uLL;
                  }
                }
              }
              else
              {
LABEL_39:
                if ( (unsigned __int8)sub_15F2ED0(v31) )
                  v29[1] |= 1uLL;
                if ( (unsigned __int8)sub_15F3040(v31) )
                  v29[1] |= 2uLL;
              }
              for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v26 + 24) )
              {
                v33 = v26 - 24;
                if ( !v26 )
                  v33 = 0;
                if ( i != v33 + 40 )
                  break;
                v26 = *(_QWORD *)(v26 + 8);
                if ( v23 == v26 )
                {
                  v118 = i;
                  v9 = v29;
                  v27 = v118;
                  goto LABEL_31;
                }
                if ( !v26 )
                  BUG();
              }
              if ( v26 != v23 )
                continue;
              break;
            }
            v9 = v29;
LABEL_20:
            v19 = v9[1];
            v20 = v19 & 7;
LABEL_21:
            v3 = (__int64)++v205;
            if ( v190 == v205 )
              goto LABEL_212;
            continue;
          }
        }
        v20 = v9[1] & 7LL;
LABEL_212:
        v122 = v19 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v19 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        {
          v206 = v20;
          v82 = (_QWORD *)sub_22077B0(272);
          v83 = v206;
          v84 = (unsigned __int64)v82;
          if ( v82 )
          {
            v85 = v82 + 2;
            *v82 = 0;
            v86 = v82 + 34;
            v82[1] = 1;
            v87 = v82 + 2;
            do
            {
              if ( v87 )
                *v87 = -8;
              v87 += 2;
            }
            while ( v87 != v86 );
            if ( (*(_BYTE *)(v84 + 8) & 1) == 0 )
            {
              j___libc_free_0(*(_QWORD *)(v84 + 16));
              v83 = v206;
            }
            v88 = *(_BYTE *)(v84 + 8) | 1;
            *(_BYTE *)(v84 + 8) = v88;
            if ( (*(_BYTE *)(v122 + 8) & 1) == 0 && *(_DWORD *)(v122 + 24) > 0x10u )
            {
              *(_BYTE *)(v84 + 8) = v88 & 0xFE;
              if ( (*(_BYTE *)(v122 + 8) & 1) != 0 )
              {
                v90 = 256;
                LODWORD(v89) = 16;
              }
              else
              {
                v89 = *(unsigned int *)(v122 + 24);
                v90 = 16 * v89;
              }
              v207 = v83;
              v91 = sub_22077B0(v90);
              *(_DWORD *)(v84 + 24) = v89;
              v83 = v207;
              *(_QWORD *)(v84 + 16) = v91;
            }
            v92 = 256;
            *(_DWORD *)(v84 + 8) = *(_DWORD *)(v122 + 8) & 0xFFFFFFFE | *(_DWORD *)(v84 + 8) & 1;
            *(_DWORD *)(v84 + 12) = *(_DWORD *)(v122 + 12);
            if ( (*(_BYTE *)(v84 + 8) & 1) == 0 )
              v92 = 16LL * *(unsigned int *)(v84 + 24);
            v3 = v122 + 16;
            if ( (*(_BYTE *)(v122 + 8) & 1) == 0 )
              v3 = *(_QWORD *)(v122 + 16);
            if ( (*(_BYTE *)(v84 + 8) & 1) == 0 )
              v85 = *(void **)(v84 + 16);
            v208 = v83;
            memcpy(v85, (const void *)v3, v92);
            v83 = v208;
          }
          v93 = v217;
          v20 = v84 | v83;
          v209 = v218 - v217;
          if ( v209 == 1 )
          {
            v94 = v20 & 0xFFFFFFFFFFFFFFF8LL;
LABEL_147:
            if ( v94 )
            {
              if ( (*(_BYTE *)(v94 + 8) & 1) == 0 )
                j___libc_free_0(*(_QWORD *)(v94 + 16));
              v3 = 272;
              j_j___libc_free_0(v94, 272);
            }
            goto LABEL_148;
          }
        }
        else
        {
          v93 = v217;
          v209 = v218 - v217;
          if ( v209 == 1 )
            goto LABEL_148;
        }
        v123 = 1;
        v124 = v20 & 0xFFFFFFFFFFFFFFF8LL;
        v202 = v20 & 7;
        srcb = (void *)((v20 & 0xFFFFFFFFFFFFFFF8LL) + 16);
        while ( 2 )
        {
          v3 = *(unsigned int *)(a1 + 288);
          v130 = *v93[v123];
          if ( !(_DWORD)v3 )
          {
            ++*(_QWORD *)(a1 + 264);
            goto LABEL_225;
          }
          v125 = *(_QWORD *)(a1 + 272);
          v126 = (v3 - 1) & (((unsigned int)v130 >> 4) ^ ((unsigned int)v130 >> 9));
          v127 = (_QWORD *)(v125 + 16LL * v126);
          v128 = *v127;
          if ( v130 == *v127 )
          {
LABEL_216:
            v129 = v127[1] & 0xFFFFFFFFFFFFFFF8LL;
            if ( v129 )
            {
              if ( (*(_BYTE *)(v129 + 8) & 1) == 0 )
                j___libc_free_0(*(_QWORD *)(v129 + 16));
              v3 = 272;
              j_j___libc_free_0(v129, 272);
            }
            goto LABEL_220;
          }
          v149 = 1;
          v150 = 0;
          while ( v128 != -8 )
          {
            if ( v128 == -16 && !v150 )
              v150 = v127;
            v126 = (v3 - 1) & (v149 + v126);
            v127 = (_QWORD *)(v125 + 16LL * v126);
            v128 = *v127;
            if ( v130 == *v127 )
              goto LABEL_216;
            ++v149;
          }
          v151 = *(_DWORD *)(a1 + 280);
          if ( v150 )
            v127 = v150;
          ++*(_QWORD *)(a1 + 264);
          v135 = v151 + 1;
          if ( 4 * (v151 + 1) >= (unsigned int)(3 * v3) )
          {
LABEL_225:
            sub_13C4B40(v179, 2 * v3);
            v131 = *(_DWORD *)(a1 + 288);
            if ( !v131 )
              goto LABEL_376;
            v132 = v131 - 1;
            v133 = *(_QWORD *)(a1 + 272);
            v3 = *(unsigned int *)(a1 + 280);
            LODWORD(v134) = (v131 - 1) & (((unsigned int)v130 >> 9) ^ ((unsigned int)v130 >> 4));
            v135 = v3 + 1;
            v127 = (_QWORD *)(v133 + 16LL * (unsigned int)v134);
            v136 = *v127;
            if ( v130 != *v127 )
            {
              v178 = 1;
              v3 = 0;
              while ( v136 != -8 )
              {
                if ( v136 == -16 && !v3 )
                  v3 = (__int64)v127;
                v134 = v132 & (unsigned int)(v134 + v178);
                v127 = (_QWORD *)(v133 + 16 * v134);
                v136 = *v127;
                if ( v130 == *v127 )
                  goto LABEL_227;
                ++v178;
              }
              goto LABEL_289;
            }
          }
          else if ( (int)v3 - *(_DWORD *)(a1 + 284) - v135 <= (unsigned int)v3 >> 3 )
          {
            v198 = ((unsigned int)v130 >> 4) ^ ((unsigned int)v130 >> 9);
            sub_13C4B40(v179, v3);
            v152 = *(_DWORD *)(a1 + 288);
            if ( !v152 )
            {
LABEL_376:
              ++*(_DWORD *)(a1 + 280);
              BUG();
            }
            v153 = v152 - 1;
            v154 = *(_QWORD *)(a1 + 272);
            v155 = 1;
            LODWORD(v156) = v153 & v198;
            v135 = *(_DWORD *)(a1 + 280) + 1;
            v3 = 0;
            v127 = (_QWORD *)(v154 + 16LL * (v153 & v198));
            v157 = *v127;
            if ( v130 != *v127 )
            {
              while ( v157 != -8 )
              {
                if ( !v3 && v157 == -16 )
                  v3 = (__int64)v127;
                v156 = v153 & (unsigned int)(v156 + v155);
                v127 = (_QWORD *)(v154 + 16 * v156);
                v157 = *v127;
                if ( v130 == *v127 )
                  goto LABEL_227;
                ++v155;
              }
LABEL_289:
              if ( v3 )
                v127 = (_QWORD *)v3;
            }
          }
LABEL_227:
          *(_DWORD *)(a1 + 280) = v135;
          if ( *v127 != -8 )
            --*(_DWORD *)(a1 + 284);
          *v127 = v130;
          v127[1] = 0;
LABEL_220:
          v94 = v124;
          v127[1] = v202;
          if ( v124 )
          {
            v137 = (_QWORD *)sub_22077B0(272);
            v138 = (unsigned __int64)v137;
            if ( v137 )
            {
              v139 = v137 + 2;
              *v137 = 0;
              v140 = v137 + 34;
              v137[1] = 1;
              v141 = v137 + 2;
              do
              {
                if ( v141 )
                  *v141 = -8;
                v141 += 2;
              }
              while ( v140 != v141 );
              if ( (*(_BYTE *)(v138 + 8) & 1) == 0 )
              {
                v188 = v139;
                v195 = (_QWORD *)v138;
                j___libc_free_0(*(_QWORD *)(v138 + 16));
                v139 = v188;
                v138 = (unsigned __int64)v195;
              }
              v142 = *(_BYTE *)(v138 + 8) | 1;
              *(_BYTE *)(v138 + 8) = v142;
              if ( (*(_BYTE *)(v124 + 8) & 1) == 0 && *(_DWORD *)(v124 + 24) > 0x10u )
              {
                *(_BYTE *)(v138 + 8) = v142 & 0xFE;
                if ( (*(_BYTE *)(v124 + 8) & 1) != 0 )
                {
                  v144 = 256;
                  LODWORD(v143) = 16;
                }
                else
                {
                  v143 = *(unsigned int *)(v124 + 24);
                  v144 = 16 * v143;
                }
                v183 = v139;
                v189 = v143;
                v196 = v138;
                v145 = sub_22077B0(v144);
                v138 = v196;
                v139 = v183;
                *(_QWORD *)(v196 + 16) = v145;
                *(_DWORD *)(v196 + 24) = v189;
              }
              v146 = 256;
              *(_DWORD *)(v138 + 8) = *(_DWORD *)(v124 + 8) & 0xFFFFFFFE | *(_DWORD *)(v138 + 8) & 1;
              *(_DWORD *)(v138 + 12) = *(_DWORD *)(v124 + 12);
              if ( (*(_BYTE *)(v138 + 8) & 1) == 0 )
                v146 = 16LL * *(unsigned int *)(v138 + 24);
              if ( (*(_BYTE *)(v124 + 8) & 1) != 0 )
                v3 = (__int64)srcb;
              else
                v3 = *(_QWORD *)(v124 + 16);
              if ( (*(_BYTE *)(v138 + 8) & 1) == 0 )
                v139 = *(void **)(v138 + 16);
              v197 = (_QWORD *)v138;
              memcpy(v139, (const void *)v3, v146);
              v138 = (unsigned __int64)v197;
            }
            v127[1] = v127[1] & 7LL | v138;
          }
          if ( ++v123 == v209 )
            goto LABEL_147;
          v93 = v217;
          continue;
        }
      }
    }
    v16 = v217;
    v34 = v217[v204];
    v3 = v34[2];
    v35 = v34[1];
    v182 = v3;
    if ( v3 == v35 )
      goto LABEL_14;
    v36 = **(_QWORD **)(v35 + 24);
    if ( !v36 )
    {
      v15 = (__int64 *)v218;
      goto LABEL_64;
    }
    v37 = v35 + 32;
    v38 = v17;
    while ( 1 )
    {
      v191 = v37;
      v199 = v38;
      v39 = sub_13C1210(a1, v36);
      v38 = v199;
      v40 = v191;
      v41 = v39;
      if ( v39 )
      {
        v3 = v199[1] & 7LL;
        v42 = v199[1] & 0xFFFFFFFFFFFFFFF8LL | v3 | *v39 & 3;
        v199[1] = v42;
        v43 = *v41;
        if ( (*v41 & 4) != 0 )
        {
          v199[1] = v42 | 4;
          v43 = *v41;
        }
        v44 = v43 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v44 )
        {
LABEL_61:
          if ( v182 == v40 )
            goto LABEL_91;
          goto LABEL_62;
        }
        v64 = *(_BYTE *)(v44 + 8) & 1;
        if ( *(_DWORD *)(v44 + 8) >> 1 )
        {
          if ( v64 )
          {
            v65 = v44 + 16;
            v66 = v44 + 272;
          }
          else
          {
            v65 = *(_QWORD *)(v44 + 16);
            v66 = v65 + 16LL * *(unsigned int *)(v44 + 24);
            if ( v65 == v66 )
              goto LABEL_61;
          }
          while ( *(_QWORD *)v65 == -8 || *(_QWORD *)v65 == -16 )
          {
            v65 += 16LL;
            if ( v65 == v66 )
              goto LABEL_61;
          }
        }
        else
        {
          if ( v64 )
          {
            v116 = v44 + 16;
            v117 = 256;
          }
          else
          {
            v116 = *(_QWORD *)(v44 + 16);
            v117 = 16LL * *(unsigned int *)(v44 + 24);
          }
          v65 = v117 + v116;
          v66 = v65;
        }
        if ( v65 == v66 )
          goto LABEL_61;
        v200 = v191;
        v67 = v38;
        while ( 1 )
        {
          v68 = *(_BYTE *)(v65 + 8);
          v69 = *(_QWORD *)v65;
          v70 = v67[1] & 0xFFFFFFFFFFFFFFF8LL;
          if ( !v70 )
          {
            v192 = *(_BYTE *)(v65 + 8);
            v76 = (_QWORD *)sub_22077B0(272);
            v68 = v192;
            v70 = (unsigned __int64)v76;
            if ( v76 )
            {
              *v76 = 0;
              v77 = v76 + 34;
              v78 = v76 + 2;
              *(v78 - 1) = 1;
              do
              {
                if ( v78 )
                  *v78 = -8;
                v78 += 2;
              }
              while ( v78 != v77 );
            }
            v67[1] = v70 | v67[1] & 7LL;
          }
          v71 = *(_BYTE *)(v70 + 8) & 1;
          if ( v71 )
          {
            v72 = v70 + 16;
            v3 = 15;
          }
          else
          {
            v3 = *(unsigned int *)(v70 + 24);
            v72 = *(_QWORD *)(v70 + 16);
            if ( !(_DWORD)v3 )
            {
              v79 = *(_DWORD *)(v70 + 8);
              ++*(_QWORD *)v70;
              v74 = 0;
              v80 = (v79 >> 1) + 1;
              goto LABEL_120;
            }
            v3 = (unsigned int)(v3 - 1);
          }
          v73 = v3 & (((unsigned int)v69 >> 9) ^ ((unsigned int)v69 >> 4));
          v74 = v72 + 16LL * v73;
          v75 = *(_QWORD *)v74;
          if ( v69 != *(_QWORD *)v74 )
            break;
          v68 |= *(_BYTE *)(v74 + 8);
LABEL_105:
          *(_BYTE *)(v74 + 8) = v68;
          do
          {
            v65 += 16LL;
            if ( v66 == v65 )
              goto LABEL_110;
          }
          while ( *(_QWORD *)v65 == -16 || *(_QWORD *)v65 == -8 );
          if ( v66 == v65 )
          {
LABEL_110:
            v40 = v200;
            v38 = v67;
            goto LABEL_61;
          }
        }
        v96 = 1;
        v97 = 0;
        while ( v75 != -8 )
        {
          if ( v97 || v75 != -16 )
            v74 = v97;
          v158 = v96 + 1;
          v73 = v3 & (v96 + v73);
          v159 = v72 + 16LL * v73;
          v75 = *(_QWORD *)v159;
          if ( v69 == *(_QWORD *)v159 )
          {
            v68 |= *(_BYTE *)(v159 + 8);
            v74 = v72 + 16LL * v73;
            goto LABEL_105;
          }
          v96 = v158;
          v97 = v74;
          v74 = v72 + 16LL * v73;
        }
        v79 = *(_DWORD *)(v70 + 8);
        v81 = 48;
        v3 = 16;
        if ( v97 )
          v74 = v97;
        ++*(_QWORD *)v70;
        v80 = (v79 >> 1) + 1;
        if ( !v71 )
        {
          v3 = *(unsigned int *)(v70 + 24);
LABEL_120:
          v81 = 3 * v3;
        }
        if ( 4 * v80 < v81 )
        {
          if ( (int)v3 - *(_DWORD *)(v70 + 12) - v80 > (unsigned int)v3 >> 3 )
          {
LABEL_123:
            *(_DWORD *)(v70 + 8) = (2 * (v79 >> 1) + 2) | v79 & 1;
            if ( *(_QWORD *)v74 != -8 )
              --*(_DWORD *)(v70 + 12);
            *(_QWORD *)v74 = v69;
            *(_BYTE *)(v74 + 8) = 0;
            goto LABEL_105;
          }
          v194 = v68;
          sub_13C4410(v70, v3);
          v68 = v194;
          if ( (*(_BYTE *)(v70 + 8) & 1) != 0 )
          {
            v111 = v70 + 16;
            v112 = 15;
            goto LABEL_181;
          }
          v148 = *(_DWORD *)(v70 + 24);
          v111 = *(_QWORD *)(v70 + 16);
          if ( v148 )
          {
            v112 = v148 - 1;
LABEL_181:
            v113 = v112 & (((unsigned int)v69 >> 9) ^ ((unsigned int)v69 >> 4));
            v74 = v111 + 16LL * v113;
            v3 = *(_QWORD *)v74;
            if ( v69 != *(_QWORD *)v74 )
            {
              v114 = 1;
              v115 = 0;
              while ( v3 != -8 )
              {
                if ( !v115 && v3 == -16 )
                  v115 = v74;
                v113 = v112 & (v114 + v113);
                v74 = v111 + 16LL * v113;
                v3 = *(_QWORD *)v74;
                if ( v69 == *(_QWORD *)v74 )
                  goto LABEL_178;
                ++v114;
              }
LABEL_184:
              if ( v115 )
                v74 = v115;
              goto LABEL_178;
            }
            goto LABEL_178;
          }
LABEL_375:
          *(_DWORD *)(v70 + 8) = (2 * (*(_DWORD *)(v70 + 8) >> 1) + 2) | *(_DWORD *)(v70 + 8) & 1;
          BUG();
        }
        v193 = v68;
        sub_13C4410(v70, 2 * v3);
        v68 = v193;
        if ( (*(_BYTE *)(v70 + 8) & 1) != 0 )
        {
          v108 = v70 + 16;
          v109 = 15;
        }
        else
        {
          v147 = *(_DWORD *)(v70 + 24);
          v108 = *(_QWORD *)(v70 + 16);
          if ( !v147 )
            goto LABEL_375;
          v109 = v147 - 1;
        }
        v110 = v109 & (((unsigned int)v69 >> 9) ^ ((unsigned int)v69 >> 4));
        v74 = v108 + 16LL * v110;
        v3 = *(_QWORD *)v74;
        if ( v69 != *(_QWORD *)v74 )
        {
          v160 = 1;
          v115 = 0;
          while ( v3 != -8 )
          {
            if ( !v115 && v3 == -16 )
              v115 = v74;
            v110 = v109 & (v160 + v110);
            v74 = v108 + 16LL * v110;
            v3 = *(_QWORD *)v74;
            if ( v69 == *(_QWORD *)v74 )
              goto LABEL_178;
            ++v160;
          }
          goto LABEL_184;
        }
LABEL_178:
        v79 = *(_DWORD *)(v70 + 8);
        goto LABEL_123;
      }
      v55 = a2 + 2;
      v56 = (_QWORD *)a2[3];
      if ( v56 )
      {
        v57 = a2 + 2;
        do
        {
          while ( 1 )
          {
            v58 = v56[2];
            v59 = v56[3];
            if ( v56[4] >= v36 )
              break;
            v56 = (_QWORD *)v56[3];
            if ( !v59 )
              goto LABEL_79;
          }
          v57 = v56;
          v56 = (_QWORD *)v56[2];
        }
        while ( v58 );
LABEL_79:
        if ( v55 != v57 && v57[4] <= v36 )
          v55 = v57;
      }
      v3 = (__int64)v218;
      v16 = v217;
      v60 = v55[5];
      v61 = ((char *)v218 - (char *)v217) >> 5;
      v62 = v218 - v217;
      if ( v61 <= 0 )
        break;
      v15 = (__int64 *)v217;
      v63 = (__int64 *)&v217[4 * v61];
      while ( 1 )
      {
        if ( v60 == *v15 )
          goto LABEL_89;
        if ( v60 == v15[1] )
        {
          if ( v218 != (__int64 **)++v15 )
            goto LABEL_90;
          goto LABEL_64;
        }
        if ( v60 == v15[2] )
        {
          v15 += 2;
          if ( v218 != (__int64 **)v15 )
            goto LABEL_90;
          goto LABEL_64;
        }
        if ( v60 == v15[3] )
          break;
        v15 += 4;
        if ( v63 == v15 )
        {
          v62 = ((char *)v218 - (char *)v15) >> 3;
          goto LABEL_265;
        }
      }
      v15 += 3;
      if ( v218 == (__int64 **)v15 )
        goto LABEL_64;
LABEL_90:
      if ( v182 == v191 )
      {
LABEL_91:
        v17 = v38;
        goto LABEL_14;
      }
LABEL_62:
      v45 = *(unsigned __int64 **)(v40 + 24);
      v37 = v40 + 32;
      v36 = *v45;
      if ( !*v45 )
      {
        v15 = (__int64 *)v218;
        v16 = v217;
        goto LABEL_64;
      }
    }
    v15 = (__int64 *)v217;
LABEL_265:
    if ( v62 == 2 )
      goto LABEL_275;
    if ( v62 != 3 )
    {
      if ( v62 != 1 )
      {
LABEL_268:
        v15 = (__int64 *)v218;
        goto LABEL_64;
      }
LABEL_277:
      if ( v60 == *v15 )
      {
        if ( v218 == (__int64 **)v15 )
          goto LABEL_64;
        goto LABEL_90;
      }
      goto LABEL_268;
    }
    if ( v60 != *v15 )
    {
      ++v15;
LABEL_275:
      if ( v60 != *v15 )
      {
        ++v15;
        goto LABEL_277;
      }
    }
LABEL_89:
    if ( v218 != (__int64 **)v15 )
      goto LABEL_90;
LABEL_64:
    if ( v16 != (__int64 **)v15 )
    {
      v46 = (__int64 **)v15;
      do
      {
        v47 = *(_DWORD *)(a1 + 288);
        if ( v47 )
        {
          v3 = (unsigned int)(v47 - 1);
          v48 = *(_QWORD *)(a1 + 272);
          v49 = 1;
          v50 = **v16;
          v51 = v3 & (((unsigned int)v50 >> 9) ^ ((unsigned int)v50 >> 4));
          v52 = (__int64 *)(v48 + 16LL * v51);
          v53 = *v52;
          if ( v50 == *v52 )
          {
LABEL_70:
            v54 = v52[1] & 0xFFFFFFFFFFFFFFF8LL;
            if ( v54 )
            {
              if ( (*(_BYTE *)(v54 + 8) & 1) == 0 )
                j___libc_free_0(*(_QWORD *)(v54 + 16));
              v3 = 272;
              j_j___libc_free_0(v54, 272);
            }
            *v52 = -16;
            --*(_DWORD *)(a1 + 280);
            ++*(_DWORD *)(a1 + 284);
          }
          else
          {
            while ( v53 != -8 )
            {
              v51 = v3 & (v49 + v51);
              v52 = (__int64 *)(v48 + 16LL * v51);
              v53 = *v52;
              if ( v50 == *v52 )
                goto LABEL_70;
              ++v49;
            }
          }
        }
        ++v16;
      }
      while ( v46 != v16 );
    }
LABEL_148:
    sub_13C6E30((__int64)v210);
    v4 = v217;
  }
  while ( v217 != v218 );
LABEL_149:
  if ( v220 )
    j_j___libc_free_0(v220, v222 - v220);
  if ( v217 )
    j_j___libc_free_0(v217, v219 - (_QWORD)v217);
  if ( v214 )
    j_j___libc_free_0(v214, v216 - v214);
  return j___libc_free_0(v211);
}
