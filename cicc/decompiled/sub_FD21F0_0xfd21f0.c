// Function: sub_FD21F0
// Address: 0xfd21f0
//
_QWORD *__fastcall sub_FD21F0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r11
  __int64 *v4; // rax
  __int64 v5; // rdx
  char v6; // cl
  _BYTE **v7; // rax
  _BYTE **v8; // r15
  _BYTE **v9; // rdx
  _BYTE *v10; // rsi
  _BYTE **v11; // r14
  __int64 v12; // rdi
  _BYTE **v13; // r13
  _BYTE *v14; // rsi
  _BYTE **v15; // rbx
  _QWORD *result; // rax
  __int64 v17; // r12
  __int64 v18; // r14
  _BYTE **v19; // rax
  _QWORD *v20; // rax
  _QWORD *v21; // rdx
  __int64 v22; // r14
  unsigned __int64 v24; // rax
  __int64 *v25; // rax
  _QWORD *v26; // rax
  __int64 v27; // rdx
  _QWORD *v28; // r13
  _QWORD *v29; // rbx
  _BYTE *v30; // r11
  unsigned int v31; // r15d
  __int64 v32; // rdx
  unsigned int v33; // esi
  __int64 v34; // r14
  unsigned int v35; // ecx
  unsigned int v36; // r8d
  _QWORD *v37; // rax
  _BYTE *v38; // rdi
  __int64 v39; // rdi
  unsigned int v40; // ecx
  __int64 v41; // rdx
  __int64 v42; // rax
  unsigned __int64 v43; // rcx
  __int64 v44; // rax
  int v45; // r10d
  __int64 *v46; // rcx
  unsigned int v47; // edi
  __int64 *v48; // rax
  _BYTE *v49; // r8
  unsigned int v50; // ecx
  __int64 v51; // rdx
  __int64 v52; // rax
  _QWORD *v53; // rax
  _QWORD *v54; // rdx
  int v55; // ecx
  int v56; // ecx
  __int64 v57; // rsi
  unsigned int v58; // edx
  __int64 *v59; // rax
  __int64 v60; // r8
  _BYTE *v61; // r9
  int j; // r10d
  int v63; // r10d
  _QWORD *v64; // rdi
  int v65; // eax
  int v66; // eax
  int v67; // eax
  int v68; // r8d
  __int64 v69; // r10
  unsigned int v70; // edx
  int v71; // eax
  _BYTE *v72; // rdi
  int v73; // r9d
  __int64 *v74; // rsi
  int v75; // eax
  int v76; // eax
  int v77; // r8d
  __int64 v78; // r10
  unsigned int v79; // edx
  int v80; // r9d
  _BYTE *v81; // rdi
  __int64 v82; // rbx
  char v83; // r13
  __int64 v84; // r12
  int v85; // eax
  __int64 v86; // rcx
  int v87; // edi
  unsigned int v88; // eax
  _BYTE *v89; // rdx
  _BYTE **v90; // rax
  int v91; // r8d
  _BYTE *v92; // rsi
  _BYTE *v93; // rsi
  int v94; // r13d
  __int64 *v95; // rdi
  int v96; // eax
  int v97; // eax
  unsigned int v98; // eax
  int v99; // eax
  int v100; // r8d
  __int64 v101; // r10
  unsigned int v102; // ecx
  int v103; // eax
  _QWORD *v104; // rdx
  __int64 v105; // rdi
  _BYTE *v106; // rdi
  int v107; // esi
  int v108; // eax
  int v109; // eax
  int v110; // r8d
  __int64 v111; // r10
  unsigned int v112; // ecx
  int v113; // r9d
  _QWORD *v114; // rsi
  _BYTE *v115; // rdi
  int v116; // eax
  int v117; // r8d
  __int64 v118; // r10
  unsigned int v119; // edx
  __int64 v120; // rsi
  int v121; // r9d
  _QWORD *v122; // rcx
  int v123; // eax
  int v124; // edi
  __int64 v125; // r9
  unsigned int v126; // edx
  _BYTE *v127; // rsi
  int v128; // r8d
  __int64 *v129; // rcx
  int v130; // eax
  int v131; // edi
  __int64 v132; // r9
  unsigned int v133; // edx
  int v134; // r8d
  __int64 v135; // rsi
  int v136; // eax
  int v137; // r8d
  __int64 v138; // r10
  unsigned int v139; // edx
  int v140; // r9d
  __int64 v141; // rsi
  _QWORD *v142; // rax
  _QWORD *v143; // rdx
  _QWORD *v144; // rdx
  _QWORD *v145; // rcx
  int v146; // r9d
  int v147; // r8d
  int v148; // edi
  int v149; // esi
  int *v150; // rdx
  int v151; // r10d
  int v152; // edx
  __int64 v153; // rdx
  int v154; // r10d
  int v155; // edx
  _QWORD *v156; // rax
  _QWORD *v157; // r12
  _QWORD *v158; // rbx
  __int64 v159; // r15
  __int64 v160; // r13
  __int64 v161; // r14
  __int64 v162; // r9
  int v163; // ecx
  unsigned __int64 v164; // rax
  int v165; // ecx
  unsigned int v166; // r10d
  unsigned __int64 v167; // r8
  int v168; // ecx
  int v169; // ecx
  unsigned __int64 v170; // rax
  int v171; // r9d
  __int64 v172; // r14
  int v173; // edi
  unsigned int v174; // r10d
  _QWORD *v175; // r8
  __int64 v176; // r9
  int v177; // r9d
  int v178; // eax
  int v179; // edi
  int v180; // [rsp+10h] [rbp-A0h]
  unsigned int v181; // [rsp+14h] [rbp-9Ch]
  __int64 v182; // [rsp+18h] [rbp-98h]
  __int64 v183; // [rsp+20h] [rbp-90h]
  unsigned int v184; // [rsp+28h] [rbp-88h]
  __int64 *v185; // [rsp+28h] [rbp-88h]
  __int64 *v187; // [rsp+38h] [rbp-78h]
  unsigned __int64 v188; // [rsp+40h] [rbp-70h]
  __int64 *v189; // [rsp+48h] [rbp-68h]
  int i; // [rsp+50h] [rbp-60h]
  __int64 *v191; // [rsp+50h] [rbp-60h]
  __int64 *v192; // [rsp+50h] [rbp-60h]
  unsigned __int64 v193; // [rsp+50h] [rbp-60h]
  unsigned int v194; // [rsp+50h] [rbp-60h]
  int v195; // [rsp+58h] [rbp-58h]
  unsigned int v196; // [rsp+5Ch] [rbp-54h]
  unsigned int v197; // [rsp+5Ch] [rbp-54h]
  _BYTE *v198; // [rsp+60h] [rbp-50h]
  _BYTE *v199; // [rsp+60h] [rbp-50h]
  __int64 v200; // [rsp+60h] [rbp-50h]
  _BYTE *v201; // [rsp+60h] [rbp-50h]
  unsigned int v202; // [rsp+60h] [rbp-50h]
  unsigned int v203; // [rsp+68h] [rbp-48h]
  _BYTE *v204; // [rsp+68h] [rbp-48h]
  _BYTE *v205; // [rsp+68h] [rbp-48h]
  __int64 v206; // [rsp+68h] [rbp-48h]
  unsigned int v207; // [rsp+68h] [rbp-48h]
  _BYTE *v208; // [rsp+68h] [rbp-48h]
  _BYTE *v209; // [rsp+68h] [rbp-48h]
  _BYTE *v210; // [rsp+68h] [rbp-48h]
  _BYTE *v211; // [rsp+68h] [rbp-48h]
  unsigned __int64 v212; // [rsp+68h] [rbp-48h]
  __int64 v213[7]; // [rsp+78h] [rbp-38h] BYREF

  v3 = a1;
  v4 = *(__int64 **)(a2 + 8);
  if ( *(_BYTE *)(a2 + 28) )
    v5 = *(unsigned int *)(a2 + 20);
  else
    v5 = *(unsigned int *)(a2 + 16);
  v189 = &v4[v5];
  if ( v4 == v189 )
    goto LABEL_6;
  while ( (unsigned __int64)*v4 >= 0xFFFFFFFFFFFFFFFELL )
  {
    if ( v189 == ++v4 )
      goto LABEL_6;
  }
  v187 = v4;
  if ( v189 == v4 )
  {
LABEL_6:
    v6 = *(_BYTE *)(a3 + 28);
    v7 = *(_BYTE ***)(a3 + 8);
    if ( !v6 )
      goto LABEL_45;
    goto LABEL_7;
  }
  v22 = *v4;
  v183 = a1 + 56;
  do
  {
    v24 = sub_FCD870(v22, *(_QWORD *)(*(_QWORD *)a1 + 40LL) + 312LL);
    if ( !*(_DWORD *)(a1 + 128) )
      goto LABEL_37;
    v195 = v24;
    v188 = HIDWORD(v24);
    v26 = *(_QWORD **)(a1 + 120);
    v27 = 2LL * *(unsigned int *)(a1 + 136);
    v28 = &v26[v27];
    if ( v26 == &v26[v27] )
      goto LABEL_37;
    while ( *v26 == -8192 || *v26 == -4096 )
    {
      v26 += 2;
      if ( v28 == v26 )
        goto LABEL_37;
    }
    if ( v28 == v26 )
    {
LABEL_37:
      if ( *(_BYTE *)v22 == 22 )
        goto LABEL_72;
      goto LABEL_38;
    }
    v29 = v26;
    v30 = (_BYTE *)v22;
    v31 = ((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4);
    do
    {
      v32 = *(_QWORD *)(a1 + 64);
      v33 = *(_DWORD *)(a1 + 80);
      v34 = v29[1];
      if ( !v33 )
        goto LABEL_67;
      v35 = v33 - 1;
      v36 = (v33 - 1) & v31;
      v196 = v33 - 1;
      v37 = (_QWORD *)(v32 + 16LL * v36);
      v38 = (_BYTE *)*v37;
      if ( (_BYTE *)*v37 == v30 )
      {
        v39 = *(_QWORD *)(*(_QWORD *)(v34 + 24) + 8LL * (*((_DWORD *)v37 + 2) >> 6));
        if ( _bittest64(&v39, *((unsigned int *)v37 + 2)) )
          goto LABEL_58;
        goto LABEL_62;
      }
      v198 = (_BYTE *)*v37;
      v184 = (v33 - 1) & v31;
      for ( i = 1; ; ++i )
      {
        if ( v198 == (_BYTE *)-4096LL )
          goto LABEL_84;
        v184 = v35 & (v184 + i);
        v198 = *(_BYTE **)(v32 + 16LL * v184);
        if ( v198 == v30 )
          break;
      }
      v185 = (__int64 *)(v32 + 16LL * (v35 & v31));
      v181 = (v33 - 1) & v31;
      v180 = 1;
      v191 = 0;
      while ( 1 )
      {
        if ( v38 == (_BYTE *)-4096LL )
        {
          v95 = v185;
          if ( v191 )
            v95 = v191;
          v96 = *(_DWORD *)(a1 + 72);
          ++*(_QWORD *)(a1 + 56);
          v97 = v96 + 1;
          v192 = v95;
          if ( 4 * v97 >= 3 * v33 )
          {
            v209 = v30;
            sub_CE2410(v183, 2 * v33);
            v123 = *(_DWORD *)(a1 + 80);
            if ( v123 )
            {
              v124 = v123 - 1;
              v125 = *(_QWORD *)(a1 + 64);
              v30 = v209;
              v126 = (v123 - 1) & v31;
              v192 = (__int64 *)(v125 + 16LL * v126);
              v127 = (_BYTE *)*v192;
              v97 = *(_DWORD *)(a1 + 72) + 1;
              if ( (_BYTE *)*v192 == v209 )
                goto LABEL_151;
              v128 = 1;
              v129 = 0;
              while ( v127 != (_BYTE *)-4096LL )
              {
                if ( !v129 && v127 == (_BYTE *)-8192LL )
                  v129 = v192;
                v126 = v124 & (v128 + v126);
                v192 = (__int64 *)(v125 + 16LL * v126);
                v127 = (_BYTE *)*v192;
                if ( (_BYTE *)*v192 == v209 )
                  goto LABEL_151;
                ++v128;
              }
LABEL_198:
              if ( !v129 )
                v129 = v192;
              v192 = v129;
              goto LABEL_151;
            }
          }
          else
          {
            if ( v33 - *(_DWORD *)(a1 + 76) - v97 > v33 >> 3 )
            {
LABEL_151:
              *(_DWORD *)(a1 + 72) = v97;
              if ( *v192 != -4096 )
                --*(_DWORD *)(a1 + 76);
              *v192 = (__int64)v30;
              *((_DWORD *)v192 + 2) = 0;
              if ( (**(_BYTE **)(v34 + 24) & 1) == 0 )
                goto LABEL_60;
              v98 = *(_DWORD *)(a1 + 80);
              v207 = v98;
              if ( v98 )
              {
                v196 = v98 - 1;
                v36 = v31 & (v98 - 1);
                v200 = *(_QWORD *)(a1 + 64);
                v37 = (_QWORD *)(v200 + 16LL * v36);
                v106 = (_BYTE *)*v37;
                if ( v30 == (_BYTE *)*v37 )
                  goto LABEL_58;
                goto LABEL_164;
              }
              ++*(_QWORD *)(a1 + 56);
              goto LABEL_156;
            }
            v210 = v30;
            sub_CE2410(v183, v33);
            v130 = *(_DWORD *)(a1 + 80);
            if ( v130 )
            {
              v131 = v130 - 1;
              v132 = *(_QWORD *)(a1 + 64);
              v30 = v210;
              v133 = (v130 - 1) & v31;
              v134 = 1;
              v129 = 0;
              v192 = (__int64 *)(v132 + 16LL * v133);
              v135 = *v192;
              v97 = *(_DWORD *)(a1 + 72) + 1;
              if ( v210 == (_BYTE *)*v192 )
                goto LABEL_151;
              while ( v135 != -4096 )
              {
                if ( v135 == -8192 && !v129 )
                  v129 = v192;
                v133 = v131 & (v134 + v133);
                v192 = (__int64 *)(v132 + 16LL * v133);
                v135 = *v192;
                if ( (_BYTE *)*v192 == v210 )
                  goto LABEL_151;
                ++v134;
              }
              goto LABEL_198;
            }
          }
          ++*(_DWORD *)(a1 + 72);
          BUG();
        }
        if ( v38 != (_BYTE *)-8192LL || v191 )
          v185 = v191;
        v181 = v35 & (v180 + v181);
        v182 = v32 + 16LL * v181;
        v38 = *(_BYTE **)v182;
        if ( *(_BYTE **)v182 == v30 )
          break;
        ++v180;
        v191 = v185;
        v185 = (__int64 *)(v32 + 16LL * v181);
      }
      v200 = *(_QWORD *)(a1 + 64);
      v207 = *(_DWORD *)(a1 + 80);
      v106 = (_BYTE *)*v37;
      v37 = (_QWORD *)(v32 + 16LL * (v35 & v31));
      v194 = *(_DWORD *)(v182 + 8);
      v176 = *(_QWORD *)(*(_QWORD *)(v34 + 24) + 8LL * (v194 >> 6));
      if ( !_bittest64(&v176, v194) )
        goto LABEL_62;
LABEL_164:
      v107 = 1;
      v104 = 0;
      while ( 2 )
      {
        if ( v106 == (_BYTE *)-4096LL )
        {
          if ( !v104 )
            v104 = v37;
          v108 = *(_DWORD *)(a1 + 72);
          ++*(_QWORD *)(a1 + 56);
          v103 = v108 + 1;
          if ( 4 * v103 < 3 * v207 )
          {
            if ( v207 - (v103 + *(_DWORD *)(a1 + 76)) > v207 >> 3 )
              goto LABEL_158;
            v201 = v30;
            sub_CE2410(v183, v207);
            v109 = *(_DWORD *)(a1 + 80);
            if ( v109 )
            {
              v110 = v109 - 1;
              v111 = *(_QWORD *)(a1 + 64);
              v30 = v201;
              v112 = v110 & v31;
              v113 = 1;
              v114 = 0;
              v103 = *(_DWORD *)(a1 + 72) + 1;
              v104 = (_QWORD *)(v111 + 16LL * (v110 & v31));
              v115 = (_BYTE *)*v104;
              if ( v201 != (_BYTE *)*v104 )
              {
                while ( v115 != (_BYTE *)-4096LL )
                {
                  if ( !v114 && v115 == (_BYTE *)-8192LL )
                    v114 = v104;
                  v112 = v110 & (v113 + v112);
                  v104 = (_QWORD *)(v111 + 16LL * v112);
                  v115 = (_BYTE *)*v104;
                  if ( (_BYTE *)*v104 == v201 )
                    goto LABEL_158;
                  ++v113;
                }
LABEL_173:
                if ( v114 )
                  v104 = v114;
              }
LABEL_158:
              *(_DWORD *)(a1 + 72) = v103;
              if ( *v104 != -4096 )
                --*(_DWORD *)(a1 + 76);
              *v104 = v30;
              v42 = 0;
              *((_DWORD *)v104 + 2) = 0;
              v41 = -2;
              goto LABEL_59;
            }
LABEL_338:
            ++*(_DWORD *)(a1 + 72);
            BUG();
          }
LABEL_156:
          v199 = v30;
          sub_CE2410(v183, 2 * v207);
          v99 = *(_DWORD *)(a1 + 80);
          if ( v99 )
          {
            v100 = v99 - 1;
            v101 = *(_QWORD *)(a1 + 64);
            v30 = v199;
            v102 = v100 & v31;
            v103 = *(_DWORD *)(a1 + 72) + 1;
            v104 = (_QWORD *)(v101 + 16LL * (v100 & v31));
            v105 = *v104;
            if ( v199 != (_BYTE *)*v104 )
            {
              v177 = 1;
              v114 = 0;
              while ( v105 != -4096 )
              {
                if ( !v114 && v105 == -8192 )
                  v114 = v104;
                v102 = v100 & (v177 + v102);
                v104 = (_QWORD *)(v101 + 16LL * v102);
                v105 = *v104;
                if ( (_BYTE *)*v104 == v199 )
                  goto LABEL_158;
                ++v177;
              }
              goto LABEL_173;
            }
            goto LABEL_158;
          }
          goto LABEL_338;
        }
        if ( v106 != (_BYTE *)-8192LL || v104 )
          v37 = v104;
        v36 = v196 & (v107 + v36);
        v106 = *(_BYTE **)(v200 + 16LL * v36);
        if ( v106 != v30 )
        {
          ++v107;
          v104 = v37;
          v37 = (_QWORD *)(v200 + 16LL * v36);
          continue;
        }
        break;
      }
      v37 = (_QWORD *)(v200 + 16LL * v36);
LABEL_58:
      v40 = *((_DWORD *)v37 + 2);
      v41 = ~(1LL << v40);
      v42 = 8LL * (v40 >> 6);
LABEL_59:
      *(_QWORD *)(*(_QWORD *)(v34 + 24) + v42) &= v41;
      *(_DWORD *)(v34 + 12) -= v188;
      *(_DWORD *)(v34 + 8) -= v195;
      *(_DWORD *)v34 -= v195;
      *(_DWORD *)(v34 + 4) -= v188;
LABEL_60:
      v33 = *(_DWORD *)(a1 + 80);
      v34 = v29[1];
      v32 = *(_QWORD *)(a1 + 64);
      if ( !v33 )
        goto LABEL_67;
      v35 = v33 - 1;
LABEL_62:
      v36 = v35 & v31;
      v37 = (_QWORD *)(v32 + 16LL * (v35 & v31));
      v38 = (_BYTE *)*v37;
      if ( v30 != (_BYTE *)*v37 )
      {
LABEL_84:
        v203 = v36;
        v61 = v38;
        for ( j = 1; ; ++j )
        {
          if ( v38 == (_BYTE *)-4096LL )
            goto LABEL_67;
          v36 = v35 & (j + v36);
          v38 = *(_BYTE **)(v32 + 16LL * v36);
          if ( v38 == v30 )
            break;
        }
        v63 = 1;
        v64 = 0;
        while ( v61 != (_BYTE *)-4096LL )
        {
          if ( v61 != (_BYTE *)-8192LL || v64 )
            v37 = v64;
          v173 = v63 + 1;
          v174 = v35 & (v203 + v63);
          v203 = v174;
          v175 = (_QWORD *)(v32 + 16LL * v174);
          v61 = (_BYTE *)*v175;
          if ( v30 == (_BYTE *)*v175 )
          {
            v37 = (_QWORD *)(v32 + 16LL * v174);
            goto LABEL_63;
          }
          v63 = v173;
          v64 = v37;
          v37 = v175;
        }
        if ( !v64 )
          v64 = v37;
        v65 = *(_DWORD *)(a1 + 72);
        ++*(_QWORD *)(a1 + 56);
        v66 = v65 + 1;
        if ( 4 * v66 >= 3 * v33 )
        {
          v208 = v30;
          sub_CE2410(v183, 2 * v33);
          v116 = *(_DWORD *)(a1 + 80);
          if ( v116 )
          {
            v117 = v116 - 1;
            v118 = *(_QWORD *)(a1 + 64);
            v30 = v208;
            v119 = v117 & v31;
            v66 = *(_DWORD *)(a1 + 72) + 1;
            v64 = (_QWORD *)(v118 + 16LL * (v117 & v31));
            v120 = *v64;
            if ( (_BYTE *)*v64 == v208 )
              goto LABEL_93;
            v121 = 1;
            v122 = 0;
            while ( v120 != -4096 )
            {
              if ( !v122 && v120 == -8192 )
                v122 = v64;
              v119 = v117 & (v121 + v119);
              v64 = (_QWORD *)(v118 + 16LL * v119);
              v120 = *v64;
              if ( v208 == (_BYTE *)*v64 )
                goto LABEL_93;
              ++v121;
            }
LABEL_204:
            if ( v122 )
              v64 = v122;
            goto LABEL_93;
          }
        }
        else
        {
          if ( v33 - (v66 + *(_DWORD *)(a1 + 76)) > v33 >> 3 )
          {
LABEL_93:
            *(_DWORD *)(a1 + 72) = v66;
            if ( *v64 != -4096 )
              --*(_DWORD *)(a1 + 76);
            *v64 = v30;
            *((_DWORD *)v64 + 2) = 0;
            if ( (**(_BYTE **)(v34 + 96) & 1) == 0 )
              goto LABEL_67;
            v33 = *(_DWORD *)(a1 + 80);
            if ( v33 )
            {
              v32 = *(_QWORD *)(a1 + 64);
              goto LABEL_64;
            }
            ++*(_QWORD *)(a1 + 56);
            goto LABEL_98;
          }
          v211 = v30;
          sub_CE2410(v183, v33);
          v136 = *(_DWORD *)(a1 + 80);
          if ( v136 )
          {
            v137 = v136 - 1;
            v138 = *(_QWORD *)(a1 + 64);
            v30 = v211;
            v139 = v137 & v31;
            v140 = 1;
            v122 = 0;
            v66 = *(_DWORD *)(a1 + 72) + 1;
            v64 = (_QWORD *)(v138 + 16LL * (v137 & v31));
            v141 = *v64;
            if ( v211 == (_BYTE *)*v64 )
              goto LABEL_93;
            while ( v141 != -4096 )
            {
              if ( v141 == -8192 && !v122 )
                v122 = v64;
              v139 = v137 & (v140 + v139);
              v64 = (_QWORD *)(v138 + 16LL * v139);
              v141 = *v64;
              if ( v211 == (_BYTE *)*v64 )
                goto LABEL_93;
              ++v140;
            }
            goto LABEL_204;
          }
        }
        ++*(_DWORD *)(a1 + 72);
        BUG();
      }
LABEL_63:
      v43 = *((unsigned int *)v37 + 2);
      v44 = *(_QWORD *)(*(_QWORD *)(v34 + 96) + 8LL * (*((_DWORD *)v37 + 2) >> 6));
      if ( !_bittest64(&v44, v43) )
        goto LABEL_67;
LABEL_64:
      v45 = 1;
      v46 = 0;
      v47 = (v33 - 1) & v31;
      v48 = (__int64 *)(v32 + 16LL * v47);
      v49 = (_BYTE *)*v48;
      if ( v30 == (_BYTE *)*v48 )
        goto LABEL_65;
      while ( 2 )
      {
        if ( v49 == (_BYTE *)-4096LL )
        {
          if ( !v46 )
            v46 = v48;
          v75 = *(_DWORD *)(a1 + 72);
          ++*(_QWORD *)(a1 + 56);
          v71 = v75 + 1;
          if ( 4 * v71 < 3 * v33 )
          {
            if ( v33 - (v71 + *(_DWORD *)(a1 + 76)) > v33 >> 3 )
              goto LABEL_111;
            v205 = v30;
            sub_CE2410(v183, v33);
            v76 = *(_DWORD *)(a1 + 80);
            if ( v76 )
            {
              v77 = v76 - 1;
              v78 = *(_QWORD *)(a1 + 64);
              v30 = v205;
              v79 = v77 & v31;
              v80 = 1;
              v74 = 0;
              v71 = *(_DWORD *)(a1 + 72) + 1;
              v46 = (__int64 *)(v78 + 16LL * (v77 & v31));
              v81 = (_BYTE *)*v46;
              if ( v205 == (_BYTE *)*v46 )
                goto LABEL_111;
              while ( v81 != (_BYTE *)-4096LL )
              {
                if ( !v74 && v81 == (_BYTE *)-8192LL )
                  v74 = v46;
                v79 = v77 & (v80 + v79);
                v46 = (__int64 *)(v78 + 16LL * v79);
                v81 = (_BYTE *)*v46;
                if ( v205 == (_BYTE *)*v46 )
                  goto LABEL_111;
                ++v80;
              }
              goto LABEL_102;
            }
LABEL_340:
            ++*(_DWORD *)(a1 + 72);
            BUG();
          }
LABEL_98:
          v204 = v30;
          sub_CE2410(v183, 2 * v33);
          v67 = *(_DWORD *)(a1 + 80);
          if ( v67 )
          {
            v68 = v67 - 1;
            v69 = *(_QWORD *)(a1 + 64);
            v30 = v204;
            v70 = v68 & v31;
            v71 = *(_DWORD *)(a1 + 72) + 1;
            v46 = (__int64 *)(v69 + 16LL * (v68 & v31));
            v72 = (_BYTE *)*v46;
            if ( v204 == (_BYTE *)*v46 )
            {
LABEL_111:
              *(_DWORD *)(a1 + 72) = v71;
              if ( *v46 != -4096 )
                --*(_DWORD *)(a1 + 76);
              *v46 = (__int64)v30;
              v51 = -2;
              v52 = 0;
              *((_DWORD *)v46 + 2) = 0;
              goto LABEL_66;
            }
            v73 = 1;
            v74 = 0;
            while ( v72 != (_BYTE *)-4096LL )
            {
              if ( !v74 && v72 == (_BYTE *)-8192LL )
                v74 = v46;
              v70 = v68 & (v73 + v70);
              v46 = (__int64 *)(v69 + 16LL * v70);
              v72 = (_BYTE *)*v46;
              if ( v204 == (_BYTE *)*v46 )
                goto LABEL_111;
              ++v73;
            }
LABEL_102:
            if ( v74 )
              v46 = v74;
            goto LABEL_111;
          }
          goto LABEL_340;
        }
        if ( v46 || v49 != (_BYTE *)-8192LL )
          v48 = v46;
        v47 = (v33 - 1) & (v45 + v47);
        v49 = *(_BYTE **)(v32 + 16LL * v47);
        if ( v30 != v49 )
        {
          ++v45;
          v46 = v48;
          v48 = (__int64 *)(v32 + 16LL * v47);
          continue;
        }
        break;
      }
      v48 = (__int64 *)(v32 + 16LL * v47);
LABEL_65:
      v50 = *((_DWORD *)v48 + 2);
      v51 = ~(1LL << v50);
      v52 = 8LL * (v50 >> 6);
LABEL_66:
      *(_QWORD *)(*(_QWORD *)(v34 + 96) + v52) &= v51;
LABEL_67:
      v29 += 2;
      if ( v29 == v28 )
        break;
      while ( *v29 == -8192 || *v29 == -4096 )
      {
        v29 += 2;
        if ( v28 == v29 )
          goto LABEL_71;
      }
    }
    while ( v28 != v29 );
LABEL_71:
    v22 = (__int64)v30;
    if ( *v30 != 22 )
      goto LABEL_38;
LABEL_72:
    if ( !*(_BYTE *)(a1 + 204) )
    {
      if ( sub_C8CA60(a1 + 176, v22) )
        goto LABEL_77;
LABEL_38:
      if ( *(_QWORD *)(v22 + 16) && *(_BYTE *)v22 > 0x1Cu )
        sub_FD1250(a1, v22);
      goto LABEL_41;
    }
    v53 = *(_QWORD **)(a1 + 184);
    v54 = &v53[*(unsigned int *)(a1 + 196)];
    if ( v53 == v54 )
      goto LABEL_41;
    while ( v22 != *v53 )
    {
      if ( v54 == ++v53 )
        goto LABEL_41;
    }
LABEL_77:
    v55 = *(_DWORD *)(a1 + 80);
    if ( v55 )
    {
      v56 = v55 - 1;
      v57 = *(_QWORD *)(a1 + 64);
      v58 = v56 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
      v59 = (__int64 *)(v57 + 16LL * v58);
      v60 = *v59;
      if ( *v59 == v22 )
      {
LABEL_79:
        *v59 = -8192;
        --*(_DWORD *)(a1 + 72);
        ++*(_DWORD *)(a1 + 76);
      }
      else
      {
        v178 = 1;
        while ( v60 != -4096 )
        {
          v179 = v178 + 1;
          v58 = v56 & (v178 + v58);
          v59 = (__int64 *)(v57 + 16LL * v58);
          v60 = *v59;
          if ( v22 == *v59 )
            goto LABEL_79;
          v178 = v179;
        }
      }
    }
LABEL_41:
    v25 = v187 + 1;
    if ( v187 + 1 == v189 )
      break;
    while ( 1 )
    {
      v22 = *v25;
      if ( (unsigned __int64)*v25 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v189 == ++v25 )
        goto LABEL_44;
    }
    v187 = v25;
  }
  while ( v189 != v25 );
LABEL_44:
  v3 = a1;
  v6 = *(_BYTE *)(a3 + 28);
  v7 = *(_BYTE ***)(a3 + 8);
  if ( !v6 )
  {
LABEL_45:
    v8 = &v7[*(unsigned int *)(a3 + 16)];
    goto LABEL_8;
  }
LABEL_7:
  v8 = &v7[*(unsigned int *)(a3 + 20)];
LABEL_8:
  if ( v7 == v8 )
    goto LABEL_12;
  v9 = v7;
  while ( 1 )
  {
    v10 = *v9;
    v11 = v9;
    if ( (unsigned __int64)*v9 < 0xFFFFFFFFFFFFFFFELL )
      break;
    if ( v8 == ++v9 )
      goto LABEL_12;
  }
  if ( v9 == v8 )
  {
LABEL_12:
    v12 = a3;
    if ( !v6 )
      goto LABEL_131;
    goto LABEL_13;
  }
  v82 = v3 + 56;
  v83 = 0;
  v206 = v3 + 176;
  v84 = v3;
  do
  {
    v85 = *(_DWORD *)(v84 + 80);
    v86 = *(_QWORD *)(v84 + 64);
    v213[0] = (__int64)v10;
    if ( v85 )
    {
      v87 = v85 - 1;
      v88 = (v85 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v89 = *(_BYTE **)(v86 + 16LL * v88);
      if ( v89 == v10 )
        goto LABEL_126;
      v91 = 1;
      while ( v89 != (_BYTE *)-4096LL )
      {
        v88 = v87 & (v91 + v88);
        v89 = *(_BYTE **)(v86 + 16LL * v88);
        if ( v89 == v10 )
          goto LABEL_126;
        ++v91;
      }
    }
    if ( *v10 != 22 )
      goto LABEL_137;
    if ( *(_BYTE *)(v84 + 204) )
    {
      v142 = *(_QWORD **)(v84 + 184);
      v143 = &v142[*(unsigned int *)(v84 + 196)];
      if ( v142 == v143 )
        goto LABEL_137;
      while ( (_BYTE *)*v142 != v10 )
      {
        if ( v143 == ++v142 )
          goto LABEL_137;
      }
    }
    else if ( !sub_C8CA60(v206, (__int64)v10) )
    {
LABEL_137:
      v92 = *(_BYTE **)(v84 + 96);
      if ( v92 == *(_BYTE **)(v84 + 104) )
      {
        sub_9281F0(v84 + 88, v92, v213);
        v93 = *(_BYTE **)(v84 + 96);
      }
      else
      {
        if ( v92 )
        {
          *(_QWORD *)v92 = v213[0];
          v92 = *(_BYTE **)(v84 + 96);
        }
        v93 = v92 + 8;
        *(_QWORD *)(v84 + 96) = v93;
      }
      v94 = ((__int64)&v93[-*(_QWORD *)(v84 + 88)] >> 3) - 1;
      *(_DWORD *)sub_FCF1D0(v82, v213) = v94;
      v83 = 1;
    }
LABEL_126:
    v90 = v11 + 1;
    if ( v11 + 1 == v8 )
      break;
    while ( 1 )
    {
      v10 = *v90;
      v11 = v90;
      if ( (unsigned __int64)*v90 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v8 == ++v90 )
        goto LABEL_129;
    }
  }
  while ( v8 != v90 );
LABEL_129:
  v3 = v84;
  if ( v83 )
  {
    if ( *(_DWORD *)(v84 + 128) )
    {
      v156 = *(_QWORD **)(v84 + 120);
      v157 = &v156[2 * *(unsigned int *)(v84 + 136)];
      if ( v156 != v157 )
      {
        while ( 1 )
        {
          v158 = v156;
          if ( *v156 != -4096 && *v156 != -8192 )
            break;
          v156 += 2;
          if ( v157 == v156 )
            goto LABEL_130;
        }
        if ( v157 != v156 )
        {
          v159 = v3;
          do
          {
            v160 = v158[1];
            v161 = (__int64)(*(_QWORD *)(v159 + 96) - *(_QWORD *)(v159 + 88)) >> 3;
            v162 = (unsigned int)v161;
            v163 = *(_DWORD *)(v160 + 88) & 0x3F;
            if ( v163 )
              *(_QWORD *)(*(_QWORD *)(v160 + 24) + 8LL * *(unsigned int *)(v160 + 32) - 8) &= ~(-1LL << v163);
            v164 = *(unsigned int *)(v160 + 32);
            *(_DWORD *)(v160 + 88) = v161;
            LOBYTE(v165) = v161;
            v166 = (unsigned int)(v161 + 63) >> 6;
            v167 = v166;
            if ( v166 != v164 )
            {
              if ( v166 >= v164 )
              {
                v212 = v166 - v164;
                if ( v166 > (unsigned __int64)*(unsigned int *)(v160 + 36) )
                {
                  sub_C8D5F0(
                    v160 + 24,
                    (const void *)(v160 + 40),
                    (unsigned int)(v161 + 63) >> 6,
                    8u,
                    v166,
                    (unsigned int)v161);
                  v164 = *(unsigned int *)(v160 + 32);
                  v166 = (unsigned int)(v161 + 63) >> 6;
                  v162 = (unsigned int)v161;
                  v167 = v166;
                }
                if ( 8 * v212 )
                {
                  v193 = v167;
                  v197 = v166;
                  v202 = v162;
                  memset((void *)(*(_QWORD *)(v160 + 24) + 8 * v164), 0, 8 * v212);
                  LODWORD(v164) = *(_DWORD *)(v160 + 32);
                  v167 = v193;
                  v166 = v197;
                  v162 = v202;
                }
                v165 = *(_DWORD *)(v160 + 88);
                *(_DWORD *)(v160 + 32) = v212 + v164;
              }
              else
              {
                *(_DWORD *)(v160 + 32) = v166;
              }
            }
            v168 = v165 & 0x3F;
            if ( v168 )
              *(_QWORD *)(*(_QWORD *)(v160 + 24) + 8LL * *(unsigned int *)(v160 + 32) - 8) &= ~(-1LL << v168);
            v169 = *(_DWORD *)(v160 + 160) & 0x3F;
            if ( v169 )
              *(_QWORD *)(*(_QWORD *)(v160 + 96) + 8LL * *(unsigned int *)(v160 + 104) - 8) &= ~(-1LL << v169);
            v170 = *(unsigned int *)(v160 + 104);
            *(_DWORD *)(v160 + 160) = v161;
            if ( v167 != v170 )
            {
              if ( v167 >= v170 )
              {
                v172 = v167 - v170;
                if ( v167 > *(unsigned int *)(v160 + 108) )
                {
                  sub_C8D5F0(v160 + 96, (const void *)(v160 + 112), v167, 8u, v167, v162);
                  v170 = *(unsigned int *)(v160 + 104);
                }
                if ( 8 * v172 )
                {
                  memset((void *)(*(_QWORD *)(v160 + 96) + 8 * v170), 0, 8 * v172);
                  LODWORD(v170) = *(_DWORD *)(v160 + 104);
                }
                LODWORD(v162) = *(_DWORD *)(v160 + 160);
                *(_DWORD *)(v160 + 104) = v172 + v170;
              }
              else
              {
                *(_DWORD *)(v160 + 104) = v166;
              }
            }
            v171 = v162 & 0x3F;
            if ( v171 )
              *(_QWORD *)(*(_QWORD *)(v160 + 96) + 8LL * *(unsigned int *)(v160 + 104) - 8) &= ~(-1LL << v171);
            v158 += 2;
            if ( v158 == v157 )
              break;
            while ( *v158 == -8192 || *v158 == -4096 )
            {
              v158 += 2;
              if ( v157 == v158 )
                goto LABEL_263;
            }
          }
          while ( v158 != v157 );
LABEL_263:
          v3 = v159;
        }
      }
    }
  }
LABEL_130:
  v7 = *(_BYTE ***)(a3 + 8);
  v12 = a3;
  if ( *(_BYTE *)(a3 + 28) )
  {
LABEL_13:
    v13 = &v7[*(unsigned int *)(v12 + 20)];
    goto LABEL_14;
  }
LABEL_131:
  v13 = &v7[*(unsigned int *)(v12 + 16)];
LABEL_14:
  if ( v7 != v13 )
  {
    while ( 1 )
    {
      v14 = *v7;
      v15 = v7;
      if ( (unsigned __int64)*v7 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v13 == ++v7 )
        goto LABEL_17;
    }
    if ( v13 != v7 )
    {
      v17 = v3 + 176;
      v18 = v3;
      if ( *v14 == 22 )
        goto LABEL_28;
LABEL_21:
      sub_FD1250(v18, (__int64)v14);
      while ( 1 )
      {
        v19 = v15 + 1;
        if ( v15 + 1 == v13 )
          break;
        while ( 1 )
        {
          v14 = *v19;
          v15 = v19;
          if ( (unsigned __int64)*v19 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v13 == ++v19 )
            goto LABEL_25;
        }
        if ( v19 == v13 )
          break;
        if ( *v14 != 22 )
          goto LABEL_21;
LABEL_28:
        if ( *(_BYTE *)(v18 + 204) )
        {
          v20 = *(_QWORD **)(v18 + 184);
          v21 = &v20[*(unsigned int *)(v18 + 196)];
          if ( v20 == v21 )
            goto LABEL_21;
          while ( v14 != (_BYTE *)*v20 )
          {
            if ( v21 == ++v20 )
              goto LABEL_21;
          }
        }
        else if ( !sub_C8CA60(v17, (__int64)v14) )
        {
          goto LABEL_21;
        }
      }
LABEL_25:
      v3 = v18;
    }
  }
LABEL_17:
  result = (_QWORD *)*(unsigned int *)(v3 + 128);
  *(_QWORD *)(v3 + 24) = 0;
  *(_QWORD *)(v3 + 32) = 0;
  if ( (_DWORD)result )
  {
    v144 = *(_QWORD **)(v3 + 120);
    v145 = &v144[2 * *(unsigned int *)(v3 + 136)];
    if ( v144 != v145 )
    {
      while ( 1 )
      {
        result = v144;
        if ( *v144 != -8192 && *v144 != -4096 )
          break;
        v144 += 2;
        if ( v145 == v144 )
          return result;
      }
      if ( v145 != v144 )
      {
        v146 = 0;
        v147 = 0;
        v148 = 0;
        v149 = 0;
        do
        {
          v150 = (int *)result[1];
          v151 = *v150;
          v152 = v150[1];
          if ( v149 < v151 )
            v149 = v151;
          if ( v148 < v152 )
            v148 = v152;
          *(_DWORD *)(v3 + 24) = v149;
          *(_DWORD *)(v3 + 28) = v148;
          v153 = result[1];
          v154 = *(_DWORD *)(v153 + 8);
          v155 = *(_DWORD *)(v153 + 12);
          if ( v147 < v154 )
            v147 = v154;
          if ( v146 < v155 )
            v146 = v155;
          result += 2;
          *(_DWORD *)(v3 + 32) = v147;
          *(_DWORD *)(v3 + 36) = v146;
          if ( result == v145 )
            break;
          while ( *result == -4096 || *result == -8192 )
          {
            result += 2;
            if ( v145 == result )
              return result;
          }
        }
        while ( result != v145 );
      }
    }
  }
  return result;
}
