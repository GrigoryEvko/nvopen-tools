// Function: sub_1858B90
// Address: 0x1858b90
//
_QWORD *__fastcall sub_1858B90(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  char v5; // al
  int v6; // r9d
  __int64 v7; // r15
  unsigned __int64 v8; // r14
  _QWORD *v9; // rax
  __int64 v10; // r15
  unsigned __int64 v11; // r14
  _QWORD *v12; // rax
  __int64 v13; // r15
  __int64 v14; // r14
  unsigned __int64 v15; // r13
  _QWORD *v16; // rax
  __int64 *v17; // rax
  __int64 (__fastcall *v18)(__int64); // rax
  __int64 (__fastcall **v19)(__int64); // r13
  __int64 **v20; // rdi
  __int64 (__fastcall **v21)(__int64); // r14
  __int64 v22; // rax
  __int64 v23; // r15
  __int64 (__fastcall *v24)(__int64); // rdi
  __int64 v25; // rcx
  int v26; // r8d
  int v27; // r9d
  char v28; // al
  __int64 (__fastcall *v29)(__int64); // rax
  __int64 (__fastcall **v30)(__int64); // r15
  __int64 **v31; // rdi
  __int64 (__fastcall **v32)(__int64); // r13
  __int64 (__fastcall *v33)(__int64); // rdi
  __int64 v34; // rbx
  char v35; // r13
  __int64 v36; // rcx
  int v37; // r8d
  int v38; // r9d
  char v39; // al
  __int64 v40; // rbx
  char v41; // r13
  __int64 v42; // rcx
  int v43; // r8d
  int v44; // r9d
  char v45; // al
  __int64 (__fastcall **v46)(__int64 *); // rbx
  __int64 v47; // rax
  __int64 (__fastcall **v48)(__int64 *); // r15
  __int64 *v49; // rbx
  _QWORD *v50; // r14
  _QWORD *v51; // rax
  __int64 v52; // rax
  _QWORD *v53; // rdx
  __int64 v54; // r15
  _QWORD *v55; // rdx
  __int64 *v56; // rsi
  __int64 v57; // r14
  __int64 *v58; // r14
  _QWORD *v59; // rbx
  _QWORD *v60; // rax
  __int64 v61; // rax
  _QWORD *v62; // rdx
  __int64 v63; // r15
  __int64 *v64; // rsi
  char v65; // al
  __int64 v66; // rbx
  __int64 v67; // r15
  _QWORD *v68; // r12
  _QWORD *v69; // rax
  __int64 v70; // rax
  _QWORD *v71; // rdx
  __int64 v72; // r14
  __int64 *v73; // rsi
  __int64 v74; // rbx
  __int64 v75; // r15
  _QWORD *v76; // r12
  _QWORD *v77; // rax
  __int64 v78; // rax
  _QWORD *v79; // rdx
  __int64 v80; // r14
  __int64 *v81; // rsi
  __int64 v82; // rdx
  unsigned __int64 v83; // rax
  __int64 *v84; // rbx
  __int64 *v85; // r14
  __int64 v86; // r13
  __int64 *v87; // rbx
  __int64 *v88; // r13
  __int64 v89; // r14
  __int64 *v90; // rbx
  __int64 *v91; // r13
  __int64 v92; // r14
  __int64 *v93; // rbx
  __int64 *v94; // r13
  __int64 v95; // r14
  void *v96; // rdi
  unsigned int v97; // eax
  __int64 v98; // rdx
  int v99; // r15d
  _QWORD *v100; // rbx
  unsigned int v101; // edx
  _QWORD *v102; // r14
  unsigned __int64 v103; // rdi
  _QWORD *v104; // rbx
  _QWORD *v105; // rdi
  _QWORD *v106; // rsi
  _QWORD *v107; // rdx
  __int64 v109; // rdx
  __int64 v110; // rsi
  __int64 v111; // rdx
  __int64 v112; // rsi
  __int64 v113; // rdx
  __int64 v114; // rsi
  __int64 v115; // rdx
  __int64 v116; // rsi
  unsigned __int64 v117; // rdi
  int v118; // edx
  __int64 v119; // rbx
  unsigned int v120; // eax
  _QWORD *v121; // rdi
  unsigned __int64 v122; // rdx
  unsigned __int64 v123; // rax
  _QWORD *v124; // rax
  __int64 v125; // rdx
  _QWORD *m; // rdx
  __int64 (__fastcall **v127)(__int64 *); // rdx
  __int64 v128; // r8
  __int64 (__fastcall **v129)(__int64 *); // rax
  __int64 (__fastcall **v130)(__int64 *); // rcx
  __int64 (__fastcall *v131)(__int64 *); // rdx
  __int64 (__fastcall **v132)(__int64 *); // rax
  unsigned int v133; // eax
  unsigned int v134; // esi
  __int64 v135; // rbx
  int v136; // r8d
  __int64 v137; // rdi
  __int64 v138; // rcx
  __int64 *v139; // rdx
  __int64 v140; // r9
  __int64 *v141; // rax
  __int64 v142; // rdx
  __int64 *v143; // r15
  __int64 v144; // rsi
  __int64 *v145; // rbx
  __int64 *v146; // rax
  int v147; // r11d
  __int64 *v148; // rax
  int v149; // ecx
  int v150; // edx
  int v151; // r8d
  int v152; // r8d
  __int64 v153; // r10
  __int64 v154; // rcx
  __int64 v155; // r9
  int v156; // edi
  __int64 *v157; // rsi
  int v158; // r8d
  int v159; // r8d
  __int64 v160; // r9
  __int64 *v161; // rdi
  __int64 v162; // r13
  int v163; // ecx
  __int64 v164; // rsi
  _QWORD *v165; // rdx
  _QWORD *v166; // rdx
  _QWORD *v167; // rdx
  _QWORD *v168; // rax
  int v169; // [rsp+8h] [rbp-158h]
  __int64 k; // [rsp+28h] [rbp-138h]
  __int64 *j; // [rsp+30h] [rbp-130h]
  char v174; // [rsp+3Fh] [rbp-121h]
  __int64 v175; // [rsp+40h] [rbp-120h]
  __int64 *i; // [rsp+48h] [rbp-118h]
  __int64 v177; // [rsp+58h] [rbp-108h] BYREF
  __int64 *v178; // [rsp+60h] [rbp-100h] BYREF
  __int64 *v179; // [rsp+68h] [rbp-F8h]
  __int64 *v180; // [rsp+70h] [rbp-F0h]
  __int64 *v181; // [rsp+80h] [rbp-E0h] BYREF
  __int64 *v182; // [rsp+88h] [rbp-D8h]
  __int64 *v183; // [rsp+90h] [rbp-D0h]
  __int64 *v184; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 *v185; // [rsp+A8h] [rbp-B8h]
  __int64 *v186; // [rsp+B0h] [rbp-B0h]
  __int64 *v187; // [rsp+C0h] [rbp-A0h] BYREF
  __int64 *v188; // [rsp+C8h] [rbp-98h]
  __int64 *v189; // [rsp+D0h] [rbp-90h]
  __int64 *v190; // [rsp+D8h] [rbp-88h]
  __int64 (__fastcall *v191)(__int64); // [rsp+E0h] [rbp-80h] BYREF
  __int64 v192; // [rsp+E8h] [rbp-78h]
  __int64 (__fastcall *v193)(__int64 *); // [rsp+F0h] [rbp-70h] BYREF
  __int64 v194; // [rsp+F8h] [rbp-68h]

  v3 = a2;
  v5 = sub_1AC3260(a3, sub_1856430, sub_18563B0);
  v7 = *(_QWORD *)(a3 + 32);
  v174 = v5;
  for ( i = (__int64 *)(a3 + 24); v7 != a3 + 24; v7 = *(_QWORD *)(v7 + 8) )
  {
    if ( !v7 )
      BUG();
    v8 = *(_QWORD *)(v7 - 8);
    if ( v8 )
    {
      v9 = (_QWORD *)sub_22077B0(24);
      if ( v9 )
        *v9 = 0;
      v9[1] = v8;
      v9[2] = v7 - 56;
      sub_17ED2A0((_QWORD *)(a2 + 384), 0, v9 + 1, v8, (__int64)v9);
    }
  }
  v10 = *(_QWORD *)(a3 + 16);
  for ( j = (__int64 *)(a3 + 8); a3 + 8 != v10; v10 = *(_QWORD *)(v10 + 8) )
  {
    if ( !v10 )
      BUG();
    v11 = *(_QWORD *)(v10 - 8);
    if ( v11 )
    {
      v12 = (_QWORD *)sub_22077B0(24);
      if ( v12 )
        *v12 = 0;
      v12[1] = v11;
      v12[2] = v10 - 56;
      sub_17ED2A0((_QWORD *)(a2 + 384), 0, v12 + 1, v11, (__int64)v12);
    }
  }
  v13 = *(_QWORD *)(a3 + 48);
  for ( k = a3 + 40; k != v13; v13 = *(_QWORD *)(v13 + 8) )
  {
    v14 = v13 - 48;
    if ( !v13 )
      v14 = 0;
    v15 = sub_15E4F10(v14);
    if ( v15 )
    {
      v16 = (_QWORD *)sub_22077B0(24);
      if ( v16 )
        *v16 = 0;
      v16[1] = v15;
      v16[2] = v14;
      sub_17ED2A0((_QWORD *)(a2 + 384), 0, v16 + 1, v15, (__int64)v16);
    }
  }
  v17 = *(__int64 **)(a3 + 32);
  v187 = *(__int64 **)(a3 + 16);
  v188 = (__int64 *)(a3 + 8);
  v189 = v17;
  v190 = (__int64 *)(a3 + 24);
  if ( v17 == i )
    goto LABEL_39;
  do
  {
    do
    {
      v18 = sub_18564A0;
      v19 = &v191;
      v192 = 0;
      v20 = &v187;
      v191 = sub_18564A0;
      v193 = sub_18564C0;
      v21 = &v191;
      v194 = 0;
      if ( ((unsigned __int8)sub_18564A0 & 1) == 0 )
        goto LABEL_26;
      while ( 1 )
      {
        v18 = *(__int64 (__fastcall **)(__int64))((char *)v18 + (_QWORD)*v20 - 1);
LABEL_26:
        v22 = v18((__int64)v20);
        v23 = v22;
        if ( v22 )
          break;
        while ( 1 )
        {
          v24 = v21[3];
          v18 = v21[2];
          v19 += 2;
          v21 = v19;
          v20 = (__int64 **)((char *)&v187 + (_QWORD)v24);
          if ( ((unsigned __int8)v18 & 1) != 0 )
            break;
          v22 = v18((__int64)v20);
          v23 = v22;
          if ( v22 )
            goto LABEL_29;
        }
      }
LABEL_29:
      v174 |= sub_1857520(a2, v22);
      if ( !sub_15E4F60(v23) )
      {
        v28 = *(_BYTE *)(v23 + 32) & 0xF;
        if ( ((v28 + 15) & 0xFu) > 2 && ((v28 + 9) & 0xFu) > 1 )
          sub_1857360(a2, v23, 0, v25, v26, v27);
      }
      sub_1858630(a2, v23);
      v29 = sub_1856440;
      v30 = &v191;
      v192 = 0;
      v31 = &v187;
      v191 = sub_1856440;
      v193 = sub_1856470;
      v32 = &v191;
      v194 = 0;
      if ( ((unsigned __int8)sub_1856440 & 1) != 0 )
        goto LABEL_34;
      while ( 2 )
      {
        if ( !(unsigned __int8)v29((__int64)v31) )
        {
          while ( 1 )
          {
            v33 = v32[3];
            v29 = v32[2];
            v30 += 2;
            v32 = v30;
            v31 = (__int64 **)((char *)&v187 + (_QWORD)v33);
            if ( ((unsigned __int8)v29 & 1) != 0 )
              break;
            if ( (unsigned __int8)v29((__int64)v31) )
              goto LABEL_38;
          }
LABEL_34:
          v29 = *(__int64 (__fastcall **)(__int64))((char *)v29 + (_QWORD)*v31 - 1);
          continue;
        }
        break;
      }
LABEL_38:
      ;
    }
    while ( v189 != i );
LABEL_39:
    ;
  }
  while ( i != v190 || j != v187 || j != v188 );
  v34 = *(_QWORD *)(a3 + 48);
  if ( k != v34 )
  {
    v35 = v174;
    do
    {
      if ( !v34 )
      {
        sub_1857520(a2, 0);
        BUG();
      }
      v35 |= sub_1857520(a2, v34 - 48);
      v39 = *(_BYTE *)(v34 - 16) & 0xF;
      if ( ((v39 + 15) & 0xFu) > 2 && ((v39 + 9) & 0xFu) > 1 )
        sub_1857360(a2, v34 - 48, 0, v36, v37, v38);
      sub_1858630(a2, v34 - 48);
      v34 = *(_QWORD *)(v34 + 8);
    }
    while ( k != v34 );
    v174 = v35;
  }
  v40 = *(_QWORD *)(a3 + 64);
  v175 = a3 + 56;
  if ( a3 + 56 != v40 )
  {
    v41 = v174;
    do
    {
      if ( !v40 )
      {
        sub_1857520(a2, 0);
        BUG();
      }
      v41 |= sub_1857520(a2, v40 - 48);
      v45 = *(_BYTE *)(v40 - 16) & 0xF;
      if ( ((v45 + 9) & 0xFu) > 1 && ((v45 + 15) & 0xFu) > 2 )
        sub_1857360(a2, v40 - 48, 0, v42, v43, v44);
      sub_1858630(a2, v40 - 48);
      v40 = *(_QWORD *)(v40 + 8);
    }
    while ( a3 + 56 != v40 );
    v174 = v41;
  }
  v46 = *(__int64 (__fastcall ***)(__int64 *))(a2 + 16);
  if ( v46 == *(__int64 (__fastcall ***)(__int64 *))(a2 + 8) )
    v47 = *(unsigned int *)(a2 + 28);
  else
    v47 = *(unsigned int *)(a2 + 24);
  v48 = &v46[v47];
  if ( v46 == v48 )
  {
LABEL_63:
    HIDWORD(v192) = 8;
    v191 = (__int64 (__fastcall *)(__int64))&v193;
LABEL_64:
    LODWORD(v192) = 0;
    goto LABEL_65;
  }
  while ( (unsigned __int64)*v46 >= 0xFFFFFFFFFFFFFFFELL )
  {
    if ( v48 == ++v46 )
      goto LABEL_63;
  }
  v191 = (__int64 (__fastcall *)(__int64))&v193;
  v192 = 0x800000000LL;
  if ( v48 == v46 )
    goto LABEL_64;
  v127 = v46;
  v128 = 0;
  while ( 1 )
  {
    v129 = v127 + 1;
    if ( v48 == v127 + 1 )
      break;
    while ( 1 )
    {
      v127 = v129;
      if ( (unsigned __int64)*v129 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v48 == ++v129 )
        goto LABEL_256;
    }
    ++v128;
    if ( v48 == v129 )
      goto LABEL_257;
  }
LABEL_256:
  ++v128;
LABEL_257:
  v130 = &v193;
  if ( v128 > 8 )
  {
    v169 = v128;
    sub_16CD150((__int64)&v191, &v193, v128, 8, v128, v6);
    LODWORD(v128) = v169;
    v130 = (__int64 (__fastcall **)(__int64 *))((char *)v191 + 8 * (unsigned int)v192);
  }
  v131 = *v46;
  do
  {
    v132 = v46 + 1;
    *v130++ = v131;
    if ( v48 == v46 + 1 )
      break;
    while ( 1 )
    {
      v131 = *v132;
      v46 = v132;
      if ( (unsigned __int64)*v132 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v48 == ++v132 )
        goto LABEL_263;
    }
  }
  while ( v48 != v132 );
LABEL_263:
  LODWORD(v192) = v192 + v128;
  v133 = v192;
  if ( (_DWORD)v192 )
  {
    while ( 1 )
    {
      v134 = *(_DWORD *)(v3 + 320);
      v135 = *((_QWORD *)v191 + v133 - 1);
      LODWORD(v192) = v133 - 1;
      if ( !v134 )
        break;
      v136 = v134 - 1;
      v137 = *(_QWORD *)(v3 + 304);
      v138 = (v134 - 1) & (((unsigned int)v135 >> 9) ^ ((unsigned int)v135 >> 4));
      v139 = (__int64 *)(v137 + 80 * v138);
      v140 = *v139;
      if ( v135 != *v139 )
      {
        v147 = 1;
        v148 = 0;
        while ( v140 != -8 )
        {
          if ( !v148 && v140 == -16 )
            v148 = v139;
          v138 = v136 & (unsigned int)(v138 + v147);
          v139 = (__int64 *)(v137 + 80 * v138);
          v140 = *v139;
          if ( v135 == *v139 )
            goto LABEL_266;
          ++v147;
        }
        v149 = *(_DWORD *)(v3 + 312);
        if ( !v148 )
          v148 = v139;
        ++*(_QWORD *)(v3 + 296);
        v150 = v149 + 1;
        if ( 4 * (v149 + 1) < 3 * v134 )
        {
          if ( v134 - *(_DWORD *)(v3 + 316) - v150 <= v134 >> 3 )
          {
            sub_1858440(v3 + 296, v134);
            v158 = *(_DWORD *)(v3 + 320);
            if ( !v158 )
            {
LABEL_343:
              ++*(_DWORD *)(v3 + 312);
              BUG();
            }
            v159 = v158 - 1;
            v160 = *(_QWORD *)(v3 + 304);
            v161 = 0;
            LODWORD(v162) = v159 & (((unsigned int)v135 >> 9) ^ ((unsigned int)v135 >> 4));
            v150 = *(_DWORD *)(v3 + 312) + 1;
            v163 = 1;
            v148 = (__int64 *)(v160 + 80LL * (unsigned int)v162);
            v164 = *v148;
            if ( v135 != *v148 )
            {
              while ( v164 != -8 )
              {
                if ( !v161 && v164 == -16 )
                  v161 = v148;
                v162 = v159 & (unsigned int)(v162 + v163);
                v148 = (__int64 *)(v160 + 80 * v162);
                v164 = *v148;
                if ( v135 == *v148 )
                  goto LABEL_290;
                ++v163;
              }
              if ( v161 )
                v148 = v161;
            }
          }
LABEL_290:
          *(_DWORD *)(v3 + 312) = v150;
          if ( *v148 != -8 )
            --*(_DWORD *)(v3 + 316);
          *v148 = v135;
          v148[1] = 0;
          v148[2] = (__int64)(v148 + 6);
          v148[3] = (__int64)(v148 + 6);
          v148[4] = 4;
          *((_DWORD *)v148 + 10) = 0;
          v133 = v192;
          if ( !(_DWORD)v192 )
            goto LABEL_65;
          continue;
        }
LABEL_295:
        sub_1858440(v3 + 296, 2 * v134);
        v151 = *(_DWORD *)(v3 + 320);
        if ( !v151 )
          goto LABEL_343;
        v152 = v151 - 1;
        v153 = *(_QWORD *)(v3 + 304);
        LODWORD(v154) = v152 & (((unsigned int)v135 >> 9) ^ ((unsigned int)v135 >> 4));
        v150 = *(_DWORD *)(v3 + 312) + 1;
        v148 = (__int64 *)(v153 + 80LL * (unsigned int)v154);
        v155 = *v148;
        if ( v135 != *v148 )
        {
          v156 = 1;
          v157 = 0;
          while ( v155 != -8 )
          {
            if ( v155 == -16 && !v157 )
              v157 = v148;
            v154 = v152 & (unsigned int)(v154 + v156);
            v148 = (__int64 *)(v153 + 80 * v154);
            v155 = *v148;
            if ( v135 == *v148 )
              goto LABEL_290;
            ++v156;
          }
          if ( v157 )
            v148 = v157;
        }
        goto LABEL_290;
      }
LABEL_266:
      v141 = (__int64 *)v139[3];
      if ( v141 == (__int64 *)v139[2] )
        v142 = *((unsigned int *)v139 + 9);
      else
        v142 = *((unsigned int *)v139 + 8);
      v143 = &v141[v142];
      if ( v141 != v143 )
      {
        while ( 1 )
        {
          v144 = *v141;
          v145 = v141;
          if ( (unsigned __int64)*v141 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v143 == ++v141 )
            goto LABEL_271;
        }
        if ( v143 != v141 )
        {
          do
          {
            sub_1857360(v3, v144, (__int64)&v191, v138, v136, v140);
            v146 = v145 + 1;
            if ( v145 + 1 == v143 )
              break;
            v144 = *v146;
            for ( ++v145; (unsigned __int64)*v146 >= 0xFFFFFFFFFFFFFFFELL; v145 = v146 )
            {
              if ( v143 == ++v146 )
                goto LABEL_271;
              v144 = *v146;
            }
          }
          while ( v143 != v145 );
        }
      }
LABEL_271:
      v133 = v192;
      if ( !(_DWORD)v192 )
        goto LABEL_65;
    }
    ++*(_QWORD *)(v3 + 296);
    goto LABEL_295;
  }
LABEL_65:
  v178 = 0;
  v179 = 0;
  v180 = 0;
  v49 = *(__int64 **)(a3 + 16);
  if ( j != v49 )
  {
    while ( 1 )
    {
      v53 = *(_QWORD **)(v3 + 16);
      v51 = *(_QWORD **)(v3 + 8);
      v54 = (__int64)(v49 - 7);
      if ( !v49 )
        v54 = 0;
      if ( v53 == v51 )
        break;
      v50 = &v53[*(unsigned int *)(v3 + 24)];
      v51 = sub_16CC9F0(v3, v54);
      if ( v54 == *v51 )
      {
        v109 = *(_QWORD *)(v3 + 16);
        if ( v109 == *(_QWORD *)(v3 + 8) )
          v110 = *(unsigned int *)(v3 + 28);
        else
          v110 = *(unsigned int *)(v3 + 24);
        v55 = (_QWORD *)(v109 + 8 * v110);
        goto LABEL_80;
      }
      v52 = *(_QWORD *)(v3 + 16);
      if ( v52 == *(_QWORD *)(v3 + 8) )
      {
        v55 = (_QWORD *)(v52 + 8LL * *(unsigned int *)(v3 + 28));
        v51 = v55;
        goto LABEL_80;
      }
      v51 = (_QWORD *)(v52 + 8LL * *(unsigned int *)(v3 + 24));
LABEL_70:
      if ( v51 == v50 )
      {
LABEL_84:
        v187 = (__int64 *)v54;
        v56 = v179;
        if ( v179 == v180 )
        {
          sub_1857F90((__int64)&v178, v179, &v187);
        }
        else
        {
          if ( v179 )
          {
            *v179 = v54;
            v56 = v179;
          }
          v179 = v56 + 1;
        }
        if ( sub_15E4F60(v54) )
          goto LABEL_71;
        v57 = *(_QWORD *)(v54 - 24);
        sub_15E5440(v54, 0);
        if ( !(unsigned __int8)sub_1ACF050(v57) )
          goto LABEL_71;
        sub_159D850(v57);
        v49 = (__int64 *)v49[1];
        if ( j == v49 )
          goto LABEL_91;
      }
      else
      {
LABEL_71:
        v49 = (__int64 *)v49[1];
        if ( j == v49 )
          goto LABEL_91;
      }
    }
    v55 = &v51[*(unsigned int *)(v3 + 28)];
    if ( v51 == v55 )
    {
      v50 = *(_QWORD **)(v3 + 8);
    }
    else
    {
      do
      {
        if ( v54 == *v51 )
          break;
        ++v51;
      }
      while ( v55 != v51 );
      v50 = v55;
    }
LABEL_80:
    if ( v51 != v55 )
    {
      while ( *v51 >= 0xFFFFFFFFFFFFFFFELL )
      {
        if ( v55 == ++v51 )
        {
          if ( v51 != v50 )
            goto LABEL_71;
          goto LABEL_84;
        }
      }
    }
    goto LABEL_70;
  }
LABEL_91:
  v181 = 0;
  v182 = 0;
  v183 = 0;
  v58 = *(__int64 **)(a3 + 32);
  if ( i != v58 )
  {
    while ( 1 )
    {
      v62 = *(_QWORD **)(v3 + 16);
      v60 = *(_QWORD **)(v3 + 8);
      v63 = (__int64)(v58 - 7);
      if ( !v58 )
        v63 = 0;
      if ( v62 == v60 )
        break;
      v59 = &v62[*(unsigned int *)(v3 + 24)];
      v60 = sub_16CC9F0(v3, v63);
      if ( v63 == *v60 )
      {
        v111 = *(_QWORD *)(v3 + 16);
        if ( v111 == *(_QWORD *)(v3 + 8) )
          v112 = *(unsigned int *)(v3 + 28);
        else
          v112 = *(unsigned int *)(v3 + 24);
        v165 = (_QWORD *)(v111 + 8 * v112);
        goto LABEL_106;
      }
      v61 = *(_QWORD *)(v3 + 16);
      if ( v61 == *(_QWORD *)(v3 + 8) )
      {
        v60 = (_QWORD *)(v61 + 8LL * *(unsigned int *)(v3 + 28));
        v165 = v60;
        goto LABEL_106;
      }
      v60 = (_QWORD *)(v61 + 8LL * *(unsigned int *)(v3 + 24));
LABEL_96:
      if ( v59 == v60 )
      {
LABEL_110:
        v187 = (__int64 *)v63;
        v64 = v182;
        if ( v182 == v183 )
        {
          sub_17E9700((__int64)&v181, v182, &v187);
        }
        else
        {
          if ( v182 )
          {
            *v182 = v63;
            v64 = v182;
          }
          v182 = v64 + 1;
        }
        if ( sub_15E4F60(v63) )
          goto LABEL_97;
        sub_15E0C30(v63);
        v65 = *(_BYTE *)(v63 + 32);
        *(_BYTE *)(v63 + 32) = v65 & 0xF0;
        if ( (v65 & 0x30) == 0 )
          goto LABEL_97;
        *(_BYTE *)(v63 + 33) |= 0x40u;
        v58 = (__int64 *)v58[1];
        if ( i == v58 )
          goto LABEL_117;
      }
      else
      {
LABEL_97:
        v58 = (__int64 *)v58[1];
        if ( i == v58 )
          goto LABEL_117;
      }
    }
    v59 = &v60[*(unsigned int *)(v3 + 28)];
    if ( v60 == v59 )
    {
      v165 = *(_QWORD **)(v3 + 8);
    }
    else
    {
      do
      {
        if ( v63 == *v60 )
          break;
        ++v60;
      }
      while ( v59 != v60 );
      v165 = v59;
    }
LABEL_106:
    if ( v60 != v165 )
    {
      while ( *v60 >= 0xFFFFFFFFFFFFFFFELL )
      {
        if ( v165 == ++v60 )
        {
          if ( v59 != v60 )
            goto LABEL_97;
          goto LABEL_110;
        }
      }
    }
    goto LABEL_96;
  }
LABEL_117:
  v184 = 0;
  v185 = 0;
  v186 = 0;
  if ( k == *(_QWORD *)(a3 + 48) )
    goto LABEL_142;
  v66 = *(_QWORD *)(a3 + 48);
  v67 = v3;
  while ( 2 )
  {
    while ( 2 )
    {
      v71 = *(_QWORD **)(v67 + 16);
      v69 = *(_QWORD **)(v67 + 8);
      v72 = v66 - 48;
      if ( !v66 )
        v72 = 0;
      if ( v71 == v69 )
      {
        v68 = &v69[*(unsigned int *)(v67 + 28)];
        if ( v69 == v68 )
        {
          v167 = *(_QWORD **)(v67 + 8);
        }
        else
        {
          do
          {
            if ( v72 == *v69 )
              break;
            ++v69;
          }
          while ( v68 != v69 );
          v167 = v68;
        }
        goto LABEL_132;
      }
      v68 = &v71[*(unsigned int *)(v67 + 24)];
      v69 = sub_16CC9F0(v67, v72);
      if ( v72 == *v69 )
      {
        v113 = *(_QWORD *)(v67 + 16);
        if ( v113 == *(_QWORD *)(v67 + 8) )
          v114 = *(unsigned int *)(v67 + 28);
        else
          v114 = *(unsigned int *)(v67 + 24);
        v167 = (_QWORD *)(v113 + 8 * v114);
      }
      else
      {
        v70 = *(_QWORD *)(v67 + 16);
        if ( v70 != *(_QWORD *)(v67 + 8) )
        {
          v69 = (_QWORD *)(v70 + 8LL * *(unsigned int *)(v67 + 24));
          goto LABEL_122;
        }
        v69 = (_QWORD *)(v70 + 8LL * *(unsigned int *)(v67 + 28));
        v167 = v69;
      }
LABEL_132:
      if ( v69 != v167 )
      {
        while ( *v69 >= 0xFFFFFFFFFFFFFFFELL )
        {
          if ( v167 == ++v69 )
          {
            if ( v69 != v68 )
              goto LABEL_123;
            goto LABEL_136;
          }
        }
      }
LABEL_122:
      if ( v69 != v68 )
      {
LABEL_123:
        v66 = *(_QWORD *)(v66 + 8);
        if ( k == v66 )
          goto LABEL_141;
        continue;
      }
      break;
    }
LABEL_136:
    v187 = (__int64 *)v72;
    v73 = v185;
    if ( v185 == v186 )
    {
      sub_1858120((__int64)&v184, v185, &v187);
    }
    else
    {
      if ( v185 )
      {
        *v185 = v72;
        v73 = v185;
      }
      v185 = v73 + 1;
    }
    sub_15E5930(v72, 0);
    v66 = *(_QWORD *)(v66 + 8);
    if ( k != v66 )
      continue;
    break;
  }
LABEL_141:
  v3 = v67;
LABEL_142:
  v187 = 0;
  v188 = 0;
  v189 = 0;
  if ( v175 == *(_QWORD *)(a3 + 64) )
    goto LABEL_170;
  v74 = *(_QWORD *)(a3 + 64);
  v75 = v3;
  while ( 2 )
  {
    while ( 2 )
    {
      v79 = *(_QWORD **)(v75 + 16);
      v77 = *(_QWORD **)(v75 + 8);
      v80 = v74 - 48;
      if ( !v74 )
        v80 = 0;
      if ( v79 == v77 )
      {
        v76 = &v77[*(unsigned int *)(v75 + 28)];
        if ( v77 == v76 )
        {
          v166 = *(_QWORD **)(v75 + 8);
        }
        else
        {
          do
          {
            if ( v80 == *v77 )
              break;
            ++v77;
          }
          while ( v76 != v77 );
          v166 = v76;
        }
        goto LABEL_157;
      }
      v76 = &v79[*(unsigned int *)(v75 + 24)];
      v77 = sub_16CC9F0(v75, v80);
      if ( v80 == *v77 )
      {
        v115 = *(_QWORD *)(v75 + 16);
        if ( v115 == *(_QWORD *)(v75 + 8) )
          v116 = *(unsigned int *)(v75 + 28);
        else
          v116 = *(unsigned int *)(v75 + 24);
        v166 = (_QWORD *)(v115 + 8 * v116);
      }
      else
      {
        v78 = *(_QWORD *)(v75 + 16);
        if ( v78 != *(_QWORD *)(v75 + 8) )
        {
          v77 = (_QWORD *)(v78 + 8LL * *(unsigned int *)(v75 + 24));
          goto LABEL_147;
        }
        v77 = (_QWORD *)(v78 + 8LL * *(unsigned int *)(v75 + 28));
        v166 = v77;
      }
LABEL_157:
      if ( v77 != v166 )
      {
        while ( *v77 >= 0xFFFFFFFFFFFFFFFELL )
        {
          if ( v166 == ++v77 )
          {
            if ( v76 != v77 )
              goto LABEL_148;
            goto LABEL_161;
          }
        }
      }
LABEL_147:
      if ( v76 != v77 )
      {
LABEL_148:
        v74 = *(_QWORD *)(v74 + 8);
        if ( v175 == v74 )
          goto LABEL_169;
        continue;
      }
      break;
    }
LABEL_161:
    v177 = v80;
    v81 = v188;
    if ( v188 == v189 )
    {
      sub_18582B0((__int64)&v187, v188, &v177);
    }
    else
    {
      if ( v188 )
      {
        *v188 = v80;
        v81 = v188;
      }
      v188 = v81 + 1;
    }
    if ( *(_QWORD *)(v80 - 24) )
    {
      v82 = *(_QWORD *)(v80 - 16);
      v83 = *(_QWORD *)(v80 - 8) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v83 = v82;
      if ( v82 )
        *(_QWORD *)(v82 + 16) = *(_QWORD *)(v82 + 16) & 3LL | v83;
    }
    *(_QWORD *)(v80 - 24) = 0;
    v74 = *(_QWORD *)(v74 + 8);
    if ( v175 != v74 )
      continue;
    break;
  }
LABEL_169:
  v3 = v75;
LABEL_170:
  v84 = v182;
  v85 = v181;
  if ( v182 != v181 )
  {
    do
    {
      v86 = *v85++;
      sub_1857520(v3, v86);
      sub_15E5B20(v86);
    }
    while ( v84 != v85 );
    v174 = 1;
  }
  v87 = v179;
  if ( v179 != v178 )
  {
    v88 = v178;
    do
    {
      v89 = *v88++;
      sub_1857520(v3, v89);
      sub_15E5B20(v89);
    }
    while ( v87 != v88 );
    v174 = 1;
  }
  v90 = v185;
  if ( v185 != v184 )
  {
    v91 = v184;
    do
    {
      v92 = *v91++;
      sub_1857520(v3, v92);
      sub_15E5B20(v92);
    }
    while ( v90 != v91 );
    v174 = 1;
  }
  v93 = v188;
  if ( v188 != v187 )
  {
    v94 = v187;
    do
    {
      v95 = *v94++;
      sub_1857520(v3, v95);
      sub_15E5B20(v95);
    }
    while ( v93 != v94 );
    v174 = 1;
  }
  ++*(_QWORD *)v3;
  v96 = *(void **)(v3 + 16);
  if ( v96 == *(void **)(v3 + 8) )
  {
LABEL_190:
    *(_QWORD *)(v3 + 28) = 0;
  }
  else
  {
    v97 = 4 * (*(_DWORD *)(v3 + 28) - *(_DWORD *)(v3 + 32));
    v98 = *(unsigned int *)(v3 + 24);
    if ( v97 < 0x20 )
      v97 = 32;
    if ( v97 >= (unsigned int)v98 )
    {
      memset(v96, -1, 8 * v98);
      goto LABEL_190;
    }
    sub_16CC920(v3);
  }
  sub_1857560(v3 + 328);
  v99 = *(_DWORD *)(v3 + 312);
  ++*(_QWORD *)(v3 + 296);
  if ( v99 || *(_DWORD *)(v3 + 316) )
  {
    v100 = *(_QWORD **)(v3 + 304);
    v101 = 4 * v99;
    v102 = &v100[10 * *(unsigned int *)(v3 + 320)];
    if ( (unsigned int)(4 * v99) < 0x40 )
      v101 = 64;
    if ( v101 >= *(_DWORD *)(v3 + 320) )
    {
      while ( v100 != v102 )
      {
        if ( *v100 != -8 )
        {
          if ( *v100 != -16 )
          {
            v103 = v100[3];
            if ( v103 != v100[2] )
              _libc_free(v103);
          }
          *v100 = -8;
        }
        v100 += 10;
      }
    }
    else
    {
      do
      {
        if ( *v100 != -16 && *v100 != -8 )
        {
          v117 = v100[3];
          if ( v117 != v100[2] )
            _libc_free(v117);
        }
        v100 += 10;
      }
      while ( v100 != v102 );
      v118 = *(_DWORD *)(v3 + 320);
      if ( v99 )
      {
        v119 = 64;
        if ( v99 != 1 )
        {
          _BitScanReverse(&v120, v99 - 1);
          v119 = (unsigned int)(1 << (33 - (v120 ^ 0x1F)));
          if ( (int)v119 < 64 )
            v119 = 64;
        }
        v121 = *(_QWORD **)(v3 + 304);
        if ( (_DWORD)v119 == v118 )
        {
          *(_QWORD *)(v3 + 312) = 0;
          v168 = &v121[10 * v119];
          do
          {
            if ( v121 )
              *v121 = -8;
            v121 += 10;
          }
          while ( v168 != v121 );
        }
        else
        {
          j___libc_free_0(v121);
          v122 = ((((((((4 * (int)v119 / 3u + 1) | ((unsigned __int64)(4 * (int)v119 / 3u + 1) >> 1)) >> 2)
                    | (4 * (int)v119 / 3u + 1)
                    | ((unsigned __int64)(4 * (int)v119 / 3u + 1) >> 1)) >> 4)
                  | (((4 * (int)v119 / 3u + 1) | ((unsigned __int64)(4 * (int)v119 / 3u + 1) >> 1)) >> 2)
                  | (4 * (int)v119 / 3u + 1)
                  | ((unsigned __int64)(4 * (int)v119 / 3u + 1) >> 1)) >> 8)
                | (((((4 * (int)v119 / 3u + 1) | ((unsigned __int64)(4 * (int)v119 / 3u + 1) >> 1)) >> 2)
                  | (4 * (int)v119 / 3u + 1)
                  | ((unsigned __int64)(4 * (int)v119 / 3u + 1) >> 1)) >> 4)
                | (((4 * (int)v119 / 3u + 1) | ((unsigned __int64)(4 * (int)v119 / 3u + 1) >> 1)) >> 2)
                | (4 * (int)v119 / 3u + 1)
                | ((unsigned __int64)(4 * (int)v119 / 3u + 1) >> 1)) >> 16;
          v123 = (v122
                | (((((((4 * (int)v119 / 3u + 1) | ((unsigned __int64)(4 * (int)v119 / 3u + 1) >> 1)) >> 2)
                    | (4 * (int)v119 / 3u + 1)
                    | ((unsigned __int64)(4 * (int)v119 / 3u + 1) >> 1)) >> 4)
                  | (((4 * (int)v119 / 3u + 1) | ((unsigned __int64)(4 * (int)v119 / 3u + 1) >> 1)) >> 2)
                  | (4 * (int)v119 / 3u + 1)
                  | ((unsigned __int64)(4 * (int)v119 / 3u + 1) >> 1)) >> 8)
                | (((((4 * (int)v119 / 3u + 1) | ((unsigned __int64)(4 * (int)v119 / 3u + 1) >> 1)) >> 2)
                  | (4 * (int)v119 / 3u + 1)
                  | ((unsigned __int64)(4 * (int)v119 / 3u + 1) >> 1)) >> 4)
                | (((4 * (int)v119 / 3u + 1) | ((unsigned __int64)(4 * (int)v119 / 3u + 1) >> 1)) >> 2)
                | (4 * (int)v119 / 3u + 1)
                | ((unsigned __int64)(4 * (int)v119 / 3u + 1) >> 1))
               + 1;
          *(_DWORD *)(v3 + 320) = v123;
          v124 = (_QWORD *)sub_22077B0(80 * v123);
          v125 = *(unsigned int *)(v3 + 320);
          *(_QWORD *)(v3 + 312) = 0;
          *(_QWORD *)(v3 + 304) = v124;
          for ( m = &v124[10 * v125]; m != v124; v124 += 10 )
          {
            if ( v124 )
              *v124 = -8;
          }
        }
        goto LABEL_205;
      }
      if ( v118 )
      {
        j___libc_free_0(*(_QWORD *)(v3 + 304));
        *(_QWORD *)(v3 + 304) = 0;
        *(_QWORD *)(v3 + 312) = 0;
        *(_DWORD *)(v3 + 320) = 0;
        goto LABEL_205;
      }
    }
    *(_QWORD *)(v3 + 312) = 0;
  }
LABEL_205:
  v104 = *(_QWORD **)(v3 + 400);
  while ( v104 )
  {
    v105 = v104;
    v104 = (_QWORD *)*v104;
    j_j___libc_free_0(v105, 24);
  }
  memset(*(void **)(v3 + 384), 0, 8LL * *(_QWORD *)(v3 + 392));
  *(_QWORD *)(v3 + 408) = 0;
  *(_QWORD *)(v3 + 400) = 0;
  v106 = a1 + 5;
  v107 = a1 + 12;
  if ( v174 )
  {
    memset(a1, 0, 0x70u);
    a1[1] = v106;
    a1[2] = v106;
    *((_DWORD *)a1 + 6) = 2;
    a1[8] = v107;
    a1[9] = v107;
    *((_DWORD *)a1 + 20) = 2;
  }
  else
  {
    a1[3] = 0x100000002LL;
    a1[1] = v106;
    a1[2] = v106;
    a1[7] = 0;
    a1[8] = v107;
    a1[9] = v107;
    a1[10] = 2;
    *((_DWORD *)a1 + 22) = 0;
    *((_DWORD *)a1 + 8) = 0;
    *a1 = 1;
    a1[5] = &unk_4F9EE48;
  }
  if ( v187 )
    j_j___libc_free_0(v187, (char *)v189 - (char *)v187);
  if ( v184 )
    j_j___libc_free_0(v184, (char *)v186 - (char *)v184);
  if ( v181 )
    j_j___libc_free_0(v181, (char *)v183 - (char *)v181);
  if ( v178 )
    j_j___libc_free_0(v178, (char *)v180 - (char *)v178);
  if ( (char *)v191 != (char *)&v193 )
    _libc_free((unsigned __int64)v191);
  return a1;
}
