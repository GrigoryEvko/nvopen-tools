// Function: sub_11D7E80
// Address: 0x11d7e80
//
__int64 __fastcall sub_11D7E80(__int64 **a1, __int64 a2)
{
  _QWORD *v2; // r12
  __int64 v3; // rax
  unsigned int v4; // esi
  __int64 v5; // rdx
  _QWORD *v6; // r15
  __int64 v7; // r9
  int v8; // r10d
  _QWORD *v9; // rax
  __int64 v10; // r8
  _QWORD *v11; // rbx
  __int64 v12; // rdi
  __int64 v13; // rdx
  unsigned __int64 *v14; // rax
  unsigned __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // r14
  __int64 v18; // r13
  __int64 v19; // rcx
  int v20; // edx
  __int64 v21; // r8
  _QWORD *v22; // rax
  __int64 v23; // rdx
  _QWORD *v24; // r15
  _BYTE *v25; // r12
  __int64 v26; // r13
  __int64 v27; // r8
  __int64 v28; // r9
  unsigned int v29; // ecx
  __int64 v30; // rdx
  __int64 *v31; // rax
  __int64 v32; // rax
  unsigned __int64 v33; // rcx
  __int64 v34; // rsi
  __int64 v35; // rdx
  __int64 v36; // r14
  __int64 v37; // r14
  __int64 *v38; // r12
  __int64 *v39; // rbx
  __int64 *v40; // rdx
  unsigned __int64 v41; // rax
  __int64 *v42; // r14
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // rbx
  __int64 *i; // r15
  __int64 v47; // rsi
  __int64 *v48; // r13
  unsigned __int64 v49; // rdx
  char **v50; // rbx
  __int64 v51; // r14
  __int64 v52; // rdx
  __int64 v53; // rax
  __int64 *v54; // r13
  __int64 *v55; // r15
  int v56; // r11d
  __int64 *v57; // rcx
  unsigned int v58; // r8d
  __int64 *v59; // rax
  __int64 v60; // rdi
  __int64 v61; // rbx
  __int64 v62; // r14
  void (*v63)(); // rax
  unsigned int v64; // edx
  int v65; // eax
  __int64 v66; // rdi
  void (*v67)(); // rax
  _QWORD *v68; // rbx
  __int64 v69; // r12
  void (*v70)(); // rax
  _QWORD *v71; // r14
  __int64 (*v72)(); // rax
  __int64 v73; // rsi
  __int64 v74; // rax
  __int64 *v75; // rdi
  int v76; // r11d
  unsigned int v77; // edx
  unsigned int v78; // r10d
  __int64 *v79; // rcx
  _QWORD *v80; // r9
  __int64 v81; // r8
  unsigned int v82; // edi
  __int64 *v83; // rcx
  __int64 v84; // r9
  __int64 *v85; // rdi
  unsigned int v86; // esi
  __int64 v87; // r10
  int v88; // ecx
  void (*v89)(); // rax
  _QWORD *v90; // r9
  int v91; // r10d
  __int64 v92; // rcx
  __int64 v93; // rdi
  __int64 v94; // rax
  __int64 v95; // rdx
  unsigned __int64 v96; // rax
  __int64 v97; // rdx
  int v98; // ecx
  __int64 *v99; // r9
  int v100; // r10d
  unsigned int v101; // edx
  __int64 v102; // rdi
  void (*v103)(); // rax
  _QWORD *v104; // rdi
  unsigned __int64 v105; // rdx
  __int64 v106; // rsi
  unsigned int v107; // eax
  _QWORD *v108; // rbx
  _QWORD *v109; // r12
  __int64 v110; // rax
  _QWORD *v111; // rax
  _QWORD *v112; // r13
  __int64 v114; // rax
  __int64 v115; // r13
  __int64 v116; // rax
  unsigned __int64 v117; // rax
  __int64 v118; // rdx
  __int64 *v119; // r14
  __int64 (*v120)(); // rax
  void (*v121)(); // rax
  char **v122; // rdx
  unsigned int v123; // edi
  __int64 *v124; // rax
  char *v125; // rcx
  __int64 *v126; // rax
  char *v127; // r12
  char v128; // al
  void (*v129)(); // rax
  __int64 (*v130)(); // rax
  __int64 v131; // rax
  unsigned __int64 v132; // rdx
  __int64 v133; // rax
  unsigned __int64 v134; // rdx
  int v135; // eax
  unsigned int v136; // edx
  __int64 v137; // r8
  int v138; // r10d
  __int64 *v139; // r9
  __int64 *v140; // r8
  unsigned int v141; // ebx
  int v142; // r9d
  __int64 v143; // rsi
  __int64 v144; // rcx
  char *v145; // rdi
  int v146; // r15d
  char **v147; // r11
  char **v148; // r10
  __int64 v149; // r15
  int v150; // r11d
  char *v151; // rsi
  int v152; // ecx
  int v153; // r10d
  int v154; // r9d
  int v155; // edx
  unsigned int v156; // eax
  _QWORD *v157; // rcx
  int v158; // r10d
  __int64 *v159; // r9
  int v160; // r10d
  unsigned int v161; // r8d
  __int64 v162; // rax
  int v163; // r10d
  int v164; // r10d
  _QWORD *v165; // [rsp+10h] [rbp-1C0h]
  _QWORD *v166; // [rsp+18h] [rbp-1B8h]
  _QWORD *v169; // [rsp+38h] [rbp-198h]
  __int64 *src; // [rsp+40h] [rbp-190h]
  void *srca; // [rsp+40h] [rbp-190h]
  char **srcb; // [rsp+40h] [rbp-190h]
  _QWORD *v173; // [rsp+48h] [rbp-188h]
  __int64 v174; // [rsp+48h] [rbp-188h]
  unsigned __int64 v175; // [rsp+48h] [rbp-188h]
  __int64 v176; // [rsp+50h] [rbp-180h] BYREF
  _QWORD *v177; // [rsp+58h] [rbp-178h]
  __int64 v178; // [rsp+60h] [rbp-170h]
  unsigned int v179; // [rsp+68h] [rbp-168h]
  __int64 v180; // [rsp+70h] [rbp-160h] BYREF
  __int64 *v181; // [rsp+78h] [rbp-158h]
  __int64 v182; // [rsp+80h] [rbp-150h]
  unsigned int v183; // [rsp+88h] [rbp-148h]
  __int64 *v184; // [rsp+90h] [rbp-140h] BYREF
  __int64 v185; // [rsp+98h] [rbp-138h]
  _BYTE v186[304]; // [rsp+A0h] [rbp-130h] BYREF

  v2 = *(_QWORD **)a2;
  v3 = *(unsigned int *)(a2 + 8);
  v176 = 0;
  v177 = 0;
  v178 = 0;
  v179 = 0;
  if ( v2 == &v2[v3] )
  {
    v180 = 0;
    v184 = (__int64 *)v186;
    v185 = 0x2000000000LL;
    v181 = 0;
    v182 = 0;
    v183 = 0;
    goto LABEL_58;
  }
  v4 = 0;
  v5 = 0;
  v6 = &v2[v3];
  while ( 1 )
  {
    v17 = *v2;
    v18 = *(_QWORD *)(*v2 + 40LL);
    if ( !v4 )
    {
      ++v176;
      goto LABEL_12;
    }
    v7 = v4 - 1;
    v8 = 1;
    v9 = 0;
    v10 = (unsigned int)v7 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
    v11 = (_QWORD *)(v5 + 16 * v10);
    v12 = *v11;
    if ( v18 == *v11 )
      break;
    while ( v12 != -4096 )
    {
      if ( !v9 && v12 == -8192 )
        v9 = v11;
      v10 = (unsigned int)v7 & (v8 + (_DWORD)v10);
      v11 = (_QWORD *)(v5 + 16LL * (unsigned int)v10);
      v12 = *v11;
      if ( v18 == *v11 )
        goto LABEL_4;
      ++v8;
    }
    if ( !v9 )
      v9 = v11;
    ++v176;
    v20 = v178 + 1;
    if ( 4 * ((int)v178 + 1) < 3 * v4 )
    {
      if ( v4 - (v20 + HIDWORD(v178)) > v4 >> 3 )
        goto LABEL_14;
      sub_11D3640((__int64)&v176, v4);
      if ( !v179 )
      {
LABEL_335:
        LODWORD(v178) = v178 + 1;
        BUG();
      }
      v90 = 0;
      v91 = 1;
      LODWORD(v92) = (v179 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
      v20 = v178 + 1;
      v9 = &v177[2 * (unsigned int)v92];
      v93 = *v9;
      if ( v18 == *v9 )
        goto LABEL_14;
      while ( v93 != -4096 )
      {
        if ( v93 == -8192 && !v90 )
          v90 = v9;
        v92 = (v179 - 1) & ((_DWORD)v92 + v91);
        v9 = &v177[2 * v92];
        v93 = *v9;
        if ( v18 == *v9 )
          goto LABEL_14;
        ++v91;
      }
      goto LABEL_93;
    }
LABEL_12:
    sub_11D3640((__int64)&v176, 2 * v4);
    if ( !v179 )
      goto LABEL_335;
    LODWORD(v19) = (v179 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
    v20 = v178 + 1;
    v9 = &v177[2 * (unsigned int)v19];
    v21 = *v9;
    if ( v18 == *v9 )
      goto LABEL_14;
    v163 = 1;
    v90 = 0;
    while ( v21 != -4096 )
    {
      if ( !v90 && v21 == -8192 )
        v90 = v9;
      v19 = (v179 - 1) & ((_DWORD)v19 + v163);
      v9 = &v177[2 * v19];
      v21 = *v9;
      if ( v18 == *v9 )
        goto LABEL_14;
      ++v163;
    }
LABEL_93:
    if ( v90 )
      v9 = v90;
LABEL_14:
    LODWORD(v178) = v20;
    if ( *v9 != -4096 )
      --HIDWORD(v178);
    *v9 = v18;
    v14 = v9 + 1;
    *v14 = 0;
LABEL_17:
    ++v2;
    *v14 = v17 & 0xFFFFFFFFFFFFFFFBLL;
    if ( v6 == v2 )
      goto LABEL_18;
LABEL_9:
    v5 = (__int64)v177;
    v4 = v179;
  }
LABEL_4:
  v13 = v11[1];
  v14 = v11 + 1;
  v15 = v13 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v13 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    goto LABEL_17;
  if ( (v13 & 4) == 0 )
  {
    v94 = sub_22077B0(48);
    if ( v94 )
    {
      *(_QWORD *)v94 = v94 + 16;
      *(_QWORD *)(v94 + 8) = 0x400000000LL;
    }
    v95 = v94;
    v96 = v94 & 0xFFFFFFFFFFFFFFF8LL;
    v11[1] = v95 | 4;
    v97 = *(unsigned int *)(v96 + 8);
    v10 = v97 + 1;
    if ( v97 + 1 > (unsigned __int64)*(unsigned int *)(v96 + 12) )
    {
      v175 = v96;
      sub_C8D5F0(v96, (const void *)(v96 + 16), v97 + 1, 8u, v10, v7);
      v96 = v175;
      v97 = *(unsigned int *)(v175 + 8);
    }
    *(_QWORD *)(*(_QWORD *)v96 + 8 * v97) = v15;
    ++*(_DWORD *)(v96 + 8);
    v15 = v11[1] & 0xFFFFFFFFFFFFFFF8LL;
  }
  v16 = *(unsigned int *)(v15 + 8);
  if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(v15 + 12) )
  {
    sub_C8D5F0(v15, (const void *)(v15 + 16), v16 + 1, 8u, v10, v7);
    v16 = *(unsigned int *)(v15 + 8);
  }
  ++v2;
  *(_QWORD *)(*(_QWORD *)v15 + 8 * v16) = v17;
  ++*(_DWORD *)(v15 + 8);
  if ( v6 != v2 )
    goto LABEL_9;
LABEL_18:
  v180 = 0;
  v181 = 0;
  v22 = *(_QWORD **)a2;
  v23 = *(unsigned int *)(a2 + 8);
  v182 = 0;
  v184 = (__int64 *)v186;
  v169 = &v22[v23];
  v185 = 0x2000000000LL;
  v183 = 0;
  if ( v22 != v169 )
  {
    v24 = v22;
    while ( 1 )
    {
      v25 = (_BYTE *)*v24;
      v26 = *(_QWORD *)(*v24 + 40LL);
      if ( !v179 )
        break;
      v27 = v179 - 1;
      v28 = 1;
      v29 = v27 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
      v30 = v177[2 * v29];
      v173 = &v177[2 * v29];
      v31 = 0;
      if ( v26 == v30 )
      {
LABEL_22:
        v32 = v173[1];
        v33 = v32 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v32 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
          goto LABEL_44;
        v34 = (v32 >> 2) & 1;
        if ( ((v32 >> 2) & 1) != 0 && !*(_DWORD *)(v33 + 8) )
          goto LABEL_44;
        v35 = (v32 >> 2) & 1;
        if ( !(_DWORD)v35 || (v36 = *(unsigned int *)(v33 + 8), (_DWORD)v36 == 1) )
        {
          if ( *v25 == 62 )
          {
            v103 = (void (*)())(*a1)[5];
            if ( v103 != nullsub_404 )
              ((void (__fastcall *)(__int64 **, _BYTE *, __int64, unsigned __int64, __int64, __int64))v103)(
                a1,
                v25,
                v35,
                v33,
                v27,
                v28);
            sub_11D33F0(a1[1], v26, *((_QWORD *)v25 - 8));
          }
          else if ( *v25 == 60 )
          {
            v118 = 0;
            v119 = a1[1];
            v120 = (__int64 (*)())(*a1)[7];
            if ( v120 != sub_11D27A0 )
              v118 = ((__int64 (__fastcall *)(__int64 **, _BYTE *, _QWORD, unsigned __int64, __int64, __int64))v120)(
                       a1,
                       v25,
                       0,
                       v33,
                       v27,
                       v28);
            sub_11D33F0(v119, v26, v118);
          }
          else
          {
            v133 = (unsigned int)v185;
            v134 = (unsigned int)v185 + 1LL;
            if ( v134 > HIDWORD(v185) )
            {
              sub_C8D5F0((__int64)&v184, v186, v134, 8u, v27, v28);
              v133 = (unsigned int)v185;
            }
            v184[v133] = (__int64)v25;
            LODWORD(v185) = v185 + 1;
          }
          v104 = v173;
          v52 = v173[1];
          v53 = (v52 >> 2) & 1;
          if ( ((v52 >> 2) & 1) != 0 )
          {
LABEL_140:
            if ( v52 && (_BYTE)v53 && (v105 = v52 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
            {
              *(_DWORD *)(v105 + 8) = 0;
              if ( v169 == ++v24 )
                goto LABEL_45;
            }
            else
            {
LABEL_44:
              if ( v169 == ++v24 )
                goto LABEL_45;
            }
          }
          else
          {
LABEL_136:
            v104[1] = 0;
            if ( v169 == ++v24 )
              goto LABEL_45;
          }
        }
        else
        {
          v37 = 8 * v36;
          v38 = (__int64 *)(*(_QWORD *)v33 + v37);
          v39 = *(__int64 **)v33;
          v40 = *(__int64 **)v33;
          if ( *(__int64 **)v33 == v38 )
          {
            v117 = v32 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (_BYTE)v34 )
              goto LABEL_170;
            goto LABEL_44;
          }
          do
          {
            if ( (*(_BYTE *)*v40 & 0xFD) == 0x3C )
            {
              src = *(__int64 **)v33;
              _BitScanReverse64(&v41, v37 >> 3);
              sub_11D29F0(*(__int64 **)v33, (__int64 *)(*(_QWORD *)v33 + v37), 2LL * (int)(63 - (v41 ^ 0x3F)));
              if ( (unsigned __int64)v37 <= 0x80 )
              {
                sub_11D2940(src, v38);
              }
              else
              {
                v42 = src + 16;
                sub_11D2940(src, src + 16);
                if ( v38 != src + 16 )
                {
                  srca = (void *)v26;
                  v166 = v24;
                  do
                  {
                    v45 = *v42;
                    for ( i = v42; ; i[1] = *i )
                    {
                      v47 = *(i - 1);
                      v48 = i--;
                      if ( !sub_B445A0(v45, v47) )
                        break;
                    }
                    ++v42;
                    *v48 = v45;
                  }
                  while ( v38 != v42 );
                  v26 = (__int64)srca;
                  v24 = v166;
                }
              }
              v49 = v173[1] & 0xFFFFFFFFFFFFFFF8LL;
              if ( (v173[1] & 4) != 0 )
              {
                v50 = *(char ***)v49;
                srcb = (char **)(*(_QWORD *)v49 + 8LL * *(unsigned int *)(v49 + 8));
              }
              else
              {
                v50 = (char **)(v173 + 1);
                if ( !v49 )
                {
                  v51 = 0;
                  goto LABEL_42;
                }
                srcb = (char **)(v173 + 2);
              }
              v51 = 0;
              if ( srcb != v50 )
              {
                v165 = v24;
                while ( 1 )
                {
                  while ( 1 )
                  {
                    v127 = *v50;
                    v128 = **v50;
                    if ( v128 == 61 )
                      break;
                    if ( v128 != 62 )
                    {
                      if ( v128 == 60 )
                      {
                        v51 = 0;
                        v130 = (__int64 (*)())(*a1)[7];
                        if ( v130 != sub_11D27A0 )
                          v51 = ((__int64 (__fastcall *)(__int64 **, char *))v130)(a1, *v50);
                      }
                      goto LABEL_186;
                    }
                    v129 = (void (*)())(*a1)[5];
                    if ( v129 != nullsub_404 )
                      ((void (__fastcall *)(__int64 **, char *))v129)(a1, *v50);
                    v51 = *((_QWORD *)v127 - 8);
                    if ( srcb == ++v50 )
                    {
LABEL_192:
                      v24 = v165;
                      goto LABEL_42;
                    }
                  }
                  if ( !v51 )
                  {
                    v131 = (unsigned int)v185;
                    v132 = (unsigned int)v185 + 1LL;
                    if ( v132 > HIDWORD(v185) )
                    {
                      sub_C8D5F0((__int64)&v184, v186, v132, 8u, v43, v44);
                      v131 = (unsigned int)v185;
                    }
                    v184[v131] = (__int64)v127;
                    LODWORD(v185) = v185 + 1;
                    goto LABEL_186;
                  }
                  v121 = (void (*)())(*a1)[3];
                  if ( v121 != nullsub_402 )
                    ((void (__fastcall *)(__int64 **, char *, __int64))v121)(a1, *v50, v51);
                  sub_BD84D0((__int64)v127, v51);
                  if ( !v183 )
                    break;
                  v43 = 1;
                  v44 = (__int64)v181;
                  v122 = 0;
                  v123 = (v183 - 1) & (((unsigned int)v127 >> 9) ^ ((unsigned int)v127 >> 4));
                  v124 = &v181[2 * v123];
                  v125 = (char *)*v124;
                  if ( v127 != (char *)*v124 )
                  {
                    while ( v125 != (char *)-4096LL )
                    {
                      if ( !v122 && v125 == (char *)-8192LL )
                        v122 = (char **)v124;
                      v123 = (v183 - 1) & (v43 + v123);
                      v124 = &v181[2 * v123];
                      v125 = (char *)*v124;
                      if ( v127 == (char *)*v124 )
                        goto LABEL_184;
                      v43 = (unsigned int)(v43 + 1);
                    }
                    if ( !v122 )
                      v122 = (char **)v124;
                    ++v180;
                    v135 = v182 + 1;
                    if ( 4 * ((int)v182 + 1) < 3 * v183 )
                    {
                      if ( v183 - HIDWORD(v182) - v135 <= v183 >> 3 )
                      {
                        sub_FAA400((__int64)&v180, v183);
                        if ( !v183 )
                        {
LABEL_337:
                          LODWORD(v182) = v182 + 1;
                          BUG();
                        }
                        v148 = 0;
                        LODWORD(v149) = (v183 - 1) & (((unsigned int)v127 >> 9) ^ ((unsigned int)v127 >> 4));
                        v150 = 1;
                        v135 = v182 + 1;
                        v122 = (char **)&v181[2 * (unsigned int)v149];
                        v151 = *v122;
                        if ( v127 != *v122 )
                        {
                          while ( v151 != (char *)-4096LL )
                          {
                            if ( !v148 && v151 == (char *)-8192LL )
                              v148 = v122;
                            v43 = (unsigned int)(v150 + 1);
                            v149 = (v183 - 1) & ((_DWORD)v149 + v150);
                            v122 = (char **)&v181[2 * v149];
                            v151 = *v122;
                            if ( v127 == *v122 )
                              goto LABEL_213;
                            ++v150;
                          }
                          if ( v148 )
                            v122 = v148;
                        }
                      }
                      goto LABEL_213;
                    }
LABEL_232:
                    sub_FAA400((__int64)&v180, 2 * v183);
                    if ( !v183 )
                      goto LABEL_337;
                    v44 = (__int64)v181;
                    LODWORD(v144) = (v183 - 1) & (((unsigned int)v127 >> 9) ^ ((unsigned int)v127 >> 4));
                    v135 = v182 + 1;
                    v122 = (char **)&v181[2 * (unsigned int)v144];
                    v145 = *v122;
                    if ( v127 != *v122 )
                    {
                      v146 = 1;
                      v147 = 0;
                      while ( v145 != (char *)-4096LL )
                      {
                        if ( !v147 && v145 == (char *)-8192LL )
                          v147 = v122;
                        v43 = (unsigned int)(v146 + 1);
                        v144 = (v183 - 1) & ((_DWORD)v144 + v146);
                        v122 = (char **)&v181[2 * v144];
                        v145 = *v122;
                        if ( v127 == *v122 )
                          goto LABEL_213;
                        ++v146;
                      }
                      if ( v147 )
                        v122 = v147;
                    }
LABEL_213:
                    LODWORD(v182) = v135;
                    if ( *v122 != (char *)-4096LL )
                      --HIDWORD(v182);
                    *v122 = v127;
                    v126 = (__int64 *)(v122 + 1);
                    v122[1] = 0;
                    goto LABEL_185;
                  }
LABEL_184:
                  v126 = v124 + 1;
LABEL_185:
                  *v126 = v51;
LABEL_186:
                  if ( srcb == ++v50 )
                    goto LABEL_192;
                }
                ++v180;
                goto LABEL_232;
              }
LABEL_42:
              sub_11D33F0(a1[1], v26, v51);
              v52 = v173[1];
              v53 = (v52 >> 2) & 1;
              if ( ((v52 >> 2) & 1) == 0 )
              {
                v173[1] = 0;
                goto LABEL_44;
              }
              goto LABEL_140;
            }
            ++v40;
          }
          while ( v38 != v40 );
          v114 = (unsigned int)v185;
          do
          {
            v115 = *v39;
            if ( v114 + 1 > (unsigned __int64)HIDWORD(v185) )
            {
              sub_C8D5F0((__int64)&v184, v186, v114 + 1, 8u, v27, v28);
              v114 = (unsigned int)v185;
            }
            ++v39;
            v184[v114] = v115;
            v114 = (unsigned int)(v185 + 1);
            LODWORD(v185) = v185 + 1;
          }
          while ( v38 != v39 );
          v104 = v173;
          v116 = v173[1];
          if ( ((v116 >> 2) & 1) == 0 )
            goto LABEL_136;
          if ( !v116 )
            goto LABEL_44;
          if ( ((v116 >> 2) & 1) == 0 )
            goto LABEL_44;
          v117 = v116 & 0xFFFFFFFFFFFFFFF8LL;
          if ( !v117 )
            goto LABEL_44;
LABEL_170:
          *(_DWORD *)(v117 + 8) = 0;
          if ( v169 == ++v24 )
            goto LABEL_45;
        }
      }
      else
      {
        while ( v30 != -4096 )
        {
          if ( !v31 && v30 == -8192 )
            v31 = v173;
          v29 = v27 & (v28 + v29);
          v173 = &v177[2 * v29];
          v30 = *v173;
          if ( v26 == *v173 )
            goto LABEL_22;
          v28 = (unsigned int)(v28 + 1);
        }
        if ( !v31 )
          v31 = v173;
        ++v176;
        v98 = v178 + 1;
        if ( 4 * ((int)v178 + 1) < 3 * v179 )
        {
          if ( v179 - HIDWORD(v178) - v98 <= v179 >> 3 )
          {
            sub_11D3640((__int64)&v176, v179);
            if ( !v179 )
            {
LABEL_336:
              LODWORD(v178) = v178 + 1;
              BUG();
            }
            v140 = 0;
            v141 = (v179 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
            v142 = 1;
            v98 = v178 + 1;
            v31 = &v177[2 * v141];
            v143 = *v31;
            if ( v26 != *v31 )
            {
              while ( v143 != -4096 )
              {
                if ( v143 == -8192 && !v140 )
                  v140 = v31;
                v141 = (v179 - 1) & (v142 + v141);
                v31 = &v177[2 * v141];
                v143 = *v31;
                if ( v26 == *v31 )
                  goto LABEL_111;
                ++v142;
              }
              if ( v140 )
                v31 = v140;
            }
          }
          goto LABEL_111;
        }
LABEL_217:
        sub_11D3640((__int64)&v176, 2 * v179);
        if ( !v179 )
          goto LABEL_336;
        v136 = (v179 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
        v98 = v178 + 1;
        v31 = &v177[2 * v136];
        v137 = *v31;
        if ( v26 != *v31 )
        {
          v138 = 1;
          v139 = 0;
          while ( v137 != -4096 )
          {
            if ( !v139 && v137 == -8192 )
              v139 = v31;
            v136 = (v179 - 1) & (v138 + v136);
            v31 = &v177[2 * v136];
            v137 = *v31;
            if ( v26 == *v31 )
              goto LABEL_111;
            ++v138;
          }
          if ( v139 )
            v31 = v139;
        }
LABEL_111:
        LODWORD(v178) = v98;
        if ( *v31 != -4096 )
          --HIDWORD(v178);
        *v31 = v26;
        ++v24;
        v31[1] = 0;
        if ( v169 == v24 )
        {
LABEL_45:
          v54 = v184;
          v55 = &v184[(unsigned int)v185];
          if ( v55 != v184 )
          {
            while ( 1 )
            {
              v61 = *v54;
              v62 = sub_11D7E40(a1[1], *(_QWORD *)(*v54 + 40));
              v63 = (void (*)())(*a1)[3];
              if ( v63 == nullsub_402 )
              {
                if ( v61 != v62 )
                  goto LABEL_51;
              }
              else
              {
                ((void (__fastcall *)(__int64 **, __int64, __int64))v63)(a1, v61, v62);
                if ( v61 != v62 )
                  goto LABEL_51;
              }
              v62 = sub_ACADE0(*(__int64 ***)(v61 + 8));
LABEL_51:
              sub_BD84D0(v61, v62);
              if ( !v183 )
              {
                ++v180;
                goto LABEL_53;
              }
              v56 = 1;
              v57 = 0;
              v58 = (v183 - 1) & (((unsigned int)v61 >> 9) ^ ((unsigned int)v61 >> 4));
              v59 = &v181[2 * v58];
              v60 = *v59;
              if ( v61 == *v59 )
              {
LABEL_48:
                ++v54;
                v59[1] = v62;
                if ( v55 == v54 )
                  goto LABEL_58;
              }
              else
              {
                while ( v60 != -4096 )
                {
                  if ( v60 == -8192 && !v57 )
                    v57 = v59;
                  v58 = (v183 - 1) & (v56 + v58);
                  v59 = &v181[2 * v58];
                  v60 = *v59;
                  if ( v61 == *v59 )
                    goto LABEL_48;
                  ++v56;
                }
                if ( !v57 )
                  v57 = v59;
                ++v180;
                v65 = v182 + 1;
                if ( 4 * ((int)v182 + 1) < 3 * v183 )
                {
                  if ( v183 - HIDWORD(v182) - v65 > v183 >> 3 )
                    goto LABEL_55;
                  sub_FAA400((__int64)&v180, v183);
                  if ( !v183 )
                  {
LABEL_338:
                    LODWORD(v182) = v182 + 1;
                    BUG();
                  }
                  v99 = 0;
                  v100 = 1;
                  v101 = (v183 - 1) & (((unsigned int)v61 >> 9) ^ ((unsigned int)v61 >> 4));
                  v65 = v182 + 1;
                  v57 = &v181[2 * v101];
                  v102 = *v57;
                  if ( v61 == *v57 )
                    goto LABEL_55;
                  while ( v102 != -4096 )
                  {
                    if ( !v99 && v102 == -8192 )
                      v99 = v57;
                    v101 = (v183 - 1) & (v100 + v101);
                    v57 = &v181[2 * v101];
                    v102 = *v57;
                    if ( v61 == *v57 )
                      goto LABEL_55;
                    ++v100;
                  }
                  goto LABEL_128;
                }
LABEL_53:
                sub_FAA400((__int64)&v180, 2 * v183);
                if ( !v183 )
                  goto LABEL_338;
                v64 = (v183 - 1) & (((unsigned int)v61 >> 9) ^ ((unsigned int)v61 >> 4));
                v65 = v182 + 1;
                v57 = &v181[2 * v64];
                v66 = *v57;
                if ( v61 == *v57 )
                  goto LABEL_55;
                v164 = 1;
                v99 = 0;
                while ( v66 != -4096 )
                {
                  if ( v66 == -8192 && !v99 )
                    v99 = v57;
                  v64 = (v183 - 1) & (v164 + v64);
                  v57 = &v181[2 * v64];
                  v66 = *v57;
                  if ( v61 == *v57 )
                    goto LABEL_55;
                  ++v164;
                }
LABEL_128:
                if ( v99 )
                  v57 = v99;
LABEL_55:
                LODWORD(v182) = v65;
                if ( *v57 != -4096 )
                  --HIDWORD(v182);
                ++v54;
                *v57 = v61;
                v57[1] = 0;
                v57[1] = v62;
                if ( v55 == v54 )
                  goto LABEL_58;
              }
            }
          }
          goto LABEL_58;
        }
      }
    }
    ++v176;
    goto LABEL_217;
  }
LABEL_58:
  v67 = (void (*)())(*a1)[2];
  if ( v67 != nullsub_401 )
    ((void (__fastcall *)(__int64 **))v67)(a1);
  v68 = *(_QWORD **)a2;
  v69 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
  if ( v69 != *(_QWORD *)a2 )
  {
    while ( 2 )
    {
      while ( 1 )
      {
        v71 = (_QWORD *)*v68;
        v72 = (__int64 (*)())(*a1)[6];
        if ( v72 != sub_11D2790 && !((unsigned __int8 (__fastcall *)(__int64 **, _QWORD))v72)(a1, *v68) )
          break;
        if ( !v71[2] )
        {
          v70 = (void (*)())(*a1)[4];
          if ( v70 != nullsub_403 )
            goto LABEL_79;
          goto LABEL_63;
        }
        v73 = v183;
        if ( !v183 )
        {
          ++v180;
          goto LABEL_265;
        }
        v74 = (__int64)v181;
        v75 = 0;
        v76 = 1;
        v77 = v183 - 1;
        v78 = (v183 - 1) & (((unsigned int)v71 >> 9) ^ ((unsigned int)v71 >> 4));
        v79 = &v181[2 * v78];
        v80 = (_QWORD *)*v79;
        if ( v71 == (_QWORD *)*v79 )
          goto LABEL_68;
        while ( 1 )
        {
          if ( v80 == (_QWORD *)-4096LL )
          {
            if ( !v75 )
              v75 = v79;
            ++v180;
            v155 = v182 + 1;
            if ( 4 * ((int)v182 + 1) >= 3 * v183 )
            {
LABEL_265:
              sub_FAA400((__int64)&v180, 2 * v183);
              if ( v183 )
              {
                v156 = (v183 - 1) & (((unsigned int)v71 >> 9) ^ ((unsigned int)v71 >> 4));
                v155 = v182 + 1;
                v75 = &v181[2 * v156];
                v157 = (_QWORD *)*v75;
                if ( v71 != (_QWORD *)*v75 )
                {
                  v158 = 1;
                  v159 = 0;
                  while ( v157 != (_QWORD *)-4096LL )
                  {
                    if ( v157 == (_QWORD *)-8192LL && !v159 )
                      v159 = v75;
                    v156 = (v183 - 1) & (v158 + v156);
                    v75 = &v181[2 * v156];
                    v157 = (_QWORD *)*v75;
                    if ( v71 == (_QWORD *)*v75 )
                      goto LABEL_258;
                    ++v158;
                  }
LABEL_277:
                  if ( v159 )
                    v75 = v159;
                }
LABEL_258:
                LODWORD(v182) = v155;
                if ( *v75 != -4096 )
                  --HIDWORD(v182);
                *v75 = (__int64)v71;
                v81 = 0;
                v75[1] = 0;
                v73 = v183;
                if ( v183 )
                {
                  v83 = v181;
                  v77 = v183 - 1;
                  v82 = 0;
                  v74 = (__int64)v181;
                  goto LABEL_69;
                }
                goto LABEL_76;
              }
            }
            else
            {
              if ( v183 - HIDWORD(v182) - v155 > v183 >> 3 )
                goto LABEL_258;
              sub_FAA400((__int64)&v180, v183);
              if ( v183 )
              {
                v159 = 0;
                v160 = 1;
                v161 = (v183 - 1) & (((unsigned int)v71 >> 9) ^ ((unsigned int)v71 >> 4));
                v155 = v182 + 1;
                v75 = &v181[2 * v161];
                v162 = *v75;
                if ( v71 != (_QWORD *)*v75 )
                {
                  while ( v162 != -4096 )
                  {
                    if ( !v159 && v162 == -8192 )
                      v159 = v75;
                    v161 = (v183 - 1) & (v160 + v161);
                    v75 = &v181[2 * v161];
                    v162 = *v75;
                    if ( v71 == (_QWORD *)*v75 )
                      goto LABEL_258;
                    ++v160;
                  }
                  goto LABEL_277;
                }
                goto LABEL_258;
              }
            }
            LODWORD(v182) = v182 + 1;
            BUG();
          }
          if ( v75 || v80 != (_QWORD *)-8192LL )
            v79 = v75;
          v78 = v77 & (v76 + v78);
          v80 = (_QWORD *)v181[2 * v78];
          if ( v71 == v80 )
            break;
          ++v76;
          v75 = v79;
          v79 = &v181[2 * v78];
        }
        v79 = &v181[2 * v78];
LABEL_68:
        v81 = v79[1];
        v82 = v77 & (((unsigned int)v81 >> 4) ^ ((unsigned int)v81 >> 9));
        v83 = &v181[2 * v82];
LABEL_69:
        v84 = *v83;
        if ( *v83 == v81 )
        {
LABEL_70:
          v85 = (__int64 *)(v74 + 16 * v73);
          if ( v83 != v85 )
          {
            while ( 1 )
            {
              v81 = v83[1];
              v86 = v77 & (((unsigned int)v81 >> 9) ^ ((unsigned int)v81 >> 4));
              v83 = (__int64 *)(v74 + 16LL * v86);
              v87 = *v83;
              if ( v81 != *v83 )
                break;
LABEL_72:
              if ( v85 == v83 )
                goto LABEL_76;
            }
            v88 = 1;
            while ( v87 != -4096 )
            {
              v154 = v88 + 1;
              v86 = v77 & (v88 + v86);
              v83 = (__int64 *)(v74 + 16LL * v86);
              v87 = *v83;
              if ( v81 == *v83 )
                goto LABEL_72;
              v88 = v154;
            }
          }
        }
        else
        {
          v152 = 1;
          while ( v84 != -4096 )
          {
            v153 = v152 + 1;
            v82 = v77 & (v152 + v82);
            v83 = (__int64 *)(v74 + 16LL * v82);
            v84 = *v83;
            if ( v81 == *v83 )
              goto LABEL_70;
            v152 = v153;
          }
        }
LABEL_76:
        v89 = (void (*)())(*a1)[3];
        if ( v89 != nullsub_402 )
        {
          v174 = v81;
          ((void (__fastcall *)(__int64 **, _QWORD *, __int64))v89)(a1, v71, v81);
          v81 = v174;
        }
        sub_BD84D0((__int64)v71, v81);
        v70 = (void (*)())(*a1)[4];
        if ( v70 != nullsub_403 )
LABEL_79:
          ((void (__fastcall *)(__int64 **, _QWORD *))v70)(a1, v71);
LABEL_63:
        ++v68;
        sub_B43D60(v71);
        if ( (_QWORD *)v69 == v68 )
          goto LABEL_147;
      }
      if ( (_QWORD *)v69 != ++v68 )
        continue;
      break;
    }
  }
LABEL_147:
  v106 = 16LL * v183;
  sub_C7D6A0((__int64)v181, v106, 8);
  if ( v184 != (__int64 *)v186 )
    _libc_free(v184, v106);
  v107 = v179;
  if ( v179 )
  {
    v108 = v177;
    v109 = &v177[2 * v179];
    do
    {
      if ( *v108 != -8192 && *v108 != -4096 )
      {
        v110 = v108[1];
        if ( v110 )
        {
          if ( (v110 & 4) != 0 )
          {
            v111 = (_QWORD *)(v110 & 0xFFFFFFFFFFFFFFF8LL);
            v112 = v111;
            if ( v111 )
            {
              if ( (_QWORD *)*v111 != v111 + 2 )
                _libc_free(*v111, v106);
              v106 = 48;
              j_j___libc_free_0(v112, 48);
            }
          }
        }
      }
      v108 += 2;
    }
    while ( v109 != v108 );
    v107 = v179;
  }
  return sub_C7D6A0((__int64)v177, 16LL * v107, 8);
}
