// Function: sub_1B40B80
// Address: 0x1b40b80
//
__int64 __fastcall sub_1B40B80(
        __int64 **a1,
        __int64 *a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10,
        __int64 a11,
        __int64 a12,
        int a13)
{
  __int64 **v13; // r15
  __int64 *v14; // r12
  __int64 v15; // rax
  __int64 *v16; // r13
  unsigned int v17; // esi
  _QWORD *v18; // rcx
  __int64 *v19; // r9
  _QWORD *v20; // rbx
  __int64 v21; // rdi
  __int64 v22; // rax
  unsigned __int64 v23; // r14
  __int64 v24; // rax
  __int64 v25; // rdx
  unsigned __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // r15
  __int64 v30; // r14
  unsigned int v31; // eax
  int v32; // ecx
  _QWORD *v33; // rdx
  __int64 *v34; // rax
  __int64 v35; // rdx
  __int64 **v36; // r14
  __int64 v37; // r12
  __int64 v38; // r13
  int v39; // r8d
  unsigned int v40; // ecx
  __int64 v41; // rdx
  __int64 v42; // rax
  unsigned __int64 v43; // rdx
  __int64 v44; // rax
  __int64 *v45; // rbx
  __int64 v46; // r12
  __int64 *v47; // rax
  __int64 v48; // rbx
  __int64 v49; // rdx
  __int64 v50; // rax
  __int64 **v51; // r14
  __int64 v52; // r13
  __int64 v53; // r15
  bool (__fastcall *v54)(__int64, __int64, __int64); // rax
  _QWORD *v55; // rdi
  __int64 v56; // rax
  int v57; // r8d
  int v58; // r9d
  double v59; // xmm4_8
  double v60; // xmm5_8
  void (*v61)(); // rax
  unsigned int v62; // edi
  _QWORD *v63; // rax
  __int64 v64; // rcx
  char v65; // al
  __int64 v66; // r12
  bool (__fastcall *v67)(__int64, __int64, __int64); // rax
  _QWORD *v68; // rdi
  __int64 v69; // rax
  __int64 v70; // r9
  void (*v71)(); // rax
  __int64 v72; // rax
  __int64 *v73; // rcx
  __int64 v74; // rax
  unsigned __int64 v75; // r13
  _BYTE *v76; // r12
  unsigned int v77; // edi
  __int64 ****v78; // rax
  __int64 ***v79; // rcx
  __int64 ***v80; // rbx
  __int64 v81; // r14
  double v82; // xmm4_8
  double v83; // xmm5_8
  void (*v84)(); // rax
  unsigned int v85; // edx
  int v86; // ecx
  __int64 ***v87; // rdi
  void (*v88)(); // rax
  __int64 v89; // rbx
  __int64 v90; // r13
  void (*v91)(); // rax
  _QWORD *v92; // r14
  __int64 v93; // r8
  unsigned int v94; // edi
  _QWORD *v95; // rsi
  unsigned int v96; // r9d
  _QWORD *v97; // rax
  _QWORD *v98; // rcx
  __int64 v99; // r12
  unsigned int v100; // edx
  __int64 *v101; // rax
  __int64 v102; // rcx
  __int64 *v103; // rcx
  unsigned int v104; // edx
  __int64 v105; // r8
  int v106; // eax
  void (*v107)(); // rax
  __int64 v108; // rax
  void (*v109)(); // rax
  unsigned __int64 v110; // rax
  __int64 v111; // rax
  __int64 v112; // r13
  _QWORD *v113; // rbx
  _QWORD *v114; // r12
  __int64 v115; // rax
  unsigned __int64 *v116; // rax
  unsigned __int64 *v117; // r13
  __int64 v119; // rax
  int v120; // r10d
  __int64 *v121; // rax
  int v122; // ecx
  int v123; // r11d
  __int64 ****v124; // r10
  __int64 ****v125; // r9
  int v126; // r10d
  unsigned int v127; // edx
  __int64 ***v128; // rdi
  _QWORD *v129; // r10
  int v130; // r11d
  unsigned int v131; // eax
  __int64 v132; // rdi
  unsigned int v133; // edx
  __int64 v134; // r8
  int v135; // r10d
  __int64 *v136; // r8
  unsigned int v137; // ebx
  __int64 v138; // rsi
  int v139; // r11d
  _QWORD *v140; // r8
  int v141; // ecx
  unsigned int v142; // edx
  __int64 v143; // rdi
  int v144; // r8d
  _QWORD *v145; // r11
  int v146; // r8d
  unsigned int v147; // edx
  __int64 v148; // rdi
  int v149; // r10d
  _QWORD *v150; // rdx
  int v151; // ecx
  int v152; // eax
  int v153; // r9d
  int v154; // r9d
  unsigned int v155; // eax
  __int64 v156; // r8
  int v157; // r10d
  _QWORD *v158; // r9
  _QWORD *v159; // r8
  unsigned int v160; // r12d
  int v161; // r9d
  _QWORD *v162; // rsi
  int v163; // r10d
  int v164; // r11d
  __int64 v165; // [rsp+10h] [rbp-1C0h]
  __int64 *v167; // [rsp+20h] [rbp-1B0h]
  __int64 v168; // [rsp+28h] [rbp-1A8h]
  __int64 *v169; // [rsp+30h] [rbp-1A0h]
  unsigned __int64 v170; // [rsp+30h] [rbp-1A0h]
  __int64 *v172; // [rsp+38h] [rbp-198h]
  __int64 v173; // [rsp+48h] [rbp-188h] BYREF
  __int64 v174; // [rsp+50h] [rbp-180h] BYREF
  _QWORD *v175; // [rsp+58h] [rbp-178h]
  __int64 v176; // [rsp+60h] [rbp-170h]
  unsigned int v177; // [rsp+68h] [rbp-168h]
  __int64 v178; // [rsp+70h] [rbp-160h] BYREF
  _QWORD *v179; // [rsp+78h] [rbp-158h]
  __int64 v180; // [rsp+80h] [rbp-150h]
  unsigned int v181; // [rsp+88h] [rbp-148h]
  _BYTE *v182; // [rsp+90h] [rbp-140h] BYREF
  __int64 v183; // [rsp+98h] [rbp-138h]
  _BYTE v184[304]; // [rsp+A0h] [rbp-130h] BYREF

  v13 = a1;
  v14 = (__int64 *)*a2;
  v15 = *((unsigned int *)a2 + 2);
  v174 = 0;
  v175 = 0;
  v16 = &v14[v15];
  v176 = 0;
  v177 = 0;
  if ( v14 == v16 )
  {
    v178 = 0;
    v182 = v184;
    v183 = 0x2000000000LL;
    v179 = 0;
    v180 = 0;
    v181 = 0;
    goto LABEL_69;
  }
  v17 = 0;
  v18 = 0;
  while ( 1 )
  {
    v29 = *v14;
    v30 = *(_QWORD *)(*v14 + 40);
    if ( !v17 )
    {
      ++v174;
      goto LABEL_17;
    }
    LODWORD(v19) = (v17 - 1) & (((unsigned int)v30 >> 4) ^ ((unsigned int)v30 >> 9));
    v20 = &v18[2 * (unsigned int)v19];
    v21 = *v20;
    if ( v30 == *v20 )
      break;
    a13 = 1;
    v33 = 0;
    while ( v21 != -8 )
    {
      if ( !v33 && v21 == -16 )
        v33 = v20;
      LODWORD(v19) = (v17 - 1) & (a13 + (_DWORD)v19);
      v20 = &v18[2 * (unsigned int)v19];
      v21 = *v20;
      if ( v30 == *v20 )
        goto LABEL_4;
      ++a13;
    }
    if ( !v33 )
      v33 = v20;
    ++v174;
    v32 = v176 + 1;
    if ( 4 * ((int)v176 + 1) < 3 * v17 )
    {
      LODWORD(v19) = v17 - (v32 + HIDWORD(v176));
      if ( (unsigned int)v19 > v17 >> 3 )
        goto LABEL_19;
      sub_1B3C430((__int64)&v174, v17);
      if ( !v177 )
        goto LABEL_308;
      v129 = 0;
      LODWORD(v19) = (_DWORD)v175;
      v130 = 1;
      v131 = (v177 - 1) & (((unsigned int)v30 >> 4) ^ ((unsigned int)v30 >> 9));
      v32 = v176 + 1;
      v33 = &v175[2 * v131];
      v132 = *v33;
      if ( v30 == *v33 )
        goto LABEL_19;
      while ( v132 != -8 )
      {
        if ( v132 == -16 && !v129 )
          v129 = v33;
        a13 = v130 + 1;
        v131 = (v177 - 1) & (v131 + v130);
        v33 = &v175[2 * v131];
        v132 = *v33;
        if ( v30 == *v33 )
          goto LABEL_19;
        ++v130;
      }
      goto LABEL_155;
    }
LABEL_17:
    sub_1B3C430((__int64)&v174, 2 * v17);
    if ( !v177 )
      goto LABEL_308;
    v31 = (v177 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
    v32 = v176 + 1;
    v33 = &v175[2 * v31];
    v19 = (__int64 *)*v33;
    if ( v30 == *v33 )
      goto LABEL_19;
    v164 = 1;
    v129 = 0;
    while ( v19 != (__int64 *)-8LL )
    {
      if ( !v129 && v19 == (__int64 *)-16LL )
        v129 = v33;
      a13 = v164 + 1;
      v31 = (v177 - 1) & (v31 + v164);
      v33 = &v175[2 * v31];
      v19 = (__int64 *)*v33;
      if ( v30 == *v33 )
        goto LABEL_19;
      ++v164;
    }
LABEL_155:
    if ( v129 )
      v33 = v129;
LABEL_19:
    LODWORD(v176) = v32;
    if ( *v33 != -8 )
      --HIDWORD(v176);
    *v33 = v30;
    v33[1] = 0;
LABEL_22:
    ++v14;
    v33[1] = v29;
    if ( v16 == v14 )
      goto LABEL_23;
LABEL_14:
    v18 = v175;
    v17 = v177;
  }
LABEL_4:
  v22 = v20[1];
  v23 = v22 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v22 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
  {
    v33 = v20;
    goto LABEL_22;
  }
  if ( (v22 & 4) == 0 )
  {
    v24 = sub_22077B0(48);
    if ( v24 )
    {
      *(_QWORD *)v24 = v24 + 16;
      *(_QWORD *)(v24 + 8) = 0x400000000LL;
    }
    v25 = v24;
    v26 = v24 & 0xFFFFFFFFFFFFFFF8LL;
    v20[1] = v25 | 4;
    v27 = *(unsigned int *)(v26 + 8);
    if ( (unsigned int)v27 >= *(_DWORD *)(v26 + 12) )
    {
      v170 = v26;
      sub_16CD150(v26, (const void *)(v26 + 16), 0, 8, a13, (int)v19);
      v26 = v170;
      v27 = *(unsigned int *)(v170 + 8);
    }
    *(_QWORD *)(*(_QWORD *)v26 + 8 * v27) = v23;
    ++*(_DWORD *)(v26 + 8);
    v23 = v20[1] & 0xFFFFFFFFFFFFFFF8LL;
  }
  v28 = *(unsigned int *)(v23 + 8);
  if ( (unsigned int)v28 >= *(_DWORD *)(v23 + 12) )
  {
    sub_16CD150(v23, (const void *)(v23 + 16), 0, 8, a13, (int)v19);
    v28 = *(unsigned int *)(v23 + 8);
  }
  ++v14;
  *(_QWORD *)(*(_QWORD *)v23 + 8 * v28) = v29;
  ++*(_DWORD *)(v23 + 8);
  if ( v16 != v14 )
    goto LABEL_14;
LABEL_23:
  v178 = 0;
  v13 = a1;
  v34 = (__int64 *)*a2;
  v35 = *((unsigned int *)a2 + 2);
  v182 = v184;
  v183 = 0x2000000000LL;
  v179 = 0;
  v167 = &v34[v35];
  v180 = 0;
  v181 = 0;
  if ( v34 != v167 )
  {
    v172 = v34;
    v36 = (__int64 **)a2;
    while ( 1 )
    {
      v37 = *v172;
      v38 = *(_QWORD *)(*v172 + 40);
      if ( !v177 )
      {
        ++v174;
        goto LABEL_162;
      }
      v39 = v177 - 1;
      v40 = (v177 - 1) & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
      v41 = v175[2 * v40];
      v169 = &v175[2 * v40];
      if ( v38 != v41 )
      {
        v120 = 1;
        v121 = 0;
        while ( v41 != -8 )
        {
          if ( !v121 && v41 == -16 )
            v121 = v169;
          LODWORD(v19) = v120 + 1;
          v40 = v39 & (v120 + v40);
          v169 = &v175[2 * v40];
          v41 = *v169;
          if ( v38 == *v169 )
            goto LABEL_27;
          ++v120;
        }
        if ( !v121 )
          v121 = v169;
        ++v174;
        v122 = v176 + 1;
        if ( 4 * ((int)v176 + 1) < 3 * v177 )
        {
          if ( v177 - HIDWORD(v176) - v122 > v177 >> 3 )
            goto LABEL_131;
          sub_1B3C430((__int64)&v174, v177);
          if ( v177 )
          {
            v136 = 0;
            v137 = (v177 - 1) & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
            LODWORD(v19) = 1;
            v122 = v176 + 1;
            v121 = &v175[2 * v137];
            v138 = *v121;
            if ( v38 != *v121 )
            {
              while ( v138 != -8 )
              {
                if ( !v136 && v138 == -16 )
                  v136 = v121;
                v137 = (v177 - 1) & ((_DWORD)v19 + v137);
                v121 = &v175[2 * v137];
                v138 = *v121;
                if ( v38 == *v121 )
                  goto LABEL_131;
                LODWORD(v19) = (_DWORD)v19 + 1;
              }
              if ( v136 )
                v121 = v136;
            }
            goto LABEL_131;
          }
          goto LABEL_308;
        }
LABEL_162:
        sub_1B3C430((__int64)&v174, 2 * v177);
        if ( v177 )
        {
          v133 = (v177 - 1) & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
          v122 = v176 + 1;
          v121 = &v175[2 * v133];
          v134 = *v121;
          if ( v38 != *v121 )
          {
            v135 = 1;
            v19 = 0;
            while ( v134 != -8 )
            {
              if ( !v19 && v134 == -16 )
                v19 = v121;
              v133 = (v177 - 1) & (v135 + v133);
              v121 = &v175[2 * v133];
              v134 = *v121;
              if ( v38 == *v121 )
                goto LABEL_131;
              ++v135;
            }
            if ( v19 )
              v121 = v19;
          }
LABEL_131:
          LODWORD(v176) = v122;
          if ( *v121 != -8 )
            --HIDWORD(v176);
          *v121 = v38;
          v121[1] = 0;
          goto LABEL_55;
        }
LABEL_308:
        LODWORD(v176) = v176 + 1;
        BUG();
      }
LABEL_27:
      v42 = v169[1];
      v43 = v42 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v42 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        goto LABEL_55;
      if ( (v42 & 4) == 0 )
        goto LABEL_94;
      v44 = *(unsigned int *)(v43 + 8);
      if ( !(_DWORD)v44 )
        goto LABEL_55;
      if ( (_DWORD)v44 == 1 )
      {
LABEL_94:
        if ( *(_BYTE *)(v37 + 16) == 55 )
        {
          v109 = (void (*)())(*v13)[6];
          if ( v109 != nullsub_638 )
            ((void (__fastcall *)(__int64 **, __int64))v109)(v13, v37);
          sub_1B3BE00(v13[1], v38, *(_QWORD *)(v37 - 48));
        }
        else
        {
          v119 = (unsigned int)v183;
          if ( (unsigned int)v183 >= HIDWORD(v183) )
          {
            sub_16CD150((__int64)&v182, v184, 0, 8, v39, (int)v19);
            v119 = (unsigned int)v183;
          }
          *(_QWORD *)&v182[8 * v119] = v37;
          LODWORD(v183) = v183 + 1;
        }
        v73 = v169;
        v74 = v169[1];
        if ( (v74 & 4) == 0 )
          goto LABEL_54;
        goto LABEL_99;
      }
      v45 = *(__int64 **)v43;
      v46 = *(_QWORD *)v43 + 8 * v44;
      v47 = *(__int64 **)v43;
      while ( *(_BYTE *)(*v47 + 16) != 55 )
      {
        if ( (__int64 *)v46 == ++v47 )
        {
          v111 = (unsigned int)v183;
          do
          {
            v112 = *v45;
            if ( HIDWORD(v183) <= (unsigned int)v111 )
            {
              sub_16CD150((__int64)&v182, v184, 0, 8, v39, (int)v19);
              v111 = (unsigned int)v183;
            }
            ++v45;
            *(_QWORD *)&v182[8 * v111] = v112;
            v111 = (unsigned int)(v183 + 1);
            LODWORD(v183) = v183 + 1;
          }
          while ( (__int64 *)v46 != v45 );
          v74 = v169[1];
          if ( (v74 & 4) != 0 )
            goto LABEL_99;
          v169[1] = 0;
          goto LABEL_55;
        }
      }
      v48 = *(_QWORD *)(v38 + 48);
      v49 = 0;
      v168 = v38 + 40;
      if ( v38 + 40 == v48 )
        goto LABEL_53;
      v50 = (__int64)v36;
      v165 = *(_QWORD *)(*v172 + 40);
      v51 = v13;
      v52 = 0;
      v53 = v50;
      do
      {
        while ( 1 )
        {
          if ( !v48 )
            BUG();
          v65 = *(_BYTE *)(v48 - 8);
          v66 = v48 - 24;
          if ( v65 != 54 )
            break;
          v54 = (bool (__fastcall *)(__int64, __int64, __int64))(*v51)[2];
          if ( v54 != sub_1B3B800 )
          {
            if ( !v54((__int64)v51, v48 - 24, v53) )
              goto LABEL_44;
            if ( !v52 )
            {
LABEL_91:
              v108 = (unsigned int)v183;
              if ( (unsigned int)v183 >= HIDWORD(v183) )
              {
                sub_16CD150((__int64)&v182, v184, 0, 8, v57, v58);
                v108 = (unsigned int)v183;
              }
              *(_QWORD *)&v182[8 * v108] = v66;
              LODWORD(v183) = v183 + 1;
              goto LABEL_44;
            }
LABEL_39:
            v61 = (void (*)())(*v51)[4];
            if ( v61 != nullsub_636 )
              ((void (__fastcall *)(__int64 **, __int64, __int64))v61)(v51, v48 - 24, v52);
            sub_164D160(v48 - 24, v52, a3, a4, a5, a6, v59, v60, a9, a10);
            if ( v181 )
            {
              v62 = (v181 - 1) & (((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4));
              v63 = &v179[2 * v62];
              v64 = *v63;
              if ( v66 == *v63 )
              {
LABEL_43:
                v63[1] = v52;
                goto LABEL_44;
              }
              v139 = 1;
              v140 = 0;
              while ( v64 != -8 )
              {
                if ( !v140 && v64 == -16 )
                  v140 = v63;
                v62 = (v181 - 1) & (v139 + v62);
                v63 = &v179[2 * v62];
                v64 = *v63;
                if ( v66 == *v63 )
                  goto LABEL_43;
                ++v139;
              }
              if ( v140 )
                v63 = v140;
              ++v178;
              v141 = v180 + 1;
              if ( 4 * ((int)v180 + 1) < 3 * v181 )
              {
                if ( v181 - HIDWORD(v180) - v141 <= v181 >> 3 )
                {
                  sub_176F940((__int64)&v178, v181);
                  if ( !v181 )
                    goto LABEL_307;
                  v145 = 0;
                  v146 = 1;
                  v147 = (v181 - 1) & (((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4));
                  v141 = v180 + 1;
                  v63 = &v179[2 * v147];
                  v148 = *v63;
                  if ( v66 != *v63 )
                  {
                    while ( v148 != -8 )
                    {
                      if ( !v145 && v148 == -16 )
                        v145 = v63;
                      v147 = (v181 - 1) & (v146 + v147);
                      v63 = &v179[2 * v147];
                      v148 = *v63;
                      if ( v66 == *v63 )
                        goto LABEL_181;
                      ++v146;
                    }
                    goto LABEL_212;
                  }
                }
                goto LABEL_181;
              }
            }
            else
            {
              ++v178;
            }
            sub_176F940((__int64)&v178, 2 * v181);
            if ( !v181 )
              goto LABEL_307;
            v142 = (v181 - 1) & (((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4));
            v141 = v180 + 1;
            v63 = &v179[2 * v142];
            v143 = *v63;
            if ( v66 != *v63 )
            {
              v144 = 1;
              v145 = 0;
              while ( v143 != -8 )
              {
                if ( !v145 && v143 == -16 )
                  v145 = v63;
                v142 = (v181 - 1) & (v144 + v142);
                v63 = &v179[2 * v142];
                v143 = *v63;
                if ( v66 == *v63 )
                  goto LABEL_181;
                ++v144;
              }
LABEL_212:
              if ( v145 )
                v63 = v145;
            }
LABEL_181:
            LODWORD(v180) = v141;
            if ( *v63 != -8 )
              --HIDWORD(v180);
            *v63 = v66;
            v63[1] = 0;
            goto LABEL_43;
          }
          v55 = *(_QWORD **)v53;
          v56 = *(unsigned int *)(v53 + 8);
          v173 = v48 - 24;
          if ( &v55[v56] != sub_1B3B740(v55, (__int64)&v55[v56], &v173) )
          {
            if ( !v52 )
              goto LABEL_91;
            goto LABEL_39;
          }
LABEL_44:
          v48 = *(_QWORD *)(v48 + 8);
          if ( v168 == v48 )
            goto LABEL_52;
        }
        if ( v65 != 55 )
          goto LABEL_44;
        v67 = (bool (__fastcall *)(__int64, __int64, __int64))(*v51)[2];
        if ( v67 != sub_1B3B800 )
        {
          if ( v67((__int64)v51, v48 - 24, v53) )
          {
            v71 = (void (*)())(*v51)[6];
            if ( v71 == nullsub_638 )
              goto LABEL_51;
            goto LABEL_160;
          }
          goto LABEL_44;
        }
        v68 = *(_QWORD **)v53;
        v69 = *(unsigned int *)(v53 + 8);
        v173 = v48 - 24;
        if ( &v68[v69] == sub_1B3B740(v68, (__int64)&v68[v69], &v173) )
          goto LABEL_44;
        v71 = *(void (**)())(v70 + 48);
        if ( v71 == nullsub_638 )
          goto LABEL_51;
LABEL_160:
        ((void (__fastcall *)(__int64 **, __int64))v71)(v51, v48 - 24);
LABEL_51:
        v52 = *(_QWORD *)(v48 - 72);
        v48 = *(_QWORD *)(v48 + 8);
      }
      while ( v168 != v48 );
LABEL_52:
      v72 = v53;
      v49 = v52;
      v38 = v165;
      v13 = v51;
      v36 = (__int64 **)v72;
LABEL_53:
      sub_1B3BE00(v13[1], v38, v49);
      v73 = v169;
      v74 = v169[1];
      if ( (v74 & 4) == 0 )
      {
LABEL_54:
        v73[1] = 0;
        goto LABEL_55;
      }
LABEL_99:
      v110 = v74 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v110 )
        *(_DWORD *)(v110 + 8) = 0;
LABEL_55:
      if ( v167 == ++v172 )
      {
        v75 = (unsigned __int64)v182;
        v76 = &v182[8 * (unsigned int)v183];
        if ( v182 != v76 )
        {
          while ( 1 )
          {
            v80 = *(__int64 ****)v75;
            v81 = sub_1B40B40(v13[1], *(_QWORD *)(*(_QWORD *)v75 + 40LL));
            v84 = (void (*)())(*v13)[4];
            if ( v84 == nullsub_636 )
            {
              if ( v80 != (__int64 ***)v81 )
                goto LABEL_62;
            }
            else
            {
              ((void (__fastcall *)(__int64 **, __int64 ***, __int64))v84)(v13, v80, v81);
              if ( v80 != (__int64 ***)v81 )
                goto LABEL_62;
            }
            v81 = sub_1599EF0(*v80);
LABEL_62:
            sub_164D160((__int64)v80, v81, a3, a4, a5, a6, v82, v83, a9, a10);
            if ( !v181 )
            {
              ++v178;
              goto LABEL_64;
            }
            v77 = (v181 - 1) & (((unsigned int)v80 >> 9) ^ ((unsigned int)v80 >> 4));
            v78 = (__int64 ****)&v179[2 * v77];
            v79 = *v78;
            if ( v80 == *v78 )
            {
LABEL_59:
              v75 += 8LL;
              v78[1] = (__int64 ***)v81;
              if ( v76 == (_BYTE *)v75 )
                goto LABEL_69;
            }
            else
            {
              v123 = 1;
              v124 = 0;
              while ( v79 != (__int64 ***)-8LL )
              {
                if ( v79 == (__int64 ***)-16LL && !v124 )
                  v124 = v78;
                v77 = (v181 - 1) & (v123 + v77);
                v78 = (__int64 ****)&v179[2 * v77];
                v79 = *v78;
                if ( v80 == *v78 )
                  goto LABEL_59;
                ++v123;
              }
              if ( v124 )
                v78 = v124;
              ++v178;
              v86 = v180 + 1;
              if ( 4 * ((int)v180 + 1) < 3 * v181 )
              {
                if ( v181 - HIDWORD(v180) - v86 > v181 >> 3 )
                  goto LABEL_66;
                sub_176F940((__int64)&v178, v181);
                if ( !v181 )
                  goto LABEL_307;
                v125 = 0;
                v126 = 1;
                v127 = (v181 - 1) & (((unsigned int)v80 >> 9) ^ ((unsigned int)v80 >> 4));
                v86 = v180 + 1;
                v78 = (__int64 ****)&v179[2 * v127];
                v128 = *v78;
                if ( v80 == *v78 )
                  goto LABEL_66;
                while ( v128 != (__int64 ***)-8LL )
                {
                  if ( !v125 && v128 == (__int64 ***)-16LL )
                    v125 = v78;
                  v127 = (v181 - 1) & (v126 + v127);
                  v78 = (__int64 ****)&v179[2 * v127];
                  v128 = *v78;
                  if ( v80 == *v78 )
                    goto LABEL_66;
                  ++v126;
                }
                goto LABEL_143;
              }
LABEL_64:
              sub_176F940((__int64)&v178, 2 * v181);
              if ( !v181 )
                goto LABEL_307;
              v85 = (v181 - 1) & (((unsigned int)v80 >> 9) ^ ((unsigned int)v80 >> 4));
              v86 = v180 + 1;
              v78 = (__int64 ****)&v179[2 * v85];
              v87 = *v78;
              if ( v80 == *v78 )
                goto LABEL_66;
              v163 = 1;
              v125 = 0;
              while ( v87 != (__int64 ***)-8LL )
              {
                if ( !v125 && v87 == (__int64 ***)-16LL )
                  v125 = v78;
                v85 = (v181 - 1) & (v163 + v85);
                v78 = (__int64 ****)&v179[2 * v85];
                v87 = *v78;
                if ( v80 == *v78 )
                  goto LABEL_66;
                ++v163;
              }
LABEL_143:
              if ( v125 )
                v78 = v125;
LABEL_66:
              LODWORD(v180) = v86;
              if ( *v78 != (__int64 ***)-8LL )
                --HIDWORD(v180);
              v75 += 8LL;
              v78[1] = 0;
              *v78 = v80;
              v78[1] = (__int64 ***)v81;
              if ( v76 == (_BYTE *)v75 )
                goto LABEL_69;
            }
          }
        }
        break;
      }
    }
  }
LABEL_69:
  v88 = (void (*)())(*v13)[3];
  if ( v88 != nullsub_656 )
    ((void (__fastcall *)(__int64 **))v88)(v13);
  v89 = *a2;
  v90 = *a2 + 8LL * *((unsigned int *)a2 + 2);
  if ( *a2 != v90 )
  {
    while ( 1 )
    {
      v92 = *(_QWORD **)v89;
      if ( !*(_QWORD *)(*(_QWORD *)v89 + 8LL) )
      {
        v91 = (void (*)())(*v13)[5];
        if ( v91 != nullsub_637 )
          goto LABEL_87;
        goto LABEL_74;
      }
      v93 = v181;
      if ( !v181 )
        break;
      v94 = v181 - 1;
      v95 = v179;
      v96 = (v181 - 1) & (((unsigned int)v92 >> 9) ^ ((unsigned int)v92 >> 4));
      v97 = &v179[2 * v96];
      v98 = (_QWORD *)*v97;
      if ( v92 == (_QWORD *)*v97 )
        goto LABEL_78;
      v149 = 1;
      v150 = 0;
      while ( 1 )
      {
        if ( v98 == (_QWORD *)-8LL )
        {
          if ( !v150 )
            v150 = v97;
          ++v178;
          v151 = v180 + 1;
          if ( 4 * ((int)v180 + 1) < 3 * v181 )
          {
            if ( v181 - HIDWORD(v180) - v151 > v181 >> 3 )
            {
LABEL_221:
              LODWORD(v180) = v151;
              if ( *v150 != -8 )
                --HIDWORD(v180);
              *v150 = v92;
              v99 = 0;
              v150[1] = 0;
              v93 = v181;
              if ( !v181 )
                goto LABEL_84;
              v101 = v179;
              v94 = v181 - 1;
              v100 = 0;
              v102 = *v179;
              v95 = v179;
              if ( !*v179 )
                goto LABEL_79;
              goto LABEL_225;
            }
            sub_176F940((__int64)&v178, v181);
            if ( v181 )
            {
              v159 = 0;
              v160 = (v181 - 1) & (((unsigned int)v92 >> 9) ^ ((unsigned int)v92 >> 4));
              v161 = 1;
              v151 = v180 + 1;
              v150 = &v179[2 * v160];
              v162 = (_QWORD *)*v150;
              if ( v92 != (_QWORD *)*v150 )
              {
                while ( v162 != (_QWORD *)-8LL )
                {
                  if ( !v159 && v162 == (_QWORD *)-16LL )
                    v159 = v150;
                  v160 = (v181 - 1) & (v161 + v160);
                  v150 = &v179[2 * v160];
                  v162 = (_QWORD *)*v150;
                  if ( v92 == (_QWORD *)*v150 )
                    goto LABEL_221;
                  ++v161;
                }
                if ( v159 )
                  v150 = v159;
              }
              goto LABEL_221;
            }
LABEL_307:
            LODWORD(v180) = v180 + 1;
            BUG();
          }
LABEL_232:
          sub_176F940((__int64)&v178, 2 * v181);
          if ( v181 )
          {
            v155 = (v181 - 1) & (((unsigned int)v92 >> 9) ^ ((unsigned int)v92 >> 4));
            v151 = v180 + 1;
            v150 = &v179[2 * v155];
            v156 = *v150;
            if ( v92 != (_QWORD *)*v150 )
            {
              v157 = 1;
              v158 = 0;
              while ( v156 != -8 )
              {
                if ( !v158 && v156 == -16 )
                  v158 = v150;
                v155 = (v181 - 1) & (v157 + v155);
                v150 = &v179[2 * v155];
                v156 = *v150;
                if ( v92 == (_QWORD *)*v150 )
                  goto LABEL_221;
                ++v157;
              }
              if ( v158 )
                v150 = v158;
            }
            goto LABEL_221;
          }
          goto LABEL_307;
        }
        if ( v150 || v98 != (_QWORD *)-16LL )
          v97 = v150;
        v96 = v94 & (v149 + v96);
        v98 = (_QWORD *)v179[2 * v96];
        if ( v92 == v98 )
          break;
        ++v149;
        v150 = v97;
        v97 = &v179[2 * v96];
      }
      v97 = &v179[2 * v96];
LABEL_78:
      v99 = v97[1];
      v100 = v94 & (((unsigned int)v99 >> 4) ^ ((unsigned int)v99 >> 9));
      v101 = &v179[2 * v100];
      v102 = *v101;
      if ( *v101 == v99 )
      {
LABEL_79:
        v103 = &v95[2 * v93];
LABEL_80:
        while ( v103 != v101 )
        {
          v99 = v101[1];
          v104 = v94 & (((unsigned int)v99 >> 9) ^ ((unsigned int)v99 >> 4));
          v101 = &v95[2 * v104];
          v105 = *v101;
          if ( v99 != *v101 )
          {
            v106 = 1;
            while ( v105 != -8 )
            {
              v154 = v106 + 1;
              v104 = v94 & (v106 + v104);
              v101 = &v95[2 * v104];
              v105 = *v101;
              if ( v99 == *v101 )
                goto LABEL_80;
              v106 = v154;
            }
            goto LABEL_84;
          }
        }
        goto LABEL_84;
      }
LABEL_225:
      v152 = 1;
      while ( v102 != -8 )
      {
        v153 = v152 + 1;
        v100 = v94 & (v152 + v100);
        v101 = &v95[2 * v100];
        v102 = *v101;
        if ( v99 == *v101 )
          goto LABEL_79;
        v152 = v153;
      }
LABEL_84:
      v107 = (void (*)())(*v13)[4];
      if ( v107 != nullsub_636 )
        ((void (__fastcall *)(__int64 **, _QWORD *, __int64))v107)(v13, v92, v99);
      sub_164D160((__int64)v92, v99, a3, a4, a5, a6, a7, a8, a9, a10);
      v91 = (void (*)())(*v13)[5];
      if ( v91 != nullsub_637 )
LABEL_87:
        ((void (__fastcall *)(__int64 **, _QWORD *))v91)(v13, v92);
LABEL_74:
      v89 += 8;
      sub_15F20C0(v92);
      if ( v90 == v89 )
        goto LABEL_109;
    }
    ++v178;
    goto LABEL_232;
  }
LABEL_109:
  j___libc_free_0(v179);
  if ( v182 != v184 )
    _libc_free((unsigned __int64)v182);
  if ( v177 )
  {
    v113 = v175;
    v114 = &v175[2 * v177];
    do
    {
      if ( *v113 != -16 && *v113 != -8 )
      {
        v115 = v113[1];
        if ( (v115 & 4) != 0 )
        {
          v116 = (unsigned __int64 *)(v115 & 0xFFFFFFFFFFFFFFF8LL);
          v117 = v116;
          if ( v116 )
          {
            if ( (unsigned __int64 *)*v116 != v116 + 2 )
              _libc_free(*v116);
            j_j___libc_free_0(v117, 48);
          }
        }
      }
      v113 += 2;
    }
    while ( v114 != v113 );
  }
  return j___libc_free_0(v175);
}
