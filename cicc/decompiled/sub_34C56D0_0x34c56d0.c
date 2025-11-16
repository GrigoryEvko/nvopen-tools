// Function: sub_34C56D0
// Address: 0x34c56d0
//
__int64 __fastcall sub_34C56D0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r15d
  __int64 v3; // rdi
  __int64 (*v4)(); // rax
  __int64 v6; // rdx
  __int64 v7; // r14
  _QWORD *v8; // r12
  unsigned __int64 v9; // rbx
  __int64 v10; // rcx
  _QWORD *v11; // r8
  __int64 v12; // r9
  __int64 v13; // r15
  __int64 i; // r14
  unsigned int v15; // edi
  unsigned __int8 v16; // al
  __int64 *v17; // rax
  __int64 *v18; // rsi
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  unsigned __int64 v22; // rbx
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rbx
  unsigned __int64 v25; // rdi
  __int64 v26; // r12
  __int64 v27; // r13
  __int64 *v28; // rbx
  unsigned __int64 v29; // rcx
  __int64 v30; // rax
  _QWORD *v31; // r13
  __int64 v32; // rbx
  _QWORD *v33; // r12
  unsigned __int64 *v34; // rcx
  unsigned __int64 v35; // rdx
  unsigned __int64 v36; // rbx
  unsigned __int64 v37; // rdi
  unsigned __int64 v38; // rbx
  unsigned __int64 v39; // rdi
  __int64 v40; // rcx
  _QWORD *v41; // r8
  __int64 v42; // rdi
  __int64 (*v43)(); // rax
  __int64 v44; // rdx
  __int64 v45; // rsi
  unsigned int v46; // eax
  unsigned __int8 v47; // di
  _DWORD *v48; // rcx
  __int64 v49; // rbx
  _BYTE *v50; // r15
  _BYTE *v51; // r14
  _BYTE *v52; // rbx
  unsigned int v53; // esi
  _DWORD *v54; // rax
  _BYTE *v55; // rdx
  _BYTE *v56; // r13
  _BYTE *v57; // rdx
  _BYTE *v58; // r14
  _BYTE *v59; // r15
  _BYTE *v60; // rbx
  __int64 v61; // rcx
  __int64 v62; // r8
  __int64 v63; // r9
  signed int v64; // r13d
  __int64 v65; // rcx
  __int64 v66; // r8
  __int64 v67; // r9
  _BYTE *v68; // r13
  _BYTE *v69; // rdi
  int *v70; // r10
  __int64 v71; // rcx
  int *v72; // r10
  unsigned __int64 v73; // rcx
  unsigned __int64 v74; // r15
  __int64 j; // r15
  __int16 v76; // ax
  _QWORD *v77; // rax
  _QWORD *v78; // rdx
  __int64 v79; // rax
  __int64 k; // r15
  int *v81; // r10
  int *v82; // r10
  __int64 v83; // rdx
  __int64 v84; // rdi
  __int64 v85; // r9
  __int64 v86; // rcx
  _DWORD *v87; // rax
  _DWORD *v88; // rsi
  __int64 (*v89)(); // rax
  __int64 v90; // r14
  __int64 m; // rbx
  unsigned int v92; // edi
  __int64 v93; // rcx
  __int64 v94; // r8
  __int64 v95; // r9
  unsigned __int64 v96; // rax
  int *v97; // rdi
  unsigned __int64 *v98; // rbx
  _QWORD *v99; // r12
  char v100; // r13
  __int64 v101; // rdx
  __int64 v102; // rcx
  __int64 v103; // r8
  __int64 v104; // r9
  __int64 v105; // rdx
  __int64 v106; // rax
  char v107; // r12
  unsigned __int64 v108; // rax
  __int64 v109; // rax
  int *v110; // r10
  int *v111; // r9
  char *v112; // rax
  __int64 v113; // rdx
  char *v114; // rdx
  char *v115; // rbx
  unsigned __int16 *v116; // r12
  int v117; // eax
  __int16 *v118; // rax
  int v119; // edx
  int v120; // eax
  __int64 v121; // r15
  int v122; // ebx
  _QWORD *v123; // r14
  __int64 v124; // r12
  _BYTE *v125; // [rsp+8h] [rbp-338h]
  __int64 v126; // [rsp+18h] [rbp-328h]
  __int64 v127; // [rsp+20h] [rbp-320h]
  __int64 v128; // [rsp+20h] [rbp-320h]
  unsigned __int8 v129; // [rsp+36h] [rbp-30Ah]
  unsigned __int8 v130; // [rsp+37h] [rbp-309h]
  __int64 *v131; // [rsp+40h] [rbp-300h]
  __int64 v132; // [rsp+50h] [rbp-2F0h]
  __int64 v133; // [rsp+78h] [rbp-2C8h]
  unsigned __int64 v134; // [rsp+78h] [rbp-2C8h]
  __int64 v135; // [rsp+80h] [rbp-2C0h]
  __int64 v136; // [rsp+80h] [rbp-2C0h]
  __int64 v137; // [rsp+80h] [rbp-2C0h]
  _QWORD *v138; // [rsp+80h] [rbp-2C0h]
  __int64 v141; // [rsp+A0h] [rbp-2A0h] BYREF
  __int64 v142; // [rsp+A8h] [rbp-298h] BYREF
  _QWORD v143[2]; // [rsp+B0h] [rbp-290h] BYREF
  unsigned __int64 v144; // [rsp+C0h] [rbp-280h] BYREF
  __int64 v145; // [rsp+C8h] [rbp-278h]
  __int64 v146; // [rsp+D0h] [rbp-270h]
  __int64 v147; // [rsp+E0h] [rbp-260h] BYREF
  _BYTE *v148; // [rsp+E8h] [rbp-258h]
  __int64 v149; // [rsp+F0h] [rbp-250h]
  __int64 v150; // [rsp+F8h] [rbp-248h]
  _BYTE v151[16]; // [rsp+100h] [rbp-240h] BYREF
  unsigned __int64 v152; // [rsp+110h] [rbp-230h]
  int v153; // [rsp+118h] [rbp-228h]
  _BYTE *v154; // [rsp+120h] [rbp-220h] BYREF
  __int64 v155; // [rsp+128h] [rbp-218h]
  _BYTE v156[24]; // [rsp+130h] [rbp-210h] BYREF
  int v157; // [rsp+148h] [rbp-1F8h] BYREF
  __int64 v158; // [rsp+150h] [rbp-1F0h]
  int *v159; // [rsp+158h] [rbp-1E8h]
  int *v160; // [rsp+160h] [rbp-1E0h]
  __int64 v161; // [rsp+168h] [rbp-1D8h]
  _BYTE *v162; // [rsp+170h] [rbp-1D0h] BYREF
  __int64 v163; // [rsp+178h] [rbp-1C8h]
  _BYTE v164[24]; // [rsp+180h] [rbp-1C0h] BYREF
  int v165; // [rsp+198h] [rbp-1A8h] BYREF
  __int64 v166; // [rsp+1A0h] [rbp-1A0h]
  int *v167; // [rsp+1A8h] [rbp-198h]
  int *v168; // [rsp+1B0h] [rbp-190h]
  __int64 v169; // [rsp+1B8h] [rbp-188h]
  _BYTE *v170; // [rsp+1C0h] [rbp-180h] BYREF
  __int64 v171; // [rsp+1C8h] [rbp-178h]
  _BYTE v172[24]; // [rsp+1D0h] [rbp-170h] BYREF
  int v173; // [rsp+1E8h] [rbp-158h] BYREF
  __int64 v174; // [rsp+1F0h] [rbp-150h]
  int *v175; // [rsp+1F8h] [rbp-148h]
  int *v176; // [rsp+200h] [rbp-140h]
  __int64 v177; // [rsp+208h] [rbp-138h]
  _BYTE *v178; // [rsp+210h] [rbp-130h] BYREF
  __int64 v179; // [rsp+218h] [rbp-128h]
  _BYTE v180[24]; // [rsp+220h] [rbp-120h] BYREF
  int v181; // [rsp+238h] [rbp-108h] BYREF
  unsigned __int64 v182; // [rsp+240h] [rbp-100h]
  int *v183; // [rsp+248h] [rbp-F8h]
  int *v184; // [rsp+250h] [rbp-F0h]
  __int64 v185; // [rsp+258h] [rbp-E8h]
  _BYTE *v186; // [rsp+260h] [rbp-E0h] BYREF
  __int64 v187; // [rsp+268h] [rbp-D8h]
  _BYTE v188[208]; // [rsp+270h] [rbp-D0h] BYREF

  v2 = 0;
  v3 = *(_QWORD *)(a1 + 136);
  v186 = v188;
  v141 = 0;
  v142 = 0;
  v187 = 0x400000000LL;
  v4 = *(__int64 (**)())(*(_QWORD *)v3 + 344LL);
  if ( v4 == sub_2DB1AE0 )
    return v2;
  if ( ((unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *, __int64 *, _BYTE **, __int64))v4)(
         v3,
         a2,
         &v141,
         &v142,
         &v186,
         1)
    || !v141
    || !(_DWORD)v187 )
  {
    goto LABEL_24;
  }
  v6 = v142;
  if ( !v142 )
  {
    v17 = *(__int64 **)(a2 + 112);
    v18 = &v17[*(unsigned int *)(a2 + 120)];
    if ( v17 == v18 )
      goto LABEL_24;
    while ( 1 )
    {
      v6 = *v17;
      if ( v141 != *v17 )
        break;
      if ( v18 == ++v17 )
        goto LABEL_24;
    }
    v142 = *v17;
    if ( !v6 )
      goto LABEL_24;
  }
  if ( *(_DWORD *)(v141 + 72) > 1u || *(_DWORD *)(v6 + 72) > 1u )
  {
LABEL_24:
    v2 = 0;
    goto LABEL_25;
  }
  v157 = 0;
  v154 = v156;
  v155 = 0x400000000LL;
  v163 = 0x400000000LL;
  v167 = &v165;
  v168 = &v165;
  v7 = *(_QWORD *)(a1 + 136);
  v159 = &v157;
  v160 = &v157;
  v8 = *(_QWORD **)(a1 + 152);
  v162 = v164;
  v158 = 0;
  v161 = 0;
  v165 = 0;
  v166 = 0;
  v169 = 0;
  v135 = v7;
  v9 = sub_2E313E0(a2);
  v130 = sub_2FDF4D0(v7, v9);
  if ( !v130 )
    goto LABEL_28;
  v13 = *(_QWORD *)(v9 + 32);
  for ( i = v13 + 40LL * (*(_DWORD *)(v9 + 40) & 0xFFFFFF); i != v13; v13 += 40 )
  {
    if ( !*(_BYTE *)v13 )
    {
      v15 = *(_DWORD *)(v13 + 8);
      if ( v15 )
      {
        v16 = *(_BYTE *)(v13 + 3);
        if ( (v16 & 0x10) != 0 )
        {
          if ( (((v16 & 0x10) != 0) & (v16 >> 6)) == 0 )
            goto LABEL_28;
          sub_34C53D0(v15, v8, (__int64)&v162, v10, (__int64)v11, v12);
        }
        else
        {
          sub_34C53D0(v15, v8, (__int64)&v154, v10, (__int64)v11, v12);
        }
      }
    }
  }
  if ( !(_DWORD)v155 && !v161 )
    goto LABEL_39;
  v131 = *(__int64 **)(a2 + 56);
  if ( (__int64 *)v9 == v131 )
    goto LABEL_40;
  v73 = *(_QWORD *)v9 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v73 )
    BUG();
  v74 = *(_QWORD *)v9 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (*(_QWORD *)v73 & 4) == 0 && (*(_BYTE *)(v73 + 44) & 4) != 0 )
  {
    for ( j = *(_QWORD *)v73; ; j = *(_QWORD *)v74 )
    {
      v74 = j & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_BYTE *)(v74 + 44) & 4) == 0 )
        break;
    }
  }
  while ( v131 != (__int64 *)v74 )
  {
    v76 = *(_WORD *)(v74 + 68);
    if ( (unsigned __int16)(v76 - 14) > 4u && v76 != 24 )
      break;
    v77 = (_QWORD *)(*(_QWORD *)v74 & 0xFFFFFFFFFFFFFFF8LL);
    v78 = v77;
    if ( !v77 )
      BUG();
    v74 = *(_QWORD *)v74 & 0xFFFFFFFFFFFFFFF8LL;
    v79 = *v77;
    if ( (v79 & 4) == 0 && (*((_BYTE *)v78 + 44) & 4) != 0 )
    {
      for ( k = v79; ; k = *(_QWORD *)v74 )
      {
        v74 = k & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_BYTE *)(v74 + 44) & 4) == 0 )
          break;
      }
    }
  }
  v83 = *(_QWORD *)(v74 + 32);
  v84 = 4LL * (unsigned int)v155;
  v85 = v83 + 40LL * (*(_DWORD *)(v74 + 40) & 0xFFFFFF);
  if ( v83 == v85 )
  {
LABEL_39:
    v131 = (__int64 *)v9;
    goto LABEL_40;
  }
  while ( 1 )
  {
    if ( *(_BYTE *)v83 == 12 )
      goto LABEL_39;
    if ( !*(_BYTE *)v83 && (*(_BYTE *)(v83 + 3) & 0x10) != 0 )
    {
      v86 = *(unsigned int *)(v83 + 8);
      if ( (_DWORD)v86 )
        break;
    }
LABEL_243:
    v83 += 40;
    if ( v85 == v83 )
      goto LABEL_39;
  }
  if ( v161 )
  {
    v109 = v158;
    if ( v158 )
    {
      v110 = &v157;
      do
      {
        v11 = *(_QWORD **)(v109 + 16);
        if ( (unsigned int)v86 > *(_DWORD *)(v109 + 32) )
        {
          v109 = *(_QWORD *)(v109 + 24);
        }
        else
        {
          v110 = (int *)v109;
          v109 = *(_QWORD *)(v109 + 16);
        }
      }
      while ( v109 );
      if ( v110 != &v157 && (unsigned int)v86 >= v110[8] )
        goto LABEL_229;
    }
    goto LABEL_243;
  }
  v87 = v154;
  v88 = &v154[v84];
  if ( v154 == &v154[v84] )
    goto LABEL_243;
  while ( (_DWORD)v86 != *v87 )
  {
    if ( v88 == ++v87 )
      goto LABEL_243;
  }
  if ( v87 == v88 )
    goto LABEL_243;
LABEL_229:
  LOBYTE(v147) = 1;
  if ( !(unsigned __int8)sub_2E8B400(v74, (__int64)&v147, v83, v86, v11)
    || (v89 = *(__int64 (**)())(*(_QWORD *)v135 + 920LL), v89 != sub_2DB1B30)
    && ((unsigned __int8 (__fastcall *)(__int64, unsigned __int64))v89)(v135, v74) )
  {
LABEL_28:
    v2 = 0;
    goto LABEL_29;
  }
  v90 = *(_QWORD *)(v74 + 32);
  for ( m = v90 + 40LL * (*(_DWORD *)(v74 + 40) & 0xFFFFFF); m != v90; v90 += 40 )
  {
    if ( !*(_BYTE *)v90 )
    {
      v92 = *(_DWORD *)(v90 + 8);
      LODWORD(v170) = v92;
      if ( v92 )
      {
        if ( (*(_BYTE *)(v90 + 3) & 0x10) != 0 )
        {
          if ( sub_34C0EE0((__int64)&v154, (unsigned int *)&v170) && (unsigned int)((_DWORD)v170 - 1) <= 0x3FFFFFFE )
          {
            v118 = (__int16 *)(v8[7] + 2LL * *(unsigned int *)(v8[1] + 24LL * (unsigned int)v170 + 4));
            v94 = (__int64)(v118 + 1);
            v119 = *v118 + (_DWORD)v170;
            if ( *v118 )
            {
              v134 = v74;
              v120 = (unsigned __int16)v119;
              v121 = m;
              v122 = v119;
              v132 = v90;
              v123 = v8;
              v124 = v94;
              while ( 1 )
              {
                v124 += 2;
                LODWORD(v178) = v120;
                sub_34C0EE0((__int64)&v154, (unsigned int *)&v178);
                if ( !*(_WORD *)(v124 - 2) )
                  break;
                v122 += *(__int16 *)(v124 - 2);
                v120 = (unsigned __int16)v122;
              }
              v8 = v123;
              m = v121;
              v90 = v132;
              v74 = v134;
            }
          }
          sub_34C53D0((unsigned int)v170, v8, (__int64)&v162, v93, v94, v95);
        }
        else
        {
          sub_34C53D0(v92, v8, (__int64)&v154, v19, v20, v21);
        }
      }
    }
  }
  v131 = (__int64 *)v74;
LABEL_40:
  if ( v131 == (__int64 *)(a2 + 48) )
    goto LABEL_28;
  v2 = 0;
  v170 = v172;
  v171 = 0x400000000LL;
  v179 = 0x400000000LL;
  v175 = &v173;
  v176 = &v173;
  v183 = &v181;
  v184 = &v181;
  v173 = 0;
  v174 = 0;
  v177 = 0;
  v178 = v180;
  v181 = 0;
  v182 = 0;
  v185 = 0;
  v26 = *(_QWORD *)(v141 + 56);
  v27 = *(_QWORD *)(v142 + 56);
  v136 = v141 + 48;
  v133 = v142 + 48;
  if ( v26 != v141 + 48 )
  {
    while ( 2 )
    {
      if ( v133 == v27 )
        goto LABEL_50;
      for ( ; v136 != v26; v26 = *(_QWORD *)(v26 + 8) )
      {
        if ( (unsigned __int16)(*(_WORD *)(v26 + 68) - 14) > 4u )
          break;
        if ( (*(_BYTE *)v26 & 4) == 0 )
        {
          while ( (*(_BYTE *)(v26 + 44) & 8) != 0 )
            v26 = *(_QWORD *)(v26 + 8);
        }
      }
      while ( (unsigned __int16)(*(_WORD *)(v27 + 68) - 14) <= 4u )
      {
        if ( (*(_BYTE *)v27 & 4) == 0 )
        {
          while ( (*(_BYTE *)(v27 + 44) & 8) != 0 )
            v27 = *(_QWORD *)(v27 + 8);
        }
        v27 = *(_QWORD *)(v27 + 8);
        if ( v133 == v27 )
          goto LABEL_50;
      }
      if ( v133 == v27
        || v136 == v26
        || !sub_2E88AF0(v26, v27, 1u)
        || (v42 = *(_QWORD *)(a1 + 136), v43 = *(__int64 (**)())(*(_QWORD *)v42 + 920LL), v43 != sub_2DB1B30)
        && ((unsigned __int8 (__fastcall *)(__int64, __int64))v43)(v42, v26) )
      {
LABEL_50:
        v137 = v27;
        if ( (_BYTE)v2 )
          goto LABEL_51;
        goto LABEL_59;
      }
      v44 = *(_QWORD *)(v26 + 32);
      v45 = v44 + 40LL * (*(_DWORD *)(v26 + 40) & 0xFFFFFF);
      if ( v44 != v45 )
      {
        while ( 2 )
        {
          if ( *(_BYTE *)v44 == 12 )
            goto LABEL_50;
          if ( *(_BYTE *)v44 )
            goto LABEL_95;
          v46 = *(_DWORD *)(v44 + 8);
          if ( !v46 )
            goto LABEL_95;
          v47 = *(_BYTE *)(v44 + 3);
          if ( (v47 & 0x10) != 0 )
          {
            if ( v161 )
            {
              v71 = v158;
              if ( v158 )
              {
                v72 = &v157;
                do
                {
                  v41 = *(_QWORD **)(v71 + 24);
                  if ( v46 > *(_DWORD *)(v71 + 32) )
                  {
                    v71 = *(_QWORD *)(v71 + 24);
                  }
                  else
                  {
                    v72 = (int *)v71;
                    v71 = *(_QWORD *)(v71 + 16);
                  }
                }
                while ( v71 );
                if ( v72 != &v157 && v46 >= v72[8] )
                  goto LABEL_50;
              }
            }
            else
            {
              v48 = v154;
              v41 = &v154[4 * (unsigned int)v155];
              if ( v154 != (_BYTE *)v41 )
              {
                while ( v46 != *v48 )
                {
                  if ( v41 == (_QWORD *)++v48 )
                    goto LABEL_88;
                }
                if ( v41 != (_QWORD *)v48 )
                  goto LABEL_50;
              }
            }
LABEL_88:
            if ( v169 )
            {
              v40 = v166;
              if ( v166 )
              {
                v81 = &v165;
                do
                {
                  v41 = *(_QWORD **)(v40 + 24);
                  if ( v46 > *(_DWORD *)(v40 + 32) )
                  {
                    v40 = *(_QWORD *)(v40 + 24);
                  }
                  else
                  {
                    v81 = (int *)v40;
                    v40 = *(_QWORD *)(v40 + 16);
                  }
                }
                while ( v40 );
                if ( v81 != &v165 && v46 >= v81[8] )
                {
LABEL_94:
                  if ( (((v47 & 0x10) != 0) & (v47 >> 6)) == 0 )
                    goto LABEL_50;
                }
              }
            }
            else
            {
              v40 = (__int64)v162;
              v41 = &v162[4 * (unsigned int)v163];
              if ( v162 != (_BYTE *)v41 )
              {
                while ( v46 != *(_DWORD *)v40 )
                {
                  v40 += 4;
                  if ( v41 == (_QWORD *)v40 )
                    goto LABEL_95;
                }
                if ( v41 != (_QWORD *)v40 )
                  goto LABEL_94;
              }
            }
LABEL_95:
            v44 += 40;
            if ( v45 == v44 )
              goto LABEL_96;
            continue;
          }
          break;
        }
        if ( v177 )
        {
          v40 = v174;
          if ( v174 )
          {
            v70 = &v173;
            do
            {
              v41 = *(_QWORD **)(v40 + 24);
              if ( v46 > *(_DWORD *)(v40 + 32) )
              {
                v40 = *(_QWORD *)(v40 + 24);
              }
              else
              {
                v70 = (int *)v40;
                v40 = *(_QWORD *)(v40 + 16);
              }
            }
            while ( v40 );
            if ( v70 != &v173 && v46 >= v70[8] )
              goto LABEL_95;
          }
        }
        else
        {
          v40 = (__int64)v170;
          v41 = &v170[4 * (unsigned int)v171];
          if ( v170 != (_BYTE *)v41 )
          {
            while ( v46 != *(_DWORD *)v40 )
            {
              v40 += 4;
              if ( v41 == (_QWORD *)v40 )
                goto LABEL_150;
            }
            if ( (_QWORD *)v40 != v41 )
              goto LABEL_95;
          }
        }
LABEL_150:
        if ( v169 )
        {
          v40 = v166;
          if ( v166 )
          {
            v82 = &v165;
            do
            {
              v41 = *(_QWORD **)(v40 + 24);
              if ( v46 > *(_DWORD *)(v40 + 32) )
              {
                v40 = *(_QWORD *)(v40 + 24);
              }
              else
              {
                v82 = (int *)v40;
                v40 = *(_QWORD *)(v40 + 16);
              }
            }
            while ( v40 );
            if ( v82 != &v165 && v46 >= v82[8] )
              goto LABEL_50;
          }
        }
        else
        {
          v40 = (__int64)v162;
          v41 = &v162[4 * (unsigned int)v163];
          if ( v162 != (_BYTE *)v41 )
          {
            while ( v46 != *(_DWORD *)v40 )
            {
              v40 += 4;
              if ( v41 == (_QWORD *)v40 )
                goto LABEL_156;
            }
            if ( v41 != (_QWORD *)v40 )
              goto LABEL_50;
          }
        }
LABEL_156:
        if ( (*(_BYTE *)(v44 + 3) & 0x40) == 0 )
          goto LABEL_95;
        if ( v161 )
        {
          v40 = v158;
          if ( !v158 )
            goto LABEL_95;
          v111 = &v157;
          do
          {
            v41 = *(_QWORD **)(v40 + 16);
            if ( v46 > *(_DWORD *)(v40 + 32) )
            {
              v40 = *(_QWORD *)(v40 + 24);
            }
            else
            {
              v111 = (int *)v40;
              v40 = *(_QWORD *)(v40 + 16);
            }
          }
          while ( v40 );
          if ( v111 == &v157 || v46 < v111[8] )
            goto LABEL_95;
        }
        else
        {
          v40 = (__int64)v154;
          v69 = &v154[4 * (unsigned int)v155];
          if ( v154 == v69 )
            goto LABEL_95;
          while ( v46 != *(_DWORD *)v40 )
          {
            v40 += 4;
            if ( v69 == (_BYTE *)v40 )
              goto LABEL_95;
          }
          if ( v69 == (_BYTE *)v40 )
            goto LABEL_95;
        }
        *(_BYTE *)(v44 + 3) &= ~0x40u;
        goto LABEL_95;
      }
LABEL_96:
      LOBYTE(v143[0]) = 1;
      v129 = sub_2E8B400(v26, (__int64)v143, v44, v40, v41);
      if ( !v129 )
        goto LABEL_50;
      v49 = *(_QWORD *)(v26 + 32);
      v50 = (_BYTE *)(v49 + 40LL * (*(_DWORD *)(v26 + 40) & 0xFFFFFF));
      v51 = (_BYTE *)(v49 + 40LL * (unsigned int)sub_2E88FE0(v26));
      if ( v50 != v51 )
      {
        while ( 1 )
        {
          v52 = v51;
          if ( (unsigned __int8)sub_2E2FA70(v51) )
            break;
          v51 += 40;
          if ( v50 == v51 )
            goto LABEL_119;
        }
        if ( v51 != v50 )
        {
          v127 = v27;
          v126 = v26;
          while ( 1 )
          {
            if ( (((v52[3] & 0x40) != 0) & ((v52[3] >> 4) ^ 1)) == 0 )
              goto LABEL_113;
            v53 = *((_DWORD *)v52 + 2);
            LODWORD(v144) = v53;
            if ( !v53 )
              goto LABEL_113;
            if ( v185 )
            {
              v96 = v182;
              if ( !v182 )
                goto LABEL_113;
              v97 = &v181;
              do
              {
                if ( v53 > *(_DWORD *)(v96 + 32) )
                {
                  v96 = *(_QWORD *)(v96 + 24);
                }
                else
                {
                  v97 = (int *)v96;
                  v96 = *(_QWORD *)(v96 + 16);
                }
              }
              while ( v96 );
              if ( v97 == &v181 || v53 < v97[8] )
                goto LABEL_113;
            }
            else
            {
              v54 = v178;
              v55 = &v178[4 * (unsigned int)v179];
              if ( v178 == v55 )
                goto LABEL_113;
              while ( v53 != *v54 )
              {
                if ( v55 == (_BYTE *)++v54 )
                  goto LABEL_113;
              }
              if ( v55 == (_BYTE *)v54 )
                goto LABEL_113;
            }
            if ( v53 - 1 <= 0x3FFFFFFE )
            {
              v112 = sub_E922F0(*(_QWORD **)(a1 + 152), v53);
              v114 = &v112[2 * v113];
              if ( v112 != v114 )
              {
                v125 = v52;
                v115 = v112;
                v116 = (unsigned __int16 *)v114;
                do
                {
                  v117 = *(unsigned __int16 *)v115;
                  v115 += 2;
                  LODWORD(v147) = v117;
                  sub_34C0EE0((__int64)&v170, (unsigned int *)&v147);
                }
                while ( v116 != (unsigned __int16 *)v115 );
                v52 = v125;
              }
            }
            else
            {
              sub_34C0EE0((__int64)&v170, (unsigned int *)&v144);
            }
LABEL_113:
            v56 = v52 + 40;
            if ( v52 + 40 != v50 )
            {
              while ( 1 )
              {
                v52 = v56;
                if ( (unsigned __int8)sub_2E2FA70(v56) )
                  break;
                v56 += 40;
                if ( v50 == v56 )
                  goto LABEL_118;
              }
              if ( v50 != v56 )
                continue;
            }
LABEL_118:
            v27 = v127;
            v26 = v126;
            break;
          }
        }
      }
LABEL_119:
      v57 = *(_BYTE **)(v26 + 32);
      v58 = &v57[40 * (*(_DWORD *)(v26 + 40) & 0xFFFFFF)];
      if ( v57 != v58 )
      {
        v59 = *(_BYTE **)(v26 + 32);
        while ( 1 )
        {
          v60 = v59;
          if ( sub_2DADC00(v59) )
            break;
          v59 += 40;
          if ( v58 == v59 )
            goto LABEL_134;
        }
        if ( v59 != v58 )
        {
          v128 = v27;
          do
          {
            if ( (((v60[3] & 0x10) != 0) & (v60[3] >> 6)) == 0 )
            {
              v64 = *((_DWORD *)v60 + 2);
              if ( v64 > 0 )
              {
                sub_34C53D0(v64, *(_QWORD **)(a1 + 152), (__int64)&v170, v61, v62, v63);
                sub_34C53D0(v64, *(_QWORD **)(a1 + 152), (__int64)&v178, v65, v66, v67);
              }
            }
            if ( v60 + 40 == v58 )
              break;
            v68 = v60 + 40;
            while ( 1 )
            {
              v60 = v68;
              if ( sub_2DADC00(v68) )
                break;
              v68 += 40;
              if ( v58 == v68 )
                goto LABEL_133;
            }
          }
          while ( v58 != v68 );
LABEL_133:
          v27 = v128;
        }
      }
LABEL_134:
      if ( (*(_BYTE *)v26 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v26 + 44) & 8) != 0 )
          v26 = *(_QWORD *)(v26 + 8);
      }
      v26 = *(_QWORD *)(v26 + 8);
      if ( (*(_BYTE *)v27 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v27 + 44) & 8) != 0 )
          v27 = *(_QWORD *)(v27 + 8);
      }
      v27 = *(_QWORD *)(v27 + 8);
      if ( v136 != v26 )
      {
        v2 = v129;
        continue;
      }
      break;
    }
    v137 = v27;
LABEL_51:
    v28 = *(__int64 **)(v141 + 56);
    if ( v28 != (__int64 *)v26 && v131 != (__int64 *)v26 )
    {
      sub_2E310C0((__int64 *)(a2 + 40), (__int64 *)(v141 + 40), *(_QWORD *)(v141 + 56), v26);
      if ( (__int64 *)v26 != v28 )
      {
        v29 = *(_QWORD *)v26 & 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)((*v28 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v26;
        *(_QWORD *)v26 = *(_QWORD *)v26 & 7LL | *v28 & 0xFFFFFFFFFFFFFFF8LL;
        v30 = *v131;
        *(_QWORD *)(v29 + 8) = v131;
        v30 &= 0xFFFFFFFFFFFFFFF8LL;
        *v28 = v30 | *v28 & 7;
        *(_QWORD *)(v30 + 8) = v28;
        *v131 = v29 | *v131 & 7;
      }
    }
    v31 = *(_QWORD **)(v142 + 56);
    v32 = v142 + 40;
    while ( v31 != (_QWORD *)v137 )
    {
      v33 = v31;
      v31 = (_QWORD *)v31[1];
      sub_2E31080(v32, (__int64)v33);
      v34 = (unsigned __int64 *)v33[1];
      v35 = *v33 & 0xFFFFFFFFFFFFFFF8LL;
      *v34 = v35 | *v34 & 7;
      *(_QWORD *)(v35 + 8) = v34;
      *v33 &= 7uLL;
      v33[1] = 0;
      sub_2E310F0(v32);
    }
    LOBYTE(v2) = *(_BYTE *)(a1 + 131);
    if ( (_BYTE)v2 )
    {
      v138 = (_QWORD *)v141;
      v143[0] = v141;
      v143[1] = v142;
      do
      {
        v98 = v143;
        v99 = v138;
        v100 = 0;
        while ( 1 )
        {
          v147 = 0;
          v148 = v151;
          v149 = 0;
          v150 = 8;
          v152 = 0;
          v153 = 0;
          v144 = 0;
          v145 = 0;
          v146 = 0;
          sub_2E330F0(v99, &v144);
          sub_3509790(&v147, v99);
          sub_2E31EE0((__int64)v99, (__int64)v99, v101, v102, v103, v104);
          v105 = v99[23];
          v106 = v99[24];
          v107 = 0;
          if ( v145 - v144 == v106 - v105 )
          {
            if ( v144 == v145 )
            {
LABEL_274:
              v107 = v2;
            }
            else
            {
              v108 = v144;
              while ( *(_DWORD *)v108 == *(_DWORD *)v105
                   && *(_QWORD *)(v108 + 8) == *(_QWORD *)(v105 + 8)
                   && *(_QWORD *)(v108 + 16) == *(_QWORD *)(v105 + 16) )
              {
                v108 += 24LL;
                v105 += 24;
                if ( v145 == v108 )
                  goto LABEL_274;
              }
              v107 = 0;
            }
          }
          if ( v144 )
            j_j___libc_free_0(v144);
          if ( v152 )
            _libc_free(v152);
          if ( v148 != v151 )
            _libc_free((unsigned __int64)v148);
          if ( !v107 )
            v100 = v2;
          if ( ++v98 == &v144 )
            break;
          v99 = (_QWORD *)*v98;
        }
      }
      while ( v100 );
      v2 = (unsigned __int8)v2;
    }
    else
    {
      v2 = v130;
    }
LABEL_59:
    v36 = v182;
    while ( v36 )
    {
      sub_34BE7F0(*(_QWORD *)(v36 + 24));
      v37 = v36;
      v36 = *(_QWORD *)(v36 + 16);
      j_j___libc_free_0(v37);
    }
    if ( v178 != v180 )
      _libc_free((unsigned __int64)v178);
  }
  v38 = v174;
  if ( v174 )
  {
    do
    {
      sub_34BE7F0(*(_QWORD *)(v38 + 24));
      v39 = v38;
      v38 = *(_QWORD *)(v38 + 16);
      j_j___libc_free_0(v39);
    }
    while ( v38 );
  }
  if ( v170 != v172 )
    _libc_free((unsigned __int64)v170);
LABEL_29:
  v22 = v166;
  if ( v166 )
  {
    do
    {
      sub_34BE7F0(*(_QWORD *)(v22 + 24));
      v23 = v22;
      v22 = *(_QWORD *)(v22 + 16);
      j_j___libc_free_0(v23);
    }
    while ( v22 );
  }
  if ( v162 != v164 )
    _libc_free((unsigned __int64)v162);
  v24 = v158;
  if ( v158 )
  {
    do
    {
      sub_34BE7F0(*(_QWORD *)(v24 + 24));
      v25 = v24;
      v24 = *(_QWORD *)(v24 + 16);
      j_j___libc_free_0(v25);
    }
    while ( v24 );
  }
  if ( v154 != v156 )
    _libc_free((unsigned __int64)v154);
LABEL_25:
  if ( v186 != v188 )
    _libc_free((unsigned __int64)v186);
  return v2;
}
