// Function: sub_DB3670
// Address: 0xdb3670
//
__int64 __fastcall sub_DB3670(__int64 a1, __int64 a2, char a3)
{
  __int64 v4; // r9
  __int64 v5; // r8
  int v6; // edi
  __int64 v7; // rsi
  unsigned int v8; // ecx
  __int64 *v9; // rax
  __int64 v10; // r10
  unsigned int v11; // eax
  __int64 v12; // rbx
  int v13; // r12d
  __int64 v14; // r14
  __int64 v15; // r12
  char v16; // cl
  __int64 v17; // rdx
  _BYTE *v18; // r13
  int v19; // r10d
  unsigned int v20; // ecx
  _QWORD *v21; // rdx
  _BYTE *v22; // rax
  __int64 *v23; // rax
  __int64 v24; // r9
  __int64 v25; // rdx
  int v26; // r13d
  __int64 v27; // rax
  unsigned int v28; // edx
  int v29; // ebx
  __int64 v30; // r12
  int v31; // r15d
  int v32; // eax
  unsigned int v33; // esi
  __int64 v34; // r13
  __int64 v35; // rdx
  unsigned int v36; // r11d
  __int64 v37; // rbx
  __int64 v38; // r10
  int v39; // r11d
  unsigned int v40; // r15d
  _BYTE *v41; // r12
  unsigned int v42; // r8d
  unsigned int v43; // r14d
  unsigned int v44; // edi
  __int64 v45; // rax
  _BYTE *v46; // rcx
  unsigned int v47; // eax
  int v48; // r12d
  _QWORD *v49; // rcx
  unsigned int v50; // r8d
  _QWORD *v51; // rax
  __int64 v52; // rdi
  unsigned int *v53; // rax
  __int64 v54; // rsi
  __int64 v56; // rsi
  int v57; // ecx
  __int64 v58; // rdi
  int v59; // ecx
  unsigned int v60; // edx
  __int64 *v61; // rax
  __int64 v62; // r8
  __int64 v63; // r14
  __int64 v64; // r13
  __int64 v65; // r15
  __int64 v66; // rax
  __int64 v67; // r12
  __int64 v68; // rsi
  _QWORD *v69; // rax
  _QWORD *v70; // rcx
  int v71; // r12d
  _BYTE **v72; // rdx
  __int64 v73; // rcx
  _BYTE **v74; // rax
  __int64 v75; // rcx
  bool v76; // cf
  __int64 v77; // rax
  unsigned __int64 v78; // rdx
  __int64 v79; // r13
  __int64 v80; // r15
  int v81; // r10d
  __int64 *v82; // rdi
  unsigned int v83; // ecx
  __int64 *v84; // rdx
  _BYTE *v85; // rax
  _BYTE *v86; // r12
  unsigned int v87; // eax
  int v88; // edx
  __int64 v89; // rax
  unsigned __int64 v90; // rdx
  int v91; // edx
  __int64 v92; // rax
  unsigned __int64 v93; // rdx
  _BYTE *v94; // r9
  __int64 v95; // r9
  int v96; // eax
  unsigned int v97; // r14d
  _BYTE *v98; // rcx
  int v99; // r8d
  __int64 v100; // rdi
  int v101; // r8d
  unsigned int v102; // r14d
  _BYTE *v103; // rcx
  __int64 v104; // rax
  unsigned __int64 v105; // rdx
  __int64 v106; // r11
  __int64 v107; // r14
  __int64 v108; // r12
  unsigned int v109; // r15d
  __int64 v110; // r10
  unsigned int v111; // edi
  __int64 v112; // rax
  _BYTE *v113; // rcx
  _BYTE *v114; // rbx
  unsigned int v115; // ecx
  int v116; // eax
  __int64 v117; // rdx
  _BYTE *v118; // rdi
  __int64 v119; // rdi
  int v120; // r9d
  unsigned int v121; // r13d
  _BYTE *v122; // rsi
  int v123; // eax
  unsigned int v124; // eax
  __int64 v125; // rcx
  int v126; // r10d
  __int64 v127; // rdi
  int v128; // edi
  unsigned int v129; // r15d
  __int64 v130; // rcx
  __int64 v131; // rax
  unsigned int v132; // edx
  __int64 v133; // rbx
  int v134; // r8d
  _QWORD *v135; // rdi
  _QWORD *v136; // rsi
  int v137; // edi
  unsigned int v138; // ebx
  __int64 v139; // r8
  unsigned int v140; // r14d
  __int64 v141; // rcx
  int v142; // eax
  int v143; // eax
  int v144; // r9d
  __int64 v145; // r8
  int v146; // r13d
  __int64 v147; // r8
  __int64 *v148; // rcx
  unsigned int v149; // r11d
  __int64 v150; // [rsp+10h] [rbp-240h]
  __int64 v151; // [rsp+28h] [rbp-228h]
  int i; // [rsp+30h] [rbp-220h]
  __int64 v153; // [rsp+30h] [rbp-220h]
  __int64 v154; // [rsp+30h] [rbp-220h]
  unsigned int v155; // [rsp+38h] [rbp-218h]
  int v156; // [rsp+38h] [rbp-218h]
  int v157; // [rsp+38h] [rbp-218h]
  int v158; // [rsp+38h] [rbp-218h]
  __int64 v159; // [rsp+38h] [rbp-218h]
  __int64 v160; // [rsp+38h] [rbp-218h]
  unsigned int v161; // [rsp+38h] [rbp-218h]
  unsigned int v162; // [rsp+38h] [rbp-218h]
  __int64 v163; // [rsp+40h] [rbp-210h]
  int v164; // [rsp+40h] [rbp-210h]
  __int64 v165; // [rsp+40h] [rbp-210h]
  unsigned int v167; // [rsp+4Ch] [rbp-204h]
  int v169; // [rsp+50h] [rbp-200h]
  int v170; // [rsp+58h] [rbp-1F8h]
  __int64 v171; // [rsp+58h] [rbp-1F8h]
  __int64 v172; // [rsp+68h] [rbp-1E8h] BYREF
  _BYTE *v173; // [rsp+70h] [rbp-1E0h] BYREF
  __int64 v174; // [rsp+78h] [rbp-1D8h]
  _BYTE v175[16]; // [rsp+80h] [rbp-1D0h] BYREF
  __int64 v176; // [rsp+90h] [rbp-1C0h] BYREF
  __int64 v177; // [rsp+98h] [rbp-1B8h]
  __int64 v178; // [rsp+A0h] [rbp-1B0h]
  unsigned int v179; // [rsp+A8h] [rbp-1A8h]
  __int64 v180; // [rsp+B0h] [rbp-1A0h] BYREF
  __int64 v181; // [rsp+B8h] [rbp-198h]
  __int64 v182; // [rsp+C0h] [rbp-190h]
  __int64 v183; // [rsp+C8h] [rbp-188h]
  _QWORD *v184; // [rsp+D0h] [rbp-180h] BYREF
  __int64 v185; // [rsp+D8h] [rbp-178h]
  _QWORD v186[4]; // [rsp+E0h] [rbp-170h] BYREF
  _BYTE *v187; // [rsp+100h] [rbp-150h] BYREF
  __int64 v188; // [rsp+108h] [rbp-148h]
  _BYTE v189[128]; // [rsp+110h] [rbp-140h] BYREF
  _BYTE *v190; // [rsp+190h] [rbp-C0h] BYREF
  __int64 v191; // [rsp+198h] [rbp-B8h]
  _BYTE v192[176]; // [rsp+1A0h] [rbp-B0h] BYREF

  v187 = v189;
  v188 = 0x1000000000LL;
  v191 = 0x1000000000LL;
  v173 = v175;
  v184 = v186;
  v174 = 0x200000000LL;
  v185 = 0x400000001LL;
  v190 = v192;
  v176 = 0;
  v177 = 0;
  v178 = 0;
  v179 = 0;
  v186[0] = a1;
  v180 = 1;
  v181 = 0;
  v182 = 0;
  v183 = 0;
  sub_CF4090((__int64)&v180, 0);
  if ( !(_DWORD)v183 )
  {
LABEL_324:
    LODWORD(v182) = v182 + 1;
    BUG();
  }
  v5 = v181;
  v6 = 1;
  v7 = 0;
  v8 = (v183 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v9 = (__int64 *)(v181 + 8LL * v8);
  v10 = *v9;
  if ( a1 != *v9 )
  {
    while ( v10 != -4096 )
    {
      if ( !v7 && v10 == -8192 )
        v7 = (__int64)v9;
      v4 = (unsigned int)(v6 + 1);
      v8 = (v183 - 1) & (v6 + v8);
      v9 = (__int64 *)(v181 + 8LL * v8);
      v10 = *v9;
      if ( a1 == *v9 )
        goto LABEL_3;
      ++v6;
    }
    if ( v7 )
      v9 = (__int64 *)v7;
  }
LABEL_3:
  LODWORD(v182) = v182 + 1;
  if ( *v9 != -4096 )
    --HIDWORD(v182);
  *v9 = a1;
  v11 = v185;
  if ( !(_DWORD)v185 )
    goto LABEL_23;
  while ( 2 )
  {
    v12 = v184[v11 - 1];
    LODWORD(v185) = v11 - 1;
    v172 = v12;
    switch ( *(_BYTE *)v12 )
    {
      case '*':
      case ',':
      case '.':
      case '0':
      case '3':
      case '6':
      case '8':
      case '9':
      case ':':
      case ';':
      case '?':
      case 'C':
      case 'D':
      case 'E':
      case 'N':
      case 'V':
        goto LABEL_8;
      case 'L':
      case 'M':
        if ( !a3 )
          goto LABEL_21;
LABEL_8:
        v13 = *(_DWORD *)(v12 + 4) & 0x7FFFFFF;
        if ( !v13 )
          goto LABEL_148;
        v14 = 32LL * (*(_DWORD *)(v12 + 4) & 0x7FFFFFF);
        v15 = 0;
        v16 = 0;
        do
        {
          if ( (*(_BYTE *)(v12 + 7) & 0x40) != 0 )
            v17 = *(_QWORD *)(v12 - 8);
          else
            v17 = v12 - 32LL * (*(_DWORD *)(v12 + 4) & 0x7FFFFFF);
          v18 = *(_BYTE **)(v17 + v15);
          if ( *v18 > 0x1Cu )
          {
            v7 = (unsigned int)v183;
            if ( (_DWORD)v183 )
            {
              v5 = (unsigned int)(v183 - 1);
              v19 = 1;
              v4 = 0;
              v20 = v5 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
              v21 = (_QWORD *)(v181 + 8LL * v20);
              v22 = (_BYTE *)*v21;
              if ( v18 == (_BYTE *)*v21 )
              {
LABEL_14:
                v16 = 1;
                goto LABEL_15;
              }
              while ( v22 != (_BYTE *)-4096LL )
              {
                if ( v4 || v22 != (_BYTE *)-8192LL )
                  v21 = (_QWORD *)v4;
                v4 = (unsigned int)(v19 + 1);
                v20 = v5 & (v19 + v20);
                v22 = *(_BYTE **)(v181 + 8LL * v20);
                if ( v22 == v18 )
                  goto LABEL_14;
                ++v19;
                v4 = (__int64)v21;
                v21 = (_QWORD *)(v181 + 8LL * v20);
              }
              if ( !v4 )
                v4 = (__int64)v21;
              ++v180;
              v91 = v182 + 1;
              if ( 4 * ((int)v182 + 1) < (unsigned int)(3 * v183) )
              {
                if ( (int)v183 - HIDWORD(v182) - v91 <= (unsigned int)v183 >> 3 )
                {
                  sub_CF4090((__int64)&v180, v183);
                  if ( !(_DWORD)v183 )
                    goto LABEL_324;
                  v7 = (unsigned int)(v183 - 1);
                  v5 = v181;
                  v128 = 1;
                  v129 = v7 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
                  v4 = v181 + 8LL * v129;
                  v130 = *(_QWORD *)v4;
                  v91 = v182 + 1;
                  v131 = 0;
                  if ( v18 != *(_BYTE **)v4 )
                  {
                    while ( v130 != -4096 )
                    {
                      if ( v130 == -8192 && !v131 )
                        v131 = v4;
                      v129 = v7 & (v128 + v129);
                      v4 = v181 + 8LL * v129;
                      v130 = *(_QWORD *)v4;
                      if ( *(_BYTE **)v4 == v18 )
                        goto LABEL_105;
                      ++v128;
                    }
                    if ( v131 )
                      v4 = v131;
                  }
                }
                goto LABEL_105;
              }
            }
            else
            {
              ++v180;
            }
            sub_CF4090((__int64)&v180, 2 * v183);
            if ( !(_DWORD)v183 )
              goto LABEL_324;
            v7 = (unsigned int)(v183 - 1);
            v5 = v181;
            v124 = v7 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
            v4 = v181 + 8LL * v124;
            v91 = v182 + 1;
            v125 = *(_QWORD *)v4;
            if ( v18 != *(_BYTE **)v4 )
            {
              v126 = 1;
              v127 = 0;
              while ( v125 != -4096 )
              {
                if ( !v127 && v125 == -8192 )
                  v127 = v4;
                v124 = v7 & (v126 + v124);
                v4 = v181 + 8LL * v124;
                v125 = *(_QWORD *)v4;
                if ( *(_BYTE **)v4 == v18 )
                  goto LABEL_105;
                ++v126;
              }
              if ( v127 )
                v4 = v127;
            }
LABEL_105:
            LODWORD(v182) = v91;
            if ( *(_QWORD *)v4 != -4096 )
              --HIDWORD(v182);
            *(_QWORD *)v4 = v18;
            v92 = (unsigned int)v185;
            v93 = (unsigned int)v185 + 1LL;
            if ( v93 > HIDWORD(v185) )
            {
              v7 = (__int64)v186;
              sub_C8D5F0((__int64)&v184, v186, v93, 8u, v5, v4);
              v92 = (unsigned int)v185;
            }
            v184[v92] = v18;
            v12 = v172;
            LODWORD(v185) = v185 + 1;
            goto LABEL_14;
          }
LABEL_15:
          v15 += 32;
        }
        while ( v14 != v15 );
        if ( v16 )
        {
          v104 = (unsigned int)v191;
          v105 = (unsigned int)v191 + 1LL;
          if ( v105 > HIDWORD(v191) )
          {
            v7 = (__int64)v192;
            sub_C8D5F0((__int64)&v190, v192, v105, 8u, v5, v4);
            v104 = (unsigned int)v191;
          }
          *(_QWORD *)&v190[8 * v104] = v12;
          LODWORD(v191) = v191 + 1;
        }
        else
        {
          v13 = *(_DWORD *)(v12 + 4) & 0x7FFFFFF;
LABEL_148:
          v7 = (__int64)&v172;
          *(_DWORD *)sub_DA65E0((__int64)&v176, &v172) = v13;
        }
LABEL_22:
        v11 = v185;
        if ( (_DWORD)v185 )
          continue;
LABEL_23:
        v26 = v191;
        if ( (_DWORD)v191 )
        {
          v27 = (unsigned int)v188;
          v28 = v191;
          v29 = 0;
          while ( 1 )
          {
            v7 = v28;
            v30 = *(_QWORD *)&v190[8 * v28 - 8];
            LODWORD(v191) = v28 - 1;
            if ( v27 + 1 > (unsigned __int64)HIDWORD(v188) )
            {
              v7 = (__int64)v189;
              sub_C8D5F0((__int64)&v187, v189, v27 + 1, 8u, v5, v4);
              v27 = (unsigned int)v188;
            }
            ++v29;
            *(_QWORD *)&v187[8 * v27] = v30;
            v27 = (unsigned int)(v188 + 1);
            LODWORD(v188) = v188 + 1;
            if ( v26 == v29 )
              break;
            v28 = v191;
          }
        }
        if ( v184 != v186 )
          _libc_free(v184, v7);
        if ( v190 != v192 )
          _libc_free(v190, v7);
        sub_C7D6A0(v181, 8LL * (unsigned int)v183, 8);
        v167 = 1;
        v170 = 0;
        v31 = 0;
        v169 = v188;
        while ( 2 )
        {
          if ( v31 == v169 )
          {
            while ( v31 != v170 )
            {
              v32 = v170;
              v31 = 0;
              v170 = 0;
              v169 = v32;
              if ( v32 )
                goto LABEL_37;
            }
            if ( dword_4F88428 < v167 || !(_DWORD)v174 )
              break;
            v106 = 0;
            v171 = 8LL * (unsigned int)v174;
            while ( 1 )
            {
              v107 = *(_QWORD *)&v173[v106];
              if ( (*(_DWORD *)(v107 + 4) & 0x7FFFFFF) != 0 )
                break;
LABEL_169:
              v106 += 8;
              if ( v171 == v106 )
                goto LABEL_54;
            }
            v108 = 0;
            v109 = 0;
            v110 = 32LL * (*(_DWORD *)(v107 + 4) & 0x7FFFFFF);
LABEL_157:
            while ( 2 )
            {
              v114 = *(_BYTE **)(*(_QWORD *)(v107 - 8) + v108);
              if ( !v114 )
                goto LABEL_126;
              if ( *v114 <= 0x1Cu )
                goto LABEL_156;
              if ( !v179 )
              {
                ++v176;
                goto LABEL_161;
              }
              v111 = (v179 - 1) & (((unsigned int)v114 >> 9) ^ ((unsigned int)v114 >> 4));
              v112 = v177 + 16LL * v111;
              v113 = *(_BYTE **)v112;
              if ( v114 == *(_BYTE **)v112 )
                goto LABEL_155;
              v164 = 1;
              v117 = 0;
              while ( v113 != (_BYTE *)-4096LL )
              {
                if ( v117 || v113 != (_BYTE *)-8192LL )
                  v112 = v117;
                v111 = (v179 - 1) & (v164 + v111);
                v113 = *(_BYTE **)(v177 + 16LL * v111);
                if ( v114 == v113 )
                {
                  v112 = v177 + 16LL * v111;
LABEL_155:
                  v109 += *(_DWORD *)(v112 + 8);
LABEL_156:
                  v108 += 32;
                  if ( v110 == v108 )
                  {
LABEL_166:
                    if ( v167 >= v109 )
                      v109 = v167;
                    v167 = v109;
                    goto LABEL_169;
                  }
                  goto LABEL_157;
                }
                ++v164;
                v117 = v112;
                v112 = v177 + 16LL * v111;
              }
              if ( !v117 )
                v117 = v112;
              ++v176;
              v116 = v178 + 1;
              if ( 4 * ((int)v178 + 1) >= 3 * v179 )
              {
LABEL_161:
                v159 = v110;
                v163 = v106;
                sub_9BAAD0((__int64)&v176, 2 * v179);
                if ( !v179 )
                  goto LABEL_325;
                v106 = v163;
                v110 = v159;
                v115 = (v179 - 1) & (((unsigned int)v114 >> 9) ^ ((unsigned int)v114 >> 4));
                v116 = v178 + 1;
                v117 = v177 + 16LL * v115;
                v118 = *(_BYTE **)v117;
                if ( v114 != *(_BYTE **)v117 )
                {
                  v146 = 1;
                  v147 = 0;
                  while ( v118 != (_BYTE *)-4096LL )
                  {
                    if ( v118 == (_BYTE *)-8192LL && !v147 )
                      v147 = v117;
                    v115 = (v179 - 1) & (v146 + v115);
                    v117 = v177 + 16LL * v115;
                    v118 = *(_BYTE **)v117;
                    if ( v114 == *(_BYTE **)v117 )
                      goto LABEL_163;
                    ++v146;
                  }
                  if ( v147 )
                    v117 = v147;
                }
              }
              else if ( v179 - HIDWORD(v178) - v116 <= v179 >> 3 )
              {
                v160 = v110;
                v165 = v106;
                sub_9BAAD0((__int64)&v176, v179);
                if ( !v179 )
                  goto LABEL_325;
                v119 = 0;
                v120 = 1;
                v121 = (v179 - 1) & (((unsigned int)v114 >> 9) ^ ((unsigned int)v114 >> 4));
                v106 = v165;
                v116 = v178 + 1;
                v110 = v160;
                v117 = v177 + 16LL * v121;
                v122 = *(_BYTE **)v117;
                if ( *(_BYTE **)v117 != v114 )
                {
                  while ( v122 != (_BYTE *)-4096LL )
                  {
                    if ( !v119 && v122 == (_BYTE *)-8192LL )
                      v119 = v117;
                    v121 = (v179 - 1) & (v120 + v121);
                    v117 = v177 + 16LL * v121;
                    v122 = *(_BYTE **)v117;
                    if ( v114 == *(_BYTE **)v117 )
                      goto LABEL_163;
                    ++v120;
                  }
                  if ( v119 )
                    v117 = v119;
                }
              }
LABEL_163:
              LODWORD(v178) = v116;
              if ( *(_QWORD *)v117 != -4096 )
                --HIDWORD(v178);
              v108 += 32;
              *(_QWORD *)v117 = v114;
              *(_DWORD *)(v117 + 8) = 0;
              if ( v110 == v108 )
                goto LABEL_166;
              continue;
            }
          }
LABEL_37:
          v33 = v179;
          v34 = *(_QWORD *)&v187[8 * v31];
          v35 = v177;
          v36 = *(_DWORD *)(v34 + 4) & 0x7FFFFFF;
          if ( !v36 )
            goto LABEL_49;
          v37 = 0;
          v38 = 32LL * v36;
          v39 = v31;
          v40 = 0;
          while ( 2 )
          {
            while ( 2 )
            {
              if ( (*(_BYTE *)(v34 + 7) & 0x40) != 0 )
              {
                v41 = *(_BYTE **)(*(_QWORD *)(v34 - 8) + v37);
                if ( *v41 <= 0x1Cu )
                  goto LABEL_47;
LABEL_40:
                if ( !v33 )
                  goto LABEL_110;
                v42 = v33 - 1;
                v43 = ((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4);
                v44 = (v33 - 1) & v43;
                v45 = v35 + 16LL * v44;
                v46 = *(_BYTE **)v45;
                if ( *(_BYTE **)v45 == v41 )
                {
                  if ( v45 == v35 + 16LL * v33 )
                    goto LABEL_110;
LABEL_43:
                  v40 += *(_DWORD *)(v45 + 8);
                  goto LABEL_44;
                }
                v155 = (v33 - 1) & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
                v94 = *(_BYTE **)v45;
                for ( i = 1; ; ++i )
                {
                  if ( v94 == (_BYTE *)-4096LL )
                    goto LABEL_110;
                  v155 = v42 & (i + v155);
                  v94 = *(_BYTE **)(v35 + 16LL * v155);
                  if ( v94 == v41 )
                    break;
                }
                v45 = v35 + 16LL * (v42 & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4)));
                if ( v35 + 16LL * v155 == v35 + 16LL * v33 )
                {
LABEL_110:
                  v31 = v39;
                  *(_QWORD *)&v187[8 * v170++] = v34;
                  goto LABEL_111;
                }
                if ( v46 == v41 )
                  goto LABEL_43;
                v156 = 1;
                v95 = 0;
                while ( v46 != (_BYTE *)-4096LL )
                {
                  if ( !v95 && v46 == (_BYTE *)-8192LL )
                    v95 = v45;
                  v44 = v42 & (v156 + v44);
                  v45 = v35 + 16LL * v44;
                  v46 = *(_BYTE **)v45;
                  if ( *(_BYTE **)v45 == v41 )
                    goto LABEL_43;
                  ++v156;
                }
                if ( !v95 )
                  v95 = v45;
                ++v176;
                v96 = v178 + 1;
                if ( 4 * ((int)v178 + 1) >= 3 * v33 )
                {
                  v153 = v38;
                  v157 = v39;
                  sub_9BAAD0((__int64)&v176, 2 * v33);
                  if ( !v179 )
                    goto LABEL_325;
                  v97 = (v179 - 1) & v43;
                  v39 = v157;
                  v38 = v153;
                  v96 = v178 + 1;
                  v95 = v177 + 16LL * v97;
                  v98 = *(_BYTE **)v95;
                  if ( v41 == *(_BYTE **)v95 )
                    goto LABEL_123;
                  v99 = 1;
                  v100 = 0;
                  while ( v98 != (_BYTE *)-4096LL )
                  {
                    if ( v98 == (_BYTE *)-8192LL && !v100 )
                      v100 = v95;
                    v97 = (v179 - 1) & (v99 + v97);
                    v95 = v177 + 16LL * v97;
                    v98 = *(_BYTE **)v95;
                    if ( *(_BYTE **)v95 == v41 )
                      goto LABEL_123;
                    ++v99;
                  }
                }
                else
                {
                  if ( v33 - (v96 + HIDWORD(v178)) > v33 >> 3 )
                    goto LABEL_123;
                  v154 = v38;
                  v158 = v39;
                  sub_9BAAD0((__int64)&v176, v33);
                  if ( !v179 )
                    goto LABEL_325;
                  v100 = 0;
                  v101 = 1;
                  v102 = (v179 - 1) & v43;
                  v39 = v158;
                  v96 = v178 + 1;
                  v38 = v154;
                  v95 = v177 + 16LL * v102;
                  v103 = *(_BYTE **)v95;
                  if ( *(_BYTE **)v95 == v41 )
                    goto LABEL_123;
                  for ( ; v103 != (_BYTE *)-4096LL; ++v101 )
                  {
                    if ( !v100 && v103 == (_BYTE *)-8192LL )
                      v100 = v95;
                    v102 = (v179 - 1) & (v101 + v102);
                    v95 = v177 + 16LL * v102;
                    v103 = *(_BYTE **)v95;
                    if ( *(_BYTE **)v95 == v41 )
                      goto LABEL_123;
                  }
                }
                if ( v100 )
                  v95 = v100;
LABEL_123:
                LODWORD(v178) = v96;
                if ( *(_QWORD *)v95 != -4096 )
                  --HIDWORD(v178);
                *(_QWORD *)v95 = v41;
                *(_DWORD *)(v95 + 8) = 0;
                v35 = v177;
                v33 = v179;
LABEL_44:
                v37 += 32;
                if ( v38 == v37 )
                  goto LABEL_48;
                continue;
              }
              break;
            }
            v41 = *(_BYTE **)(v34 - 32LL * (*(_DWORD *)(v34 + 4) & 0x7FFFFFF) + v37);
            if ( *v41 > 0x1Cu )
              goto LABEL_40;
LABEL_47:
            v37 += 32;
            ++v40;
            if ( v38 != v37 )
              continue;
            break;
          }
LABEL_48:
          v47 = v40;
          v31 = v39;
          v36 = v47;
LABEL_49:
          if ( !v33 )
          {
            ++v176;
            goto LABEL_214;
          }
          v48 = 1;
          v49 = 0;
          v50 = (v33 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
          v51 = (_QWORD *)(v35 + 16LL * v50);
          v52 = *v51;
          if ( v34 == *v51 )
            goto LABEL_51;
          while ( 2 )
          {
            if ( v52 == -4096 )
            {
              if ( !v49 )
                v49 = v51;
              ++v176;
              v123 = v178 + 1;
              if ( 4 * ((int)v178 + 1) < 3 * v33 )
              {
                if ( v33 - (v123 + HIDWORD(v178)) > v33 >> 3 )
                {
LABEL_189:
                  LODWORD(v178) = v123;
                  if ( *v49 != -4096 )
                    --HIDWORD(v178);
                  *v49 = v34;
                  v53 = (unsigned int *)(v49 + 1);
                  *((_DWORD *)v49 + 2) = 0;
                  goto LABEL_52;
                }
                v162 = v36;
                sub_9BAAD0((__int64)&v176, v33);
                if ( v179 )
                {
                  v136 = 0;
                  v137 = 1;
                  v138 = (v179 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
                  v36 = v162;
                  v123 = v178 + 1;
                  v49 = (_QWORD *)(v177 + 16LL * v138);
                  v139 = *v49;
                  if ( v34 != *v49 )
                  {
                    while ( v139 != -4096 )
                    {
                      if ( !v136 && v139 == -8192 )
                        v136 = v49;
                      v138 = (v179 - 1) & (v137 + v138);
                      v49 = (_QWORD *)(v177 + 16LL * v138);
                      v139 = *v49;
                      if ( v34 == *v49 )
                        goto LABEL_189;
                      ++v137;
                    }
                    if ( v136 )
                      v49 = v136;
                  }
                  goto LABEL_189;
                }
LABEL_325:
                LODWORD(v178) = v178 + 1;
                BUG();
              }
LABEL_214:
              v161 = v36;
              sub_9BAAD0((__int64)&v176, 2 * v33);
              if ( v179 )
              {
                v36 = v161;
                v132 = (v179 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
                v123 = v178 + 1;
                v49 = (_QWORD *)(v177 + 16LL * v132);
                v133 = *v49;
                if ( v34 != *v49 )
                {
                  v134 = 1;
                  v135 = 0;
                  while ( v133 != -4096 )
                  {
                    if ( !v135 && v133 == -8192 )
                      v135 = v49;
                    v132 = (v179 - 1) & (v134 + v132);
                    v49 = (_QWORD *)(v177 + 16LL * v132);
                    v133 = *v49;
                    if ( v34 == *v49 )
                      goto LABEL_189;
                    ++v134;
                  }
                  if ( v135 )
                    v49 = v135;
                }
                goto LABEL_189;
              }
              goto LABEL_325;
            }
            if ( v49 || v52 != -8192 )
              v51 = v49;
            v50 = (v33 - 1) & (v48 + v50);
            v52 = *(_QWORD *)(v35 + 16LL * v50);
            if ( v34 != v52 )
            {
              ++v48;
              v49 = v51;
              v51 = (_QWORD *)(v35 + 16LL * v50);
              continue;
            }
            break;
          }
          v51 = (_QWORD *)(v35 + 16LL * v50);
LABEL_51:
          v53 = (unsigned int *)(v51 + 1);
LABEL_52:
          *v53 = v36;
          if ( v36 <= v167 || (v167 = v36, dword_4F88428 >= v36) )
          {
LABEL_111:
            ++v31;
            continue;
          }
          break;
        }
LABEL_54:
        v54 = 16LL * v179;
        sub_C7D6A0(v177, v54, 8);
        if ( v173 != v175 )
          _libc_free(v173, v54);
        if ( v187 != v189 )
          _libc_free(v187, v54);
        return v167;
      case 'T':
        v56 = *(_QWORD *)(v12 + 40);
        v57 = *(_DWORD *)(a2 + 24);
        v58 = *(_QWORD *)(a2 + 8);
        if ( !v57 )
          goto LABEL_21;
        v59 = v57 - 1;
        v60 = v59 & (((unsigned int)v56 >> 9) ^ ((unsigned int)v56 >> 4));
        v61 = (__int64 *)(v58 + 16LL * v60);
        v62 = *v61;
        if ( v56 == *v61 )
          goto LABEL_61;
        v143 = 1;
        if ( v62 == -4096 )
          goto LABEL_21;
        while ( 1 )
        {
          v144 = v143 + 1;
          v60 = v59 & (v143 + v60);
          v61 = (__int64 *)(v58 + 16LL * v60);
          v145 = *v61;
          if ( v56 == *v61 )
            break;
          v143 = v144;
          if ( v145 == -4096 )
            goto LABEL_21;
        }
LABEL_61:
        v63 = v61[1];
        if ( !v63 || v56 != **(_QWORD **)(v63 + 32) || (*(_DWORD *)(v12 + 4) & 0x7FFFFFF) == 0 )
          goto LABEL_21;
        v64 = 0;
        v65 = 0;
        v24 = 0;
        v25 = 8LL * (*(_DWORD *)(v12 + 4) & 0x7FFFFFF);
        break;
      default:
        goto LABEL_21;
    }
    break;
  }
  while ( 1 )
  {
    v66 = *(_QWORD *)(v12 - 8);
    v67 = *(_QWORD *)(v66 + 4 * v64);
    v68 = *(_QWORD *)(32LL * *(unsigned int *)(v12 + 72) + v66 + v64);
    if ( *(_BYTE *)(v63 + 84) )
    {
      v69 = *(_QWORD **)(v63 + 64);
      v70 = &v69[*(unsigned int *)(v63 + 76)];
      if ( v69 != v70 )
      {
        while ( v68 != *v69 )
        {
          if ( v70 == ++v69 )
            goto LABEL_19;
        }
LABEL_70:
        if ( v65 )
        {
          if ( v67 != v65 )
            goto LABEL_21;
        }
        else
        {
          v65 = v67;
        }
        goto LABEL_72;
      }
    }
    else
    {
      v150 = v25;
      v151 = v24;
      v23 = sub_C8CA60(v63 + 56, v68);
      v24 = v151;
      v25 = v150;
      if ( v23 )
        goto LABEL_70;
    }
LABEL_19:
    if ( v24 )
    {
      if ( v67 != v24 )
        goto LABEL_21;
    }
    else
    {
      v24 = v67;
    }
LABEL_72:
    v64 += 8;
    if ( v25 == v64 )
    {
      if ( v65 && v24 )
      {
        v71 = *(_DWORD *)(v12 + 4) & 0x7FFFFFF;
        if ( v71 )
        {
          v72 = *(_BYTE ***)(v12 - 8);
          v73 = (unsigned int)(v71 - 1);
          v71 = 0;
          v74 = v72 + 4;
          v75 = (__int64)&v72[4 * v73 + 4];
          while ( 1 )
          {
            v76 = **v72 < 0x1Du;
            v72 = v74;
            v71 += v76;
            if ( (_BYTE **)v75 == v74 )
              break;
            v74 += 4;
          }
        }
        v7 = (__int64)&v172;
        *(_DWORD *)sub_DA65E0((__int64)&v176, &v172) = v71;
        v77 = (unsigned int)v174;
        v78 = (unsigned int)v174 + 1LL;
        if ( v78 > HIDWORD(v174) )
        {
          v7 = (__int64)v175;
          sub_C8D5F0((__int64)&v173, v175, v78, 8u, v5, v4);
          v77 = (unsigned int)v174;
        }
        v79 = 0;
        *(_QWORD *)&v173[8 * v77] = v12;
        LODWORD(v174) = v174 + 1;
        v80 = 32LL * (*(_DWORD *)(v12 + 4) & 0x7FFFFFF);
        if ( (*(_DWORD *)(v12 + 4) & 0x7FFFFFF) == 0 )
          goto LABEL_22;
        while ( 2 )
        {
          v86 = *(_BYTE **)(*(_QWORD *)(v12 - 8) + v79);
          if ( !v86 )
LABEL_126:
            BUG();
          if ( *v86 > 0x1Cu )
          {
            v7 = (unsigned int)v183;
            if ( !(_DWORD)v183 )
            {
              ++v180;
              goto LABEL_89;
            }
            v4 = (unsigned int)(v183 - 1);
            v81 = 1;
            v5 = v181;
            v82 = 0;
            v83 = v4 & (((unsigned int)v86 >> 9) ^ ((unsigned int)v86 >> 4));
            v84 = (__int64 *)(v181 + 8LL * v83);
            v85 = (_BYTE *)*v84;
            if ( v86 != (_BYTE *)*v84 )
            {
              while ( v85 != (_BYTE *)-4096LL )
              {
                if ( v85 != (_BYTE *)-8192LL || v82 )
                  v84 = v82;
                v83 = v4 & (v81 + v83);
                v85 = *(_BYTE **)(v181 + 8LL * v83);
                if ( v86 == v85 )
                  goto LABEL_84;
                ++v81;
                v82 = v84;
                v84 = (__int64 *)(v181 + 8LL * v83);
              }
              if ( !v82 )
                v82 = v84;
              ++v180;
              v88 = v182 + 1;
              if ( 4 * ((int)v182 + 1) >= (unsigned int)(3 * v183) )
              {
LABEL_89:
                sub_CF4090((__int64)&v180, 2 * v183);
                if ( !(_DWORD)v183 )
                  goto LABEL_324;
                v4 = (unsigned int)(v183 - 1);
                v7 = (unsigned int)v182;
                v87 = v4 & (((unsigned int)v86 >> 9) ^ ((unsigned int)v86 >> 4));
                v82 = (__int64 *)(v181 + 8LL * v87);
                v88 = v182 + 1;
                v5 = *v82;
                if ( v86 != (_BYTE *)*v82 )
                {
                  v7 = 1;
                  v148 = 0;
                  while ( v5 != -4096 )
                  {
                    if ( !v148 && v5 == -8192 )
                      v148 = v82;
                    v149 = v7 + 1;
                    v87 = v4 & (v7 + v87);
                    v7 = v87;
                    v82 = (__int64 *)(v181 + 8LL * v87);
                    v5 = *v82;
                    if ( v86 == (_BYTE *)*v82 )
                      goto LABEL_91;
                    v7 = v149;
                  }
                  if ( v148 )
                    v82 = v148;
                }
              }
              else if ( (int)v183 - HIDWORD(v182) - v88 <= (unsigned int)v183 >> 3 )
              {
                sub_CF4090((__int64)&v180, v183);
                if ( !(_DWORD)v183 )
                  goto LABEL_324;
                v5 = (unsigned int)(v183 - 1);
                v4 = v181;
                v7 = 0;
                v140 = v5 & (((unsigned int)v86 >> 9) ^ ((unsigned int)v86 >> 4));
                v82 = (__int64 *)(v181 + 8LL * v140);
                v141 = *v82;
                v88 = v182 + 1;
                v142 = 1;
                if ( v86 != (_BYTE *)*v82 )
                {
                  while ( v141 != -4096 )
                  {
                    if ( !v7 && v141 == -8192 )
                      v7 = (__int64)v82;
                    v140 = v5 & (v142 + v140);
                    v82 = (__int64 *)(v181 + 8LL * v140);
                    v141 = *v82;
                    if ( v86 == (_BYTE *)*v82 )
                      goto LABEL_91;
                    ++v142;
                  }
                  if ( v7 )
                    v82 = (__int64 *)v7;
                }
              }
LABEL_91:
              LODWORD(v182) = v88;
              if ( *v82 != -4096 )
                --HIDWORD(v182);
              *v82 = (__int64)v86;
              v89 = (unsigned int)v185;
              v90 = (unsigned int)v185 + 1LL;
              if ( v90 > HIDWORD(v185) )
              {
                v7 = (__int64)v186;
                sub_C8D5F0((__int64)&v184, v186, v90, 8u, v5, v4);
                v89 = (unsigned int)v185;
              }
              v184[v89] = v86;
              LODWORD(v185) = v185 + 1;
            }
          }
LABEL_84:
          v79 += 32;
          if ( v80 == v79 )
            goto LABEL_22;
          continue;
        }
      }
LABEL_21:
      v7 = (__int64)&v172;
      *(_DWORD *)sub_DA65E0((__int64)&v176, &v172) = 1;
      goto LABEL_22;
    }
  }
}
