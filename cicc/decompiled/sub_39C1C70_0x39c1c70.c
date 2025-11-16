// Function: sub_39C1C70
// Address: 0x39c1c70
//
void __fastcall sub_39C1C70(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v5; // rbx
  size_t v6; // r12
  char *v7; // r15
  __int16 v8; // ax
  __int64 v9; // r8
  __int64 v10; // r9
  _QWORD *v11; // rdx
  _QWORD *v12; // rsi
  unsigned __int64 v13; // rcx
  unsigned __int64 v14; // rax
  __int64 i; // rax
  __int64 v16; // rax
  __int64 (*v17)(); // rax
  int v18; // eax
  char *v19; // r11
  unsigned int v20; // ebx
  char *v21; // r13
  __int64 v22; // r12
  __int16 v23; // ax
  __int64 v24; // r14
  _QWORD *v25; // r10
  unsigned int v26; // esi
  __int64 v27; // rdx
  unsigned int v30; // eax
  int v31; // edx
  unsigned int v32; // eax
  unsigned int v33; // r8d
  __int64 v34; // rdx
  int v35; // ecx
  unsigned __int64 v36; // rdi
  unsigned __int64 v37; // rdi
  char *v40; // rdi
  char *v41; // r11
  char v42; // dl
  int v43; // esi
  __int64 v44; // rbx
  __int64 v45; // r12
  unsigned int v46; // ecx
  unsigned int v47; // edx
  __int16 v48; // r8
  _WORD *v49; // r9
  unsigned int v50; // esi
  unsigned __int16 *v51; // rcx
  unsigned __int16 v52; // r8
  unsigned __int16 *v53; // r10
  unsigned __int16 *v54; // r9
  unsigned __int16 *v55; // rsi
  unsigned __int64 v56; // rcx
  __int64 v57; // rdx
  int v58; // r10d
  int *v59; // rsi
  char *v60; // r9
  int *v61; // r8
  int v62; // ecx
  int v63; // edx
  __int64 v64; // rdx
  unsigned int v65; // ecx
  char *v66; // r9
  _DWORD *v67; // r8
  int v68; // ecx
  __int64 v69; // rsi
  _DWORD *v70; // r10
  unsigned __int64 v71; // rdx
  unsigned __int16 *v72; // rdx
  __int64 v73; // rdx
  unsigned __int16 v74; // r9
  _QWORD *v75; // rdi
  __int64 v76; // r15
  __int64 v77; // r14
  __int64 v78; // rax
  unsigned int v79; // edx
  int v80; // r9d
  __int64 v81; // rax
  unsigned int v82; // ecx
  _QWORD *v83; // rax
  __int64 *v84; // r8
  __int64 v85; // rsi
  __int64 v86; // rdx
  __int64 v87; // rax
  unsigned int v88; // ecx
  __int64 v89; // r8
  __int64 v90; // rax
  __int64 v91; // rdx
  char v92; // di
  __int64 v93; // rax
  __int64 *v94; // rax
  __int16 v95; // ax
  __int64 v96; // rax
  _QWORD *v97; // rdx
  __int64 *v98; // r8
  __int64 v99; // rsi
  __int64 v100; // rcx
  _QWORD *v101; // rax
  __int64 *v102; // rdi
  __int64 v103; // rsi
  __int64 v104; // rcx
  char *v105; // rax
  __int64 v106; // r8
  char *v107; // rsi
  __int64 v108; // rcx
  char *v109; // rdx
  char *v110; // rcx
  __int64 v111; // rsi
  __int64 v112; // rdx
  int v113; // r8d
  int *v114; // rax
  unsigned __int64 v115; // rdi
  unsigned __int64 v116; // r8
  int v117; // r15d
  __int64 v118; // rdi
  int v119; // eax
  _WORD *v120; // rdx
  unsigned __int16 *v121; // rcx
  int v122; // r9d
  int v123; // eax
  unsigned __int16 *v124; // rdx
  unsigned __int16 *v125; // r8
  unsigned __int16 *v126; // rax
  int v127; // r11d
  unsigned __int16 *v128; // r15
  int v129; // edx
  unsigned __int16 *v130; // rax
  __int64 v131; // rdx
  __int64 v132; // rdx
  char *v133; // r14
  __int64 v134; // rbx
  __int64 v135; // r13
  __int64 v136; // rax
  int v137; // ecx
  unsigned __int64 v138; // rdx
  unsigned __int64 v139; // rcx
  __int64 j; // rcx
  unsigned __int64 v141; // rdi
  int v142; // esi
  __int64 v143; // rax
  _QWORD *v146; // [rsp+18h] [rbp-D8h]
  _QWORD *v147; // [rsp+20h] [rbp-D0h]
  unsigned __int64 v148; // [rsp+28h] [rbp-C8h]
  _QWORD *v149; // [rsp+30h] [rbp-C0h]
  _QWORD *v150; // [rsp+38h] [rbp-B8h]
  char *v151; // [rsp+40h] [rbp-B0h]
  unsigned __int16 *v152; // [rsp+40h] [rbp-B0h]
  __int64 v153; // [rsp+48h] [rbp-A8h]
  int v154; // [rsp+48h] [rbp-A8h]
  _QWORD *v155; // [rsp+50h] [rbp-A0h]
  int v156; // [rsp+50h] [rbp-A0h]
  __int64 v157; // [rsp+58h] [rbp-98h]
  _QWORD *v158; // [rsp+58h] [rbp-98h]
  _QWORD *v159; // [rsp+58h] [rbp-98h]
  unsigned int v160; // [rsp+58h] [rbp-98h]
  int v161; // [rsp+58h] [rbp-98h]
  __int64 v162; // [rsp+68h] [rbp-88h]
  _QWORD *v163; // [rsp+68h] [rbp-88h]
  int v165; // [rsp+78h] [rbp-78h]
  unsigned int v166; // [rsp+7Ch] [rbp-74h]
  int v167; // [rsp+7Ch] [rbp-74h]
  __int16 v168; // [rsp+80h] [rbp-70h]
  unsigned int v169; // [rsp+80h] [rbp-70h]
  unsigned __int64 v170; // [rsp+88h] [rbp-68h]
  _QWORD *v171; // [rsp+88h] [rbp-68h]
  __int64 v172; // [rsp+88h] [rbp-68h]
  unsigned int v173; // [rsp+88h] [rbp-68h]
  __int64 v174; // [rsp+88h] [rbp-68h]
  int *v175; // [rsp+88h] [rbp-68h]
  unsigned int v176; // [rsp+88h] [rbp-68h]
  __int64 v177; // [rsp+88h] [rbp-68h]
  __int64 *v178; // [rsp+88h] [rbp-68h]
  __int64 v179; // [rsp+90h] [rbp-60h] BYREF
  __int64 v180; // [rsp+98h] [rbp-58h] BYREF
  _QWORD *v181; // [rsp+A0h] [rbp-50h]
  __int64 *v182; // [rsp+A8h] [rbp-48h]
  __int64 *v183; // [rsp+B0h] [rbp-40h]
  __int64 v184; // [rsp+B8h] [rbp-38h]

  v5 = (unsigned int)(*(_DWORD *)(a2 + 16) + 63) >> 6;
  v165 = *(_DWORD *)(a2 + 16);
  v6 = 8 * v5;
  v7 = (char *)malloc(8 * v5);
  if ( !v7 )
  {
    if ( v6 || (v143 = malloc(1u)) == 0 )
      sub_16BD1C0("Allocation failed", 1u);
    else
      v7 = (char *)v143;
  }
  if ( (_DWORD)v5 )
    memset(v7, 0, v6);
  v146 = (_QWORD *)(a1 + 320);
  v153 = *(_QWORD *)(a1 + 328);
  if ( v153 != a1 + 320 )
  {
    v166 = (unsigned int)(v165 + 31) >> 5;
    v151 = &v7[8 * (unsigned int)(v5 - 1)];
    while ( 1 )
    {
      v170 = sub_1DD6160(v153);
      v162 = v153 + 24;
      if ( v170 == v153 + 24 )
        break;
      v8 = *(_WORD *)(v170 + 46);
      if ( (v8 & 4) != 0 || (v8 & 8) == 0 )
      {
        if ( (*(_QWORD *)(*(_QWORD *)(v170 + 16) + 8LL) & 8LL) == 0 )
          break;
      }
      else if ( !sub_1E15D00(v170, 8u, 1) )
      {
        break;
      }
      v9 = 0;
      v10 = *(_QWORD *)(v170 + 64);
      v179 = v10;
      if ( v10 )
      {
        sub_1623A60((__int64)&v179, v10, 2);
        v9 = v179;
        v10 = *(_QWORD *)(v170 + 64);
      }
      v11 = (_QWORD *)v170;
      v12 = (_QWORD *)v170;
      if ( v10 == v9 )
      {
        while ( 1 )
        {
          v13 = *v12 & 0xFFFFFFFFFFFFFFF8LL;
          if ( !v13 )
            BUG();
          v14 = *v12 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_QWORD *)v13 & 4) == 0 && (*(_BYTE *)(v13 + 46) & 4) != 0 )
          {
            for ( i = *(_QWORD *)v13; ; i = *(_QWORD *)v14 )
            {
              v14 = i & 0xFFFFFFFFFFFFFFF8LL;
              if ( (*(_BYTE *)(v14 + 46) & 4) == 0 )
                break;
            }
          }
          if ( v162 == v14 )
            break;
          v11 = v12;
          v12 = (_QWORD *)v14;
          if ( *(_QWORD *)(v14 + 64) != v9 )
            goto LABEL_21;
        }
        v171 = *(_QWORD **)(v153 + 32);
      }
      else
      {
LABEL_21:
        v171 = v11;
      }
      if ( v9 )
        sub_161E7C0((__int64)&v179, v9);
LABEL_27:
      v16 = *(_QWORD *)(v153 + 32);
      if ( v16 != v162 && (_QWORD *)v16 != v171 )
      {
        do
        {
          v168 = *(_WORD *)(v16 + 46);
          if ( (v168 & 1) == 0 )
          {
            v40 = *(char **)(v16 + 32);
            v41 = &v40[40 * *(unsigned int *)(v16 + 40)];
            if ( v40 != v41 )
            {
              v42 = *v40;
              if ( !*v40 )
              {
LABEL_75:
                if ( (v40[3] & 0x10) == 0 )
                  goto LABEL_84;
                v43 = *((_DWORD *)v40 + 2);
                if ( v43 <= 0 )
                  goto LABEL_84;
                v44 = *(_QWORD *)(a2 + 8);
                v45 = *(_QWORD *)(a2 + 56);
                v46 = *(_DWORD *)(v44 + 24LL * (unsigned int)v43 + 16);
                v47 = 0;
                v48 = v43 * (v46 & 0xF);
                v49 = (_WORD *)(v45 + 2LL * (v46 >> 4));
                v50 = 0;
                v51 = v49 + 1;
                v52 = *v49 + v48;
                while ( 1 )
                {
                  v53 = v51;
                  v54 = v51;
                  if ( !v51 )
                    break;
                  while ( 1 )
                  {
                    v55 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 4LL * v52);
                    v56 = *v55;
                    v50 = v55[1];
                    if ( (_WORD)v56 )
                    {
                      while ( 1 )
                      {
                        v57 = v45 + 2LL * *(unsigned int *)(v44 + 24LL * (unsigned __int16)v56 + 8);
                        if ( v57 )
                          goto LABEL_81;
                        if ( !(_WORD)v50 )
                          break;
                        v56 = v50;
                        v50 = 0;
                      }
                      v47 = v56;
                    }
                    v74 = *v53;
                    v51 = 0;
                    ++v53;
                    v52 += v74;
                    if ( !v74 )
                      break;
                    v54 = v53;
                    if ( !v53 )
                      goto LABEL_107;
                  }
                }
LABEL_107:
                v56 = v47;
                v57 = 0;
LABEL_81:
                while ( v54 )
                {
                  while ( 1 )
                  {
                    v57 += 2;
                    *(_QWORD *)&v7[(v56 >> 3) & 0x1FF8] |= 1LL << v56;
                    v58 = *(unsigned __int16 *)(v57 - 2);
                    if ( !(_WORD)v58 )
                      break;
                    v56 = (unsigned int)(v58 + v56);
                  }
                  if ( (_WORD)v50 )
                  {
                    v73 = (unsigned __int16)v50;
                    v56 = v50;
                    v50 = 0;
                    v57 = v45 + 2LL * *(unsigned int *)(v44 + 24 * v73 + 8);
                  }
                  else
                  {
                    v50 = *v54;
                    v52 += v50;
                    if ( (_WORD)v50 )
                    {
                      ++v54;
                      v72 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 4LL * v52);
                      v56 = *v72;
                      v50 = v72[1];
                      v57 = v45 + 2LL * *(unsigned int *)(v44 + 24LL * (unsigned __int16)v56 + 8);
                    }
                    else
                    {
                      v57 = 0;
                      v54 = 0;
                    }
                  }
                }
                goto LABEL_84;
              }
              while ( 1 )
              {
                if ( v42 != 12 )
                  goto LABEL_84;
                v59 = (int *)*((_QWORD *)v40 + 3);
                v60 = v7;
                v61 = &v59[2 * ((v166 - 2) >> 1) + 2];
                if ( v166 <= 1 )
                {
                  v65 = (unsigned int)(v165 + 31) >> 5;
                  v61 = (int *)*((_QWORD *)v40 + 3);
                  v64 = 0;
                }
                else
                {
                  do
                  {
                    v62 = *v59;
                    v63 = v59[1];
                    v59 += 2;
                    v60 += 8;
                    *((_QWORD *)v60 - 1) |= (unsigned int)~v62 | ((unsigned __int64)(unsigned int)~v63 << 32);
                  }
                  while ( v59 != v61 );
                  v64 = ((v166 - 2) >> 1) + 1;
                  v65 = ((v165 + 31) & 0x20) != 0;
                }
                if ( v65 )
                {
                  v66 = &v7[8 * v64];
                  v67 = v61 + 1;
                  v68 = 0;
                  v69 = *(_QWORD *)v66;
                  v70 = v67;
                  while ( 1 )
                  {
                    v71 = (unsigned __int64)(unsigned int)~*(v67 - 1) << v68;
                    v68 += 32;
                    v69 |= v71;
                    if ( v67 == v70 )
                      break;
                    ++v67;
                  }
                  *(_QWORD *)v66 = v69;
                }
                if ( (v165 & 0x3F) != 0 )
                {
                  v40 += 40;
                  *(_QWORD *)v151 &= ~(-1LL << v165);
                  if ( v41 == v40 )
                    break;
                }
                else
                {
LABEL_84:
                  v40 += 40;
                  if ( v41 == v40 )
                    break;
                }
                v42 = *v40;
                if ( !*v40 )
                  goto LABEL_75;
              }
            }
          }
          if ( (*(_BYTE *)v16 & 4) == 0 && (v168 & 8) != 0 )
          {
            do
              v16 = *(_QWORD *)(v16 + 8);
            while ( (*(_BYTE *)(v16 + 46) & 8) != 0 );
          }
          v16 = *(_QWORD *)(v16 + 8);
        }
        while ( v162 != v16 && v171 != (_QWORD *)v16 );
      }
      v153 = *(_QWORD *)(v153 + 8);
      if ( v146 == (_QWORD *)v153 )
        goto LABEL_36;
    }
    v171 = 0;
    goto LABEL_27;
  }
LABEL_36:
  v17 = *(__int64 (**)())(**(_QWORD **)(a1 + 16) + 56LL);
  if ( v17 == sub_1D12D20 )
    BUG();
  v18 = *(_DWORD *)(v17() + 112);
  LODWORD(v180) = 0;
  v181 = 0;
  v167 = v18;
  v182 = &v180;
  v183 = &v180;
  v184 = 0;
  v149 = *(_QWORD **)(a1 + 328);
  if ( v149 == v146 )
  {
    v75 = 0;
    goto LABEL_118;
  }
  v147 = (_QWORD *)a2;
  v19 = v7;
  v20 = (unsigned int)(v165 - 1) >> 6;
  v148 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v165;
  do
  {
    v21 = v19;
    v22 = v149[4];
    v163 = v149 + 3;
    if ( (_QWORD *)v22 != v149 + 3 )
    {
      while ( 1 )
      {
        v23 = **(_WORD **)(v22 + 16);
        if ( v23 != 12 )
        {
          if ( v23 != 13 )
          {
            v24 = *(_QWORD *)(v22 + 32);
            v172 = v24 + 40LL * *(unsigned int *)(v22 + 40);
            if ( v172 != v24 )
            {
              v25 = v147;
              do
              {
                if ( *(_BYTE *)v24 )
                {
                  if ( *(_BYTE *)v24 == 12 && v165 )
                  {
                    v27 = 0;
                    while ( 1 )
                    {
                      _RCX = *(_QWORD *)&v21[8 * v27];
                      if ( v20 == (_DWORD)v27 )
                        _RCX = v148 & *(_QWORD *)&v21[8 * v27];
                      if ( _RCX )
                        break;
                      if ( v20 < (unsigned int)++v27 )
                        goto LABEL_46;
                    }
                    __asm { tzcnt   rcx, rcx }
                    if ( ((_DWORD)v27 << 6) + (_DWORD)_RCX != -1 )
                    {
                      v30 = ((_DWORD)v27 << 6) + _RCX;
                      do
                      {
                        if ( (int)v30 > 0 && v167 != v30 )
                        {
                          v31 = *(_DWORD *)(*(_QWORD *)(v24 + 24) + 4LL * (v30 >> 5));
                          if ( !_bittest(&v31, v30) )
                          {
                            v97 = v181;
                            if ( v181 )
                            {
                              v98 = &v180;
                              do
                              {
                                while ( 1 )
                                {
                                  v99 = v97[2];
                                  v100 = v97[3];
                                  if ( v30 <= *((_DWORD *)v97 + 8) )
                                    break;
                                  v97 = (_QWORD *)v97[3];
                                  if ( !v100 )
                                    goto LABEL_153;
                                }
                                v98 = v97;
                                v97 = (_QWORD *)v97[2];
                              }
                              while ( v99 );
LABEL_153:
                              if ( v98 != &v180 && v30 >= *((_DWORD *)v98 + 8) )
                              {
                                v155 = v25;
                                v160 = v30;
                                sub_39C1B70((__int64)&v179, (__int64)v98, a3, v22);
                                v30 = v160;
                                v25 = v155;
                              }
                            }
                          }
                        }
                        v32 = v30 + 1;
                        if ( v165 == v32 )
                          break;
                        v33 = v32 >> 6;
                        if ( v20 < v32 >> 6 )
                          break;
                        v34 = v33;
                        v35 = 64 - (v32 & 0x3F);
                        v36 = 0xFFFFFFFFFFFFFFFFLL >> v35;
                        if ( v35 == 64 )
                          v36 = 0;
                        v37 = ~v36;
                        while ( 1 )
                        {
                          _RAX = *(_QWORD *)&v21[8 * v34];
                          if ( v33 == (_DWORD)v34 )
                            _RAX = v37 & *(_QWORD *)&v21[8 * v34];
                          if ( v20 == (_DWORD)v34 )
                            _RAX &= v148;
                          if ( _RAX )
                            break;
                          if ( v20 < (unsigned int)++v34 )
                            goto LABEL_46;
                        }
                        __asm { tzcnt   rax, rax }
                        v30 = ((_DWORD)v34 << 6) + _RAX;
                      }
                      while ( v30 != -1 );
                    }
                  }
                }
                else if ( (*(_BYTE *)(v24 + 3) & 0x10) != 0 )
                {
                  v26 = *(_DWORD *)(v24 + 8);
                  if ( v26 )
                  {
                    v95 = *(_WORD *)(v22 + 46);
                    if ( (v95 & 4) != 0 || (v95 & 8) == 0 )
                    {
                      v96 = (*(_QWORD *)(*(_QWORD *)(v22 + 16) + 8LL) >> 4) & 1LL;
                    }
                    else
                    {
                      v158 = v25;
                      LOBYTE(v96) = sub_1E15D00(v22, 0x10u, 1);
                      v26 = *(_DWORD *)(v24 + 8);
                      v25 = v158;
                    }
                    if ( !(_BYTE)v96 || v167 != v26 )
                    {
                      if ( (v26 & 0x80000000) == 0 )
                      {
                        v117 = 0;
                        v118 = v25[7];
                        v119 = v26 * (*(_DWORD *)(v25[1] + 24LL * v26 + 16) & 0xF);
                        v120 = (_WORD *)(v118 + 2LL * (*(_DWORD *)(v25[1] + 24LL * v26 + 16) >> 4));
                        LOWORD(v119) = *v120 + v119;
                        v121 = v120 + 1;
                        v122 = v119;
                        v123 = 0;
LABEL_191:
                        v124 = v121;
                        v125 = v121;
                        if ( v121 )
                        {
                          while ( 1 )
                          {
                            v126 = (unsigned __int16 *)(v25[6] + 4LL * (unsigned __int16)v122);
                            v127 = *v126;
                            v123 = v126[1];
                            if ( (_WORD)v127 )
                              break;
LABEL_233:
                            v142 = *v124;
                            v121 = 0;
                            ++v124;
                            if ( !(_WORD)v142 )
                              goto LABEL_191;
                            v122 += v142;
                            v125 = v124;
                            if ( !v124 )
                              goto LABEL_235;
                          }
                          while ( 1 )
                          {
                            v128 = (unsigned __int16 *)(v118
                                                      + 2LL
                                                      * *(unsigned int *)(v25[1] + 24LL * (unsigned __int16)v127 + 8));
                            if ( v128 )
                              break;
                            if ( !(_WORD)v123 )
                            {
                              v117 = v127;
                              goto LABEL_233;
                            }
                            v127 = v123;
                            v123 = 0;
                          }
                        }
                        else
                        {
LABEL_235:
                          v127 = v117;
                          v128 = 0;
                        }
                        while ( v125 )
                        {
                          while ( 1 )
                          {
                            if ( (*(_QWORD *)&v21[8 * ((unsigned __int16)v127 >> 6)] & (1LL << v127)) != 0 )
                            {
                              v150 = v25;
                              v152 = v125;
                              v154 = v123;
                              v156 = v122;
                              v161 = v127;
                              sub_39C1C10((__int64)&v179, (unsigned __int16)v127, a3, v22);
                              v25 = v150;
                              v125 = v152;
                              v123 = v154;
                              v122 = v156;
                              v127 = v161;
                            }
                            v129 = *v128++;
                            if ( !(_WORD)v129 )
                              break;
                            v127 += v129;
                          }
                          if ( (_WORD)v123 )
                          {
                            v132 = (unsigned __int16)v123;
                            v127 = v123;
                            v123 = 0;
                            v128 = (unsigned __int16 *)(v25[7] + 2LL * *(unsigned int *)(v25[1] + 24 * v132 + 8));
                          }
                          else
                          {
                            v123 = *v125;
                            v122 += v123;
                            if ( (_WORD)v123 )
                            {
                              ++v125;
                              v130 = (unsigned __int16 *)(v25[6] + 4LL * (unsigned __int16)v122);
                              v131 = *v130;
                              v123 = v130[1];
                              v127 = v131;
                              v128 = (unsigned __int16 *)(v25[7] + 2LL * *(unsigned int *)(v25[1] + 24 * v131 + 8));
                            }
                            else
                            {
                              v128 = 0;
                              v125 = 0;
                            }
                          }
                        }
                      }
                      else
                      {
                        v159 = v25;
                        sub_39C1C10((__int64)&v179, v26, a3, v22);
                        v25 = v159;
                      }
                    }
                  }
                }
LABEL_46:
                v24 += 40;
              }
              while ( v172 != v24 );
            }
          }
          goto LABEL_110;
        }
        v76 = 0;
        v77 = sub_1E16500(v22);
        v78 = sub_15C70A0(v22 + 64);
        if ( *(_DWORD *)(v78 + 8) == 2 )
          v76 = *(_QWORD *)(v78 - 8);
        v79 = sub_39C0E30(a3, v77, v76);
        if ( v79 )
          break;
LABEL_125:
        sub_39C1A80(a3, v77, v76, v22);
        v81 = *(_QWORD *)(v22 + 32);
        if ( *(_BYTE *)v81 )
          goto LABEL_110;
        v82 = *(_DWORD *)(v81 + 8);
        if ( !v82 )
          goto LABEL_110;
        v83 = v181;
        v84 = &v180;
        if ( v181 )
        {
          do
          {
            while ( 1 )
            {
              v85 = v83[2];
              v86 = v83[3];
              if ( v82 <= *((_DWORD *)v83 + 8) )
                break;
              v83 = (_QWORD *)v83[3];
              if ( !v86 )
                goto LABEL_132;
            }
            v84 = v83;
            v83 = (_QWORD *)v83[2];
          }
          while ( v85 );
LABEL_132:
          if ( v84 != &v180 && v82 >= *((_DWORD *)v84 + 8) )
            goto LABEL_139;
        }
        v173 = v82;
        v157 = (__int64)v84;
        v87 = sub_22077B0(0x48u);
        v88 = v173;
        v89 = v87;
        v87 += 56;
        *(_DWORD *)(v87 - 24) = v173;
        *(_QWORD *)(v89 + 40) = v87;
        *(_QWORD *)(v89 + 48) = 0x100000000LL;
        v174 = v89;
        v169 = v88;
        v90 = sub_39C12B0(&v179, v157, (unsigned int *)(v89 + 32));
        if ( v91 )
        {
          v92 = v90 || &v180 == (__int64 *)v91 || v169 < *(_DWORD *)(v91 + 32);
          sub_220F040(v92, v174, (_QWORD *)v91, &v180);
          ++v184;
          v84 = (__int64 *)v174;
LABEL_139:
          v93 = *((unsigned int *)v84 + 12);
          if ( (unsigned int)v93 >= *((_DWORD *)v84 + 13) )
            goto LABEL_222;
          goto LABEL_140;
        }
        v141 = v174;
        v177 = v90;
        j_j___libc_free_0(v141);
        v84 = (__int64 *)v177;
        v93 = *(unsigned int *)(v177 + 48);
        if ( (unsigned int)v93 >= *(_DWORD *)(v177 + 52) )
        {
LABEL_222:
          v178 = v84;
          sub_16CD150((__int64)(v84 + 5), v84 + 7, 0, 16, (int)v84, v80);
          v84 = v178;
          v93 = *((unsigned int *)v178 + 12);
        }
LABEL_140:
        v94 = (__int64 *)(v84[5] + 16 * v93);
        *v94 = v77;
        v94[1] = v76;
        ++*((_DWORD *)v84 + 12);
LABEL_110:
        if ( (*(_BYTE *)v22 & 4) == 0 )
        {
          while ( (*(_BYTE *)(v22 + 46) & 8) != 0 )
            v22 = *(_QWORD *)(v22 + 8);
        }
        v22 = *(_QWORD *)(v22 + 8);
        if ( v163 == (_QWORD *)v22 )
        {
          v19 = v21;
          goto LABEL_113;
        }
      }
      v101 = v181;
      v102 = &v180;
      if ( v181 )
      {
        do
        {
          while ( 1 )
          {
            v103 = v101[2];
            v104 = v101[3];
            if ( v79 <= *((_DWORD *)v101 + 8) )
              break;
            v101 = (_QWORD *)v101[3];
            if ( !v104 )
              goto LABEL_162;
          }
          v102 = v101;
          v101 = (_QWORD *)v101[2];
        }
        while ( v103 );
LABEL_162:
        if ( v102 != &v180 && v79 < *((_DWORD *)v102 + 8) )
          v102 = &v180;
      }
      v105 = (char *)v102[5];
      v106 = *((unsigned int *)v102 + 12);
      v107 = &v105[16 * v106];
      v108 = (16 * v106) >> 4;
      if ( (16 * v106) >> 6 )
      {
        v109 = &v105[64 * ((16 * v106) >> 6)];
        while ( 1 )
        {
          if ( v77 == *(_QWORD *)v105 )
          {
            if ( v76 == *((_QWORD *)v105 + 1) )
              goto LABEL_177;
            if ( v77 != *((_QWORD *)v105 + 2) )
              goto LABEL_169;
          }
          else if ( v77 != *((_QWORD *)v105 + 2) )
          {
            goto LABEL_169;
          }
          if ( v76 == *((_QWORD *)v105 + 3) )
          {
            v105 += 16;
            goto LABEL_177;
          }
LABEL_169:
          if ( v77 == *((_QWORD *)v105 + 4) && v76 == *((_QWORD *)v105 + 5) )
          {
            v105 += 32;
            goto LABEL_177;
          }
          if ( v77 == *((_QWORD *)v105 + 6) && v76 == *((_QWORD *)v105 + 7) )
          {
            v105 += 48;
            goto LABEL_177;
          }
          v105 += 64;
          if ( v109 == v105 )
          {
            v108 = (v107 - v105) >> 4;
            break;
          }
        }
      }
      if ( v108 != 2 )
      {
        if ( v108 != 3 )
        {
          if ( v108 != 1 )
            goto LABEL_176;
          goto LABEL_228;
        }
        if ( v77 == *(_QWORD *)v105 && v76 == *((_QWORD *)v105 + 1) )
        {
LABEL_177:
          v110 = v105 + 16;
          v111 = v107 - (v105 + 16);
          v112 = v111 >> 4;
          if ( v111 > 0 )
          {
            while ( 1 )
            {
              *(_QWORD *)v105 = *((_QWORD *)v105 + 2);
              *((_QWORD *)v105 + 1) = *((_QWORD *)v105 + 3);
              v105 = v110;
              if ( !--v112 )
                break;
              v110 += 16;
            }
            LODWORD(v106) = *((_DWORD *)v102 + 12);
          }
          v113 = v106 - 1;
          *((_DWORD *)v102 + 12) = v113;
          if ( !v113 )
          {
            v114 = sub_220F330((int *)v102, &v180);
            v115 = *((_QWORD *)v114 + 5);
            v116 = (unsigned __int64)v114;
            if ( (int *)v115 != v114 + 14 )
            {
              v175 = v114;
              _libc_free(v115);
              v116 = (unsigned __int64)v175;
            }
            j_j___libc_free_0(v116);
            --v184;
          }
          goto LABEL_125;
        }
        v105 += 16;
      }
      if ( v77 != *(_QWORD *)v105 || v76 != *((_QWORD *)v105 + 1) )
      {
        v105 += 16;
LABEL_228:
        if ( v77 != *(_QWORD *)v105 )
        {
LABEL_176:
          v105 = v107;
          goto LABEL_177;
        }
        if ( v76 != *((_QWORD *)v105 + 1) )
          v105 = v107;
        goto LABEL_177;
      }
      goto LABEL_177;
    }
LABEL_113:
    if ( !a4
      && v163 != (_QWORD *)(v149[3] & 0xFFFFFFFFFFFFFFF8LL)
      && v149 != (_QWORD *)(*(_QWORD *)(a1 + 320) & 0xFFFFFFFFFFFFFFF8LL)
      && v182 != &v180 )
    {
      v176 = v20;
      v133 = v19;
      v134 = (__int64)v182;
      do
      {
        v135 = v134;
        v136 = sub_220EEE0(v134);
        v137 = *(_DWORD *)(v134 + 32);
        v134 = v136;
        if ( v137 < 0 || (*(_QWORD *)&v133[8 * ((unsigned int)v137 >> 6)] & (1LL << v137)) != 0 )
        {
          v138 = v149[3] & 0xFFFFFFFFFFFFFFF8LL;
          if ( !v138 )
            BUG();
          v139 = v149[3] & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_QWORD *)v138 & 4) == 0 && (*(_BYTE *)(v138 + 46) & 4) != 0 )
          {
            for ( j = *(_QWORD *)v138; ; j = *(_QWORD *)v139 )
            {
              v139 = j & 0xFFFFFFFFFFFFFFF8LL;
              if ( (*(_BYTE *)(v139 + 46) & 4) == 0 )
                break;
            }
          }
          sub_39C1B70((__int64)&v179, v135, a3, v139);
        }
      }
      while ( (__int64 *)v134 != &v180 );
      v20 = v176;
      v19 = v133;
    }
    v149 = (_QWORD *)v149[1];
  }
  while ( v149 != v146 );
  v75 = v181;
  v7 = v19;
LABEL_118:
  sub_39C0B90(v75);
  _libc_free((unsigned __int64)v7);
}
