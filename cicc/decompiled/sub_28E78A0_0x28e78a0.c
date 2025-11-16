// Function: sub_28E78A0
// Address: 0x28e78a0
//
__int64 __fastcall sub_28E78A0(unsigned __int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r15
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v11; // rdi
  __int64 v12; // r13
  __int64 v13; // rax
  _BYTE *v14; // rax
  _QWORD *v15; // rax
  _QWORD *v16; // r14
  _QWORD *v17; // rax
  char v18; // r12
  __int64 v19; // rbx
  __int64 v20; // rax
  __int64 v21; // r12
  _QWORD *v22; // rdi
  int v23; // r11d
  unsigned int v24; // ecx
  _QWORD *v25; // rdx
  __int64 v26; // rsi
  unsigned __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rdi
  unsigned __int64 v30; // rdi
  __int64 *v31; // rbx
  __int64 *i; // r13
  __int64 v33; // r14
  _BYTE *v34; // rsi
  _QWORD *v35; // rbx
  _QWORD *v36; // r12
  __int64 v37; // rax
  void *v38; // r14
  signed __int64 v39; // r13
  _BYTE *v40; // rbx
  _BYTE *v41; // r12
  unsigned __int64 v42; // r13
  unsigned __int64 v43; // rdi
  size_t v45; // rdx
  __int64 v46; // rax
  __int64 v47; // rcx
  bool v48; // cf
  unsigned __int64 v49; // rax
  unsigned __int64 v50; // rbx
  __int64 v51; // rax
  char *v52; // r12
  unsigned __int64 v53; // rbx
  __int64 v54; // rax
  _QWORD *v55; // rbx
  int v56; // eax
  __int64 v57; // rdi
  __int64 v58; // rbx
  char v59; // bl
  __int64 v60; // rax
  char v61; // bl
  __int64 *v62; // r12
  __int64 v63; // rax
  __int64 v64; // r12
  __int128 *v65; // rax
  int v66; // eax
  __int64 v67; // r8
  __int64 v68; // r9
  char *v69; // rcx
  char *v70; // r13
  __int64 v71; // rbx
  unsigned __int64 v72; // rax
  char *v73; // r15
  __int64 v74; // r13
  const char *v75; // rax
  size_t v76; // rdx
  size_t v77; // r12
  const char *v78; // r14
  const char *v79; // rax
  size_t v80; // rdx
  size_t v81; // rbx
  int v82; // eax
  __int64 v83; // rax
  char *v84; // rax
  __int64 v85; // rax
  unsigned __int64 v86; // rdx
  __int64 v87; // r13
  __int64 v88; // r15
  unsigned __int64 v89; // rax
  __int64 v90; // rbx
  unsigned int v91; // r12d
  __int64 v92; // r13
  __int64 v93; // r15
  __int64 v94; // rbx
  __int64 v95; // r9
  int v96; // r13d
  __int64 *v97; // r8
  __int64 v98; // rdx
  __int64 *v99; // rcx
  __int64 v100; // rdi
  int v101; // edx
  __int64 v102; // rax
  unsigned __int64 v103; // rdx
  __int64 *v104; // r15
  __int64 *v105; // rbx
  __int64 v106; // rsi
  __int64 v107; // rax
  __int64 v108; // r9
  unsigned __int64 v109; // rdx
  unsigned __int64 v110; // rax
  __int64 v111; // rdx
  unsigned __int64 v112; // r8
  __int64 v113; // rax
  int v114; // eax
  __int64 v115; // rax
  unsigned __int64 v116; // rdx
  __int64 v117; // rax
  char *v118; // rcx
  char *v119; // rdx
  __int64 v120; // rsi
  _QWORD *v121; // rax
  char *v122; // rax
  char *v123; // rcx
  unsigned int v124; // eax
  __int64 v125; // rdi
  int v126; // esi
  __int64 *v127; // rcx
  int v128; // esi
  unsigned int v129; // eax
  __int64 v130; // rdi
  unsigned __int8 v131; // [rsp+1Fh] [rbp-3C1h]
  char *v132; // [rsp+30h] [rbp-3B0h]
  unsigned __int64 v133; // [rsp+38h] [rbp-3A8h]
  __int64 *v134; // [rsp+40h] [rbp-3A0h]
  __int64 *v135; // [rsp+40h] [rbp-3A0h]
  char *dest; // [rsp+48h] [rbp-398h]
  size_t n; // [rsp+50h] [rbp-390h]
  size_t na; // [rsp+50h] [rbp-390h]
  size_t nb; // [rsp+50h] [rbp-390h]
  __int64 v140; // [rsp+60h] [rbp-380h]
  __int64 v142; // [rsp+70h] [rbp-370h]
  char *v143; // [rsp+70h] [rbp-370h]
  char *v144; // [rsp+78h] [rbp-368h]
  __int64 v145; // [rsp+78h] [rbp-368h]
  int v146; // [rsp+80h] [rbp-360h]
  unsigned __int64 v147; // [rsp+80h] [rbp-360h]
  __int64 *v148; // [rsp+88h] [rbp-358h]
  char *v149; // [rsp+88h] [rbp-358h]
  char *v150; // [rsp+88h] [rbp-358h]
  void *src; // [rsp+90h] [rbp-350h] BYREF
  _BYTE *v152; // [rsp+98h] [rbp-348h]
  _BYTE *v153; // [rsp+A0h] [rbp-340h]
  __int64 *v154; // [rsp+B0h] [rbp-330h] BYREF
  __int64 *v155; // [rsp+B8h] [rbp-328h]
  __int64 v156; // [rsp+C0h] [rbp-320h]
  unsigned __int64 v157; // [rsp+D0h] [rbp-310h] BYREF
  unsigned __int64 v158; // [rsp+D8h] [rbp-308h]
  __int64 v159; // [rsp+E0h] [rbp-300h]
  __int64 v160; // [rsp+F0h] [rbp-2F0h] BYREF
  __int64 v161; // [rsp+F8h] [rbp-2E8h]
  __int64 v162; // [rsp+100h] [rbp-2E0h]
  __int64 v163; // [rsp+108h] [rbp-2D8h]
  __int16 v164; // [rsp+110h] [rbp-2D0h]
  unsigned __int64 v165[2]; // [rsp+120h] [rbp-2C0h] BYREF
  char v166; // [rsp+130h] [rbp-2B0h] BYREF
  _BYTE *v167; // [rsp+138h] [rbp-2A8h]
  __int64 v168; // [rsp+140h] [rbp-2A0h]
  _BYTE v169[56]; // [rsp+148h] [rbp-298h] BYREF
  __int64 v170; // [rsp+180h] [rbp-260h]
  unsigned __int64 v171; // [rsp+188h] [rbp-258h]
  char v172; // [rsp+190h] [rbp-250h]
  int v173; // [rsp+194h] [rbp-24Ch]
  int v174; // [rsp+198h] [rbp-248h]
  __int64 *v175; // [rsp+1A0h] [rbp-240h] BYREF
  __int64 v176; // [rsp+1A8h] [rbp-238h]
  _BYTE v177[128]; // [rsp+1B0h] [rbp-230h] BYREF
  __int64 v178; // [rsp+230h] [rbp-1B0h] BYREF
  __int64 v179; // [rsp+238h] [rbp-1A8h]
  __int64 v180; // [rsp+240h] [rbp-1A0h]
  __int64 v181; // [rsp+248h] [rbp-198h]
  __int64 *v182; // [rsp+250h] [rbp-190h] BYREF
  unsigned __int64 v183; // [rsp+258h] [rbp-188h]
  __int64 v184; // [rsp+260h] [rbp-180h] BYREF
  _BYTE v185[32]; // [rsp+268h] [rbp-178h] BYREF
  _BYTE *v186; // [rsp+288h] [rbp-158h]
  __int64 v187; // [rsp+290h] [rbp-150h]
  _BYTE v188[192]; // [rsp+298h] [rbp-148h] BYREF
  _BYTE *v189; // [rsp+358h] [rbp-88h]
  __int64 v190; // [rsp+360h] [rbp-80h]
  _BYTE v191[120]; // [rsp+368h] [rbp-78h] BYREF

  v6 = a2;
  v170 = 0;
  v131 = sub_F62E00(a1, 0, 0, a4, a5, a6);
  v165[0] = (unsigned __int64)&v166;
  v165[1] = 0x100000000LL;
  v167 = v169;
  v168 = 0x600000000LL;
  v174 = *(_DWORD *)(a1 + 92);
  v172 = 0;
  v173 = 0;
  v171 = a1;
  sub_B1F440((__int64)v165);
  v175 = (__int64 *)v177;
  v176 = 0x1000000000LL;
  if ( (_BYTE)qword_5004708 )
    goto LABEL_2;
  sub_B84820(&v157, *(_QWORD *)(a1 + 40));
  v59 = qword_50047E8;
  v60 = sub_22077B0(0x1F0u);
  v61 = v59 ^ 1;
  v62 = (__int64 *)v60;
  if ( v60 )
    sub_980840(v60, a2);
  sub_B8B4C0((__int64)&v157, v62, 0);
  v63 = sub_22077B0(0xF0u);
  v64 = v63;
  if ( v63 )
  {
    *(_QWORD *)(v63 + 8) = 0;
    *(_QWORD *)(v63 + 16) = &unk_5004678;
    *(_QWORD *)(v63 + 56) = v63 + 104;
    *(_QWORD *)(v63 + 112) = v63 + 160;
    *(_DWORD *)(v63 + 24) = 2;
    *(_QWORD *)(v63 + 32) = 0;
    *(_QWORD *)(v63 + 40) = 0;
    *(_QWORD *)(v63 + 48) = 0;
    *(_QWORD *)(v63 + 64) = 1;
    *(_QWORD *)(v63 + 72) = 0;
    *(_QWORD *)(v63 + 80) = 0;
    *(_QWORD *)(v63 + 96) = 0;
    *(_QWORD *)(v63 + 104) = 0;
    *(_QWORD *)(v63 + 120) = 1;
    *(_QWORD *)(v63 + 128) = 0;
    *(_QWORD *)(v63 + 136) = 0;
    *(_QWORD *)(v63 + 152) = 0;
    *(_QWORD *)(v63 + 160) = 0;
    *(_BYTE *)(v63 + 168) = 0;
    *(_QWORD *)v63 = off_4A21C58;
    *(_QWORD *)(v63 + 176) = 0;
    *(_QWORD *)(v63 + 184) = 0;
    *(_QWORD *)(v63 + 192) = 0;
    *(_BYTE *)(v63 + 200) = v61;
    *(_QWORD *)(v63 + 208) = 0;
    *(_QWORD *)(v63 + 216) = 0;
    *(_QWORD *)(v63 + 224) = 0;
    *(_QWORD *)(v63 + 232) = 0;
    *(_DWORD *)(v63 + 88) = 1065353216;
    *(_DWORD *)(v63 + 144) = 1065353216;
    v65 = sub_BC2B00();
    sub_28E6450((__int64)v65);
  }
  sub_B8B4C0((__int64)&v157, (__int64 *)v64, 0);
  sub_B8A620((__int64)&v157, a1);
  v66 = *(_DWORD *)(a1 + 92);
  v171 = a1;
  v174 = v66;
  sub_B1F440((__int64)v165);
  v69 = *(char **)(v64 + 184);
  v70 = *(char **)(v64 + 176);
  if ( v69 == v70 )
  {
    v84 = *(char **)(v64 + 176);
LABEL_123:
    v143 = v70;
    goto LABEL_124;
  }
  v71 = v69 - v70;
  v144 = *(char **)(v64 + 184);
  _BitScanReverse64(&v72, (v69 - v70) >> 3);
  sub_28E6000(*(char **)(v64 + 176), v144, 2LL * (int)(63 - (v72 ^ 0x3F)));
  if ( v71 <= 128 )
  {
    sub_28E5E80(v70, v144);
  }
  else
  {
    sub_28E5E80(v70, (_QWORD *)v70 + 16);
    v149 = v70 + 128;
    if ( v144 != v70 + 128 )
    {
      v142 = v64;
      do
      {
        v73 = v149;
        v74 = *(_QWORD *)v149;
        while ( 1 )
        {
          while ( 1 )
          {
            v75 = sub_BD5D20(*(_QWORD *)(*((_QWORD *)v73 - 1) + 40LL));
            v77 = v76;
            v78 = v75;
            v79 = sub_BD5D20(*(_QWORD *)(v74 + 40));
            v81 = v80;
            if ( v77 <= v80 )
              v80 = v77;
            if ( !v80 )
              break;
            v82 = memcmp(v79, v78, v80);
            if ( !v82 )
              break;
            if ( v82 >= 0 )
              goto LABEL_113;
            v117 = *((_QWORD *)v73 - 1);
            v73 -= 8;
            *((_QWORD *)v73 + 1) = v117;
          }
          if ( v77 == v81 || v77 <= v81 )
            break;
          v83 = *((_QWORD *)v73 - 1);
          v73 -= 8;
          *((_QWORD *)v73 + 1) = v83;
        }
LABEL_113:
        v149 += 8;
        *(_QWORD *)v73 = v74;
      }
      while ( v144 != v149 );
      v64 = v142;
      v6 = a2;
    }
  }
  v70 = *(char **)(v64 + 184);
  v84 = *(char **)(v64 + 176);
  if ( v70 == v84 )
    goto LABEL_123;
  v118 = *(char **)(v64 + 176);
  do
  {
    v119 = v118;
    v118 += 8;
    if ( v118 == v70 )
      goto LABEL_123;
    v120 = *((_QWORD *)v118 - 1);
  }
  while ( v120 != *(_QWORD *)v118 );
  if ( v119 == v70 )
    goto LABEL_123;
  v121 = v119 + 16;
  if ( v70 == v119 + 16 )
    goto LABEL_196;
  while ( 1 )
  {
    if ( *v121 != v120 )
    {
      *((_QWORD *)v119 + 1) = *v121;
      v119 += 8;
    }
    if ( v70 == (char *)++v121 )
      break;
    v120 = *(_QWORD *)v119;
  }
  v118 = v119 + 8;
  v143 = *(char **)(v64 + 184);
  if ( v119 + 8 != v70 )
  {
    if ( v143 != v70 )
    {
      v122 = (char *)memmove(v119 + 8, v70, v143 - v70);
      v143 = *(char **)(v64 + 184);
      v123 = &v122[v143 - v70];
      v84 = *(char **)(v64 + 176);
      if ( v123 == v143 )
        goto LABEL_124;
      v143 = v123;
      goto LABEL_195;
    }
LABEL_196:
    v143 = v118;
    v84 = *(char **)(v64 + 176);
LABEL_195:
    *(_QWORD *)(v64 + 184) = v143;
    goto LABEL_124;
  }
  v84 = *(char **)(v64 + 176);
LABEL_124:
  if ( v84 != v143 )
  {
    v150 = v84;
    v135 = v6;
    do
    {
      v87 = *(_QWORD *)v150;
      if ( (_BYTE)qword_50049A8 )
      {
        v178 = 0;
        v179 = 0;
        v180 = 0;
        v181 = 0;
        v182 = &v184;
        v183 = 0;
        v88 = *(_QWORD *)(v87 + 40);
        v89 = *(_QWORD *)(v88 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v89 != v88 + 48 )
        {
          if ( !v89 )
            BUG();
          v90 = v89 - 24;
          if ( (unsigned int)*(unsigned __int8 *)(v89 - 24) - 30 <= 0xA )
          {
            v146 = sub_B46E30(v90);
            if ( v146 )
            {
              v145 = v87;
              v91 = 0;
              v92 = v88;
              v93 = v90;
              while ( 1 )
              {
                v94 = sub_B46EC0(v93, v91);
                if ( !(unsigned __int8)sub_B19720((__int64)v165, v94, v92) )
                {
LABEL_136:
                  if ( v146 == ++v91 )
                    goto LABEL_151;
                  goto LABEL_137;
                }
                if ( (_DWORD)v181 )
                {
                  v95 = v179;
                  v96 = 1;
                  v97 = 0;
                  LODWORD(v98) = (v181 - 1) & (((unsigned int)v94 >> 4) ^ ((unsigned int)v94 >> 9));
                  v99 = (__int64 *)(v179 + 8LL * (unsigned int)v98);
                  v100 = *v99;
                  if ( v94 == *v99 )
                    goto LABEL_136;
                  while ( v100 != -4096 )
                  {
                    if ( v100 == -8192 && !v97 )
                      v97 = v99;
                    v98 = ((_DWORD)v181 - 1) & (unsigned int)(v98 + v96);
                    v99 = (__int64 *)(v179 + 8 * v98);
                    v100 = *v99;
                    if ( v94 == *v99 )
                      goto LABEL_136;
                    ++v96;
                  }
                  if ( !v97 )
                    v97 = v99;
                  ++v178;
                  v101 = v180 + 1;
                  if ( 4 * ((int)v180 + 1) < (unsigned int)(3 * v181) )
                  {
                    if ( (int)v181 - HIDWORD(v180) - v101 <= (unsigned int)v181 >> 3 )
                    {
                      sub_CF28B0((__int64)&v178, v181);
                      if ( !(_DWORD)v181 )
                      {
LABEL_238:
                        LODWORD(v180) = v180 + 1;
                        BUG();
                      }
                      v128 = 1;
                      v95 = v179;
                      v129 = (v181 - 1) & (((unsigned int)v94 >> 4) ^ ((unsigned int)v94 >> 9));
                      v97 = (__int64 *)(v179 + 8LL * v129);
                      v101 = v180 + 1;
                      v127 = 0;
                      v130 = *v97;
                      if ( v94 != *v97 )
                      {
                        while ( v130 != -4096 )
                        {
                          if ( !v127 && v130 == -8192 )
                            v127 = v97;
                          v129 = (v181 - 1) & (v128 + v129);
                          v97 = (__int64 *)(v179 + 8LL * v129);
                          v130 = *v97;
                          if ( v94 == *v97 )
                            goto LABEL_146;
                          ++v128;
                        }
LABEL_212:
                        if ( v127 )
                          v97 = v127;
                        goto LABEL_146;
                      }
                    }
                    goto LABEL_146;
                  }
                }
                else
                {
                  ++v178;
                }
                sub_CF28B0((__int64)&v178, 2 * v181);
                if ( !(_DWORD)v181 )
                  goto LABEL_238;
                v95 = v179;
                v124 = (v181 - 1) & (((unsigned int)v94 >> 9) ^ ((unsigned int)v94 >> 4));
                v97 = (__int64 *)(v179 + 8LL * v124);
                v101 = v180 + 1;
                v125 = *v97;
                if ( v94 != *v97 )
                {
                  v126 = 1;
                  v127 = 0;
                  while ( v125 != -4096 )
                  {
                    if ( !v127 && v125 == -8192 )
                      v127 = v97;
                    v124 = (v181 - 1) & (v126 + v124);
                    v97 = (__int64 *)(v179 + 8LL * v124);
                    v125 = *v97;
                    if ( v94 == *v97 )
                      goto LABEL_146;
                    ++v126;
                  }
                  goto LABEL_212;
                }
LABEL_146:
                LODWORD(v180) = v101;
                if ( *v97 != -4096 )
                  --HIDWORD(v180);
                *v97 = v94;
                v102 = (unsigned int)v183;
                v103 = (unsigned int)v183 + 1LL;
                if ( v103 > HIDWORD(v183) )
                {
                  sub_C8D5F0((__int64)&v182, &v184, v103, 8u, (__int64)v97, v95);
                  v102 = (unsigned int)v183;
                }
                ++v91;
                v182[v102] = v94;
                LODWORD(v183) = v183 + 1;
                if ( v146 == v91 )
                {
LABEL_151:
                  v104 = v182;
                  v105 = &v182[(unsigned int)v183];
                  if ( v182 != v105 )
                  {
                    do
                    {
                      v106 = *v104;
                      v164 = 257;
                      v107 = sub_F41C30(*(_QWORD *)(v145 + 40), v106, (__int64)v165, 0, 0, (void **)&v160);
                      v109 = *(_QWORD *)(v107 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                      if ( v109 == v107 + 48 )
                      {
                        v110 = 0;
                      }
                      else
                      {
                        if ( !v109 )
                          BUG();
                        v110 = v109 - 24;
                        if ( (unsigned int)*(unsigned __int8 *)(v109 - 24) - 30 > 0xA )
                          v110 = 0;
                      }
                      v111 = (unsigned int)v176;
                      v112 = (unsigned int)v176 + 1LL;
                      if ( v112 > HIDWORD(v176) )
                      {
                        v147 = v110;
                        sub_C8D5F0((__int64)&v175, v177, (unsigned int)v176 + 1LL, 8u, v112, v108);
                        v111 = (unsigned int)v176;
                        v110 = v147;
                      }
                      ++v104;
                      v175[v111] = v110;
                      LODWORD(v176) = v176 + 1;
                    }
                    while ( v105 != v104 );
                    v104 = v182;
                  }
                  if ( v104 != &v184 )
                    _libc_free((unsigned __int64)v104);
                  break;
                }
LABEL_137:
                v92 = *(_QWORD *)(v145 + 40);
              }
            }
          }
        }
        sub_C7D6A0(v179, 8LL * (unsigned int)v181, 8);
      }
      else
      {
        v85 = (unsigned int)v176;
        v86 = (unsigned int)v176 + 1LL;
        if ( v86 > HIDWORD(v176) )
        {
          sub_C8D5F0((__int64)&v175, v177, v86, 8u, v67, v68);
          v85 = (unsigned int)v176;
        }
        v175[v85] = v87;
        LODWORD(v176) = v176 + 1;
      }
      v150 += 8;
    }
    while ( v143 != v150 );
    v131 = 1;
    v6 = v135;
  }
  sub_B82610(&v157);
LABEL_2:
  if ( (_BYTE)qword_50048C8 )
  {
    v9 = (unsigned int)v176;
    goto LABEL_4;
  }
  v54 = *(_QWORD *)(a1 + 80);
  if ( !v54 )
    BUG();
  v55 = *(_QWORD **)(v54 + 32);
  if ( v55 )
    v55 -= 3;
  while ( 1 )
  {
    v56 = *(unsigned __int8 *)v55;
    if ( (unsigned int)(v56 - 30) <= 0xA )
      break;
LABEL_99:
    if ( (_BYTE)v56 == 85 )
    {
      v113 = *(v55 - 4);
      if ( !v113 )
        goto LABEL_169;
      if ( *(_BYTE *)v113 )
        goto LABEL_169;
      if ( *(_QWORD *)(v113 + 24) != v55[10] )
        goto LABEL_169;
      if ( (*(_BYTE *)(v113 + 33) & 0x20) == 0 )
        goto LABEL_169;
      v114 = *(_DWORD *)(v113 + 36);
      if ( v114 == 151 || (unsigned int)(v114 - 156) <= 1 )
        goto LABEL_169;
LABEL_103:
      v58 = v55[4];
      if ( v58 )
        goto LABEL_104;
LABEL_173:
      v55 = 0;
    }
    else
    {
      if ( (_BYTE)v56 == 40 || (_BYTE)v56 == 34 )
        goto LABEL_169;
      if ( (unsigned int)(v56 - 30) > 0xA )
        goto LABEL_103;
      v58 = *(_QWORD *)(sub_AA5780(v55[5]) + 56);
      if ( !v58 )
        goto LABEL_173;
LABEL_104:
      v55 = (_QWORD *)(v58 - 24);
    }
  }
  v57 = sub_AA5780(v55[5]);
  if ( v57 && sub_AA5510(v57) )
  {
    v56 = *(unsigned __int8 *)v55;
    goto LABEL_99;
  }
LABEL_169:
  v115 = (unsigned int)v176;
  v116 = (unsigned int)v176 + 1LL;
  if ( v116 > HIDWORD(v176) )
  {
    sub_C8D5F0((__int64)&v175, v177, v116, 8u, v7, v8);
    v115 = (unsigned int)v176;
  }
  v131 = 1;
  v175[v115] = (__int64)v55;
  v9 = (unsigned int)(v176 + 1);
  LODWORD(v176) = v176 + 1;
LABEL_4:
  v134 = &v175[v9];
  if ( v175 == v134 )
    goto LABEL_56;
  v148 = v175;
  v133 = 0;
  dest = 0;
  v132 = 0;
  while ( 2 )
  {
    v10 = 0;
    v11 = *v148;
    src = 0;
    v152 = 0;
    v12 = *(_QWORD *)(v11 + 40);
    v153 = 0;
    v13 = sub_B43CA0(v11);
    v14 = sub_BA8CB0(v13, (__int64)"gc.safepoint_poll", 0x11u);
    LOWORD(v182) = 257;
    if ( v14 )
      v10 = *((_QWORD *)v14 + 3);
    v140 = (__int64)v14;
    v15 = sub_BD2C40(88, 1u);
    v16 = v15;
    if ( v15 )
    {
      sub_B4A410((__int64)v15, v10, v140, (__int64)&v178, 1u, 0, v11 + 24, 0);
      v17 = v16 + 3;
    }
    else
    {
      v17 = 0;
    }
    if ( *(_QWORD **)(v12 + 56) == v17 )
    {
      n = (size_t)v17;
      v18 = 1;
    }
    else
    {
      v18 = 0;
      n = *v17 & 0xFFFFFFFFFFFFFFF8LL;
    }
    v19 = v17[1];
    v191[64] = 1;
    v183 = (unsigned __int64)v185;
    v184 = 0x400000000LL;
    v189 = v191;
    v186 = v188;
    v178 = 0;
    v179 = 0;
    v180 = 0;
    v181 = 0;
    v182 = 0;
    v187 = 0x800000000LL;
    v190 = 0x800000000LL;
    sub_29F2700(v16, &v178, 0, 0, 1, 0);
    v154 = 0;
    v155 = 0;
    v156 = 0;
    v160 = 0;
    v161 = 0;
    v162 = 0;
    v163 = 0;
    if ( v18 )
      v20 = *(_QWORD *)(v12 + 56);
    else
      v20 = *(_QWORD *)(n + 8);
    if ( v19 )
      v19 -= 24;
    if ( !v20 )
    {
      v157 = 0;
      v158 = 0;
      v159 = 0;
      BUG();
    }
    v157 = 0;
    na = v20 - 24;
    v158 = 0;
    v159 = 0;
    v21 = *(_QWORD *)(v20 + 16);
    v160 = 1;
    sub_CF28B0((__int64)&v160, 0);
    if ( !(_DWORD)v163 )
    {
      LODWORD(v162) = v162 + 1;
      BUG();
    }
    v22 = 0;
    v23 = 1;
    v24 = (v163 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
    v25 = (_QWORD *)(v161 + 8LL * v24);
    v26 = *v25;
    if ( *v25 != v21 )
    {
      while ( v26 != -4096 )
      {
        if ( !v22 && v26 == -8192 )
          v22 = v25;
        v24 = (v163 - 1) & (v23 + v24);
        v25 = (_QWORD *)(v161 + 8LL * v24);
        v26 = *v25;
        if ( v21 == *v25 )
          goto LABEL_19;
        ++v23;
      }
      if ( v22 )
        v25 = v22;
    }
LABEL_19:
    LODWORD(v162) = v162 + 1;
    if ( *v25 != -4096 )
      --HIDWORD(v162);
    *v25 = v21;
    sub_28E74C0(na, v19, (__int64)&v154, (__int64)&v160, (__int64)&v157);
    v27 = v158;
    if ( v157 == v158 )
    {
      v30 = v158;
    }
    else
    {
      do
      {
        v28 = *(_QWORD *)(v27 - 8);
        v158 = v27 - 8;
        v29 = *(_QWORD *)(v28 + 56);
        if ( v29 )
          v29 -= 24;
        sub_28E74C0(v29, v19, (__int64)&v154, (__int64)&v160, (__int64)&v157);
        v27 = v158;
        v30 = v157;
      }
      while ( v158 != v157 );
    }
    if ( v30 )
      j_j___libc_free_0(v30);
    v31 = v155;
    for ( i = v154; v31 != i; ++i )
    {
      v33 = *i;
      if ( sub_28E5B80(*i, v6) )
      {
        v157 = v33;
        v34 = v152;
        if ( v152 == v153 )
        {
          sub_2445670((__int64)&src, v152, &v157);
        }
        else
        {
          if ( v152 )
          {
            *(_QWORD *)v152 = v33;
            v34 = v152;
          }
          v152 = v34 + 8;
        }
      }
    }
    sub_C7D6A0(v161, 8LL * (unsigned int)v163, 8);
    if ( v154 )
      j_j___libc_free_0((unsigned __int64)v154);
    if ( v189 != v191 )
      _libc_free((unsigned __int64)v189);
    v35 = v186;
    v36 = &v186[24 * (unsigned int)v187];
    if ( v186 != (_BYTE *)v36 )
    {
      do
      {
        v37 = *(v36 - 1);
        v36 -= 3;
        if ( v37 != 0 && v37 != -4096 && v37 != -8192 )
          sub_BD60C0(v36);
      }
      while ( v35 != v36 );
      v36 = v186;
    }
    if ( v36 != (_QWORD *)v188 )
      _libc_free((unsigned __int64)v36);
    if ( (_BYTE *)v183 != v185 )
      _libc_free(v183);
    v38 = src;
    if ( v152 != src )
    {
      v39 = v152 - (_BYTE *)src;
      if ( v133 - (unsigned __int64)dest >= v152 - (_BYTE *)src )
      {
        memmove(dest, src, v152 - (_BYTE *)src);
        dest += v39;
        goto LABEL_51;
      }
      v45 = dest - v132;
      v46 = v39 >> 3;
      v47 = (dest - v132) >> 3;
      if ( v39 >> 3 > (unsigned __int64)(0xFFFFFFFFFFFFFFFLL - v47) )
        sub_4262D8((__int64)"vector::_M_range_insert");
      if ( v45 >= v39 )
        v46 = (dest - v132) >> 3;
      v48 = __CFADD__(v47, v46);
      v49 = v47 + v46;
      if ( v48 )
      {
        v50 = 0x7FFFFFFFFFFFFFF8LL;
        goto LABEL_81;
      }
      if ( v49 )
      {
        if ( v49 > 0xFFFFFFFFFFFFFFFLL )
          v49 = 0xFFFFFFFFFFFFFFFLL;
        v50 = 8 * v49;
LABEL_81:
        v51 = sub_22077B0(v50);
        v45 = dest - v132;
        v52 = (char *)v51;
        v53 = v51 + v50;
      }
      else
      {
        v53 = 0;
        v52 = 0;
      }
      if ( v132 != dest )
      {
        nb = v45;
        memmove(v52, v132, v45);
        dest = (char *)memcpy(&v52[nb], v38, v39) + v39;
        goto LABEL_84;
      }
      dest = (char *)memcpy(&v52[v45], v38, v39) + v39;
      if ( v132 )
LABEL_84:
        j_j___libc_free_0((unsigned __int64)v132);
      v133 = v53;
      v38 = src;
      v132 = v52;
    }
    if ( v38 )
LABEL_51:
      j_j___libc_free_0((unsigned __int64)v38);
    if ( v134 != ++v148 )
      continue;
    break;
  }
  if ( v132 )
    j_j___libc_free_0((unsigned __int64)v132);
  v134 = v175;
LABEL_56:
  if ( v134 != (__int64 *)v177 )
    _libc_free((unsigned __int64)v134);
  v40 = v167;
  v41 = &v167[8 * (unsigned int)v168];
  if ( v167 != v41 )
  {
    do
    {
      v42 = *((_QWORD *)v41 - 1);
      v41 -= 8;
      if ( v42 )
      {
        v43 = *(_QWORD *)(v42 + 24);
        if ( v43 != v42 + 40 )
          _libc_free(v43);
        j_j___libc_free_0(v42);
      }
    }
    while ( v40 != v41 );
    v41 = v167;
  }
  if ( v41 != v169 )
    _libc_free((unsigned __int64)v41);
  if ( (char *)v165[0] != &v166 )
    _libc_free(v165[0]);
  return v131;
}
