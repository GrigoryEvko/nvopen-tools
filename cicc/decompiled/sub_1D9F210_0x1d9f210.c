// Function: sub_1D9F210
// Address: 0x1d9f210
//
__int64 __fastcall sub_1D9F210(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // rax
  __int64 v11; // r12
  __int64 *v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rdi
  __int64 (*v17)(); // rax
  __int64 v18; // rdi
  __int64 (*v19)(); // rax
  __int64 v20; // rdi
  __int64 (*v21)(void); // rdx
  int v22; // eax
  __int64 v23; // r12
  __int64 v24; // r13
  __int64 k; // r15
  __int64 v26; // r14
  char v27; // al
  __int64 v28; // r13
  __int64 v29; // rax
  int v30; // r8d
  int v31; // r9d
  _QWORD *v32; // r12
  unsigned __int8 v33; // al
  __int64 v34; // rax
  __int64 v35; // rax
  unsigned int v36; // r12d
  _QWORD *v37; // rax
  unsigned __int64 v38; // rdi
  unsigned int v39; // r8d
  __int64 v40; // r13
  int v41; // ebx
  int v42; // ecx
  __int64 v43; // rax
  unsigned int v44; // edx
  int v45; // eax
  _BYTE *v46; // rdi
  _QWORD **v47; // r12
  _QWORD **v48; // rbx
  _QWORD *v49; // rdi
  __int64 v50; // rdi
  __int64 v51; // rax
  __int64 v52; // rbx
  unsigned int v53; // r12d
  unsigned int v54; // eax
  int v55; // ebx
  unsigned int v56; // r13d
  unsigned int v57; // eax
  unsigned int v58; // r11d
  __int64 v59; // r12
  int v60; // r14d
  int v61; // esi
  int v62; // r9d
  int v63; // r8d
  int v64; // edx
  int v65; // edi
  int v66; // edx
  unsigned int v67; // edx
  unsigned __int64 v68; // rbx
  __int64 v69; // rax
  unsigned __int64 v70; // rax
  __int64 v71; // rdx
  int v72; // r12d
  _QWORD *v73; // rax
  int v74; // r8d
  int v75; // r9d
  int v76; // ecx
  __int64 v77; // rax
  unsigned int v78; // edx
  int v79; // eax
  _DWORD *v80; // r8
  __int64 v81; // rax
  int v82; // edx
  __int64 v83; // r12
  __int64 v84; // rax
  unsigned int v85; // edx
  __int64 *v86; // rax
  __int64 v87; // rbx
  __int64 *v88; // r15
  __int64 v89; // r13
  _QWORD *v90; // r10
  unsigned int v91; // eax
  unsigned __int8 *v92; // rdi
  __int64 v93; // r14
  unsigned int v94; // r9d
  __int64 *v95; // rax
  __int64 v96; // r8
  unsigned int v97; // esi
  __int64 v98; // rcx
  unsigned int v99; // eax
  unsigned int v100; // edx
  __int64 v101; // rsi
  int v102; // ecx
  _QWORD *v103; // rax
  __int64 v104; // rdi
  __int64 (*v105)(); // rax
  unsigned __int8 v106; // bl
  int v107; // r8d
  int v108; // r9d
  __int64 *v109; // r12
  __int64 *v110; // r13
  __int64 v111; // rax
  __int64 v112; // rbx
  __int64 v113; // rdi
  __int64 (*v114)(); // rax
  unsigned __int8 v115; // bl
  int v116; // r8d
  int v117; // r9d
  __int64 v118; // rax
  __int64 v119; // rax
  __int64 *v120; // rdx
  int v121; // ecx
  int v122; // edx
  int v123; // r8d
  unsigned int i; // esi
  __int64 *v125; // rcx
  __int64 v126; // rdx
  __int64 *v127; // rcx
  int v128; // edi
  unsigned int j; // edx
  __int64 v130; // r8
  unsigned int v131; // esi
  unsigned int v132; // edx
  _QWORD *v133; // rax
  __int64 v134; // rdx
  _QWORD *v135; // rbx
  _QWORD *v136; // r12
  __int64 v137; // r15
  int v138; // r14d
  unsigned __int8 *v139; // rsi
  __int64 v140; // rsi
  __int64 v141; // rax
  __int64 v142; // rax
  _QWORD *v143; // r14
  double v144; // xmm4_8
  double v145; // xmm5_8
  _QWORD *v146; // rax
  unsigned __int64 *v147; // r15
  __int64 v148; // rax
  unsigned __int64 v149; // rcx
  __int64 v150; // rsi
  __int64 v151; // rdx
  unsigned __int8 *v152; // rsi
  int v153; // edx
  __int64 *v154; // [rsp+8h] [rbp-318h]
  unsigned int v155; // [rsp+10h] [rbp-310h]
  __int64 *v156; // [rsp+18h] [rbp-308h]
  __int64 *v157; // [rsp+20h] [rbp-300h]
  __int64 *v158; // [rsp+30h] [rbp-2F0h]
  __int64 v159; // [rsp+38h] [rbp-2E8h]
  int v160; // [rsp+38h] [rbp-2E8h]
  int v161; // [rsp+38h] [rbp-2E8h]
  __int64 v162; // [rsp+38h] [rbp-2E8h]
  __int64 v163; // [rsp+38h] [rbp-2E8h]
  __int64 v164; // [rsp+40h] [rbp-2E0h]
  __int64 *v165; // [rsp+48h] [rbp-2D8h]
  __int64 v166; // [rsp+48h] [rbp-2D8h]
  unsigned int v167; // [rsp+50h] [rbp-2D0h]
  unsigned int v168; // [rsp+60h] [rbp-2C0h]
  __int64 v169; // [rsp+60h] [rbp-2C0h]
  unsigned int v170; // [rsp+60h] [rbp-2C0h]
  _QWORD *v171; // [rsp+68h] [rbp-2B8h]
  int v172; // [rsp+78h] [rbp-2A8h]
  unsigned __int8 v173; // [rsp+7Fh] [rbp-2A1h]
  unsigned __int8 v174; // [rsp+7Fh] [rbp-2A1h]
  __int64 v175; // [rsp+80h] [rbp-2A0h]
  __int64 v177; // [rsp+90h] [rbp-290h]
  __int64 v178; // [rsp+98h] [rbp-288h]
  __int64 v179; // [rsp+A8h] [rbp-278h] BYREF
  __int64 v180; // [rsp+B0h] [rbp-270h] BYREF
  __int16 v181; // [rsp+C0h] [rbp-260h]
  unsigned __int8 *v182[2]; // [rsp+D0h] [rbp-250h] BYREF
  __int16 v183; // [rsp+E0h] [rbp-240h]
  _DWORD *v184; // [rsp+F0h] [rbp-230h] BYREF
  __int64 v185; // [rsp+F8h] [rbp-228h]
  _DWORD v186[4]; // [rsp+100h] [rbp-220h] BYREF
  __int64 v187; // [rsp+110h] [rbp-210h] BYREF
  _QWORD *v188; // [rsp+118h] [rbp-208h]
  __int64 v189; // [rsp+120h] [rbp-200h]
  unsigned int v190; // [rsp+128h] [rbp-1F8h]
  __int64 *v191; // [rsp+130h] [rbp-1F0h] BYREF
  __int64 v192; // [rsp+138h] [rbp-1E8h]
  _BYTE v193[32]; // [rsp+140h] [rbp-1E0h] BYREF
  __int64 *v194; // [rsp+160h] [rbp-1C0h] BYREF
  __int64 v195; // [rsp+168h] [rbp-1B8h]
  _BYTE v196[32]; // [rsp+170h] [rbp-1B0h] BYREF
  unsigned __int8 *v197; // [rsp+190h] [rbp-190h] BYREF
  __int64 v198; // [rsp+198h] [rbp-188h]
  unsigned __int64 *v199; // [rsp+1A0h] [rbp-180h] BYREF
  _QWORD *v200; // [rsp+1A8h] [rbp-178h]
  __int64 v201; // [rsp+1B0h] [rbp-170h]
  int v202; // [rsp+1B8h] [rbp-168h]
  __int64 v203; // [rsp+1C0h] [rbp-160h]
  __int64 v204; // [rsp+1C8h] [rbp-158h]
  _BYTE *v205; // [rsp+1E0h] [rbp-140h] BYREF
  __int64 v206; // [rsp+1E8h] [rbp-138h]
  _BYTE v207[304]; // [rsp+1F0h] [rbp-130h] BYREF

  v10 = sub_160F9A0(*(_QWORD *)(a1 + 8), (__int64)&unk_4FCBA30, 1u);
  if ( !v10 )
    return 0;
  v11 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v10 + 104LL))(v10, &unk_4FCBA30);
  if ( !v11 || !byte_4FC4320 )
    return 0;
  v13 = *(__int64 **)(a1 + 8);
  v14 = *v13;
  v15 = v13[1];
  if ( v14 == v15 )
LABEL_278:
    BUG();
  while ( *(_UNKNOWN **)v14 != &unk_4F9E06C )
  {
    v14 += 16;
    if ( v15 == v14 )
      goto LABEL_278;
  }
  *(_QWORD *)(a1 + 160) = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v14 + 8) + 104LL))(
                            *(_QWORD *)(v14 + 8),
                            &unk_4F9E06C)
                        + 160;
  v16 = *(_QWORD *)(v11 + 208);
  v17 = *(__int64 (**)())(*(_QWORD *)v16 + 16LL);
  if ( v17 == sub_16FF750 )
    BUG();
  v18 = ((__int64 (__fastcall *)(__int64, __int64))v17)(v16, a2);
  v19 = *(__int64 (**)())(*(_QWORD *)v18 + 56LL);
  if ( v19 == sub_1D12D20 )
  {
    *(_QWORD *)(a1 + 168) = 0;
    BUG();
  }
  v20 = ((__int64 (__fastcall *)(__int64))v19)(v18);
  *(_QWORD *)(a1 + 168) = v20;
  v21 = *(__int64 (**)(void))(*(_QWORD *)v20 + 848LL);
  v22 = 2;
  if ( v21 != sub_1D9ED50 )
    v22 = v21();
  v23 = a2 + 72;
  *(_DWORD *)(a1 + 176) = v22;
  v24 = *(_QWORD *)(a2 + 80);
  v205 = v207;
  v206 = 0x2000000000LL;
  if ( a2 + 72 == v24 )
    return 0;
  if ( !v24 )
    BUG();
  while ( *(_QWORD *)(v24 + 24) == v24 + 16 )
  {
    v24 = *(_QWORD *)(v24 + 8);
    if ( v23 == v24 )
      return 0;
    if ( !v24 )
      BUG();
  }
  if ( v24 == v23 )
    return 0;
  v173 = 0;
  k = *(_QWORD *)(v24 + 24);
  v26 = v24;
  v178 = a2 + 72;
  do
  {
    if ( !k )
      BUG();
    v27 = *(_BYTE *)(k - 8);
    v177 = k - 24;
    if ( v27 != 54 )
      goto LABEL_41;
    if ( sub_15F32D0(k - 24) || (*(_BYTE *)(k - 6) & 1) != 0 )
      goto LABEL_40;
    v191 = (__int64 *)v193;
    v192 = 0x400000000LL;
    v194 = (__int64 *)v196;
    v195 = 0x400000000LL;
    v28 = *(_QWORD *)(k - 16);
    if ( !v28 )
      goto LABEL_38;
    do
    {
      while ( 1 )
      {
        v32 = sub_1648700(v28);
        v33 = *((_BYTE *)v32 + 16);
        if ( v33 <= 0x17u )
          goto LABEL_36;
        if ( v33 != 83 )
          break;
        if ( *(_BYTE *)(*(v32 - 3) + 16LL) != 13 )
          goto LABEL_36;
        v35 = (unsigned int)v195;
        if ( (unsigned int)v195 >= HIDWORD(v195) )
        {
          sub_16CD150((__int64)&v194, v196, 0, 8, v30, v31);
          v35 = (unsigned int)v195;
        }
        v194[v35] = (__int64)v32;
        LODWORD(v195) = v195 + 1;
        v28 = *(_QWORD *)(v28 + 8);
        if ( !v28 )
          goto LABEL_54;
      }
      if ( v33 != 85 || *(_BYTE *)(*(v32 - 6) + 16LL) != 9 )
        goto LABEL_36;
      v29 = (unsigned int)v192;
      if ( (unsigned int)v192 >= HIDWORD(v192) )
      {
        sub_16CD150((__int64)&v191, v193, 0, 8, v30, v31);
        v29 = (unsigned int)v192;
      }
      v191[v29] = (__int64)v32;
      LODWORD(v192) = v192 + 1;
      v28 = *(_QWORD *)(v28 + 8);
    }
    while ( v28 );
LABEL_54:
    if ( !(_DWORD)v192 )
      goto LABEL_36;
    v36 = *(_DWORD *)(a1 + 176);
    v37 = (_QWORD *)*v191;
    v197 = (unsigned __int8 *)&v199;
    v198 = 0x1000000000LL;
    sub_15FAA20((unsigned __int8 *)*(v37 - 3), (__int64)&v197);
    v38 = (unsigned __int64)v197;
    if ( (unsigned int)v198 <= 1uLL || v36 <= 1 )
    {
LABEL_106:
      if ( v197 != (unsigned __int8 *)&v199 )
        goto LABEL_107;
      goto LABEL_36;
    }
    v39 = 3;
    v40 = 2;
LABEL_58:
    if ( !(_DWORD)v40 )
      goto LABEL_67;
    v41 = 0;
    while ( 1 )
    {
      v42 = v41;
      v43 = 0;
      v44 = 0;
      do
      {
        v45 = *(_DWORD *)&v197[4 * v43];
        if ( v45 >= 0 && v45 != v42 )
          goto LABEL_65;
        v42 += v40;
        v43 = ++v44;
      }
      while ( (unsigned int)v198 > (unsigned __int64)v44 );
      if ( (unsigned int)v198 == (unsigned __int64)v44 )
        break;
LABEL_65:
      if ( ++v41 == (_DWORD)v40 )
      {
        if ( v36 < v39 )
          goto LABEL_106;
LABEL_67:
        v40 = (unsigned int)(v40 + 1);
        ++v39;
        goto LABEL_58;
      }
    }
    if ( v197 != (unsigned __int8 *)&v199 )
      _libc_free((unsigned __int64)v197);
    v186[0] = v41;
    v68 = (unsigned __int64)v191;
    v184 = v186;
    v185 = 0x400000001LL;
    v169 = *(_QWORD *)*v191;
    v69 = (unsigned int)v192;
    if ( (unsigned int)v192 > 1 )
    {
      v70 = (unsigned __int64)v191;
      v71 = 1;
      v72 = 1;
      while ( 1 )
      {
        v73 = *(_QWORD **)(v70 + 8 * v71);
        if ( v169 != *v73 )
        {
          v80 = v184;
          goto LABEL_123;
        }
        v197 = (unsigned __int8 *)&v199;
        v198 = 0x1000000000LL;
        sub_15FAA20((unsigned __int8 *)*(v73 - 3), (__int64)&v197);
        v75 = 0;
        while ( (_DWORD)v198 )
        {
          v76 = v75;
          v77 = 0;
          v78 = 0;
          while ( 1 )
          {
            v79 = *(_DWORD *)&v197[4 * v77];
            if ( v79 >= 0 && v79 != v76 )
              break;
            v76 += v40;
            v77 = ++v78;
            if ( (unsigned int)v198 <= (unsigned __int64)v78 )
            {
              if ( (unsigned int)v198 == (unsigned __int64)v78 )
                goto LABEL_125;
              break;
            }
          }
          if ( ++v75 >= (unsigned int)v40 )
          {
            if ( v197 == (unsigned __int8 *)&v199 )
              goto LABEL_139;
            _libc_free((unsigned __int64)v197);
            v80 = v184;
            goto LABEL_123;
          }
        }
LABEL_125:
        if ( v197 != (unsigned __int8 *)&v199 )
        {
          v161 = v75;
          _libc_free((unsigned __int64)v197);
          v75 = v161;
        }
        v81 = (unsigned int)v185;
        if ( (unsigned int)v185 >= HIDWORD(v185) )
        {
          v160 = v75;
          sub_16CD150((__int64)&v184, v186, 0, 4, v74, v75);
          v81 = (unsigned int)v185;
          v75 = v160;
        }
        v184[v81] = v75;
        v69 = (unsigned int)v192;
        v71 = (unsigned int)(v72 + 1);
        LODWORD(v185) = v185 + 1;
        v72 = v71;
        if ( (unsigned int)v192 <= (unsigned int)v71 )
          break;
        v70 = (unsigned __int64)v191;
      }
      v40 = (unsigned int)v40;
      v68 = (unsigned __int64)v191;
    }
    if ( !(_DWORD)v195 )
      goto LABEL_169;
    v187 = 0;
    v188 = 0;
    v157 = v194;
    v189 = 0;
    v190 = 0;
    v156 = &v194[(unsigned int)v195];
    v158 = (__int64 *)(v68 + 8 * v69);
    v165 = v194;
    v164 = k;
    v154 = (__int64 *)v68;
    v162 = v26;
    v155 = v40;
    do
    {
      v83 = *v165;
      v84 = *(_QWORD *)(*v165 - 24);
      v85 = *(_DWORD *)(v84 + 32);
      v86 = *(__int64 **)(v84 + 24);
      if ( v85 > 0x40 )
        v87 = *v86;
      else
        v87 = (__int64)((_QWORD)v86 << (64 - (unsigned __int8)v85)) >> (64 - (unsigned __int8)v85);
      v88 = v154;
      if ( v158 == v154 )
      {
LABEL_149:
        v90 = v188;
        v91 = v190;
        goto LABEL_162;
      }
      v170 = ((unsigned int)v83 >> 9) ^ ((unsigned int)v83 >> 4);
      while ( 1 )
      {
        v89 = *v88;
        if ( sub_15CCEE0(*(_QWORD *)(a1 + 160), *v88, v83) )
          break;
LABEL_148:
        if ( v158 == ++v88 )
          goto LABEL_149;
      }
      v197 = (unsigned __int8 *)&v199;
      v198 = 0x400000000LL;
      sub_15FAA20(*(unsigned __int8 **)(v89 - 24), (__int64)&v197);
      v92 = v197;
      if ( (_DWORD)v198 )
      {
        v93 = 0;
        while ( *(_DWORD *)&v197[4 * v93] != v87 )
        {
          if ( ++v93 == (unsigned int)v198 )
            goto LABEL_158;
        }
        if ( v190 )
        {
          v94 = (v190 - 1) & v170;
          v95 = &v188[3 * v94];
          v96 = *v95;
          if ( v83 == *v95 )
          {
LABEL_157:
            v95[1] = v89;
            *((_DWORD *)v95 + 4) = v93;
            goto LABEL_158;
          }
          v120 = 0;
          v121 = 1;
          while ( v96 != -8 )
          {
            if ( !v120 && v96 == -16 )
              v120 = v95;
            v94 = (v190 - 1) & (v121 + v94);
            v95 = &v188[3 * v94];
            v96 = *v95;
            if ( v83 == *v95 )
              goto LABEL_157;
            ++v121;
          }
          if ( v120 )
            v95 = v120;
          ++v187;
          v122 = v189 + 1;
          if ( 4 * ((int)v189 + 1) < 3 * v190 )
          {
            if ( v190 - HIDWORD(v189) - v122 <= v190 >> 3 )
            {
              sub_1D9F040((__int64)&v187, v190);
              if ( !v190 )
                goto LABEL_277;
              v95 = 0;
              v123 = 1;
              for ( i = (v190 - 1) & v170; ; i = (v190 - 1) & v131 )
              {
                v125 = &v188[3 * i];
                v126 = *v125;
                if ( v83 == *v125 )
                {
                  v122 = v189 + 1;
                  v95 = &v188[3 * i];
                  goto LABEL_199;
                }
                if ( v126 == -8 )
                  break;
                if ( v95 || v126 != -16 )
                  v125 = v95;
                v131 = v123 + i;
                v95 = v125;
                ++v123;
              }
              if ( !v95 )
                v95 = &v188[3 * i];
              v122 = v189 + 1;
            }
LABEL_199:
            LODWORD(v189) = v122;
            if ( *v95 != -8 )
              --HIDWORD(v189);
            *v95 = v83;
            v95[1] = 0;
            *((_DWORD *)v95 + 4) = 0;
            v92 = v197;
            goto LABEL_157;
          }
        }
        else
        {
          ++v187;
        }
        sub_1D9F040((__int64)&v187, 2 * v190);
        if ( !v190 )
        {
LABEL_277:
          LODWORD(v189) = v189 + 1;
          BUG();
        }
        v127 = 0;
        v128 = 1;
        for ( j = (v190 - 1) & v170; ; j = (v190 - 1) & v132 )
        {
          v95 = &v188[3 * j];
          v130 = *v95;
          if ( v83 == *v95 )
          {
            v122 = v189 + 1;
            goto LABEL_199;
          }
          if ( v130 == -8 )
            break;
          if ( v130 != -16 || v127 )
            v95 = v127;
          v132 = v128 + j;
          v127 = v95;
          ++v128;
        }
        if ( v127 )
          v95 = v127;
        v122 = v189 + 1;
        goto LABEL_199;
      }
LABEL_158:
      v91 = v190;
      if ( !v190 )
        goto LABEL_180;
      v90 = v188;
      v97 = (v190 - 1) & v170;
      v98 = v188[3 * v97];
      if ( v83 != v98 )
      {
        v153 = 1;
        while ( v98 != -8 )
        {
          v97 = (v190 - 1) & (v153 + v97);
          v98 = v188[3 * v97];
          if ( v83 == v98 )
            goto LABEL_160;
          ++v153;
        }
LABEL_180:
        if ( v92 != (unsigned __int8 *)&v199 )
          _libc_free((unsigned __int64)v92);
        goto LABEL_148;
      }
LABEL_160:
      if ( v92 != (unsigned __int8 *)&v199 )
      {
        _libc_free((unsigned __int64)v92);
        v90 = v188;
        v91 = v190;
      }
LABEL_162:
      if ( !v91 )
        goto LABEL_138;
      v99 = v91 - 1;
      v100 = v99 & (((unsigned int)v83 >> 9) ^ ((unsigned int)v83 >> 4));
      v101 = v90[3 * v100];
      v102 = 1;
      if ( v101 != v83 )
      {
        while ( v101 != -8 )
        {
          v100 = v99 & (v102 + v100);
          v101 = v90[3 * v100];
          if ( v83 == v101 )
            goto LABEL_164;
          ++v102;
        }
LABEL_138:
        k = v164;
        v26 = v162;
        j___libc_free_0(v90);
LABEL_139:
        v80 = v184;
        goto LABEL_123;
      }
LABEL_164:
      ++v165;
    }
    while ( v156 != v165 );
    k = v164;
    v26 = v162;
    v40 = v155;
    v103 = (_QWORD *)sub_16498A0(*v157);
    v197 = 0;
    v199 = 0;
    v200 = v103;
    v201 = 0;
    v202 = 0;
    v203 = 0;
    v204 = 0;
    v198 = 0;
    if ( (_DWORD)v189 )
    {
      v133 = v188;
      v134 = 3LL * v190;
      v171 = &v188[v134];
      if ( v188 != &v188[v134] )
      {
        while ( 1 )
        {
          v135 = v133;
          if ( *v133 != -8 && *v133 != -16 )
            break;
          v133 += 3;
          if ( v171 == v133 )
            goto LABEL_166;
        }
        if ( v133 != v171 )
        {
          v136 = (_QWORD *)*v133;
          v166 = v162;
          while ( 1 )
          {
            v137 = v135[1];
            v138 = *((_DWORD *)v135 + 4);
            v198 = v136[5];
            v199 = v136 + 3;
            v139 = (unsigned __int8 *)v136[6];
            v182[0] = v139;
            if ( v139 )
            {
              sub_1623A60((__int64)v182, (__int64)v139, 2);
              v140 = (__int64)v197;
              if ( !v197 )
                goto LABEL_242;
            }
            else
            {
              v140 = (__int64)v197;
              if ( !v197 )
                goto LABEL_244;
            }
            sub_161E7C0((__int64)&v197, v140);
LABEL_242:
            v197 = v182[0];
            if ( v182[0] )
              sub_1623210((__int64)v182, v182[0], (__int64)&v197);
LABEL_244:
            v181 = 257;
            v141 = sub_1643360(v200);
            v142 = sub_159C470(v141, v138, 0);
            if ( *(_BYTE *)(v137 + 16) > 0x10u || *(_BYTE *)(v142 + 16) > 0x10u )
            {
              v163 = v142;
              v183 = 257;
              v146 = sub_1648A60(56, 2u);
              v143 = v146;
              if ( v146 )
                sub_15FA320((__int64)v146, (_QWORD *)v137, v163, (__int64)v182, 0);
              if ( v198 )
              {
                v147 = v199;
                sub_157E9D0(v198 + 40, (__int64)v143);
                v148 = v143[3];
                v149 = *v147;
                v143[4] = v147;
                v149 &= 0xFFFFFFFFFFFFFFF8LL;
                v143[3] = v149 | v148 & 7;
                *(_QWORD *)(v149 + 8) = v143 + 3;
                *v147 = *v147 & 7 | (unsigned __int64)(v143 + 3);
              }
              sub_164B780((__int64)v143, &v180);
              if ( v197 )
              {
                v179 = (__int64)v197;
                sub_1623A60((__int64)&v179, (__int64)v197, 2);
                v150 = v143[6];
                v151 = (__int64)(v143 + 6);
                if ( v150 )
                {
                  sub_161E7C0((__int64)(v143 + 6), v150);
                  v151 = (__int64)(v143 + 6);
                }
                v152 = (unsigned __int8 *)v179;
                v143[6] = v179;
                if ( v152 )
                  sub_1623210((__int64)&v179, v152, v151);
              }
            }
            else
            {
              v143 = (_QWORD *)sub_15A37D0((_BYTE *)v137, v142, 0);
            }
            v135 += 3;
            sub_164D160((__int64)v136, (__int64)v143, a3, a4, a5, a6, v144, v145, a9, a10);
            sub_15F20C0(v136);
            if ( v135 != v171 )
            {
              while ( 1 )
              {
                v136 = (_QWORD *)*v135;
                if ( *v135 != -16 && v136 != (_QWORD *)-8LL )
                  break;
                v135 += 3;
                if ( v171 == v135 )
                  goto LABEL_251;
              }
              if ( v135 != v171 )
                continue;
            }
LABEL_251:
            k = v164;
            v26 = v166;
            v40 = v155;
            break;
          }
        }
      }
    }
LABEL_166:
    if ( v197 )
      sub_161E7C0((__int64)&v197, (__int64)v197);
    j___libc_free_0(v188);
    v68 = (unsigned __int64)v191;
LABEL_169:
    v80 = v184;
    v104 = *(_QWORD *)(a1 + 168);
    v105 = *(__int64 (**)())(*(_QWORD *)v104 + 856LL);
    if ( v105 != sub_1D9ED60 )
    {
      v106 = ((__int64 (__fastcall *)(__int64, __int64, unsigned __int64, _QWORD, _DWORD *, _QWORD, __int64))v105)(
               v104,
               v177,
               v68,
               (unsigned int)v192,
               v184,
               (unsigned int)v185,
               v40);
      if ( !v106 )
        goto LABEL_139;
      v109 = v191;
      v110 = &v191[(unsigned int)v192];
      v111 = (unsigned int)v206;
      if ( v191 != v110 )
      {
        v174 = v106;
        do
        {
          v112 = *v109;
          if ( (unsigned int)v111 >= HIDWORD(v206) )
          {
            sub_16CD150((__int64)&v205, v207, 0, 8, v107, v108);
            v111 = (unsigned int)v206;
          }
          ++v109;
          *(_QWORD *)&v205[8 * v111] = v112;
          v111 = (unsigned int)(v206 + 1);
          LODWORD(v206) = v206 + 1;
        }
        while ( v110 != v109 );
        v106 = v174;
      }
      if ( HIDWORD(v206) <= (unsigned int)v111 )
      {
        sub_16CD150((__int64)&v205, v207, 0, 8, v107, v108);
        v111 = (unsigned int)v206;
      }
      v173 = v106;
      *(_QWORD *)&v205[8 * v111] = v177;
      v80 = v184;
      LODWORD(v206) = v206 + 1;
    }
LABEL_123:
    if ( v80 != v186 )
    {
      v38 = (unsigned __int64)v80;
LABEL_107:
      _libc_free(v38);
    }
LABEL_36:
    if ( v194 != (__int64 *)v196 )
      _libc_free((unsigned __int64)v194);
LABEL_38:
    if ( v191 != (__int64 *)v193 )
      _libc_free((unsigned __int64)v191);
LABEL_40:
    v27 = *(_BYTE *)(k - 8);
LABEL_41:
    if ( v27 != 55 )
      goto LABEL_45;
    if ( sub_15F32D0(v177) )
      goto LABEL_45;
    if ( (*(_BYTE *)(k - 6) & 1) != 0 )
      goto LABEL_45;
    v34 = *(_QWORD *)(k - 72);
    v175 = v34;
    if ( *(_BYTE *)(v34 + 16) != 85 )
      goto LABEL_45;
    v50 = *(_QWORD *)(k - 72);
    v51 = *(_QWORD *)(v34 + 8);
    if ( !v51 || *(_QWORD *)(v51 + 8) )
      goto LABEL_45;
    v52 = *(_QWORD *)(**(_QWORD **)(v50 - 72) + 32LL);
    v53 = *(_DWORD *)(a1 + 176);
    v197 = (unsigned __int8 *)&v199;
    v198 = 0x1000000000LL;
    v167 = v53;
    sub_15FAA20(*(unsigned __int8 **)(v50 - 24), (__int64)&v197);
    if ( (unsigned int)v198 <= 3 || v53 <= 1 )
    {
LABEL_104:
      if ( v197 != (unsigned __int8 *)&v199 )
        _libc_free((unsigned __int64)v197);
      goto LABEL_45;
    }
    v54 = 2 * v52;
    v55 = v172;
    v56 = 2;
    v168 = v54;
    v159 = v26;
    while ( 1 )
    {
      v57 = (unsigned int)v198 / v56;
      if ( !((unsigned int)v198 % v56) && (unsigned int)v198 >= v56 )
      {
        v58 = v57 - 1;
        if ( ((v57 - 1) & v57) == 0 )
          break;
      }
LABEL_102:
      if ( v167 < ++v56 )
      {
        v172 = v55;
        v26 = v159;
        goto LABEL_104;
      }
    }
    v59 = 0;
    while ( 2 )
    {
      v60 = v59;
      if ( v57 != 1 )
      {
        v61 = v59;
        v62 = 0;
        v63 = 0;
        while ( 1 )
        {
          v64 = *(_DWORD *)&v197[4 * v61];
          v65 = *(_DWORD *)&v197[4 * v61 + 4 * v56];
          v61 += v56;
          if ( v65 < 0 )
          {
            if ( v64 < 0 )
              goto LABEL_98;
            ++v63;
            v55 = v64;
            v62 = 1;
            if ( v58 == v63 )
              goto LABEL_93;
          }
          else
          {
            if ( v64 >= 0 )
            {
              if ( v65 != v64 + 1 )
                goto LABEL_101;
              goto LABEL_87;
            }
LABEL_98:
            if ( v62 )
            {
              ++v62;
              if ( v65 >= 0 && v55 + v62 != v65 )
                goto LABEL_101;
            }
LABEL_87:
            if ( v58 == ++v63 )
              goto LABEL_93;
          }
        }
      }
      v62 = 0;
      v63 = 0;
LABEL_93:
      v66 = *(_DWORD *)&v197[4 * v59];
      if ( v66 < 0 )
      {
        v82 = *(_DWORD *)&v197[4 * v56 * v58 + 4 * (unsigned int)v59];
        if ( v82 < 0 )
        {
          if ( !v62 )
          {
            v67 = (unsigned int)v198 / v56;
            goto LABEL_95;
          }
          v66 = v62 + 1 - v57 + v55;
        }
        else
        {
          v66 = v82 - v63;
        }
        if ( v66 < 0 )
          break;
      }
      v67 = v57 + v66;
LABEL_95:
      if ( v67 <= v168 )
      {
        ++v59;
        ++v60;
        if ( v59 != v56 )
          continue;
      }
      break;
    }
LABEL_101:
    if ( v60 != v56 )
      goto LABEL_102;
    v172 = v55;
    v26 = v159;
    if ( v197 != (unsigned __int8 *)&v199 )
      _libc_free((unsigned __int64)v197);
    v113 = *(_QWORD *)(a1 + 168);
    v114 = *(__int64 (**)())(*(_QWORD *)v113 + 864LL);
    if ( v114 != sub_1D9ED70 )
    {
      v115 = ((__int64 (__fastcall *)(__int64, __int64, __int64, _QWORD))v114)(v113, v177, v175, v56);
      if ( v115 )
      {
        v118 = (unsigned int)v206;
        if ( (unsigned int)v206 >= HIDWORD(v206) )
        {
          sub_16CD150((__int64)&v205, v207, 0, 8, v116, v117);
          v118 = (unsigned int)v206;
        }
        *(_QWORD *)&v205[8 * v118] = v177;
        v119 = (unsigned int)(v206 + 1);
        LODWORD(v206) = v119;
        if ( HIDWORD(v206) <= (unsigned int)v119 )
        {
          sub_16CD150((__int64)&v205, v207, 0, 8, v116, v117);
          v119 = (unsigned int)v206;
        }
        v173 = v115;
        *(_QWORD *)&v205[8 * v119] = v175;
        LODWORD(v206) = v206 + 1;
      }
    }
LABEL_45:
    for ( k = *(_QWORD *)(k + 8); k == v26 - 24 + 40; k = *(_QWORD *)(v26 + 24) )
    {
      v26 = *(_QWORD *)(v26 + 8);
      if ( v178 == v26 )
        goto LABEL_69;
      if ( !v26 )
        BUG();
    }
  }
  while ( v178 != v26 );
LABEL_69:
  v46 = v205;
  v47 = (_QWORD **)&v205[8 * (unsigned int)v206];
  if ( v47 != (_QWORD **)v205 )
  {
    v48 = (_QWORD **)v205;
    do
    {
      v49 = *v48++;
      sub_15F20C0(v49);
    }
    while ( v47 != v48 );
    v46 = v205;
  }
  if ( v46 != v207 )
    _libc_free((unsigned __int64)v46);
  return v173;
}
