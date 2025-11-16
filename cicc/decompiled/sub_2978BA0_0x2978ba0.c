// Function: sub_2978BA0
// Address: 0x2978ba0
//
__int64 __fastcall sub_2978BA0(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4, __int64 a5, char a6)
{
  __int64 v6; // rax
  __int64 v8; // rax
  unsigned int v9; // esi
  unsigned __int8 *v10; // r14
  __int64 v11; // r15
  __int64 v12; // rcx
  int v13; // r10d
  unsigned __int8 **v14; // rdx
  unsigned int v15; // r8d
  unsigned __int8 **v16; // rax
  unsigned __int8 *v17; // rdi
  unsigned __int8 **v18; // rax
  unsigned __int8 *v19; // rbx
  unsigned int v20; // ecx
  int v21; // eax
  unsigned __int8 *v22; // rdi
  int v23; // r12d
  unsigned __int8 **v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  void *v28; // rsi
  unsigned __int8 **v29; // rdx
  unsigned __int8 **v30; // rax
  unsigned __int8 **v32; // rbx
  __m128i v33; // xmm0
  __m128i v34; // xmm1
  __m128i v35; // xmm2
  __int64 *v36; // rax
  __int64 *v37; // rax
  unsigned __int8 **v38; // rax
  unsigned __int8 *v39; // rsi
  unsigned __int64 v40; // rax
  __int64 v41; // rdi
  unsigned __int64 v42; // r12
  __int64 v43; // rax
  unsigned __int8 **v44; // rax
  unsigned __int8 **v45; // r13
  unsigned __int8 *v46; // rsi
  unsigned __int8 **v47; // r12
  __int64 v48; // r13
  __int64 v49; // r14
  __int64 v50; // rbx
  __int64 v51; // r12
  __int64 v52; // rcx
  __int64 v53; // rdx
  unsigned int v54; // eax
  unsigned int v55; // esi
  __int64 v56; // rdi
  __int64 v57; // rdx
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // rcx
  __int64 v62; // r12
  __int64 v63; // rax
  unsigned __int64 v64; // rax
  __int64 v65; // rdi
  __int64 v66; // rax
  __int64 v67; // r13
  int v68; // edx
  __int64 v69; // rcx
  int v70; // edx
  unsigned int v71; // esi
  __int64 *v72; // rax
  __int64 v73; // rdi
  __int64 v74; // r8
  bool v75; // r9
  __int64 v76; // rdi
  unsigned int v77; // esi
  __int64 *v78; // rax
  __int64 v79; // r11
  bool v80; // al
  __int64 v81; // rdx
  unsigned int v82; // eax
  __int64 *v83; // rax
  unsigned __int8 **v84; // rax
  unsigned __int8 **v85; // r14
  unsigned __int8 **v86; // r8
  unsigned int v87; // r12d
  int v88; // r10d
  unsigned __int8 *v89; // rsi
  unsigned __int8 **v90; // rax
  unsigned int v91; // eax
  __int64 v92; // r13
  __int64 v93; // rax
  __int64 v94; // rdi
  __int64 v95; // rax
  __int64 v96; // rax
  __int64 v97; // rsi
  __int64 v98; // rax
  __int64 v99; // rdx
  __int64 v100; // rcx
  int v101; // ecx
  unsigned __int64 v102; // rcx
  __int64 v103; // rsi
  __int64 v104; // r13
  __int64 v105; // rax
  __int64 v106; // rax
  __int64 v107; // r8
  __int64 v108; // r9
  __int64 v109; // rax
  unsigned __int64 v110; // rdx
  __int64 v111; // rsi
  _QWORD *v112; // rax
  char *v113; // rdx
  __int64 v114; // rax
  __int64 v115; // rax
  __int64 *v116; // r14
  __int64 *v117; // r12
  __int64 *v118; // r15
  _QWORD *v119; // rdx
  __int64 v120; // rcx
  __int64 v121; // r8
  __int64 v122; // r9
  __int64 *v123; // rdi
  __int64 *v124; // rax
  __int64 v125; // rax
  unsigned int v126; // eax
  unsigned int v127; // eax
  _QWORD *v128; // rax
  unsigned __int8 *v129; // rax
  unsigned __int8 v130; // dl
  unsigned __int8 v131; // al
  __int64 v132; // rdx
  int v133; // eax
  int v134; // eax
  __int64 v135; // rcx
  unsigned __int64 v136; // rax
  int v137; // edx
  unsigned __int64 v138; // rax
  int v139; // r11d
  unsigned __int8 **v140; // r9
  unsigned __int8 **v141; // rax
  int v142; // r8d
  int v143; // r10d
  __int64 v144; // [rsp+20h] [rbp-520h]
  char v148; // [rsp+3Bh] [rbp-505h]
  int v149; // [rsp+3Ch] [rbp-504h]
  unsigned __int8 *v151; // [rsp+50h] [rbp-4F0h]
  _QWORD *v153; // [rsp+80h] [rbp-4C0h]
  int v154; // [rsp+88h] [rbp-4B8h]
  unsigned __int8 v155; // [rsp+8Eh] [rbp-4B2h]
  char v156; // [rsp+8Fh] [rbp-4B1h]
  unsigned __int8 v157; // [rsp+8Fh] [rbp-4B1h]
  __int64 v158; // [rsp+90h] [rbp-4B0h]
  __int64 v159; // [rsp+98h] [rbp-4A8h]
  unsigned __int64 v160; // [rsp+A0h] [rbp-4A0h]
  _QWORD *v161; // [rsp+A8h] [rbp-498h]
  _QWORD v162[4]; // [rsp+B0h] [rbp-490h] BYREF
  __int64 v163; // [rsp+D0h] [rbp-470h] BYREF
  __int64 v164; // [rsp+D8h] [rbp-468h]
  __int64 v165; // [rsp+E0h] [rbp-460h]
  unsigned int v166; // [rsp+E8h] [rbp-458h]
  _BYTE *v167; // [rsp+F0h] [rbp-450h] BYREF
  __int64 v168; // [rsp+F8h] [rbp-448h]
  _BYTE v169[32]; // [rsp+100h] [rbp-440h] BYREF
  __m128i v170; // [rsp+120h] [rbp-420h] BYREF
  __m128i v171; // [rsp+130h] [rbp-410h] BYREF
  __m128i v172; // [rsp+140h] [rbp-400h] BYREF
  __m128i v173[3]; // [rsp+150h] [rbp-3F0h] BYREF
  char v174; // [rsp+180h] [rbp-3C0h]
  __int64 v175; // [rsp+190h] [rbp-3B0h] BYREF
  void *s; // [rsp+198h] [rbp-3A8h]
  _BYTE v177[12]; // [rsp+1A0h] [rbp-3A0h]
  char v178; // [rsp+1ACh] [rbp-394h]
  char v179; // [rsp+1B0h] [rbp-390h] BYREF
  __int64 v180; // [rsp+1D0h] [rbp-370h] BYREF
  void *v181; // [rsp+1D8h] [rbp-368h]
  _BYTE v182[12]; // [rsp+1E0h] [rbp-360h]
  char v183; // [rsp+1ECh] [rbp-354h]
  char v184; // [rsp+1F0h] [rbp-350h] BYREF
  __int64 v185; // [rsp+210h] [rbp-330h] BYREF
  unsigned __int8 **v186; // [rsp+218h] [rbp-328h]
  __int64 v187; // [rsp+220h] [rbp-320h]
  int v188; // [rsp+228h] [rbp-318h]
  char v189; // [rsp+22Ch] [rbp-314h]
  char v190; // [rsp+230h] [rbp-310h] BYREF
  _QWORD v191[2]; // [rsp+270h] [rbp-2D0h] BYREF
  __int64 v192; // [rsp+280h] [rbp-2C0h]
  __int64 v193; // [rsp+288h] [rbp-2B8h] BYREF
  unsigned int v194; // [rsp+290h] [rbp-2B0h]
  _QWORD v195[2]; // [rsp+3C8h] [rbp-178h] BYREF
  char v196; // [rsp+3D8h] [rbp-168h]
  _BYTE *v197; // [rsp+3E0h] [rbp-160h]
  __int64 v198; // [rsp+3E8h] [rbp-158h]
  _BYTE v199[128]; // [rsp+3F0h] [rbp-150h] BYREF
  __int16 v200; // [rsp+470h] [rbp-D0h]
  _QWORD v201[2]; // [rsp+478h] [rbp-C8h] BYREF
  __int64 v202; // [rsp+488h] [rbp-B8h]
  __int64 v203; // [rsp+490h] [rbp-B0h] BYREF
  unsigned int v204; // [rsp+498h] [rbp-A8h]
  char v205; // [rsp+510h] [rbp-30h] BYREF

  v6 = (unsigned int)(*(_DWORD *)(a1 + 44) + 1);
  if ( (unsigned int)v6 >= *(_DWORD *)(a2 + 32) || !*(_QWORD *)(*(_QWORD *)(a2 + 24) + 8 * v6) )
    return 0;
  v8 = *(_QWORD *)(a1 + 48);
  v9 = 0;
  v10 = 0;
  v11 = a2;
  v162[0] = &v167;
  v12 = 0;
  v161 = (_QWORD *)(v8 & 0xFFFFFFFFFFFFFFF8LL);
  v186 = (unsigned __int8 **)&v190;
  v167 = v169;
  v168 = 0x400000000LL;
  s = &v179;
  v181 = &v184;
  v185 = 0;
  v187 = 8;
  v188 = 0;
  v189 = 1;
  v175 = 0;
  *(_QWORD *)v177 = 4;
  *(_DWORD *)&v177[8] = 0;
  v178 = 1;
  v180 = 0;
  *(_QWORD *)v182 = 4;
  *(_DWORD *)&v182[8] = 0;
  v183 = 1;
  v163 = 0;
  v164 = 0;
  v165 = 0;
  v166 = 0;
  v162[1] = &v163;
  v162[2] = a1;
  v148 = 0;
  v154 = 0;
  v149 = 0;
  v155 = 0;
  while ( 2 )
  {
    v19 = (unsigned __int8 *)(v161 - 3);
    if ( !v161 )
      v19 = 0;
    if ( !v9 )
    {
      ++v163;
      goto LABEL_13;
    }
    v13 = 1;
    v14 = 0;
    v15 = (v9 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
    v16 = (unsigned __int8 **)(v12 + 16LL * v15);
    v17 = *v16;
    if ( v19 != *v16 )
    {
      while ( v17 != (unsigned __int8 *)-4096LL )
      {
        if ( v17 != (unsigned __int8 *)-8192LL || v14 )
          v16 = v14;
        v15 = (v9 - 1) & (v13 + v15);
        v17 = *(unsigned __int8 **)(v12 + 16LL * v15);
        if ( v19 == v17 )
        {
          v16 = (unsigned __int8 **)(v12 + 16LL * v15);
          goto LABEL_5;
        }
        ++v13;
        v14 = v16;
        v16 = (unsigned __int8 **)(v12 + 16LL * v15);
      }
      if ( !v14 )
        v14 = v16;
      ++v163;
      v21 = v165 + 1;
      if ( 4 * ((int)v165 + 1) < 3 * v9 )
      {
        if ( v9 - (v21 + HIDWORD(v165)) <= v9 >> 3 )
        {
          sub_2978470((__int64)&v163, v9);
          if ( !v166 )
          {
LABEL_328:
            LODWORD(v165) = v165 + 1;
            BUG();
          }
          v86 = 0;
          v87 = (v166 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
          v88 = 1;
          v21 = v165 + 1;
          v14 = (unsigned __int8 **)(v164 + 16LL * v87);
          v89 = *v14;
          if ( v19 != *v14 )
          {
            while ( v89 != (unsigned __int8 *)-4096LL )
            {
              if ( !v86 && v89 == (unsigned __int8 *)-8192LL )
                v86 = v14;
              v87 = (v166 - 1) & (v88 + v87);
              v14 = (unsigned __int8 **)(v164 + 16LL * v87);
              v89 = *v14;
              if ( v19 == *v14 )
                goto LABEL_15;
              ++v88;
            }
            if ( v86 )
              v14 = v86;
          }
        }
        goto LABEL_15;
      }
LABEL_13:
      sub_2978470((__int64)&v163, 2 * v9);
      if ( !v166 )
        goto LABEL_328;
      v20 = (v166 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
      v21 = v165 + 1;
      v14 = (unsigned __int8 **)(v164 + 16LL * v20);
      v22 = *v14;
      if ( v19 != *v14 )
      {
        v139 = 1;
        v140 = 0;
        while ( v22 != (unsigned __int8 *)-4096LL )
        {
          if ( !v140 && v22 == (unsigned __int8 *)-8192LL )
            v140 = v14;
          v20 = (v166 - 1) & (v139 + v20);
          v14 = (unsigned __int8 **)(v164 + 16LL * v20);
          v22 = *v14;
          if ( v19 == *v14 )
            goto LABEL_15;
          ++v139;
        }
        if ( v140 )
          v14 = v140;
      }
LABEL_15:
      LODWORD(v165) = v21;
      if ( *v14 != (unsigned __int8 *)-4096LL )
        --HIDWORD(v165);
      *v14 = v19;
      v18 = v14 + 1;
      v14[1] = 0;
      goto LABEL_6;
    }
LABEL_5:
    v18 = v16 + 1;
LABEL_6:
    *v18 = v10;
    v153 = *(_QWORD **)(a1 + 56);
    if ( v161 == v153 )
    {
      if ( sub_B46AA0((__int64)v19) )
        break;
      v160 = (unsigned __int64)v161;
    }
    else
    {
      v160 = *v161 & 0xFFFFFFFFFFFFFFF8LL;
      if ( sub_B46AA0((__int64)v19) )
        goto LABEL_8;
    }
    v23 = *v19;
    if ( (_BYTE)v23 == 60 )
    {
      if ( sub_B4D040((__int64)v19) )
        goto LABEL_26;
      v23 = *v19;
    }
    if ( (unsigned __int8)sub_B46490((__int64)v19) )
    {
      v148 |= (_BYTE)v23 == 85;
      if ( v189 )
      {
        v90 = v186;
        v25 = HIDWORD(v187);
        v24 = &v186[HIDWORD(v187)];
        if ( v186 != v24 )
        {
          do
          {
            if ( v19 == *v90 )
              goto LABEL_26;
            ++v90;
          }
          while ( v24 != v90 );
        }
        if ( HIDWORD(v187) < (unsigned int)v187 )
        {
          ++HIDWORD(v187);
          *v24 = v19;
          ++v185;
          goto LABEL_26;
        }
      }
      sub_C8CC70((__int64)&v185, (__int64)v19, (__int64)v24, v25, v26, v27);
      goto LABEL_26;
    }
    if ( (_BYTE)v23 != 61 )
      goto LABEL_62;
    if ( v148 )
      goto LABEL_26;
    sub_D665A0(&v170, (__int64)v19);
    v84 = v186;
    if ( v189 )
      v85 = &v186[HIDWORD(v187)];
    else
      v85 = &v186[(unsigned int)v187];
    if ( v186 == v85 )
    {
LABEL_61:
      v23 = *v19;
      goto LABEL_62;
    }
    do
    {
      v39 = *v84;
      if ( (unsigned __int64)*v84 < 0xFFFFFFFFFFFFFFFELL )
      {
        if ( v85 == v84 )
          goto LABEL_61;
        v151 = v19;
        v32 = v84;
        while ( 1 )
        {
          v33 = _mm_loadu_si128(&v170);
          v34 = _mm_loadu_si128(&v171);
          v174 = 1;
          v35 = _mm_loadu_si128(&v172);
          v191[0] = a4;
          v36 = &v193;
          v173[0] = v33;
          v191[1] = 0;
          v192 = 1;
          v173[1] = v34;
          v173[2] = v35;
          do
          {
            *v36 = -4;
            v36 += 5;
            *(v36 - 4) = -3;
            *(v36 - 3) = -4;
            *(v36 - 2) = -3;
          }
          while ( v36 != v195 );
          v195[0] = v201;
          v195[1] = 0;
          v197 = v199;
          v198 = 0x400000000LL;
          v196 = 0;
          v200 = 256;
          v201[1] = 0;
          v202 = 1;
          v201[0] = &unk_49DDBE8;
          v37 = &v203;
          do
          {
            *v37 = -4096;
            v37 += 2;
          }
          while ( v37 != (__int64 *)&v205 );
          v156 = sub_CF63E0(a4, v39, v173, (__int64)v191);
          v201[0] = &unk_49DDBE8;
          if ( (v202 & 1) == 0 )
            sub_C7D6A0(v203, 16LL * v204, 8);
          nullsub_184();
          if ( v197 != v199 )
            _libc_free((unsigned __int64)v197);
          if ( (v192 & 1) == 0 )
            sub_C7D6A0(v193, 40LL * v194, 8);
          if ( (v156 & 2) != 0 )
            break;
          v38 = v32 + 1;
          if ( v32 + 1 != v85 )
          {
            while ( 1 )
            {
              v39 = *v38;
              v32 = v38;
              if ( (unsigned __int64)*v38 < 0xFFFFFFFFFFFFFFFELL )
                break;
              if ( v85 == ++v38 )
                goto LABEL_60;
            }
            if ( v85 != v38 )
              continue;
          }
LABEL_60:
          v19 = v151;
          goto LABEL_61;
        }
        v19 = v151;
        goto LABEL_26;
      }
      ++v84;
    }
    while ( v85 != v84 );
    v23 = *v19;
LABEL_62:
    if ( (unsigned int)(unsigned __int8)v23 - 30 <= 0xA )
      goto LABEL_26;
    if ( (_BYTE)v23 == 84 )
      goto LABEL_26;
    v40 = (unsigned int)(unsigned __int8)v23 - 39;
    if ( (unsigned int)v40 <= 0x38 )
    {
      v41 = 0x100060000000001LL;
      if ( _bittest64(&v41, v40) )
        goto LABEL_26;
    }
    if ( (unsigned __int8)sub_B46790(v19, 0) )
      goto LABEL_26;
    v157 = sub_B46900(v19);
    if ( !v157 )
      goto LABEL_26;
    v42 = (unsigned int)(v23 - 34);
    if ( (unsigned __int8)v42 > 0x33u )
      goto LABEL_77;
    v43 = 0x8000000000041LL;
    if ( !_bittest64(&v43, v42) )
      goto LABEL_77;
    if ( (unsigned __int8)sub_A73ED0((_QWORD *)v19 + 9, 6) || (unsigned __int8)sub_B49560((__int64)v19, 6) )
      goto LABEL_26;
    v44 = v186;
    if ( v189 )
      v45 = &v186[HIDWORD(v187)];
    else
      v45 = &v186[(unsigned int)v187];
    if ( v186 != v45 )
    {
      while ( 1 )
      {
        v46 = *v44;
        v47 = v44;
        if ( (unsigned __int64)*v44 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v45 == ++v44 )
          goto LABEL_77;
      }
      if ( v45 != v44 )
      {
        while ( 1 )
        {
          if ( (sub_CF5B00(a4, v46, v19) & 2) != 0 )
            goto LABEL_26;
          v141 = v47 + 1;
          if ( v47 + 1 == v45 )
            goto LABEL_77;
          v46 = *v141;
          ++v47;
          if ( (unsigned __int64)*v141 >= 0xFFFFFFFFFFFFFFFELL )
            break;
LABEL_298:
          if ( v45 == v47 )
            goto LABEL_77;
        }
        while ( v45 != ++v141 )
        {
          v46 = *v141;
          v47 = v141;
          if ( (unsigned __int64)*v141 < 0xFFFFFFFFFFFFFFFELL )
            goto LABEL_298;
        }
      }
    }
LABEL_77:
    if ( sub_CEA920((__int64)v19) )
      goto LABEL_26;
    if ( *v19 != 43 )
      goto LABEL_79;
    if ( (v19[7] & 0x40) != 0 )
      v129 = (unsigned __int8 *)*((_QWORD *)v19 - 1);
    else
      v129 = &v19[-32 * (*((_DWORD *)v19 + 1) & 0x7FFFFFF)];
    v130 = **(_BYTE **)v129;
    v131 = **((_BYTE **)v129 + 4);
    if ( v130 <= 0x1Cu )
    {
      if ( v131 > 0x1Cu )
      {
LABEL_248:
        if ( v131 == 47 )
          goto LABEL_26;
      }
    }
    else
    {
      if ( v131 > 0x1Cu )
      {
        if ( v130 != 47 )
          goto LABEL_248;
LABEL_26:
        if ( !v183 )
          goto LABEL_122;
        goto LABEL_27;
      }
      if ( v130 == 47 )
        goto LABEL_26;
    }
LABEL_79:
    v48 = *((_QWORD *)v19 + 2);
    v49 = *((_QWORD *)v19 + 5);
    if ( !v48 )
      goto LABEL_26;
    v159 = (__int64)v19;
    v50 = 0;
    do
    {
      v51 = *(_QWORD *)(v48 + 24);
      if ( *(_BYTE *)v51 == 84 )
      {
        v52 = *(_QWORD *)(*(_QWORD *)(v51 - 8) + 32LL * *(unsigned int *)(v51 + 72)
                                               + 8LL * (unsigned int)sub_BD2910(v48));
        if ( !v52 )
          goto LABEL_163;
      }
      else
      {
        v52 = *(_QWORD *)(v51 + 40);
        if ( !v52 )
        {
LABEL_163:
          v53 = 0;
          v54 = 0;
          goto LABEL_84;
        }
      }
      v53 = (unsigned int)(*(_DWORD *)(v52 + 44) + 1);
      v54 = *(_DWORD *)(v52 + 44) + 1;
LABEL_84:
      v55 = *(_DWORD *)(v11 + 32);
      if ( v54 >= v55 )
        goto LABEL_98;
      v56 = *(_QWORD *)(v11 + 24);
      v57 = *(_QWORD *)(v56 + 8 * v53);
      if ( !v57 )
        goto LABEL_98;
      if ( v50 )
      {
        v58 = *(_QWORD *)(*(_QWORD *)(v50 + 72) + 80LL);
        if ( v58 )
          v58 -= 24;
        if ( v58 != v50 && v52 != v58 )
        {
          v59 = (unsigned int)(*(_DWORD *)(v50 + 44) + 1);
          if ( v55 <= (unsigned int)v59 )
          {
            v60 = 0;
          }
          else
          {
            v60 = *(_QWORD *)(v56 + 8 * v59);
            if ( v57 == v60 )
            {
LABEL_96:
              v50 = *(_QWORD *)v57;
              goto LABEL_97;
            }
          }
          do
          {
            if ( *(_DWORD *)(v60 + 16) < *(_DWORD *)(v57 + 16) )
            {
              v61 = v60;
              v60 = v57;
              v57 = v61;
            }
            v60 = *(_QWORD *)(v60 + 8);
          }
          while ( v60 != v57 );
          goto LABEL_96;
        }
        v50 = v58;
      }
      else
      {
        v50 = v52;
      }
LABEL_97:
      if ( !(unsigned __int8)sub_B19720(v11, v49, v50) )
      {
        v19 = (unsigned __int8 *)v159;
        goto LABEL_26;
      }
LABEL_98:
      v48 = *(_QWORD *)(v48 + 8);
    }
    while ( v48 );
    v62 = v50;
    v19 = (unsigned __int8 *)v159;
    if ( !v62 || v49 == v62 )
      goto LABEL_26;
    while ( 2 )
    {
      v63 = sub_AA4FF0(v62);
      if ( !v63 )
        BUG();
      v64 = (unsigned int)*(unsigned __int8 *)(v63 - 24) - 39;
      if ( (unsigned int)v64 <= 0x38 )
      {
        v65 = 0x100060000000001LL;
        if ( _bittest64(&v65, v64) )
          goto LABEL_117;
      }
      v66 = sub_AA5510(v62);
      v67 = *(_QWORD *)(v159 + 40);
      if ( v67 == v66 )
        goto LABEL_166;
      if ( !(unsigned __int8)sub_B46420(v159) )
      {
LABEL_109:
        if ( !(unsigned __int8)sub_B19720(v11, v67, v62) )
          goto LABEL_117;
        v68 = *(_DWORD *)(a3 + 24);
        v69 = *(_QWORD *)(a3 + 8);
        if ( v68 )
        {
          v70 = v68 - 1;
          v71 = v70 & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
          v72 = (__int64 *)(v69 + 16LL * v71);
          v73 = *v72;
          if ( *v72 == v62 )
          {
LABEL_112:
            v74 = v72[1];
            v75 = v74 != 0;
          }
          else
          {
            v134 = 1;
            while ( v73 != -4096 )
            {
              v142 = v134 + 1;
              v71 = v70 & (v134 + v71);
              v72 = (__int64 *)(v69 + 16LL * v71);
              v73 = *v72;
              if ( *v72 == v62 )
                goto LABEL_112;
              v134 = v142;
            }
            v75 = 0;
            v74 = 0;
          }
          v76 = *(_QWORD *)(v159 + 40);
          v77 = v70 & (((unsigned int)v76 >> 9) ^ ((unsigned int)v76 >> 4));
          v78 = (__int64 *)(v69 + 16LL * v77);
          v79 = *v78;
          if ( *v78 == v76 )
          {
LABEL_114:
            v80 = v78[1] != v74;
          }
          else
          {
            v133 = 1;
            while ( v79 != -4096 )
            {
              v143 = v133 + 1;
              v77 = v70 & (v133 + v77);
              v78 = (__int64 *)(v69 + 16LL * v77);
              v79 = *v78;
              if ( *v78 == v76 )
                goto LABEL_114;
              v133 = v143;
            }
            v80 = v75;
          }
          if ( v80 && v75 )
            goto LABEL_117;
        }
LABEL_166:
        v92 = 0;
        if ( !v62 )
          goto LABEL_26;
        if ( !byte_5006F68 )
          goto LABEL_171;
        v93 = *(_QWORD *)(v159 + 16);
        if ( v93 )
        {
          if ( *(_QWORD *)(v93 + 8) && *(_BYTE *)v159 != 93 && *(_BYTE *)v159 != 63 )
          {
            do
            {
              v132 = *(_QWORD *)(v93 + 24);
              if ( *(_BYTE *)v132 > 0x1Cu )
              {
                if ( v92 )
                {
                  if ( *(_QWORD *)(v132 + 40) != v92 )
                    goto LABEL_26;
                }
                else
                {
                  v92 = *(_QWORD *)(v132 + 40);
                }
              }
              v93 = *(_QWORD *)(v93 + 8);
            }
            while ( v93 );
            goto LABEL_185;
          }
LABEL_171:
          if ( !byte_5007048 || *(_BYTE *)v159 != 63 )
            goto LABEL_185;
          v94 = *(_QWORD *)(v159 + 16);
          if ( v94 )
          {
            while ( 1 )
            {
              v95 = *(_QWORD *)(v94 + 24);
              if ( *(_BYTE *)v95 > 0x1Cu )
              {
                v96 = *(_QWORD *)(v95 + 40);
                v97 = v96 + 48;
                v98 = *(_QWORD *)(v96 + 56);
                if ( v97 != v98 )
                  break;
              }
LABEL_256:
              v94 = *(_QWORD *)(v94 + 8);
              if ( !v94 )
                goto LABEL_26;
            }
            v99 = v98;
            v100 = 0;
            do
            {
              v99 = *(_QWORD *)(v99 + 8);
              ++v100;
            }
            while ( v97 != v99 );
            if ( v100 <= 10 )
            {
              while ( 1 )
              {
                if ( !v98 )
                  BUG();
                v101 = *(unsigned __int8 *)(v98 - 24);
                if ( (((_BYTE)v101 - 61) & 0xDE) != 0 )
                {
                  v102 = (unsigned int)(v101 - 30);
                  if ( (unsigned __int8)v102 > 0x37u )
                    break;
                  v103 = 0xD10000000007FFLL;
                  if ( !_bittest64(&v103, v102) )
                    break;
                }
                v98 = *(_QWORD *)(v98 + 8);
                if ( v99 == v98 )
                  goto LABEL_256;
              }
            }
            goto LABEL_185;
          }
          goto LABEL_26;
        }
        if ( *(_BYTE *)v159 == 63 || *(_BYTE *)v159 == 93 )
          goto LABEL_171;
LABEL_185:
        v104 = sub_AA5190(v62);
        if ( v104 )
          v104 -= 24;
        if ( !(unsigned __int8)sub_B46420(v159) )
        {
          v105 = *(_QWORD *)(v159 + 16);
          if ( v105 )
          {
            v135 = *(_QWORD *)(v105 + 8);
            if ( !v135 )
            {
              v104 = *(_QWORD *)(v105 + 24);
              if ( *(_BYTE *)v104 <= 0x1Cu )
                BUG();
              if ( *(_QWORD *)(v104 + 40) != v62 )
              {
                v136 = *(_QWORD *)(v62 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                if ( v136 == v62 + 48 )
                {
                  v104 = 0;
                }
                else
                {
                  if ( !v136 )
                    BUG();
                  v137 = *(unsigned __int8 *)(v136 - 24);
                  v138 = v136 - 24;
                  if ( (unsigned int)(v137 - 30) < 0xB )
                    v135 = v138;
                  v104 = v135;
                }
              }
            }
          }
        }
        v106 = v144;
        LOWORD(v106) = 0;
        v144 = v106;
        sub_B444E0((_QWORD *)v159, v104 + 24, v106);
        v109 = (unsigned int)v168;
        v110 = (unsigned int)v168 + 1LL;
        if ( v110 > HIDWORD(v168) )
        {
          sub_C8D5F0((__int64)&v167, v169, v110, 8u, v107, v108);
          v109 = (unsigned int)v168;
        }
        v111 = v159;
        *(_QWORD *)&v167[8 * v109] = v159;
        LODWORD(v168) = v168 + 1;
        if ( (unsigned __int8)sub_2978150(a5, v159, a1) )
          goto LABEL_197;
        if ( v178 )
        {
          v112 = s;
          v113 = (char *)s + 8 * *(unsigned int *)&v177[4];
          if ( s == v113 )
            goto LABEL_198;
          while ( v159 != *v112 )
          {
            if ( v113 == (char *)++v112 )
              goto LABEL_198;
          }
LABEL_197:
          v111 = *(_QWORD *)(*(_QWORD *)a5 + 40LL) + 312LL;
          v114 = sub_FCD870(v159, v111);
          v149 -= v114;
          v154 -= HIDWORD(v114);
          goto LABEL_198;
        }
        v111 = v159;
        if ( sub_C8CA60((__int64)&v175, v159) )
          goto LABEL_197;
LABEL_198:
        v115 = 4LL * (*(_DWORD *)(v159 + 4) & 0x7FFFFFF);
        if ( (*(_BYTE *)(v159 + 7) & 0x40) != 0 )
        {
          v116 = *(__int64 **)(v159 - 8);
          v117 = &v116[v115];
        }
        else
        {
          v117 = (__int64 *)v159;
          v116 = (__int64 *)(v159 - v115 * 8);
        }
        if ( v117 != v116 )
        {
          v158 = v11;
          v118 = v116;
          while ( 1 )
          {
            if ( (unsigned __int8)sub_2978150(a5, *v118, a1) )
              goto LABEL_202;
            v121 = *v118;
            if ( !v178 )
              break;
            v119 = s;
            v123 = (__int64 *)((char *)s + 8 * *(unsigned int *)&v177[4]);
            v122 = *(unsigned int *)&v177[4];
            v124 = (__int64 *)s;
            if ( s == v123 )
              goto LABEL_254;
            while ( 1 )
            {
              v111 = *v124;
              if ( v121 == *v124 )
                break;
              if ( v123 == ++v124 )
              {
                if ( v121 == *(_QWORD *)s )
                  goto LABEL_214;
                goto LABEL_212;
              }
            }
LABEL_203:
            if ( *(_BYTE *)v111 > 0x1Cu && a1 == *(_QWORD *)(v111 + 40) )
            {
              if ( !v183 )
                goto LABEL_250;
              v128 = v181;
              v119 = (char *)v181 + 8 * *(unsigned int *)&v182[4];
              if ( v181 != v119 )
              {
                while ( v111 != *v128 )
                {
                  if ( v119 == ++v128 )
                    goto LABEL_237;
                }
                goto LABEL_205;
              }
LABEL_237:
              if ( *(_DWORD *)&v182[4] < *(_DWORD *)v182 )
              {
                ++*(_DWORD *)&v182[4];
                *v119 = v111;
                ++v180;
              }
              else
              {
LABEL_250:
                sub_C8CC70((__int64)&v180, v111, (__int64)v119, v120, v121, v122);
              }
            }
LABEL_205:
            v118 += 4;
            if ( v117 == v118 )
            {
              v11 = v158;
              goto LABEL_218;
            }
          }
          if ( sub_C8CA60((__int64)&v175, v121) )
          {
LABEL_202:
            v111 = *v118;
            goto LABEL_203;
          }
          v121 = *v118;
          if ( !v178 )
            goto LABEL_241;
          v119 = s;
          v122 = *(unsigned int *)&v177[4];
          v124 = (__int64 *)((char *)s + 8 * *(unsigned int *)&v177[4]);
          if ( v124 != s )
          {
            while ( v121 != *v119 )
            {
LABEL_212:
              if ( v124 == ++v119 )
                goto LABEL_254;
            }
            goto LABEL_214;
          }
LABEL_254:
          if ( (unsigned int)v122 < *(_DWORD *)v177 )
          {
            *(_DWORD *)&v177[4] = v122 + 1;
            *v124 = v121;
            ++v175;
          }
          else
          {
LABEL_241:
            sub_C8CC70((__int64)&v175, v121, (__int64)v119, v120, v121, v122);
          }
LABEL_214:
          v125 = sub_FCD870(*v118, *(_QWORD *)(*(_QWORD *)a5 + 40LL) + 312LL);
          v149 += v125;
          v154 += HIDWORD(v125);
          goto LABEL_202;
        }
LABEL_218:
        if ( a6 && (v154 > 0 || v149 > 0) )
          goto LABEL_26;
        ++v180;
        LODWORD(v168) = 0;
        if ( !v183 )
        {
          v126 = 4 * (*(_DWORD *)&v182[4] - *(_DWORD *)&v182[8]);
          if ( v126 < 0x20 )
            v126 = 32;
          if ( v126 < *(_DWORD *)v182 )
          {
            sub_C8C990((__int64)&v180, v111);
LABEL_227:
            ++v175;
            if ( !v178 )
            {
              v127 = 4 * (*(_DWORD *)&v177[4] - *(_DWORD *)&v177[8]);
              if ( v127 < 0x20 )
                v127 = 32;
              if ( *(_DWORD *)v177 > v127 )
              {
                sub_C8C990((__int64)&v175, v111);
                v154 = 0;
                v149 = 0;
                v155 = v157;
                goto LABEL_34;
              }
              memset(s, -1, 8LL * *(unsigned int *)v177);
            }
            *(_QWORD *)&v177[4] = 0;
            v154 = 0;
            v155 = v157;
            v149 = 0;
            goto LABEL_34;
          }
          v111 = 0xFFFFFFFFLL;
          memset(v181, -1, 8LL * *(unsigned int *)v182);
        }
        *(_QWORD *)&v182[4] = 0;
        goto LABEL_227;
      }
      if ( (*(_BYTE *)(v159 + 7) & 0x20) != 0 && sub_B91C10(v159, 6) )
      {
        v67 = *(_QWORD *)(v159 + 40);
        goto LABEL_109;
      }
LABEL_117:
      if ( v62 )
      {
        v81 = (unsigned int)(*(_DWORD *)(v62 + 44) + 1);
        v82 = *(_DWORD *)(v62 + 44) + 1;
      }
      else
      {
        v81 = 0;
        v82 = 0;
      }
      if ( v82 >= *(_DWORD *)(v11 + 32) )
        BUG();
      v62 = **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v11 + 24) + 8 * v81) + 8LL);
      if ( v49 != v62 )
        continue;
      break;
    }
    if ( !v183 )
    {
LABEL_122:
      v28 = v19;
      v83 = sub_C8CA60((__int64)&v180, (__int64)v19);
      if ( v83 )
      {
        *v83 = -2;
        ++*(_DWORD *)&v182[8];
        ++v180;
      }
      goto LABEL_32;
    }
LABEL_27:
    v28 = v181;
    v29 = (unsigned __int8 **)((char *)v181 + 8 * *(unsigned int *)&v182[4]);
    v30 = (unsigned __int8 **)v181;
    if ( v181 != v29 )
    {
      while ( v19 != *v30 )
      {
        if ( v29 == ++v30 )
          goto LABEL_32;
      }
      --*(_DWORD *)&v182[4];
      *v30 = (unsigned __int8 *)*((_QWORD *)v181 + *(unsigned int *)&v182[4]);
      ++v180;
    }
LABEL_32:
    if ( (_DWORD)v168 && *(_DWORD *)&v182[4] == *(_DWORD *)&v182[8] )
    {
      sub_2978650((__int64)v162);
      ++v175;
      LODWORD(v168) = 0;
      if ( !v178 )
      {
        v91 = 4 * (*(_DWORD *)&v177[4] - *(_DWORD *)&v177[8]);
        if ( v91 < 0x20 )
          v91 = 32;
        if ( *(_DWORD *)v177 > v91 )
        {
          sub_C8C990((__int64)&v175, (__int64)v28);
          v154 = 0;
          v149 = 0;
          goto LABEL_34;
        }
        memset(s, -1, 8LL * *(unsigned int *)v177);
      }
      *(_QWORD *)&v177[4] = 0;
      v154 = 0;
      v149 = 0;
    }
LABEL_34:
    if ( v161 != v153 )
    {
LABEL_8:
      v12 = v164;
      v10 = v19;
      v9 = v166;
      v161 = (_QWORD *)v160;
      continue;
    }
    break;
  }
  sub_2978650((__int64)v162);
  sub_C7D6A0(v164, 16LL * v166, 8);
  if ( !v183 )
    _libc_free((unsigned __int64)v181);
  if ( !v178 )
    _libc_free((unsigned __int64)s);
  if ( v167 != v169 )
    _libc_free((unsigned __int64)v167);
  if ( !v189 )
    _libc_free((unsigned __int64)v186);
  return v155;
}
