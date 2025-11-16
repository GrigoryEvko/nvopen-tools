// Function: sub_351A710
// Address: 0x351a710
//
__int64 __fastcall sub_351A710(__int64 a1, __int64 a2, __int64 ***a3, __int64 a4)
{
  __int64 v4; // r15
  __int64 v6; // r12
  unsigned int v8; // eax
  __int64 v9; // rdx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 *v14; // r13
  __int64 v15; // rsi
  __int64 ***v16; // rbx
  __int64 v17; // rdi
  int v18; // eax
  unsigned int v19; // eax
  __int64 *v20; // rsi
  __int64 v21; // rax
  __int64 *v22; // rbx
  __int64 *v23; // rax
  __int64 v24; // rdi
  unsigned int v25; // r13d
  unsigned int v26; // eax
  unsigned int v27; // r14d
  unsigned __int8 v28; // al
  __int64 v29; // rax
  int v30; // edx
  __int64 *v31; // rax
  __m128i *v32; // r14
  unsigned int v33; // r13d
  __int64 v34; // rax
  unsigned int *v35; // r9
  __int64 v36; // rbx
  __int64 v37; // r12
  __int64 v38; // rax
  int *v39; // r8
  __int64 v40; // rcx
  __int64 v41; // rbx
  unsigned int *v42; // r9
  __int64 v43; // rdx
  __int64 v44; // rax
  __m128i v45; // xmm0
  __int64 v46; // rax
  unsigned __int64 v47; // r8
  __int64 *v48; // rdi
  __int64 v49; // r14
  __int64 v50; // rbx
  unsigned __int8 v51; // al
  __int64 v52; // r14
  __int64 v53; // r15
  __int64 *v55; // r14
  char v56; // di
  __int64 *v57; // rbx
  __int64 v58; // rsi
  __int64 *v59; // rax
  __int64 v60; // rdx
  __int64 v61; // r14
  __int64 v62; // rax
  __int64 *v63; // r15
  __int64 v64; // rsi
  __int64 *v65; // rax
  __int64 *v66; // rdi
  __int64 v67; // rax
  __int64 *v68; // r13
  __int64 *v69; // r12
  __int64 v70; // rsi
  __int64 *v71; // rax
  char v72; // bl
  char v73; // al
  _QWORD *v74; // rdi
  _QWORD *v75; // rsi
  __int64 ***v76; // r12
  _QWORD *v77; // rdi
  _QWORD *v78; // rsi
  __int64 v79; // r8
  _QWORD *v80; // rax
  __int64 v81; // rax
  __int64 *v82; // r13
  __int64 *v83; // r14
  char v84; // di
  __int64 v85; // rsi
  __int64 *v86; // rax
  __int64 v87; // rdi
  int v88; // esi
  int v89; // r10d
  __int64 **v90; // r13
  __int64 *v91; // rbx
  _QWORD *v92; // rdi
  _QWORD *v93; // rsi
  __int64 *v94; // r8
  __int64 v95; // rsi
  unsigned __int64 v96; // rax
  __int64 v97; // r9
  unsigned __int64 v98; // rcx
  __int64 v99; // rax
  unsigned __int64 v100; // rdx
  __int64 *v101; // rcx
  const __m128i *v102; // rdx
  __m128i *v103; // rax
  int v104; // eax
  __int64 v105; // rdi
  int v106; // edx
  unsigned int v107; // eax
  __int64 v108; // rcx
  int v109; // r8d
  _BYTE *v110; // rdi
  __int64 v111; // rdx
  __int64 v112; // r13
  __int64 v113; // r14
  __int64 v114; // rcx
  __int64 v115; // rbx
  __int64 v116; // r8
  unsigned int v117; // esi
  int v118; // ecx
  __int64 v119; // r11
  unsigned int v120; // r15d
  __int64 v121; // rdx
  unsigned int v122; // r8d
  __int64 v123; // rax
  __int64 v124; // rdi
  __int64 v125; // rax
  unsigned __int64 v126; // r9
  __int64 v127; // r8
  unsigned __int64 v128; // r13
  __int64 *v129; // rax
  __int64 v130; // rax
  bool v131; // cf
  unsigned __int64 v132; // rax
  unsigned __int8 v133; // si
  unsigned __int8 v134; // si
  unsigned int v135; // eax
  __int64 v136; // rt1
  int v137; // esi
  int v138; // esi
  int v139; // edi
  __int64 v140; // rcx
  unsigned int j; // edx
  __int64 v142; // r9
  int v143; // edx
  int v144; // ecx
  int v145; // edi
  int v146; // edi
  __int64 v147; // r8
  unsigned int v148; // esi
  int i; // r9d
  __int64 *v150; // rcx
  __int64 v151; // rdx
  unsigned int v152; // edx
  unsigned int v153; // esi
  unsigned __int64 v154; // [rsp+8h] [rbp-2E8h]
  __int64 *v155; // [rsp+10h] [rbp-2E0h]
  __int64 v156; // [rsp+18h] [rbp-2D8h]
  __int64 v157; // [rsp+18h] [rbp-2D8h]
  __int64 *v158; // [rsp+20h] [rbp-2D0h]
  unsigned int v159; // [rsp+30h] [rbp-2C0h]
  const void *v160; // [rsp+30h] [rbp-2C0h]
  __int64 v161; // [rsp+38h] [rbp-2B8h]
  unsigned __int64 v162; // [rsp+38h] [rbp-2B8h]
  __int64 *v163; // [rsp+40h] [rbp-2B0h]
  __int64 v164; // [rsp+48h] [rbp-2A8h]
  __int64 *v165; // [rsp+48h] [rbp-2A8h]
  unsigned int v166; // [rsp+50h] [rbp-2A0h]
  __int64 v167; // [rsp+50h] [rbp-2A0h]
  __int64 v168; // [rsp+50h] [rbp-2A0h]
  __int64 *v169; // [rsp+50h] [rbp-2A0h]
  unsigned int v170; // [rsp+58h] [rbp-298h]
  unsigned int *v171; // [rsp+58h] [rbp-298h]
  int *v172; // [rsp+58h] [rbp-298h]
  __int64 *v173; // [rsp+58h] [rbp-298h]
  __int64 v174; // [rsp+58h] [rbp-298h]
  __int64 *v175; // [rsp+58h] [rbp-298h]
  __int64 v176; // [rsp+58h] [rbp-298h]
  unsigned int v177; // [rsp+58h] [rbp-298h]
  __int64 *v178; // [rsp+58h] [rbp-298h]
  int v181; // [rsp+7Ch] [rbp-274h] BYREF
  __int64 v182; // [rsp+80h] [rbp-270h] BYREF
  __int64 v183; // [rsp+88h] [rbp-268h] BYREF
  __int64 v184[4]; // [rsp+90h] [rbp-260h] BYREF
  __int64 *v185; // [rsp+B0h] [rbp-240h] BYREF
  __int64 v186; // [rsp+B8h] [rbp-238h]
  _BYTE v187[32]; // [rsp+C0h] [rbp-230h] BYREF
  __int64 v188; // [rsp+E0h] [rbp-210h] BYREF
  __int64 *v189; // [rsp+E8h] [rbp-208h]
  __int64 v190; // [rsp+F0h] [rbp-200h]
  int v191; // [rsp+F8h] [rbp-1F8h]
  char v192; // [rsp+FCh] [rbp-1F4h]
  _BYTE v193[32]; // [rsp+100h] [rbp-1F0h] BYREF
  __int64 *v194; // [rsp+120h] [rbp-1D0h] BYREF
  unsigned __int64 v195; // [rsp+128h] [rbp-1C8h]
  __int64 v196; // [rsp+130h] [rbp-1C0h] BYREF
  int v197; // [rsp+138h] [rbp-1B8h]
  char v198; // [rsp+13Ch] [rbp-1B4h]
  char v199; // [rsp+140h] [rbp-1B0h] BYREF
  _QWORD v200[2]; // [rsp+1F0h] [rbp-100h] BYREF
  _BYTE v201[192]; // [rsp+200h] [rbp-F0h] BYREF
  char v202; // [rsp+2C0h] [rbp-30h] BYREF

  v4 = a2;
  v6 = a1;
  sub_F02DB0(&v181, qword_501F308[8], 0x64u);
  v185 = (__int64 *)v187;
  v186 = 0x400000000LL;
  v8 = sub_3516000(a1, a2, a3, a4, (__int64)&v185);
  v12 = *(_QWORD *)(a1 + 496);
  v170 = v8;
  v13 = *(unsigned int *)(a1 + 512);
  if ( (_DWORD)v13 )
  {
    v9 = ((_DWORD)v13 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v14 = (__int64 *)(v12 + 24 * v9);
    v15 = *v14;
    if ( v4 == *v14 )
    {
LABEL_3:
      if ( v14 != (__int64 *)(v12 + 24 * v13) )
      {
        v194 = (__int64 *)v14[1];
        *v14 = -8192;
        --*(_DWORD *)(a1 + 504);
        ++*(_DWORD *)(a1 + 508);
        v16 = (__int64 ***)*sub_3515040(a1 + 888, (__int64 *)&v194);
        if ( sub_2E322C0(v4, (__int64)v194) )
        {
          if ( !a4 )
          {
LABEL_9:
            if ( a3 != v16 )
            {
              v12 = (__int64)v194;
              if ( **v16 == v194 )
              {
                v53 = v14[1];
                goto LABEL_47;
              }
            }
            goto LABEL_11;
          }
          if ( *(_DWORD *)(a4 + 16) )
          {
            v12 = (__int64)v194;
            v17 = *(_QWORD *)(a4 + 8);
            v18 = *(_DWORD *)(a4 + 24);
            if ( v18 )
            {
              v9 = (unsigned int)(v18 - 1);
              v19 = v9 & (((unsigned int)v194 >> 9) ^ ((unsigned int)v194 >> 4));
              v20 = *(__int64 **)(v17 + 8LL * v19);
              if ( v194 == v20 )
                goto LABEL_9;
              v10 = 1;
              while ( v20 != (__int64 *)-4096LL )
              {
                v11 = (unsigned int)(v10 + 1);
                v19 = v9 & (v10 + v19);
                v20 = *(__int64 **)(v17 + 8LL * v19);
                if ( v194 == v20 )
                  goto LABEL_9;
                v10 = (unsigned int)v11;
              }
            }
          }
          else
          {
            v74 = *(_QWORD **)(a4 + 32);
            v75 = &v74[*(unsigned int *)(a4 + 40)];
            if ( v75 != sub_3510810(v74, (__int64)v75, (__int64 *)&v194) )
              goto LABEL_9;
          }
        }
      }
    }
    else
    {
      v10 = 1;
      while ( v15 != -4096 )
      {
        v11 = (unsigned int)(v10 + 1);
        v9 = ((_DWORD)v13 - 1) & (unsigned int)(v10 + v9);
        v14 = (__int64 *)(v12 + 24LL * (unsigned int)v9);
        v15 = *v14;
        if ( v4 == *v14 )
          goto LABEL_3;
        v10 = (unsigned int)v11;
      }
    }
  }
LABEL_11:
  v21 = (unsigned int)v186;
  if ( *(_DWORD *)(v4 + 120) != 2 || (_DWORD)v186 != 2 )
    goto LABEL_12;
  v55 = *(__int64 **)(v4 + 112);
  v188 = 0;
  v56 = 1;
  v189 = (__int64 *)v193;
  v57 = v55 + 2;
  v190 = 2;
  v191 = 0;
  v192 = 1;
  do
  {
    v58 = *v55;
    if ( !v56 )
    {
LABEL_123:
      sub_C8CC70((__int64)&v188, v58, v9, v12, v10, v11);
      v56 = v192;
      goto LABEL_57;
    }
    v59 = v189;
    v12 = HIDWORD(v190);
    v9 = (__int64)&v189[HIDWORD(v190)];
    if ( v189 == (__int64 *)v9 )
    {
LABEL_130:
      if ( HIDWORD(v190) >= (unsigned int)v190 )
        goto LABEL_123;
      v12 = (unsigned int)++HIDWORD(v190);
      *(_QWORD *)v9 = v58;
      v56 = v192;
      ++v188;
    }
    else
    {
      while ( v58 != *v59 )
      {
        if ( (__int64 *)v9 == ++v59 )
          goto LABEL_130;
      }
    }
LABEL_57:
    ++v55;
  }
  while ( v57 != v55 );
  v60 = (__int64)v185;
  v194 = 0;
  v61 = v6 + 888;
  v195 = (unsigned __int64)&v199;
  v196 = 8;
  v198 = 1;
  v197 = 0;
  v155 = &v185[(unsigned int)v186];
  v158 = v185;
  if ( v185 == v155 )
  {
    if ( !v192 )
      _libc_free((unsigned __int64)v189);
    goto LABEL_102;
  }
  v161 = v6;
  v168 = v4;
LABEL_60:
  v183 = *v158;
  v12 = *(_QWORD *)(v183 + 64);
  v62 = *(unsigned int *)(v183 + 72);
  if ( v12 != v12 + 8 * v62 )
  {
    v159 = 0;
    v63 = *(__int64 **)(v183 + 64);
    v165 = (__int64 *)(v12 + 8 * v62);
    while ( 1 )
    {
      v64 = *v63;
      v184[0] = *v63;
      if ( v192 )
      {
        v65 = v189;
        v66 = &v189[HIDWORD(v190)];
        if ( v189 != v66 )
        {
          do
          {
            v60 = *v65;
            if ( v64 == *v65 )
              goto LABEL_67;
            ++v65;
          }
          while ( v66 != v65 );
        }
      }
      else if ( sub_C8CA60((__int64)&v188, v64) )
      {
        v60 = v184[0];
LABEL_67:
        v10 = *(_QWORD *)(v60 + 112);
        v67 = *(unsigned int *)(v60 + 120);
        v11 = v10 + 8 * v67;
        if ( v10 != v11 )
        {
          v68 = *(__int64 **)(v60 + 112);
          v69 = (__int64 *)(v10 + 8 * v67);
          while ( 1 )
          {
            v70 = *v68;
            if ( v192 )
            {
              v71 = v189;
              v60 = (__int64)&v189[HIDWORD(v190)];
              if ( v189 == (__int64 *)v60 )
                goto LABEL_120;
              while ( v70 != *v71 )
              {
                if ( (__int64 *)v60 == ++v71 )
                  goto LABEL_120;
              }
            }
            else if ( !sub_C8CA60((__int64)&v188, v70) )
            {
              goto LABEL_120;
            }
            if ( v69 == ++v68 )
              goto LABEL_75;
          }
        }
        goto LABEL_75;
      }
      v76 = (__int64 ***)*sub_3515040(v61, v184);
      if ( v168 != v184[0] )
      {
        if ( a4 )
        {
          v11 = *(unsigned int *)(a4 + 16);
          if ( (_DWORD)v11 )
          {
            v60 = *(unsigned int *)(a4 + 24);
            v87 = *(_QWORD *)(a4 + 8);
            if ( !(_DWORD)v60 )
              goto LABEL_75;
            v88 = v60 - 1;
            v60 = ((_DWORD)v60 - 1) & (unsigned int)((LODWORD(v184[0]) >> 9) ^ (LODWORD(v184[0]) >> 4));
            v11 = *(_QWORD *)(v87 + 8 * v60);
            if ( v184[0] != v11 )
            {
              v89 = 1;
              while ( v11 != -4096 )
              {
                v12 = (unsigned int)(v89 + 1);
                v60 = v88 & (unsigned int)(v89 + v60);
                v11 = *(_QWORD *)(v87 + 8LL * (unsigned int)v60);
                if ( v184[0] == v11 )
                  goto LABEL_86;
                ++v89;
              }
              goto LABEL_75;
            }
          }
          else
          {
            v77 = *(_QWORD **)(a4 + 32);
            v78 = &v77[*(unsigned int *)(a4 + 40)];
            if ( v78 == sub_3510810(v77, (__int64)v78, v184) )
              goto LABEL_75;
          }
        }
LABEL_86:
        if ( a3 != v76 && v76 != (__int64 ***)*sub_3515040(v61, &v183) )
        {
          ++v159;
          if ( v198 )
          {
            v80 = (_QWORD *)v195;
            v60 = v195 + 8LL * HIDWORD(v196);
            if ( v195 != v60 )
            {
              while ( v184[0] != *v80 )
              {
                if ( (_QWORD *)v60 == ++v80 )
                  goto LABEL_92;
              }
              goto LABEL_75;
            }
LABEL_92:
            if ( HIDWORD(v196) < (unsigned int)v196 )
            {
              ++HIDWORD(v196);
              *(_QWORD *)v60 = v184[0];
              v73 = v198;
              v194 = (__int64 *)((char *)v194 + 1);
LABEL_94:
              v10 = v184[0];
              v60 = (unsigned int)(HIDWORD(v190) - v191);
              if ( *(_DWORD *)(v184[0] + 120) != (_DWORD)v60 )
              {
                v6 = v161;
                v4 = v168;
                goto LABEL_96;
              }
              if ( !(unsigned __int8)sub_3512080(v184[0], (__int64)&v188) )
                break;
              goto LABEL_75;
            }
          }
          sub_C8CC70((__int64)&v194, v184[0], v60, v12, v79, v11);
          v73 = v198;
          if ( (_BYTE)v60 )
            goto LABEL_94;
        }
      }
LABEL_75:
      if ( v165 == ++v63 )
      {
        v10 = v159;
        if ( !v159 )
          break;
        if ( v155 == ++v158 )
        {
          v6 = v161;
          v4 = v168;
          v72 = 1;
          v73 = v198;
          goto LABEL_97;
        }
        goto LABEL_60;
      }
    }
  }
LABEL_120:
  v6 = v161;
  v4 = v168;
  v73 = v198;
LABEL_96:
  v72 = 0;
LABEL_97:
  if ( !v73 )
    _libc_free(v195);
  if ( !v192 )
    _libc_free((unsigned __int64)v189);
  if ( v72 )
  {
LABEL_102:
    v81 = *(unsigned int *)(v4 + 120);
    v82 = *(__int64 **)(v4 + 112);
    v188 = 0;
    v190 = 4;
    v83 = &v82[v81];
    v191 = 0;
    v192 = 1;
    v189 = (__int64 *)v193;
    if ( v82 == v83 )
    {
      v174 = 0;
LABEL_113:
      v53 = v174;
      goto LABEL_47;
    }
    v84 = 1;
    while ( 2 )
    {
      v85 = *v82;
      if ( !v84 )
        goto LABEL_114;
      v86 = v189;
      v12 = HIDWORD(v190);
      v60 = (__int64)&v189[HIDWORD(v190)];
      if ( v189 == (__int64 *)v60 )
      {
LABEL_117:
        if ( HIDWORD(v190) < (unsigned int)v190 )
        {
          v12 = (unsigned int)++HIDWORD(v190);
          *(_QWORD *)v60 = v85;
          v84 = v192;
          ++v188;
          goto LABEL_109;
        }
LABEL_114:
        sub_C8CC70((__int64)&v188, v85, v60, v12, v10, v11);
        v84 = v192;
        goto LABEL_109;
      }
      while ( v85 != *v86 )
      {
        if ( (__int64 *)v60 == ++v86 )
          goto LABEL_117;
      }
LABEL_109:
      if ( v83 != ++v82 )
        continue;
      break;
    }
    v174 = 0;
    if ( HIDWORD(v190) - v191 != 2 || (_DWORD)v186 != 2 )
    {
LABEL_111:
      if ( !v84 )
        _libc_free((unsigned __int64)v189);
      goto LABEL_113;
    }
    v90 = &v194;
    v194 = &v196;
    v195 = 0x800000000LL;
    v200[1] = 0x800000000LL;
    v162 = (unsigned __int64)v185;
    v200[0] = v201;
    while ( 2 )
    {
      v160 = v90 + 2;
      v91 = *(__int64 **)(*(_QWORD *)v162 + 64LL);
      v182 = *(_QWORD *)v162;
      v169 = &v91[*(unsigned int *)(v182 + 72)];
      if ( v91 == v169 )
      {
LABEL_168:
        v162 += 8LL;
        v90 += 26;
        if ( &v202 != (char *)v90 )
          continue;
        sub_3513940((__int64)&v194);
        sub_3513940((__int64)v200);
        v110 = (_BYTE *)v200[0];
        v111 = (__int64)v194;
        v112 = *(_QWORD *)(v200[0] + 8LL);
        v113 = v194[1];
        v114 = v200[0];
        if ( v113 == v112 )
        {
          v114 = v200[0] + 24LL;
          v130 = v194[3];
          v131 = __CFADD__(*(_QWORD *)v200[0], v130);
          v132 = *(_QWORD *)v200[0] + v130;
          if ( __CFADD__(*(_QWORD *)(v200[0] + 24LL), *v194) )
            goto LABEL_209;
          if ( v131 )
            v132 = -1;
          if ( *(_QWORD *)(v200[0] + 24LL) + *v194 >= v132 )
          {
LABEL_209:
            v112 = *(_QWORD *)(v200[0] + 32LL);
          }
          else
          {
            v113 = v194[4];
            v114 = v200[0];
            v111 = (__int64)(v194 + 3);
          }
        }
        if ( v4 == v112 )
        {
          v115 = *(_QWORD *)(v111 + 16);
          v174 = *(_QWORD *)(v114 + 16);
        }
        else
        {
          v115 = *(_QWORD *)(v114 + 16);
          v174 = *(_QWORD *)(v111 + 16);
          if ( v4 != v113 )
          {
            v174 = 0;
LABEL_173:
            if ( v110 != v201 )
              _libc_free((unsigned __int64)v110);
            if ( v194 != &v196 )
              _libc_free((unsigned __int64)v194);
            v84 = v192;
            goto LABEL_111;
          }
          v136 = v113;
          v113 = v112;
          v112 = v136;
        }
        if ( v174 == v113 )
        {
          if ( (_BYTE)qword_503C648 )
          {
            if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v6 + 520) + 8LL) + 688LL) & 1) == 0 )
            {
              v133 = sub_2FD62C0(v115);
              if ( *(_DWORD *)(v115 + 120) != 1 )
              {
                if ( (unsigned __int8)sub_2FD64C0((__int64 *)(v6 + 600), v133, (__int64 *)v115) )
                {
                  v134 = sub_2FD62C0(v115);
                  if ( *(_DWORD *)(v115 + 120) != 1 )
                  {
                    if ( (unsigned __int8)sub_2FD64C0((__int64 *)(v6 + 600), v134, (__int64 *)v115) )
                    {
                      if ( (unsigned __int8)sub_3515CB0(v6, v112, (__int64 *)v115, (__int64)a3, a4) )
                      {
                        v135 = sub_2E441D0(*(_QWORD *)(v6 + 528), v112, v174);
                        if ( (unsigned __int8)sub_35161F0(v6, v112, v115, v135, a3, a4) )
                        {
                          v174 = v115;
                          v110 = (_BYTE *)v200[0];
                          goto LABEL_173;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
        v117 = *(_DWORD *)(v6 + 512);
        if ( v117 )
        {
          v118 = 1;
          v119 = *(_QWORD *)(v6 + 496);
          v120 = ((unsigned int)v113 >> 9) ^ ((unsigned int)v113 >> 4);
          v121 = 0;
          v122 = (v117 - 1) & v120;
          v123 = v119 + 24LL * v122;
          v124 = *(_QWORD *)v123;
          if ( *(_QWORD *)v123 == v113 )
          {
LABEL_189:
            v125 = v123 + 8;
            *(_QWORD *)v125 = v115;
            *(_BYTE *)(v125 + 8) = 0;
            v110 = (_BYTE *)v200[0];
            goto LABEL_173;
          }
          while ( v124 != -4096 )
          {
            if ( v124 == -8192 && !v121 )
              v121 = v123;
            v122 = (v117 - 1) & (v118 + v122);
            v123 = v119 + 24LL * v122;
            v124 = *(_QWORD *)v123;
            if ( v113 == *(_QWORD *)v123 )
              goto LABEL_189;
            ++v118;
          }
          v144 = *(_DWORD *)(v6 + 504);
          if ( v121 )
            v123 = v121;
          ++*(_QWORD *)(v6 + 488);
          v143 = v144 + 1;
          if ( 4 * (v144 + 1) < 3 * v117 )
          {
            if ( v117 - *(_DWORD *)(v6 + 508) - v143 > v117 >> 3 )
            {
LABEL_219:
              *(_DWORD *)(v6 + 504) = v143;
              if ( *(_QWORD *)v123 != -4096 )
                --*(_DWORD *)(v6 + 508);
              *(_QWORD *)v123 = v113;
              *(_QWORD *)(v123 + 8) = 0;
              *(_BYTE *)(v123 + 16) = 0;
              goto LABEL_189;
            }
            sub_35124E0(v6 + 488, v117);
            v145 = *(_DWORD *)(v6 + 512);
            if ( v145 )
            {
              v146 = v145 - 1;
              v147 = *(_QWORD *)(v6 + 496);
              v123 = 0;
              v148 = v146 & v120;
              for ( i = 1; ; ++i )
              {
                v150 = (__int64 *)(v147 + 24LL * v148);
                v151 = *v150;
                if ( v113 == *v150 )
                {
                  v143 = *(_DWORD *)(v6 + 504) + 1;
                  v123 = v147 + 24LL * v148;
                  goto LABEL_219;
                }
                if ( v151 == -4096 )
                  break;
                if ( v151 != -8192 || v123 )
                  v150 = (__int64 *)v123;
                v153 = i + v148;
                v123 = (__int64)v150;
                v148 = v146 & v153;
              }
              if ( !v123 )
                v123 = v147 + 24LL * v148;
              v143 = *(_DWORD *)(v6 + 504) + 1;
              goto LABEL_219;
            }
LABEL_252:
            ++*(_DWORD *)(v6 + 504);
            BUG();
          }
        }
        else
        {
          ++*(_QWORD *)(v6 + 488);
        }
        sub_35124E0(v6 + 488, 2 * v117);
        v137 = *(_DWORD *)(v6 + 512);
        if ( v137 )
        {
          v138 = v137 - 1;
          v139 = 1;
          v140 = 0;
          for ( j = v138 & (((unsigned int)v113 >> 9) ^ ((unsigned int)v113 >> 4)); ; j = v138 & v152 )
          {
            v123 = *(_QWORD *)(v6 + 496) + 24LL * j;
            v142 = *(_QWORD *)v123;
            if ( v113 == *(_QWORD *)v123 )
            {
              v143 = *(_DWORD *)(v6 + 504) + 1;
              goto LABEL_219;
            }
            if ( v142 == -4096 )
              break;
            if ( v142 != -8192 || v140 )
              v123 = v140;
            v152 = v139 + j;
            v140 = v123;
            ++v139;
          }
          if ( v140 )
            v123 = v140;
          v143 = *(_DWORD *)(v6 + 504) + 1;
          goto LABEL_219;
        }
        goto LABEL_252;
      }
      break;
    }
    while ( 2 )
    {
      v95 = *v91;
      v183 = v95;
      if ( v4 != v95 )
      {
        if ( a4 )
        {
          if ( *(_DWORD *)(a4 + 16) )
          {
            v104 = *(_DWORD *)(a4 + 24);
            v105 = *(_QWORD *)(a4 + 8);
            if ( !v104 )
              goto LABEL_147;
            v106 = v104 - 1;
            v94 = &v183;
            v107 = (v104 - 1) & (((unsigned int)v95 >> 9) ^ ((unsigned int)v95 >> 4));
            v108 = *(_QWORD *)(v105 + 8LL * v107);
            if ( v108 != v95 )
            {
              v109 = 1;
              while ( v108 != -4096 )
              {
                v107 = v106 & (v109 + v107);
                v108 = *(_QWORD *)(v105 + 8LL * v107);
                if ( v95 == v108 )
                  goto LABEL_156;
                ++v109;
              }
LABEL_147:
              if ( v169 == ++v91 )
                goto LABEL_168;
              continue;
            }
          }
          else
          {
            v92 = *(_QWORD **)(a4 + 32);
            v93 = &v92[*(unsigned int *)(a4 + 40)];
            if ( v93 == sub_3510810(v92, (__int64)v93, &v183) )
              goto LABEL_147;
          }
        }
        else
        {
LABEL_156:
          v94 = &v183;
        }
        v175 = v94;
        if ( a3 == (__int64 ***)*sub_3515040(v6 + 888, v94) )
          goto LABEL_147;
        v176 = *sub_3515040(v6 + 888, v175);
        if ( v176 == *sub_3515040(v6 + 888, &v182) )
          goto LABEL_147;
        v95 = v183;
      }
      break;
    }
    v177 = sub_2E441D0(*(_QWORD *)(v6 + 528), v95, v182);
    v184[0] = sub_2F06CB0(*(_QWORD *)(v6 + 536), v183);
    v96 = sub_1098D20((unsigned __int64 *)v184, v177);
    v98 = *((unsigned int *)v90 + 3);
    v184[0] = v96;
    v184[1] = v183;
    v184[2] = v182;
    v99 = *((unsigned int *)v90 + 2);
    v100 = v99 + 1;
    if ( v99 + 1 > v98 )
    {
      v116 = (__int64)*v90;
      if ( *v90 > v184 || (v178 = *v90, (unsigned __int64)v184 >= v116 + 24 * v99) )
      {
        sub_C8D5F0((__int64)v90, v160, v100, 0x18u, v116, v97);
        v101 = *v90;
        v99 = *((unsigned int *)v90 + 2);
        v102 = (const __m128i *)v184;
      }
      else
      {
        sub_C8D5F0((__int64)v90, v160, v100, 0x18u, v116, v97);
        v101 = *v90;
        v99 = *((unsigned int *)v90 + 2);
        v102 = (const __m128i *)((char *)*v90 + (char *)v184 - (char *)v178);
      }
    }
    else
    {
      v101 = *v90;
      v102 = (const __m128i *)v184;
    }
    v103 = (__m128i *)&v101[3 * v99];
    *v103 = _mm_loadu_si128(v102);
    v103[1].m128i_i64[0] = v102[1].m128i_i64[0];
    ++*((_DWORD *)v90 + 2);
    goto LABEL_147;
  }
  v21 = (unsigned int)v186;
LABEL_12:
  v22 = v185;
  v194 = &v196;
  v195 = 0x400000000LL;
  v163 = &v185[v21];
  if ( v163 == v185 )
  {
    v32 = (__m128i *)&v196;
    v33 = 0;
    v164 = 0;
    v35 = (unsigned int *)&v196;
LABEL_135:
    sub_3512C90(v32->m128i_i8, v35);
    v47 = 0;
    goto LABEL_36;
  }
  v164 = 0;
  v166 = 0;
  do
  {
    v24 = *(_QWORD *)(v6 + 528);
    v25 = 0x80000000;
    v188 = *v22;
    v26 = sub_2E441D0(v24, v4, v188);
    v27 = v26;
    if ( v26 < v170 )
    {
      sub_F02DB0(v184, v26, v170);
      v25 = v184[0];
    }
    v23 = sub_3515040(v6 + 888, &v188);
    if ( (unsigned __int8)sub_35144C0(v6, v4, v188, *v23, v27, (__int64)a3, a4) )
    {
      if ( (_BYTE)qword_503C648 )
      {
        if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v6 + 520) + 8LL) + 688LL) & 1) == 0 )
        {
          v156 = v188;
          v28 = sub_2FD62C0(v188);
          if ( *(_DWORD *)(v156 + 120) != 1 )
          {
            if ( (unsigned __int8)sub_2FD64C0((__int64 *)(v6 + 600), v28, (__int64 *)v156) )
            {
              v29 = (unsigned int)v195;
              v30 = v195;
              if ( (unsigned int)v195 >= (unsigned __int64)HIDWORD(v195) )
              {
                v126 = v25 | v154 & 0xFFFFFFFF00000000LL;
                v127 = v188;
                v154 = v126;
                v128 = v126;
                if ( HIDWORD(v195) < (unsigned __int64)(unsigned int)v195 + 1 )
                {
                  v157 = v188;
                  sub_C8D5F0((__int64)&v194, &v196, (unsigned int)v195 + 1LL, 0x10u, v188, v126);
                  v29 = (unsigned int)v195;
                  v127 = v157;
                }
                v129 = &v194[2 * v29];
                *v129 = v128;
                v129[1] = v127;
                LODWORD(v195) = v195 + 1;
              }
              else
              {
                v31 = &v194[2 * (unsigned int)v195];
                if ( v31 )
                {
                  *(_DWORD *)v31 = v25;
                  v31[1] = v188;
                  v30 = v195;
                }
                LODWORD(v195) = v30 + 1;
              }
            }
          }
        }
      }
    }
    else if ( v25 > v166 || !v164 )
    {
      v166 = v25;
      v164 = v188;
    }
    ++v22;
  }
  while ( v163 != v22 );
  v32 = (__m128i *)v194;
  v33 = v166;
  v34 = 2LL * (unsigned int)v195;
  v35 = (unsigned int *)&v194[v34];
  if ( !(v34 * 8) )
    goto LABEL_135;
  v171 = (unsigned int *)&v194[v34];
  v36 = (v34 * 8) >> 4;
  v167 = v6;
  while ( 1 )
  {
    v37 = 16 * v36;
    v38 = sub_2207800(16 * v36);
    v39 = (int *)v38;
    if ( v38 )
      break;
    v36 >>= 1;
    if ( !v36 )
    {
      v35 = v171;
      v6 = v167;
      goto LABEL_135;
    }
  }
  v40 = v36;
  v41 = 4 * v36;
  v42 = v171;
  v43 = v38 + v37;
  v44 = v38 + 16;
  v6 = v167;
  *(__m128i *)(v44 - 16) = _mm_loadu_si128(v32);
  if ( v43 == v44 )
  {
    v46 = (__int64)v39;
  }
  else
  {
    do
    {
      v45 = _mm_loadu_si128((const __m128i *)(v44 - 16));
      v44 += 16;
      *(__m128i *)(v44 - 16) = v45;
    }
    while ( v43 != v44 );
    v46 = (__int64)&v39[v41 - 4];
  }
  v172 = v39;
  v32->m128i_i32[0] = *(_DWORD *)v46;
  v32->m128i_i64[1] = *(_QWORD *)(v46 + 8);
  sub_351A640(v32->m128i_i32, v42, v39, v40);
  v47 = (unsigned __int64)v172;
LABEL_36:
  j_j___libc_free_0(v47);
  v48 = v194;
  v49 = 2LL * (unsigned int)v195;
  v173 = &v194[v49];
  if ( &v194[v49] != v194 )
  {
    v50 = (__int64)v194;
    do
    {
      v52 = *(_QWORD *)(v50 + 8);
      if ( v33 > *(_DWORD *)v50 )
        goto LABEL_44;
      v51 = sub_2FD62C0(*(_QWORD *)(v50 + 8));
      if ( *(_DWORD *)(v52 + 120) != 1
        && (unsigned __int8)sub_2FD64C0((__int64 *)(v6 + 600), v51, (__int64 *)v52)
        && (unsigned __int8)sub_3515CB0(v6, v4, (__int64 *)v52, (__int64)a3, a4)
        && (unsigned __int8)sub_35161F0(v6, v4, v52, v33, a3, a4) )
      {
        v164 = v52;
LABEL_44:
        v48 = v194;
        goto LABEL_45;
      }
      v50 += 16;
    }
    while ( v173 != (__int64 *)v50 );
    v48 = v194;
  }
LABEL_45:
  v53 = v164;
  if ( v48 != &v196 )
    _libc_free((unsigned __int64)v48);
LABEL_47:
  if ( v185 != (__int64 *)v187 )
    _libc_free((unsigned __int64)v185);
  return v53;
}
