// Function: sub_1D64280
// Address: 0x1d64280
//
_BOOL8 __fastcall sub_1D64280(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        __m128i a4,
        __m128i a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10,
        __int64 a11,
        __int64 a12,
        int a13,
        int a14)
{
  __int64 v14; // rdx
  _BYTE *v15; // rax
  _BYTE *v16; // r14
  int v17; // edx
  _BYTE *v18; // rdi
  size_t v19; // r15
  __int64 v20; // r13
  unsigned int i; // eax
  __int64 v22; // rax
  __int64 v23; // rax
  unsigned __int64 v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // r15
  _BYTE *v27; // r9
  _BYTE *v28; // r8
  size_t v29; // r10
  unsigned __int64 v30; // r14
  __int64 v31; // rbx
  unsigned __int64 v32; // rax
  int v33; // eax
  unsigned __int64 *v34; // rdi
  unsigned __int64 v35; // rax
  bool v36; // zf
  _QWORD *v37; // r14
  __int64 v38; // r12
  __int64 v39; // rbx
  _QWORD *v40; // rax
  _BYTE *v41; // r13
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // r13
  unsigned __int64 v45; // r15
  __int64 v46; // rax
  double v47; // xmm4_8
  double v48; // xmm5_8
  __int64 v49; // rax
  double v50; // xmm4_8
  double v51; // xmm5_8
  __int64 v52; // rdx
  __int64 v53; // rbx
  unsigned int v54; // ecx
  __int64 v55; // rsi
  __int64 v56; // rax
  __int64 v57; // rdx
  __int64 v58; // rax
  __int64 v59; // r8
  __int64 v60; // r9
  __int64 v61; // r13
  __int64 v62; // rdx
  __int64 v63; // rsi
  __int64 v64; // r9
  __int64 v65; // r15
  __int64 v66; // r8
  __int64 v67; // r12
  __int64 v68; // rcx
  __int64 v69; // r14
  __int64 v70; // r13
  __int64 v71; // rdx
  __int64 v72; // rdx
  int v73; // eax
  __int64 v74; // rax
  int v75; // esi
  __int64 v76; // rsi
  __int64 *v77; // rax
  __int64 v78; // rdi
  unsigned __int64 v79; // rsi
  __int64 v80; // rsi
  __int64 v81; // rsi
  __int64 v82; // rax
  __int64 v83; // rdi
  __int64 v84; // rax
  _QWORD *v85; // rbx
  __int64 v86; // rax
  __int64 v88; // r14
  _QWORD *v89; // rax
  __int64 v90; // r8
  __int64 v91; // r9
  __int64 v92; // rcx
  __int64 v93; // rdx
  __int64 v94; // rdx
  __int64 *v95; // rax
  __int64 v96; // rsi
  unsigned __int64 v97; // rdx
  __int64 v98; // rdx
  __int64 v99; // rdx
  __int64 v100; // r12
  int v101; // eax
  __int64 v102; // rax
  int v103; // edx
  __int64 v104; // r14
  __int64 v105; // r15
  __int64 v106; // rax
  __int64 v107; // rcx
  __int64 v108; // rdx
  _BYTE *v109; // rdx
  __int64 v110; // rax
  __int64 j; // r15
  _QWORD *v112; // rax
  __int64 v113; // r14
  __int64 v114; // rax
  __int64 v115; // rdx
  __int64 v116; // r9
  __int64 v117; // rdi
  __int64 v118; // rax
  char v119; // r8
  unsigned int v120; // ecx
  __int64 v121; // rdx
  __int64 v122; // rax
  __int64 v123; // rsi
  __int64 v124; // rsi
  __int64 v125; // r8
  __int64 v126; // rax
  __int64 v127; // rax
  __int64 v128; // rax
  __int64 *v129; // r15
  __int64 *v130; // rax
  __int64 *v131; // rdx
  __int64 v132; // rax
  __int64 *v133; // r15
  __int64 v134; // rcx
  __int64 *v135; // rcx
  __int64 *v136; // r14
  __int64 v137; // r12
  unsigned __int64 v138; // rbx
  __int64 v139; // rax
  __int64 v141; // [rsp+20h] [rbp-420h]
  _BYTE *v142; // [rsp+28h] [rbp-418h]
  __int64 v143; // [rsp+30h] [rbp-410h]
  __int64 v144; // [rsp+38h] [rbp-408h]
  __int64 v145; // [rsp+38h] [rbp-408h]
  __int64 v146; // [rsp+38h] [rbp-408h]
  __int64 v147; // [rsp+40h] [rbp-400h]
  unsigned __int64 v148; // [rsp+40h] [rbp-400h]
  size_t v149; // [rsp+48h] [rbp-3F8h]
  __int64 v150; // [rsp+48h] [rbp-3F8h]
  _BYTE *v151; // [rsp+50h] [rbp-3F0h]
  bool v152; // [rsp+50h] [rbp-3F0h]
  __int64 v153; // [rsp+50h] [rbp-3F0h]
  __int64 v154; // [rsp+50h] [rbp-3F0h]
  _BYTE *v155; // [rsp+58h] [rbp-3E8h]
  _BYTE *v156; // [rsp+58h] [rbp-3E8h]
  _BYTE *v157; // [rsp+58h] [rbp-3E8h]
  __int64 v158; // [rsp+68h] [rbp-3D8h] BYREF
  __int64 v159[4]; // [rsp+70h] [rbp-3D0h] BYREF
  _BYTE *v160; // [rsp+90h] [rbp-3B0h] BYREF
  __int64 v161; // [rsp+98h] [rbp-3A8h]
  _BYTE dest[128]; // [rsp+A0h] [rbp-3A0h] BYREF
  __int64 v163; // [rsp+120h] [rbp-320h] BYREF
  _BYTE *v164; // [rsp+128h] [rbp-318h]
  _BYTE *v165; // [rsp+130h] [rbp-310h]
  __int64 v166; // [rsp+138h] [rbp-308h]
  int v167; // [rsp+140h] [rbp-300h]
  _BYTE v168[136]; // [rsp+148h] [rbp-2F8h] BYREF
  __int64 v169; // [rsp+1D0h] [rbp-270h] BYREF
  __int64 *v170; // [rsp+1D8h] [rbp-268h]
  __int64 *v171; // [rsp+1E0h] [rbp-260h]
  __int64 v172; // [rsp+1E8h] [rbp-258h]
  int v173; // [rsp+1F0h] [rbp-250h]
  _BYTE v174[136]; // [rsp+1F8h] [rbp-248h] BYREF
  _BYTE *v175; // [rsp+280h] [rbp-1C0h] BYREF
  __int64 v176; // [rsp+288h] [rbp-1B8h]
  _BYTE v177[432]; // [rsp+290h] [rbp-1B0h] BYREF

  v14 = *(_QWORD *)(a1 + 208);
  v164 = v168;
  v165 = v168;
  v15 = *(_BYTE **)(v14 + 40);
  v167 = 0;
  v16 = *(_BYTE **)(v14 + 32);
  v17 = 0;
  v18 = dest;
  v19 = v15 - v16;
  v160 = dest;
  v161 = 0x1000000000LL;
  v163 = 0;
  v20 = (v15 - v16) >> 3;
  v166 = 16;
  if ( (unsigned __int64)(v15 - v16) > 0x80 )
  {
    v157 = v15;
    sub_16CD150((__int64)&v160, dest, (v15 - v16) >> 3, 8, a13, a14);
    v17 = v161;
    v15 = v157;
    v18 = &v160[8 * (unsigned int)v161];
  }
  if ( v16 != v15 )
  {
    memmove(v18, v16, v19);
    v17 = v161;
  }
  LODWORD(v161) = v20 + v17;
  for ( i = v20 + v17; i; i = v161 )
  {
    v24 = (unsigned __int64)v160;
    v25 = i;
    v22 = i - 1;
    v26 = *(_QWORD *)&v160[8 * v25 - 8];
    LODWORD(v161) = v22;
    v27 = *(_BYTE **)(v26 + 16);
    v28 = *(_BYTE **)(v26 + 8);
    v29 = v27 - v28;
    v30 = (v27 - v28) >> 3;
    if ( v30 > (unsigned __int64)HIDWORD(v161) - v22 )
    {
      v149 = *(_QWORD *)(v26 + 16) - (_QWORD)v28;
      v151 = *(_BYTE **)(v26 + 16);
      v155 = *(_BYTE **)(v26 + 8);
      sub_16CD150((__int64)&v160, dest, v22 + v30, 8, (int)v28, (int)v27);
      v24 = (unsigned __int64)v160;
      v22 = (unsigned int)v161;
      v29 = v149;
      v27 = v151;
      v28 = v155;
    }
    if ( v28 != v27 )
    {
      memmove((void *)(v24 + 8 * v22), v28, v29);
      LODWORD(v22) = v161;
    }
    LODWORD(v161) = v30 + v22;
    v23 = sub_13FC520(v26);
    if ( v23 )
      sub_1412190((__int64)&v163, v23);
  }
  v175 = v177;
  v176 = 0x1000000000LL;
  v31 = *(_QWORD *)(*(_QWORD *)(a2 + 80) + 8LL);
  if ( a2 + 72 == v31 )
  {
    v152 = 0;
  }
  else
  {
    do
    {
      v32 = v31 - 24;
      v169 = 6;
      if ( !v31 )
        v32 = 0;
      v170 = 0;
      v171 = (__int64 *)v32;
      if ( v32 != -8 && v32 != 0 && v32 != -16 )
        sub_164C220((__int64)&v169);
      v33 = v176;
      if ( (unsigned int)v176 >= HIDWORD(v176) )
      {
        sub_170B450((__int64)&v175, 0);
        v33 = v176;
      }
      v34 = (unsigned __int64 *)&v175[24 * v33];
      if ( v34 )
      {
        *v34 = 6;
        v34[1] = 0;
        v35 = (unsigned __int64)v171;
        v36 = v171 + 1 == 0;
        v34[2] = (unsigned __int64)v171;
        if ( v35 != 0 && !v36 && v35 != -16 )
          sub_1649AC0(v34, v169 & 0xFFFFFFFFFFFFFFF8LL);
        v33 = v176;
      }
      LODWORD(v176) = v33 + 1;
      if ( v171 + 1 != 0 && v171 != 0 && v171 != (__int64 *)-16LL )
        sub_1649B30(&v169);
      v31 = *(_QWORD *)(v31 + 8);
    }
    while ( a2 + 72 != v31 );
    v37 = v175;
    v152 = 0;
    v142 = &v175[24 * (unsigned int)v176];
    if ( v142 != v175 )
    {
      v156 = v175;
      while ( 1 )
      {
        v38 = *((_QWORD *)v156 + 2);
        if ( !v38 )
          goto LABEL_82;
        v39 = sub_1D5D650(*((_QWORD *)v156 + 2));
        if ( !v39 )
          goto LABEL_82;
        v40 = v164;
        if ( v165 == v164 )
          break;
        v41 = &v165[8 * (unsigned int)v166];
        v40 = sub_16CC9F0((__int64)&v163, v38);
        if ( v38 == *v40 )
        {
          if ( v165 == v164 )
            v109 = &v165[8 * HIDWORD(v166)];
          else
            v109 = &v165[8 * (unsigned int)v166];
          goto LABEL_131;
        }
        if ( v165 == v164 )
        {
          v109 = &v165[8 * HIDWORD(v166)];
          v40 = v109;
          goto LABEL_131;
        }
        v40 = &v165[8 * (unsigned int)v166];
LABEL_39:
        if ( v41 != (_BYTE *)v40 && !byte_4FC2B20 )
        {
          if ( !sub_157F0B0(v38) )
            goto LABEL_82;
          v42 = sub_157F0B0(v38);
          if ( !sub_157F1C0(v42) )
            goto LABEL_82;
        }
        v43 = sub_157F120(v38);
        v44 = v43;
        if ( !v43 )
          goto LABEL_46;
        if ( (unsigned __int8)(*(_BYTE *)(sub_157EBA0(v43) + 16) - 27) > 1u )
          goto LABEL_46;
        v45 = sub_157EBA0(v38);
        if ( v45 != sub_157ED20(v38) )
          goto LABEL_46;
        v110 = *(_QWORD *)(v39 + 48);
        if ( !v110 )
          BUG();
        if ( *(_BYTE *)(v110 - 8) != 77 )
          goto LABEL_46;
        v169 = 0;
        v170 = (__int64 *)v174;
        v171 = (__int64 *)v174;
        v172 = 16;
        v173 = 0;
        v159[0] = *(_QWORD *)(v39 + 8);
        sub_15CDD40(v159);
        j = v159[0];
        if ( !v159[0] )
          goto LABEL_166;
        v112 = sub_1648700(v159[0]);
LABEL_141:
        v113 = v112[5];
        if ( v38 == v113 )
          goto LABEL_163;
        v114 = sub_157F280(v39);
        v116 = v115;
        v117 = v114;
        if ( v114 == v115 )
          goto LABEL_162;
        while ( 1 )
        {
          v118 = 0x17FFFFFFE8LL;
          v119 = *(_BYTE *)(v117 + 23) & 0x40;
          v120 = *(_DWORD *)(v117 + 20) & 0xFFFFFFF;
          if ( v120 )
          {
            v121 = 24LL * *(unsigned int *)(v117 + 56) + 8;
            v122 = 0;
            do
            {
              v123 = v117 - 24LL * v120;
              if ( v119 )
                v123 = *(_QWORD *)(v117 - 8);
              if ( v38 == *(_QWORD *)(v123 + v121) )
              {
                v118 = 24 * v122;
                goto LABEL_150;
              }
              ++v122;
              v121 += 8;
            }
            while ( v120 != (_DWORD)v122 );
            v118 = 0x17FFFFFFE8LL;
          }
LABEL_150:
          v124 = v119 ? *(_QWORD *)(v117 - 8) : v117 - 24LL * v120;
          v125 = *(_QWORD *)(v124 + v118);
          v126 = 0x17FFFFFFE8LL;
          if ( v120 )
            break;
LABEL_157:
          if ( v125 != *(_QWORD *)(v124 + v126) )
            goto LABEL_181;
LABEL_158:
          v128 = *(_QWORD *)(v117 + 32);
          if ( !v128 )
            BUG();
          v117 = 0;
          if ( *(_BYTE *)(v128 - 8) == 77 )
            v117 = v128 - 24;
          if ( v116 == v117 )
            goto LABEL_162;
        }
        v127 = 0;
        do
        {
          if ( v113 == *(_QWORD *)(v124 + 24LL * *(unsigned int *)(v117 + 56) + 8 * v127 + 8) )
          {
            v126 = 24 * v127;
            goto LABEL_157;
          }
          ++v127;
        }
        while ( v120 != (_DWORD)v127 );
        if ( v125 == *(_QWORD *)(v124 + 0x17FFFFFFE8LL) )
          goto LABEL_158;
LABEL_181:
        if ( v116 == v117 )
LABEL_162:
          sub_1412190((__int64)&v169, v113);
LABEL_163:
        for ( j = *(_QWORD *)(j + 8); j; j = *(_QWORD *)(j + 8) )
        {
          v112 = sub_1648700(j);
          if ( (unsigned __int8)(*((_BYTE *)v112 + 16) - 25) <= 9u )
            goto LABEL_141;
        }
LABEL_166:
        if ( v171 == v170 )
          v129 = &v171[HIDWORD(v172)];
        else
          v129 = &v171[(unsigned int)v172];
        v130 = sub_15CC2D0((__int64)&v169, v44);
        if ( v171 == v170 )
          v131 = &v171[HIDWORD(v172)];
        else
          v131 = &v171[(unsigned int)v172];
        for ( ; v131 != v130; ++v130 )
        {
          if ( (unsigned __int64)*v130 < 0xFFFFFFFFFFFFFFFELL )
            break;
        }
        if ( v130 == v129 )
        {
          v148 = sub_1368AA0(*(__int64 **)(a1 + 216), v44);
          v132 = sub_1368AA0(*(__int64 **)(a1 + 216), v38);
          v133 = v171;
          v158 = v132;
          if ( v171 == v170 )
            v134 = (__int64)&v171[HIDWORD(v172)];
          else
            v134 = (__int64)&v171[(unsigned int)v172];
          if ( v171 != (__int64 *)v134 )
          {
            do
            {
              if ( (unsigned __int64)*v133 < 0xFFFFFFFFFFFFFFFELL )
                break;
              ++v133;
            }
            while ( (__int64 *)v134 != v133 );
          }
          v159[0] = v134;
          v159[1] = v134;
          v145 = v134;
          sub_19E4730((__int64)v159);
          v159[2] = (__int64)&v169;
          v135 = (__int64 *)v145;
          v159[3] = v169;
          if ( (__int64 *)v159[0] != v133 )
          {
            v146 = v38;
            v136 = v135;
            do
            {
              v137 = *v133;
              if ( v44 == sub_157F120(*v133) && v39 == sub_1D5D650(v137) )
              {
                v139 = sub_1368AA0(*(__int64 **)(a1 + 216), v137);
                sub_16AF570(&v158, v139);
              }
              for ( ++v133; v136 != v133; ++v133 )
              {
                if ( (unsigned __int64)*v133 < 0xFFFFFFFFFFFFFFFELL )
                  break;
              }
            }
            while ( (__int64 *)v159[0] != v133 );
            v38 = v146;
          }
          v138 = v158 * (unsigned int)dword_4FC2960;
          if ( v171 != v170 )
            _libc_free((unsigned __int64)v171);
          if ( v148 <= v138 )
            goto LABEL_46;
        }
        else
        {
          if ( v171 != v170 )
            _libc_free((unsigned __int64)v171);
LABEL_46:
          v141 = *(_QWORD *)(sub_157EBA0(v38) - 24);
          v46 = sub_157F0B0(v141);
          v152 = v46 != 0 && v141 != v46;
          if ( v152 )
          {
            sub_1AA7EA0(v141, 0, 0, 0, 0, a3, a4, a5, a6, v47, v48, a9, a10);
          }
          else
          {
            v49 = sub_157F280(v141);
            v144 = v52;
            v53 = v49;
            while ( v144 != v53 )
            {
              v54 = *(_DWORD *)(v53 + 20) & 0xFFFFFFF;
              if ( v54 )
              {
                v55 = 0;
                v56 = 24LL * *(unsigned int *)(v53 + 56) + 8;
                while ( 1 )
                {
                  v57 = v53 - 24LL * v54;
                  if ( (*(_BYTE *)(v53 + 23) & 0x40) != 0 )
                    v57 = *(_QWORD *)(v53 - 8);
                  if ( v38 == *(_QWORD *)(v57 + v56) )
                    break;
                  v55 = (unsigned int)(v55 + 1);
                  v56 += 8;
                  if ( v54 == (_DWORD)v55 )
                    goto LABEL_96;
                }
              }
              else
              {
LABEL_96:
                v55 = 0xFFFFFFFFLL;
              }
              v58 = sub_15F5350(v53, v55, 0);
              v61 = v58;
              if ( *(_BYTE *)(v58 + 16) == 77 && v38 == *(_QWORD *)(v58 + 40) )
              {
                if ( (*(_DWORD *)(v58 + 20) & 0xFFFFFFF) != 0 )
                {
                  v104 = 0;
                  v105 = 8LL * (*(_DWORD *)(v58 + 20) & 0xFFFFFFF);
                  do
                  {
                    v108 = v104 + 24LL * *(unsigned int *)(v61 + 56) + 8;
                    if ( (*(_BYTE *)(v61 + 23) & 0x40) != 0 )
                      v106 = *(_QWORD *)(v61 - 8);
                    else
                      v106 = v61 - 24LL * (*(_DWORD *)(v61 + 20) & 0xFFFFFFF);
                    v107 = 3 * v104;
                    v104 += 8;
                    sub_1704F80(v53, *(_QWORD *)(v106 + v107), *(_QWORD *)(v106 + v108), v107, v59, v60);
                  }
                  while ( v104 != v105 );
                }
              }
              else
              {
                v62 = *(_QWORD *)(v38 + 48);
                if ( !v62 )
                  BUG();
                if ( *(_BYTE *)(v62 - 8) == 77 )
                {
                  LODWORD(v63) = *(_DWORD *)(v62 - 4) & 0xFFFFFFF;
                  if ( (_DWORD)v63 )
                  {
                    v63 = (unsigned int)v63;
                    v64 = v62 - 24;
                    v143 = v38;
                    v65 = 0;
                    v66 = v58 + 8;
                    v67 = v62 - 24;
                    v68 = 8LL * (unsigned int)v63;
                    v69 = v58;
                    v70 = v62;
                    do
                    {
                      if ( (*(_BYTE *)(v70 - 1) & 0x40) != 0 )
                        v71 = *(_QWORD *)(v70 - 32);
                      else
                        v71 = v67 - 24LL * (*(_DWORD *)(v70 - 4) & 0xFFFFFFF);
                      v72 = *(_QWORD *)(v65 + v71 + 24LL * *(unsigned int *)(v70 + 32) + 8);
                      v73 = *(_DWORD *)(v53 + 20) & 0xFFFFFFF;
                      if ( v73 == *(_DWORD *)(v53 + 56) )
                      {
                        v147 = v68;
                        v150 = v66;
                        v153 = v72;
                        sub_15F55D0(v53, v63, v72, v68, v66, v64);
                        v68 = v147;
                        v66 = v150;
                        v72 = v153;
                        v73 = *(_DWORD *)(v53 + 20) & 0xFFFFFFF;
                      }
                      v74 = (v73 + 1) & 0xFFFFFFF;
                      v75 = v74 | *(_DWORD *)(v53 + 20) & 0xF0000000;
                      *(_DWORD *)(v53 + 20) = v75;
                      if ( (v75 & 0x40000000) != 0 )
                        v76 = *(_QWORD *)(v53 - 8);
                      else
                        v76 = v53 - 24 * v74;
                      v77 = (__int64 *)(v76 + 24LL * (unsigned int)(v74 - 1));
                      if ( *v77 )
                      {
                        v78 = v77[1];
                        v79 = v77[2] & 0xFFFFFFFFFFFFFFFCLL;
                        *(_QWORD *)v79 = v78;
                        if ( v78 )
                          *(_QWORD *)(v78 + 16) = *(_QWORD *)(v78 + 16) & 3LL | v79;
                      }
                      *v77 = v69;
                      v80 = *(_QWORD *)(v69 + 8);
                      v77[1] = v80;
                      if ( v80 )
                        *(_QWORD *)(v80 + 16) = (unsigned __int64)(v77 + 1) | *(_QWORD *)(v80 + 16) & 3LL;
                      v77[2] = v66 | v77[2] & 3;
                      *(_QWORD *)(v69 + 8) = v77;
                      v81 = *(_DWORD *)(v53 + 20) & 0xFFFFFFF;
                      v82 = (unsigned int)(v81 - 1);
                      if ( (*(_BYTE *)(v53 + 23) & 0x40) != 0 )
                        v83 = *(_QWORD *)(v53 - 8);
                      else
                        v83 = v53 - 24 * v81;
                      v65 += 8;
                      v63 = 3LL * *(unsigned int *)(v53 + 56);
                      *(_QWORD *)(v83 + 8 * v82 + 24LL * *(unsigned int *)(v53 + 56) + 8) = v72;
                    }
                    while ( v65 != v68 );
                    v38 = v143;
                  }
                }
                else
                {
                  v88 = *(_QWORD *)(v38 + 8);
                  if ( v88 )
                  {
                    while ( 1 )
                    {
                      v89 = sub_1648700(v88);
                      v92 = *((unsigned __int8 *)v89 + 16);
                      v93 = (unsigned int)(v92 - 25);
                      if ( (unsigned __int8)(v92 - 25) <= 9u )
                        break;
                      v88 = *(_QWORD *)(v88 + 8);
                      if ( !v88 )
                        goto LABEL_77;
                    }
                    v154 = v38;
LABEL_112:
                    v100 = v89[5];
                    v101 = *(_DWORD *)(v53 + 20) & 0xFFFFFFF;
                    if ( v101 == *(_DWORD *)(v53 + 56) )
                    {
                      sub_15F55D0(v53, v55, v93, v92, v90, v91);
                      v101 = *(_DWORD *)(v53 + 20) & 0xFFFFFFF;
                    }
                    v102 = (v101 + 1) & 0xFFFFFFF;
                    v103 = v102 | *(_DWORD *)(v53 + 20) & 0xF0000000;
                    *(_DWORD *)(v53 + 20) = v103;
                    if ( (v103 & 0x40000000) != 0 )
                      v94 = *(_QWORD *)(v53 - 8);
                    else
                      v94 = v53 - 24 * v102;
                    v95 = (__int64 *)(v94 + 24LL * (unsigned int)(v102 - 1));
                    if ( *v95 )
                    {
                      v96 = v95[1];
                      v97 = v95[2] & 0xFFFFFFFFFFFFFFFCLL;
                      *(_QWORD *)v97 = v96;
                      if ( v96 )
                        *(_QWORD *)(v96 + 16) = *(_QWORD *)(v96 + 16) & 3LL | v97;
                    }
                    *v95 = v61;
                    v98 = *(_QWORD *)(v61 + 8);
                    v95[1] = v98;
                    if ( v98 )
                      *(_QWORD *)(v98 + 16) = (unsigned __int64)(v95 + 1) | *(_QWORD *)(v98 + 16) & 3LL;
                    v95[2] = (v61 + 8) | v95[2] & 3;
                    *(_QWORD *)(v61 + 8) = v95;
                    v99 = *(_DWORD *)(v53 + 20) & 0xFFFFFFF;
                    if ( (*(_BYTE *)(v53 + 23) & 0x40) != 0 )
                      v55 = *(_QWORD *)(v53 - 8);
                    else
                      v55 = v53 - 24 * v99;
                    *(_QWORD *)(v55 + 8LL * (unsigned int)(v99 - 1) + 24LL * *(unsigned int *)(v53 + 56) + 8) = v100;
                    while ( 1 )
                    {
                      v88 = *(_QWORD *)(v88 + 8);
                      if ( !v88 )
                        break;
                      v89 = sub_1648700(v88);
                      v92 = *((unsigned __int8 *)v89 + 16);
                      v93 = (unsigned int)(v92 - 25);
                      if ( (unsigned __int8)(v92 - 25) <= 9u )
                        goto LABEL_112;
                    }
                    v38 = v154;
                  }
                }
              }
LABEL_77:
              v84 = *(_QWORD *)(v53 + 32);
              if ( !v84 )
                BUG();
              v53 = 0;
              if ( *(_BYTE *)(v84 - 8) == 77 )
                v53 = v84 - 24;
            }
            sub_164D160(v38, v141, a3, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6, v50, v51, a9, a10);
            sub_157F980(v38);
            v152 = 1;
          }
        }
LABEL_82:
        v156 += 24;
        if ( v156 == v142 )
        {
          v85 = v175;
          v37 = &v175[24 * (unsigned int)v176];
          if ( v175 != (_BYTE *)v37 )
          {
            do
            {
              v86 = *(v37 - 1);
              v37 -= 3;
              if ( v86 != 0 && v86 != -8 && v86 != -16 )
                sub_1649B30(v37);
            }
            while ( v85 != v37 );
            v37 = v175;
          }
          goto LABEL_89;
        }
      }
      v109 = &v164[8 * HIDWORD(v166)];
      if ( v164 == v109 )
      {
        v41 = v164;
      }
      else
      {
        do
        {
          if ( v38 == *v40 )
            break;
          ++v40;
        }
        while ( v109 != (_BYTE *)v40 );
        v41 = &v164[8 * HIDWORD(v166)];
      }
LABEL_131:
      while ( v109 != (_BYTE *)v40 )
      {
        if ( *v40 < 0xFFFFFFFFFFFFFFFELL )
          break;
        ++v40;
      }
      goto LABEL_39;
    }
LABEL_89:
    if ( v37 != (_QWORD *)v177 )
      _libc_free((unsigned __int64)v37);
  }
  if ( v160 != dest )
    _libc_free((unsigned __int64)v160);
  if ( v165 != v164 )
    _libc_free((unsigned __int64)v165);
  return v152;
}
