// Function: sub_1BD9F80
// Address: 0x1bd9f80
//
__int64 __fastcall sub_1BD9F80(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        __int64 *a5,
        __int64 *a6,
        __m128i a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14)
{
  __int64 *v15; // rax
  unsigned __int64 *v18; // rdi
  unsigned __int64 v19; // r13
  _DWORD *v20; // rax
  _BYTE *v21; // rdx
  _DWORD *i; // rdx
  unsigned int v23; // eax
  unsigned __int64 v24; // r11
  unsigned __int64 v25; // r12
  _QWORD *v26; // r15
  _QWORD *v27; // r13
  unsigned int v28; // edx
  unsigned int v29; // esi
  __int64 v30; // rcx
  __int64 v31; // rcx
  unsigned int *v32; // r14
  __int64 *v33; // rbx
  _QWORD *v34; // rdi
  int v35; // edx
  __int64 v36; // rax
  _QWORD *v37; // rcx
  __int64 v38; // r8
  unsigned int v39; // esi
  __int64 v40; // rax
  int v41; // r10d
  __int64 v42; // rdx
  __int64 *v43; // rcx
  __int64 v44; // rdi
  _QWORD *v45; // rdi
  int v46; // edx
  unsigned int v47; // ecx
  __int64 *v48; // rax
  unsigned int v49; // esi
  unsigned int v50; // ecx
  unsigned int v51; // edx
  __int64 v52; // rdx
  unsigned int v53; // esi
  unsigned int v54; // eax
  unsigned int v55; // edx
  __int64 v56; // rdx
  int v57; // edx
  __int64 v58; // rsi
  int v59; // r14d
  __int64 *v60; // rdi
  __int64 *v61; // r12
  _QWORD *v62; // rdi
  int v63; // ecx
  unsigned int v64; // edx
  __int64 v65; // rsi
  __int64 v66; // r13
  _QWORD *v67; // rsi
  int v68; // edx
  unsigned int v69; // eax
  __int64 v70; // rcx
  _QWORD *v71; // rax
  _BYTE *v72; // r14
  unsigned int v73; // r14d
  unsigned __int8 v74; // al
  unsigned __int8 v75; // r13
  unsigned __int64 v76; // r10
  __int64 v77; // rbx
  __int64 *v78; // r12
  __int64 *v79; // r14
  __int64 v80; // rsi
  __int64 *v81; // rdi
  __int64 *v82; // rax
  __int64 *v83; // rcx
  __int64 v84; // rax
  _QWORD *v85; // rdi
  int v86; // esi
  unsigned int v87; // edx
  __int64 *v88; // rax
  int v89; // edi
  unsigned int v90; // eax
  __int64 v91; // rcx
  unsigned int v92; // esi
  unsigned int v93; // eax
  unsigned int v94; // edx
  unsigned int v95; // edi
  int v96; // edi
  int v97; // r10d
  _QWORD *v98; // rdi
  int v99; // ecx
  unsigned int v100; // edx
  __int64 v101; // rsi
  __int64 *v102; // rax
  _QWORD *v103; // rdi
  int v104; // ecx
  unsigned int v105; // edx
  __int64 v106; // rsi
  _BYTE *v107; // rdx
  __int64 **v109; // r11
  int v110; // r10d
  _QWORD *v111; // r9
  _QWORD *v112; // r8
  int v113; // r14d
  unsigned int v114; // esi
  int v115; // r9d
  _QWORD *v116; // rax
  int v117; // r10d
  _BYTE *v118; // rsi
  int v119; // edi
  unsigned int v120; // esi
  int v121; // r14d
  __int64 *v122; // rcx
  _QWORD *v123; // r11
  unsigned int v124; // esi
  int v125; // r14d
  int v126; // r14d
  __int64 v127; // rsi
  __int64 *v128; // rdi
  __int64 v129; // r8
  _QWORD *v130; // r8
  int v131; // r14d
  unsigned int v132; // esi
  __int64 v133; // rdi
  int v134; // r9d
  __int64 v135; // [rsp+8h] [rbp-298h]
  unsigned int v137; // [rsp+28h] [rbp-278h]
  unsigned int v138; // [rsp+2Ch] [rbp-274h]
  int v139; // [rsp+30h] [rbp-270h]
  unsigned __int8 v140; // [rsp+30h] [rbp-270h]
  __int64 v141; // [rsp+30h] [rbp-270h]
  int v142; // [rsp+38h] [rbp-268h]
  _BYTE *v143; // [rsp+38h] [rbp-268h]
  unsigned int v144; // [rsp+40h] [rbp-260h]
  unsigned int *v145; // [rsp+58h] [rbp-248h]
  _BYTE *v146; // [rsp+58h] [rbp-248h]
  __int64 v147; // [rsp+60h] [rbp-240h] BYREF
  __int64 v148; // [rsp+68h] [rbp-238h]
  _QWORD *v149; // [rsp+70h] [rbp-230h] BYREF
  unsigned int v150; // [rsp+78h] [rbp-228h]
  __int64 v151; // [rsp+90h] [rbp-210h] BYREF
  __int64 v152; // [rsp+98h] [rbp-208h]
  __int64 v153; // [rsp+A0h] [rbp-200h]
  __int64 v154; // [rsp+A8h] [rbp-1F8h]
  _BYTE *v155; // [rsp+B0h] [rbp-1F0h] BYREF
  _BYTE *v156; // [rsp+B8h] [rbp-1E8h]
  _BYTE *v157; // [rsp+C0h] [rbp-1E0h]
  __int64 v158; // [rsp+D0h] [rbp-1D0h] BYREF
  __int64 v159; // [rsp+D8h] [rbp-1C8h]
  _QWORD *v160; // [rsp+E0h] [rbp-1C0h] BYREF
  unsigned int v161; // [rsp+E8h] [rbp-1B8h]
  _BYTE *v162; // [rsp+120h] [rbp-180h] BYREF
  __int64 v163; // [rsp+128h] [rbp-178h]
  _BYTE v164[64]; // [rsp+130h] [rbp-170h] BYREF
  __int64 *v165; // [rsp+170h] [rbp-130h] BYREF
  __int64 v166; // [rsp+178h] [rbp-128h]
  _BYTE v167[64]; // [rsp+180h] [rbp-120h] BYREF
  __int64 v168; // [rsp+1C0h] [rbp-E0h] BYREF
  _BYTE *v169; // [rsp+1C8h] [rbp-D8h]
  _BYTE *v170; // [rsp+1D0h] [rbp-D0h]
  __int64 v171; // [rsp+1D8h] [rbp-C8h]
  int v172; // [rsp+1E0h] [rbp-C0h]
  _BYTE v173[184]; // [rsp+1E8h] [rbp-B8h] BYREF

  v15 = (__int64 *)&v149;
  v151 = 0;
  v152 = 0;
  v153 = 0;
  v154 = 0;
  v155 = 0;
  v156 = 0;
  v157 = 0;
  v147 = 0;
  v148 = 1;
  do
    *v15++ = -8;
  while ( v15 != &v151 );
  v158 = 0;
  v159 = 1;
  v18 = (unsigned __int64 *)&v160;
  do
  {
    *v18 = -8;
    v18 += 2;
  }
  while ( v18 != (unsigned __int64 *)&v162 );
  v19 = a3 - 1;
  v168 = 0;
  v169 = v173;
  v170 = v173;
  v20 = v164;
  v171 = 16;
  v21 = v164;
  v172 = 0;
  v162 = v164;
  v163 = 0x1000000000LL;
  v137 = a3;
  v138 = a3 - 1;
  if ( a3 == 1 )
  {
    a6 = (__int64 *)v164;
    v23 = 1;
    goto LABEL_15;
  }
  if ( v19 > 0x10 )
  {
    sub_16CD150((__int64)v18, v164, a3 - 1, 4, (int)a5, (int)a6);
    v21 = v162;
    v20 = &v162[4 * (unsigned int)v163];
  }
  for ( i = &v21[4 * v19]; i != v20; ++v20 )
  {
    if ( v20 )
      *v20 = 0;
  }
  LODWORD(v163) = a3 - 1;
  if ( a3 )
  {
    a6 = (__int64 *)v162;
    v23 = a3;
LABEL_15:
    v135 = a4;
    v24 = (unsigned __int64)a6;
    v25 = v23 - 1;
    v26 = (_QWORD *)a1;
    v27 = (_QWORD *)(a2 + 8 * v25);
    v144 = v23 + v138;
    v142 = 2 * v23;
    while ( 1 )
    {
      if ( v138 )
      {
        LODWORD(a5) = v137;
        v28 = 0;
        LODWORD(a6) = v144;
        v29 = 1 - v23;
        do
        {
          if ( (unsigned int)v25 >= v29 + v23 )
          {
            v30 = v28++;
            *(_DWORD *)(v24 + 4 * v30) = v142 - 2 - v23;
            v24 = (unsigned __int64)v162;
          }
          if ( v137 > v23 )
          {
            v31 = v28++;
            *(_DWORD *)(v24 + 4 * v31) = v23;
            v24 = (unsigned __int64)v162;
          }
          ++v23;
        }
        while ( v144 != v23 );
      }
      v145 = (unsigned int *)(v24 + 4LL * (unsigned int)v163);
      if ( v145 != (unsigned int *)v24 )
        break;
LABEL_36:
      --v144;
      --v27;
      v142 -= 2;
      if ( !(_DWORD)v25 )
      {
        a1 = (__int64)v26;
        a4 = v135;
        goto LABEL_65;
      }
      v23 = v25;
      v24 = (unsigned __int64)v162;
      LODWORD(v25) = v25 - 1;
    }
    v139 = v25;
    v32 = (unsigned int *)v24;
    while ( 1 )
    {
      v25 = a2 + 8LL * *v32;
      if ( (unsigned __int8)sub_385F290(*(_QWORD *)v25, *v27, v26[8], *v26, 1) )
        break;
      if ( v145 == ++v32 )
      {
        LODWORD(v25) = v139;
        goto LABEL_36;
      }
    }
    v33 = (__int64 *)v25;
    LODWORD(v25) = v139;
    if ( (v148 & 1) != 0 )
    {
      v34 = &v149;
      v35 = 3;
    }
    else
    {
      v53 = v150;
      v34 = v149;
      v35 = v150 - 1;
      if ( !v150 )
      {
        v54 = v148;
        ++v147;
        v37 = 0;
        v55 = ((unsigned int)v148 >> 1) + 1;
        goto LABEL_50;
      }
    }
    LODWORD(v36) = v35 & (((unsigned int)*v27 >> 9) ^ ((unsigned int)*v27 >> 4));
    v37 = &v34[(unsigned int)v36];
    v38 = *v37;
    if ( *v37 == *v27 )
    {
LABEL_30:
      v39 = v154;
      if ( (_DWORD)v154 )
        goto LABEL_31;
      goto LABEL_56;
    }
    v110 = 1;
    v111 = 0;
    while ( v38 != -8 )
    {
      if ( !v111 && v38 == -16 )
        v111 = v37;
      v36 = v35 & (unsigned int)(v36 + v110);
      v37 = &v34[v36];
      v38 = *v37;
      if ( *v27 == *v37 )
        goto LABEL_30;
      ++v110;
    }
    v54 = v148;
    if ( v111 )
      v37 = v111;
    ++v147;
    v55 = ((unsigned int)v148 >> 1) + 1;
    if ( (v148 & 1) != 0 )
    {
      v53 = 4;
      if ( 4 * v55 >= 0xC )
        goto LABEL_194;
      goto LABEL_51;
    }
    v53 = v150;
LABEL_50:
    if ( 4 * v55 >= 3 * v53 )
    {
LABEL_194:
      sub_1BCDA80((__int64)&v147, 2 * v53);
      if ( (v148 & 1) != 0 )
      {
        v112 = &v149;
        v113 = 3;
        goto LABEL_196;
      }
      v112 = v149;
      if ( v150 )
      {
        v113 = v150 - 1;
LABEL_196:
        v114 = v113 & (((unsigned int)*v27 >> 9) ^ ((unsigned int)*v27 >> 4));
        v37 = &v112[v114];
        v54 = v148;
        v56 = *v37;
        if ( *v27 == *v37 )
          goto LABEL_53;
        v115 = 1;
        v116 = 0;
        while ( v56 != -8 )
        {
          if ( !v116 && v56 == -16 )
            v116 = v37;
          v114 = v113 & (v115 + v114);
          v37 = &v112[v114];
          v56 = *v37;
          if ( *v27 == *v37 )
            goto LABEL_259;
          ++v115;
        }
        v56 = *v27;
        if ( !v116 )
          goto LABEL_259;
        goto LABEL_258;
      }
LABEL_312:
      LODWORD(v148) = (2 * ((unsigned int)v148 >> 1) + 2) | v148 & 1;
      BUG();
    }
LABEL_51:
    if ( v53 - HIDWORD(v148) - v55 > v53 >> 3 )
    {
      v56 = *v27;
      goto LABEL_53;
    }
    sub_1BCDA80((__int64)&v147, v53);
    if ( (v148 & 1) != 0 )
    {
      v130 = &v149;
      v131 = 3;
    }
    else
    {
      v130 = v149;
      if ( !v150 )
        goto LABEL_312;
      v131 = v150 - 1;
    }
    v56 = *v27;
    v132 = v131 & (((unsigned int)*v27 >> 9) ^ ((unsigned int)*v27 >> 4));
    v37 = &v130[v132];
    v54 = v148;
    v133 = *v37;
    if ( *v37 == *v27 )
      goto LABEL_53;
    v134 = 1;
    v116 = 0;
    while ( v133 != -8 )
    {
      if ( v133 == -16 && !v116 )
        v116 = v37;
      v132 = v131 & (v134 + v132);
      v37 = &v130[v132];
      v133 = *v37;
      if ( v56 == *v37 )
        goto LABEL_259;
      ++v134;
    }
    if ( !v116 )
    {
LABEL_259:
      v54 = v148;
LABEL_53:
      LODWORD(v148) = (2 * (v54 >> 1) + 2) | v54 & 1;
      if ( *v37 != -8 )
        --HIDWORD(v148);
      *v37 = v56;
      v39 = v154;
      if ( (_DWORD)v154 )
      {
LABEL_31:
        v40 = *v33;
        v41 = 1;
        a6 = 0;
        LODWORD(v42) = (v39 - 1) & (((unsigned int)*v33 >> 9) ^ ((unsigned int)*v33 >> 4));
        v43 = (__int64 *)(v152 + 8LL * (unsigned int)v42);
        v44 = *v43;
        if ( *v33 == *v43 )
          goto LABEL_32;
        while ( v44 != -8 )
        {
          if ( v44 == -16 && !a6 )
            a6 = v43;
          v42 = (v39 - 1) & ((_DWORD)v42 + v41);
          v43 = (__int64 *)(v152 + 8 * v42);
          v44 = *v43;
          if ( v40 == *v43 )
            goto LABEL_32;
          ++v41;
        }
        if ( a6 )
          v43 = a6;
        ++v151;
        v57 = v153 + 1;
        if ( 4 * ((int)v153 + 1) < 3 * v39 )
        {
          if ( v39 - HIDWORD(v153) - v57 > v39 >> 3 )
          {
LABEL_217:
            LODWORD(v153) = v57;
            if ( *v43 != -8 )
              --HIDWORD(v153);
            *v43 = v40;
            v118 = v156;
            if ( v156 == v157 )
            {
              sub_190D490((__int64)&v155, v156, v33);
            }
            else
            {
              if ( v156 )
              {
                *(_QWORD *)v156 = *v33;
                v118 = v156;
              }
              v156 = v118 + 8;
            }
LABEL_32:
            if ( (v159 & 1) != 0 )
            {
              v45 = &v160;
              v46 = 3;
            }
            else
            {
              v49 = v161;
              v45 = v160;
              v46 = v161 - 1;
              if ( !v161 )
              {
                v50 = v159;
                ++v158;
                v48 = 0;
                v51 = ((unsigned int)v159 >> 1) + 1;
                goto LABEL_41;
              }
            }
            v47 = v46 & (((unsigned int)*v33 >> 9) ^ ((unsigned int)*v33 >> 4));
            v48 = &v45[2 * v47];
            a5 = (__int64 *)*v48;
            if ( *v48 == *v33 )
            {
LABEL_35:
              v48[1] = *v27;
              goto LABEL_36;
            }
            v117 = 1;
            a6 = 0;
            while ( a5 != (__int64 *)-8LL )
            {
              if ( !a6 && a5 == (__int64 *)-16LL )
                a6 = v48;
              v47 = v46 & (v117 + v47);
              v48 = &v45[2 * v47];
              a5 = (__int64 *)*v48;
              if ( *v33 == *v48 )
                goto LABEL_35;
              ++v117;
            }
            v50 = v159;
            LODWORD(a5) = 12;
            v49 = 4;
            if ( a6 )
              v48 = a6;
            ++v158;
            v51 = ((unsigned int)v159 >> 1) + 1;
            if ( (v159 & 1) != 0 )
            {
LABEL_42:
              if ( (unsigned int)a5 > 4 * v51 )
              {
                if ( v49 - HIDWORD(v159) - v51 > v49 >> 3 )
                {
                  v52 = *v33;
LABEL_45:
                  LODWORD(v159) = (2 * (v50 >> 1) + 2) | v50 & 1;
                  if ( *v48 != -8 )
                    --HIDWORD(v159);
                  *v48 = v52;
                  v48[1] = 0;
                  goto LABEL_35;
                }
                sub_196D4F0((__int64)&v158, v49);
                if ( (v159 & 1) != 0 )
                {
                  v123 = &v160;
                  LODWORD(a5) = 3;
LABEL_237:
                  v50 = v159;
                  v124 = (unsigned int)a5 & (((unsigned int)*v33 >> 9) ^ ((unsigned int)*v33 >> 4));
                  v48 = &v123[2 * v124];
                  v52 = *v48;
                  if ( *v33 == *v48 )
                    goto LABEL_45;
                  v125 = 1;
                  v122 = 0;
                  while ( v52 != -8 )
                  {
                    if ( v52 == -16 && !v122 )
                      v122 = v48;
                    LODWORD(a6) = v125 + 1;
                    v124 = (unsigned int)a5 & (v125 + v124);
                    v48 = &v123[2 * v124];
                    v52 = *v48;
                    if ( *v33 == *v48 )
                      goto LABEL_243;
                    ++v125;
                  }
                  v52 = *v33;
                  goto LABEL_241;
                }
                v123 = v160;
                if ( v161 )
                {
                  LODWORD(a5) = v161 - 1;
                  goto LABEL_237;
                }
LABEL_310:
                LODWORD(v159) = (2 * ((unsigned int)v159 >> 1) + 2) | v159 & 1;
                BUG();
              }
              sub_196D4F0((__int64)&v158, 2 * v49);
              if ( (v159 & 1) != 0 )
              {
                a5 = (__int64 *)&v160;
                v119 = 3;
              }
              else
              {
                a5 = v160;
                if ( !v161 )
                  goto LABEL_310;
                v119 = v161 - 1;
              }
              v52 = *v33;
              v50 = v159;
              v120 = v119 & (((unsigned int)*v33 >> 9) ^ ((unsigned int)*v33 >> 4));
              v48 = &a5[2 * v120];
              a6 = (__int64 *)*v48;
              if ( *v48 == *v33 )
                goto LABEL_45;
              v121 = 1;
              v122 = 0;
              while ( a6 != (__int64 *)-8LL )
              {
                if ( !v122 && a6 == (__int64 *)-16LL )
                  v122 = v48;
                v120 = v119 & (v121 + v120);
                v48 = &a5[2 * v120];
                a6 = (__int64 *)*v48;
                if ( v52 == *v48 )
                  goto LABEL_243;
                ++v121;
              }
LABEL_241:
              if ( v122 )
                v48 = v122;
LABEL_243:
              v50 = v159;
              goto LABEL_45;
            }
            v49 = v161;
LABEL_41:
            LODWORD(a5) = 3 * v49;
            goto LABEL_42;
          }
          sub_196DBF0((__int64)&v151, v39);
          if ( (_DWORD)v154 )
          {
            v40 = *v33;
            LODWORD(a6) = v152;
            v126 = 1;
            LODWORD(v127) = (v154 - 1) & (((unsigned int)*v33 >> 9) ^ ((unsigned int)*v33 >> 4));
            v43 = (__int64 *)(v152 + 8LL * (unsigned int)v127);
            v57 = v153 + 1;
            v128 = 0;
            v129 = *v43;
            if ( *v43 != *v33 )
            {
              while ( v129 != -8 )
              {
                if ( v129 == -16 && !v128 )
                  v128 = v43;
                v127 = ((_DWORD)v154 - 1) & (unsigned int)(v127 + v126);
                v43 = (__int64 *)(v152 + 8 * v127);
                v129 = *v43;
                if ( v40 == *v43 )
                  goto LABEL_217;
                ++v126;
              }
              if ( v128 )
                v43 = v128;
            }
            goto LABEL_217;
          }
LABEL_311:
          LODWORD(v153) = v153 + 1;
          BUG();
        }
LABEL_57:
        sub_196DBF0((__int64)&v151, 2 * v39);
        if ( (_DWORD)v154 )
        {
          LODWORD(a6) = v152;
          v57 = v153 + 1;
          LODWORD(v58) = (v154 - 1) & (((unsigned int)*v33 >> 9) ^ ((unsigned int)*v33 >> 4));
          v43 = (__int64 *)(v152 + 8LL * (unsigned int)v58);
          v40 = *v43;
          if ( *v33 != *v43 )
          {
            v59 = 1;
            v60 = 0;
            while ( v40 != -8 )
            {
              if ( v40 == -16 && !v60 )
                v60 = v43;
              v58 = ((_DWORD)v154 - 1) & (unsigned int)(v58 + v59);
              v43 = (__int64 *)(v152 + 8 * v58);
              v40 = *v43;
              if ( *v33 == *v43 )
                goto LABEL_217;
              ++v59;
            }
            v40 = *v33;
            if ( v60 )
              v43 = v60;
          }
          goto LABEL_217;
        }
        goto LABEL_311;
      }
LABEL_56:
      ++v151;
      goto LABEL_57;
    }
LABEL_258:
    v37 = v116;
    goto LABEL_259;
  }
LABEL_65:
  v140 = 0;
  v143 = v155;
  v146 = v156;
  if ( v155 != v156 )
  {
    v61 = &v168;
    do
    {
      v66 = *((_QWORD *)v146 - 1);
      if ( (v148 & 1) != 0 )
      {
        v62 = &v149;
        v63 = 3;
      }
      else
      {
        v62 = v149;
        if ( !v150 )
          goto LABEL_76;
        v63 = v150 - 1;
      }
      LODWORD(a5) = 1;
      v64 = v63 & (((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4));
      v65 = v62[v64];
      if ( v66 == v65 )
        goto LABEL_69;
      while ( v65 != -8 )
      {
        LODWORD(a6) = (_DWORD)a5 + 1;
        v64 = v63 & ((_DWORD)a5 + v64);
        v65 = v62[v64];
        if ( v66 == v65 )
          goto LABEL_69;
        LODWORD(a5) = (_DWORD)a5 + 1;
      }
LABEL_76:
      v165 = (__int64 *)v167;
      v166 = 0x800000000LL;
      if ( (v148 & 1) != 0 )
      {
LABEL_77:
        v67 = &v149;
        v68 = 3;
        goto LABEL_78;
      }
      while ( 1 )
      {
        v67 = v149;
        v68 = v150 - 1;
        if ( v150 )
        {
LABEL_78:
          v69 = v68 & (((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4));
          v70 = v67[v69];
          if ( v70 == v66 )
            goto LABEL_79;
          v96 = 1;
          while ( v70 != -8 )
          {
            LODWORD(a5) = v96 + 1;
            v69 = v68 & (v96 + v69);
            v70 = v67[v69];
            if ( v70 == v66 )
              goto LABEL_79;
            ++v96;
          }
        }
        if ( !(_DWORD)v154 )
          goto LABEL_84;
        v89 = 1;
        v90 = (v154 - 1) & (((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4));
        v91 = *(_QWORD *)(v152 + 8LL * v90);
        if ( v91 != v66 )
          break;
LABEL_79:
        v71 = v169;
        if ( v170 == v169 )
        {
          v72 = &v169[8 * HIDWORD(v171)];
          if ( v169 == v72 )
          {
            v107 = v169;
          }
          else
          {
            do
            {
              if ( *v71 == v66 )
                break;
              ++v71;
            }
            while ( v72 != (_BYTE *)v71 );
            v107 = &v169[8 * HIDWORD(v171)];
          }
        }
        else
        {
          v72 = &v170[8 * (unsigned int)v171];
          v71 = sub_16CC9F0((__int64)v61, v66);
          if ( *v71 == v66 )
          {
            if ( v170 == v169 )
              v107 = &v170[8 * HIDWORD(v171)];
            else
              v107 = &v170[8 * (unsigned int)v171];
          }
          else
          {
            if ( v170 != v169 )
            {
              v71 = &v170[8 * (unsigned int)v171];
              goto LABEL_83;
            }
            v107 = &v170[8 * HIDWORD(v171)];
            v71 = v107;
          }
        }
        while ( v107 != (_BYTE *)v71 && *v71 >= 0xFFFFFFFFFFFFFFFELL )
          ++v71;
LABEL_83:
        if ( v72 != (_BYTE *)v71 )
          goto LABEL_84;
        v84 = (unsigned int)v166;
        if ( (unsigned int)v166 >= HIDWORD(v166) )
        {
          sub_16CD150((__int64)&v165, v167, 0, 8, (int)a5, (int)a6);
          v84 = (unsigned int)v166;
        }
        v165[v84] = v66;
        LODWORD(v166) = v166 + 1;
        if ( (v159 & 1) != 0 )
        {
          v85 = &v160;
          v86 = 3;
        }
        else
        {
          v92 = v161;
          v85 = v160;
          if ( !v161 )
          {
            v93 = v159;
            ++v158;
            a5 = 0;
            v94 = ((unsigned int)v159 >> 1) + 1;
LABEL_129:
            v95 = 3 * v92;
            goto LABEL_130;
          }
          v86 = v161 - 1;
        }
        v87 = v86 & (((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4));
        v88 = &v85[2 * v87];
        a6 = (__int64 *)*v88;
        if ( *v88 == v66 )
        {
          v66 = v88[1];
          goto LABEL_108;
        }
        v97 = 1;
        a5 = 0;
        while ( a6 != (__int64 *)-8LL )
        {
          if ( a6 != (__int64 *)-16LL || a5 )
            v88 = a5;
          LODWORD(a5) = v97 + 1;
          v87 = v86 & (v97 + v87);
          v109 = (__int64 **)&v85[2 * v87];
          a6 = *v109;
          if ( *v109 == (__int64 *)v66 )
          {
            v66 = (__int64)v109[1];
            goto LABEL_108;
          }
          ++v97;
          a5 = v88;
          v88 = &v85[2 * v87];
        }
        v95 = 12;
        v92 = 4;
        if ( !a5 )
          a5 = v88;
        v93 = v159;
        ++v158;
        v94 = ((unsigned int)v159 >> 1) + 1;
        if ( (v159 & 1) == 0 )
        {
          v92 = v161;
          goto LABEL_129;
        }
LABEL_130:
        if ( 4 * v94 >= v95 )
        {
          sub_196D4F0((__int64)&v158, 2 * v92);
          if ( (v159 & 1) != 0 )
          {
            v98 = &v160;
            v99 = 3;
          }
          else
          {
            v98 = v160;
            if ( !v161 )
              goto LABEL_313;
            v99 = v161 - 1;
          }
          v93 = v159;
          v100 = v99 & (((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4));
          a5 = &v98[2 * v100];
          v101 = *a5;
          if ( *a5 == v66 )
            goto LABEL_132;
          LODWORD(a6) = 1;
          v102 = 0;
          while ( v101 != -8 )
          {
            if ( !v102 && v101 == -16 )
              v102 = a5;
            v100 = v99 & ((_DWORD)a6 + v100);
            a5 = &v98[2 * v100];
            v101 = *a5;
            if ( *a5 == v66 )
              goto LABEL_153;
            LODWORD(a6) = (_DWORD)a6 + 1;
          }
        }
        else
        {
          if ( v92 - HIDWORD(v159) - v94 > v92 >> 3 )
            goto LABEL_132;
          sub_196D4F0((__int64)&v158, v92);
          if ( (v159 & 1) != 0 )
          {
            v103 = &v160;
            v104 = 3;
          }
          else
          {
            v103 = v160;
            if ( !v161 )
            {
LABEL_313:
              LODWORD(v159) = (2 * ((unsigned int)v159 >> 1) + 2) | v159 & 1;
              BUG();
            }
            v104 = v161 - 1;
          }
          v93 = v159;
          v105 = v104 & (((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4));
          a5 = &v103[2 * v105];
          v106 = *a5;
          if ( *a5 == v66 )
            goto LABEL_132;
          LODWORD(a6) = 1;
          v102 = 0;
          while ( v106 != -8 )
          {
            if ( v106 == -16 && !v102 )
              v102 = a5;
            v105 = v104 & ((_DWORD)a6 + v105);
            a5 = &v103[2 * v105];
            v106 = *a5;
            if ( *a5 == v66 )
              goto LABEL_153;
            LODWORD(a6) = (_DWORD)a6 + 1;
          }
        }
        if ( v102 )
          a5 = v102;
LABEL_153:
        v93 = v159;
LABEL_132:
        LODWORD(v159) = (2 * (v93 >> 1) + 2) | v93 & 1;
        if ( *a5 != -8 )
          --HIDWORD(v159);
        *a5 = v66;
        v66 = 0;
        a5[1] = 0;
LABEL_108:
        if ( (v148 & 1) != 0 )
          goto LABEL_77;
      }
      while ( v91 != -8 )
      {
        LODWORD(a5) = v89 + 1;
        v90 = (v154 - 1) & (v89 + v90);
        v91 = *(_QWORD *)(v152 + 8LL * v90);
        if ( v91 == v66 )
          goto LABEL_79;
        ++v89;
      }
LABEL_84:
      v73 = *(_DWORD *)(a4 + 1392);
      if ( v73 < *(_DWORD *)(a4 + 1396) )
      {
LABEL_154:
        a5 = v165;
        goto LABEL_155;
      }
      while ( 1 )
      {
        v74 = sub_1BD9A10(a7, a8, a9, a10, a11, a12, a13, a14, a1, v165, (unsigned int)v166, a4, v73, (int)a6);
        if ( v74 )
          break;
        v73 >>= 1;
        if ( v73 < *(_DWORD *)(a4 + 1396) )
          goto LABEL_154;
      }
      v75 = v74;
      a5 = &v165[(unsigned int)v166];
      if ( v165 == a5 )
      {
        v140 = v74;
        goto LABEL_155;
      }
      v141 = a1;
      v76 = (unsigned __int64)v170;
      v77 = (__int64)v61;
      v78 = v165;
      a6 = (__int64 *)v169;
      v79 = &v165[(unsigned int)v166];
LABEL_92:
      while ( 2 )
      {
        v80 = *v78;
        if ( a6 != (__int64 *)v76 )
          goto LABEL_90;
        v81 = &a6[HIDWORD(v171)];
        if ( a6 == v81 )
        {
LABEL_182:
          if ( HIDWORD(v171) < (unsigned int)v171 )
          {
            ++HIDWORD(v171);
            *v81 = v80;
            a6 = (__int64 *)v169;
            ++v168;
            v76 = (unsigned __int64)v170;
            goto LABEL_91;
          }
LABEL_90:
          sub_16CCBA0(v77, v80);
          v76 = (unsigned __int64)v170;
          a6 = (__int64 *)v169;
          goto LABEL_91;
        }
        v82 = a6;
        v83 = 0;
        while ( v80 != *v82 )
        {
          if ( *v82 == -2 )
            v83 = v82;
          if ( v81 == ++v82 )
          {
            if ( !v83 )
              goto LABEL_182;
            ++v78;
            *v83 = v80;
            v76 = (unsigned __int64)v170;
            --v172;
            a6 = (__int64 *)v169;
            ++v168;
            if ( v79 != v78 )
              goto LABEL_92;
            goto LABEL_101;
          }
        }
LABEL_91:
        if ( v79 != ++v78 )
          continue;
        break;
      }
LABEL_101:
      v61 = (__int64 *)v77;
      a5 = v165;
      a1 = v141;
      v140 = v75;
LABEL_155:
      if ( a5 != (__int64 *)v167 )
        _libc_free((unsigned __int64)a5);
LABEL_69:
      v146 -= 8;
    }
    while ( v143 != v146 );
  }
  if ( v162 != v164 )
    _libc_free((unsigned __int64)v162);
  if ( v170 != v169 )
    _libc_free((unsigned __int64)v170);
  if ( (v159 & 1) == 0 )
    j___libc_free_0(v160);
  if ( (v148 & 1) == 0 )
    j___libc_free_0(v149);
  if ( v155 )
    j_j___libc_free_0(v155, v157 - v155);
  j___libc_free_0(v152);
  return v140;
}
