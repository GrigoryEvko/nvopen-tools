// Function: sub_265CC40
// Address: 0x265cc40
//
__int64 __fastcall sub_265CC40(__int64 *a1, __int64 *a2, __m128i a3)
{
  __int64 *v3; // r13
  __int64 *v4; // r14
  __int64 v5; // r13
  __int64 v6; // r8
  __int64 v7; // r9
  _BYTE *v8; // r15
  __int64 *v9; // rax
  __int64 *v10; // r12
  __int64 v11; // rcx
  __int64 *v12; // rdx
  __int64 *v13; // rax
  __int64 *v14; // rdx
  char v15; // di
  __int64 *v16; // rax
  unsigned int v17; // r12d
  __int64 v18; // rax
  __int64 v19; // rbx
  __int64 *v20; // r15
  const char *v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rsi
  __int64 v24; // rdx
  unsigned __int64 v25; // rax
  const void *v26; // r12
  size_t v27; // rbx
  __int64 *v28; // r14
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 *v32; // r13
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 *v35; // rcx
  int v36; // eax
  _QWORD *v37; // rax
  __int64 v38; // rdx
  __int64 *v39; // r12
  __int64 v40; // r14
  __int64 v41; // r13
  const char *v42; // rax
  __int64 v43; // rdx
  __int64 v44; // r12
  __int64 m; // r13
  int v46; // eax
  __int64 v48; // rax
  __int64 v49; // r8
  __int64 v50; // r9
  __int64 v51; // rdx
  __int64 v52; // rcx
  unsigned __int64 v53; // rdx
  unsigned __int64 v54; // rbx
  _QWORD *v55; // rax
  int v56; // edi
  __int64 v57; // rax
  __int64 v58; // rcx
  unsigned __int64 *v59; // rax
  unsigned __int64 v60; // r10
  __int64 *v61; // r14
  __int64 v62; // rax
  __int64 v63; // r12
  __int64 v64; // rax
  __int64 v65; // rbx
  unsigned __int8 v66; // al
  __int64 *v67; // r13
  __int64 v68; // rdx
  __int64 v69; // rbx
  __m128i v70; // rax
  __int64 v71; // rdx
  unsigned int v72; // ebx
  _QWORD *v73; // rdx
  _BYTE *v74; // rdx
  __int64 v75; // rax
  __int64 v76; // rbx
  __int64 v77; // rax
  __int64 v78; // rdx
  _BYTE *v79; // rbx
  _BYTE *v80; // r12
  unsigned __int64 v81; // rdi
  __int64 v82; // rax
  _QWORD *v83; // rbx
  _QWORD *v84; // r12
  unsigned __int64 v85; // rdi
  unsigned __int64 v86; // rdi
  unsigned __int64 *v87; // r12
  int v88; // eax
  unsigned int v89; // ebx
  __int64 v90; // rax
  _BYTE *v91; // r13
  size_t v92; // r12
  _QWORD *v93; // rax
  __int64 v94; // r14
  __int64 *v95; // rax
  unsigned __int8 *v96; // rax
  char *v97; // r12
  unsigned __int64 *v98; // r13
  unsigned __int64 v99; // rdi
  __int64 v100; // rcx
  char v101; // al
  __int64 v102; // rcx
  __int64 v103; // rdx
  _QWORD *v104; // r14
  unsigned int v105; // esi
  int v106; // eax
  int v107; // eax
  __int64 v108; // r14
  __int64 v109; // rsi
  unsigned __int64 *v110; // r8
  unsigned __int64 v111; // rsi
  __int64 *v112; // rax
  __int64 v113; // rdx
  __int64 v114; // rcx
  unsigned int v115; // ebx
  bool v116; // al
  unsigned __int64 v117; // rax
  unsigned __int64 v118; // rdx
  unsigned int v119; // r9d
  unsigned int v120; // ecx
  _QWORD *v121; // rsi
  unsigned __int64 v122; // rax
  int v123; // r11d
  __int64 v124; // rdx
  __int64 v125; // rdx
  unsigned __int8 v126; // al
  _BYTE **v127; // rax
  _BYTE *v128; // rdi
  __int64 v129; // r13
  size_t v130; // rdx
  size_t v131; // rbx
  __int64 *v132; // r14
  __int64 *v133; // r12
  const void *v134; // r15
  __int64 v135; // r13
  unsigned __int64 v136; // rdi
  __int64 v137; // rbx
  int v138; // esi
  int v139; // edi
  __int64 v140; // [rsp+10h] [rbp-510h]
  __int64 v141; // [rsp+18h] [rbp-508h]
  __int64 v144; // [rsp+48h] [rbp-4D8h]
  __int64 v145; // [rsp+50h] [rbp-4D0h]
  __int64 v146; // [rsp+50h] [rbp-4D0h]
  __int64 *v147; // [rsp+58h] [rbp-4C8h]
  __int64 v148; // [rsp+60h] [rbp-4C0h]
  __int64 v149; // [rsp+78h] [rbp-4A8h]
  __int64 v150; // [rsp+88h] [rbp-498h]
  __int64 v151; // [rsp+88h] [rbp-498h]
  __int64 v152; // [rsp+88h] [rbp-498h]
  __int64 *v153; // [rsp+90h] [rbp-490h]
  _QWORD *v154; // [rsp+98h] [rbp-488h]
  char *v155; // [rsp+A0h] [rbp-480h]
  unsigned __int64 v156; // [rsp+A8h] [rbp-478h]
  __int64 *i; // [rsp+A8h] [rbp-478h]
  __int64 v158; // [rsp+A8h] [rbp-478h]
  __int64 v159; // [rsp+B0h] [rbp-470h]
  __int64 k; // [rsp+B0h] [rbp-470h]
  __int64 *v161; // [rsp+B0h] [rbp-470h]
  __int64 v162; // [rsp+B0h] [rbp-470h]
  __int64 *v163; // [rsp+B8h] [rbp-468h]
  __int64 *v164; // [rsp+B8h] [rbp-468h]
  __int64 v165; // [rsp+B8h] [rbp-468h]
  __int64 v166; // [rsp+B8h] [rbp-468h]
  unsigned __int8 v167; // [rsp+CAh] [rbp-456h] BYREF
  char v168; // [rsp+CBh] [rbp-455h] BYREF
  int v169; // [rsp+CCh] [rbp-454h] BYREF
  __int64 v170; // [rsp+D0h] [rbp-450h] BYREF
  __int64 v171; // [rsp+D8h] [rbp-448h] BYREF
  __int64 v172[2]; // [rsp+E0h] [rbp-440h] BYREF
  __int64 *v173; // [rsp+F0h] [rbp-430h]
  _QWORD v174[4]; // [rsp+100h] [rbp-420h] BYREF
  __int64 v175; // [rsp+120h] [rbp-400h] BYREF
  _QWORD *v176; // [rsp+128h] [rbp-3F8h]
  __int64 v177; // [rsp+130h] [rbp-3F0h]
  unsigned int v178; // [rsp+138h] [rbp-3E8h]
  _BYTE *v179; // [rsp+140h] [rbp-3E0h] BYREF
  size_t v180; // [rsp+148h] [rbp-3D8h]
  __int64 v181; // [rsp+160h] [rbp-3C0h] BYREF
  __int64 v182; // [rsp+168h] [rbp-3B8h] BYREF
  __int64 *v183; // [rsp+170h] [rbp-3B0h]
  __int64 *v184; // [rsp+178h] [rbp-3A8h]
  __int64 *v185; // [rsp+180h] [rbp-3A0h]
  __int64 v186; // [rsp+188h] [rbp-398h]
  _BYTE *v187; // [rsp+190h] [rbp-390h] BYREF
  __int64 v188; // [rsp+198h] [rbp-388h]
  _BYTE v189[32]; // [rsp+1A0h] [rbp-380h] BYREF
  char *v190; // [rsp+1C0h] [rbp-360h] BYREF
  _BYTE **v191; // [rsp+1C8h] [rbp-358h]
  unsigned __int64 v192; // [rsp+1D0h] [rbp-350h]
  __int64 *v193; // [rsp+1D8h] [rbp-348h]
  __int64 *v194; // [rsp+1E0h] [rbp-340h]
  __int64 *v195; // [rsp+1E8h] [rbp-338h]
  unsigned __int8 *v196; // [rsp+1F0h] [rbp-330h]
  unsigned int *v197; // [rsp+1F8h] [rbp-328h]
  _BYTE *v198; // [rsp+200h] [rbp-320h] BYREF
  __int64 v199; // [rsp+208h] [rbp-318h]
  _BYTE v200[64]; // [rsp+210h] [rbp-310h] BYREF
  __int64 v201[4]; // [rsp+250h] [rbp-2D0h] BYREF
  unsigned __int64 v202[6]; // [rsp+270h] [rbp-2B0h] BYREF
  __m128i v203; // [rsp+2A0h] [rbp-280h] BYREF
  unsigned __int64 v204[6]; // [rsp+2C0h] [rbp-260h] BYREF
  __m128i j; // [rsp+2F0h] [rbp-230h] BYREF
  __int64 v206; // [rsp+300h] [rbp-220h]
  __int64 v207; // [rsp+308h] [rbp-218h]
  unsigned __int64 v208[6]; // [rsp+310h] [rbp-210h] BYREF
  unsigned __int64 v209; // [rsp+340h] [rbp-1E0h] BYREF
  __int64 v210; // [rsp+348h] [rbp-1D8h] BYREF
  char *v211; // [rsp+350h] [rbp-1D0h] BYREF
  __int64 v212; // [rsp+358h] [rbp-1C8h]
  _QWORD v213[6]; // [rsp+360h] [rbp-1C0h] BYREF
  char *v214; // [rsp+390h] [rbp-190h] BYREF
  __int64 v215; // [rsp+398h] [rbp-188h]
  _BYTE v216[384]; // [rsp+3A0h] [rbp-180h] BYREF

  v3 = (__int64 *)a2[6];
  v167 = 0;
  v4 = v3;
  LODWORD(v182) = 0;
  v183 = 0;
  v184 = &v182;
  v185 = &v182;
  v186 = 0;
  v163 = a2 + 5;
  if ( v3 != a2 + 5 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v5 = (__int64)(v4 - 6);
        if ( !v4 )
          v5 = 0;
        v8 = (_BYTE *)sub_B325F0(v5);
        if ( !*v8 )
          break;
LABEL_22:
        v4 = (__int64 *)v4[1];
        if ( v163 == v4 )
          goto LABEL_23;
      }
      v9 = v183;
      v10 = &v182;
      if ( !v183 )
        goto LABEL_12;
      do
      {
        while ( 1 )
        {
          v11 = v9[2];
          v12 = (__int64 *)v9[3];
          if ( v9[4] >= (unsigned __int64)v8 )
            break;
          v9 = (__int64 *)v9[3];
          if ( !v12 )
            goto LABEL_10;
        }
        v10 = v9;
        v9 = (__int64 *)v9[2];
      }
      while ( v11 );
LABEL_10:
      if ( v10 == &v182 || v10[4] > (unsigned __int64)v8 )
      {
LABEL_12:
        v159 = (__int64)v10;
        v10 = (__int64 *)sub_22077B0(0x50u);
        v10[4] = (__int64)v8;
        v10[5] = 0;
        v10[6] = (__int64)(v10 + 9);
        v10[7] = 1;
        *((_DWORD *)v10 + 16) = 0;
        *((_BYTE *)v10 + 68) = 1;
        v13 = sub_264B4B0(&v181, v159, (unsigned __int64 *)v10 + 4);
        if ( v14 )
        {
          v15 = v13 || &v182 == v14 || (unsigned __int64)v8 < v14[4];
          sub_220F040(v15, (__int64)v10, v14, &v182);
          ++v186;
        }
        else
        {
          v161 = v13;
          j_j___libc_free_0((unsigned __int64)v10);
          v10 = v161;
        }
      }
      if ( !*((_BYTE *)v10 + 68) )
        goto LABEL_75;
      v16 = (__int64 *)v10[6];
      v11 = *((unsigned int *)v10 + 15);
      v12 = &v16[v11];
      if ( v16 != v12 )
      {
        while ( *v16 != v5 )
        {
          if ( v12 == ++v16 )
            goto LABEL_77;
        }
        goto LABEL_22;
      }
LABEL_77:
      if ( (unsigned int)v11 >= *((_DWORD *)v10 + 14) )
      {
LABEL_75:
        sub_C8CC70((__int64)(v10 + 5), v5, (__int64)v12, v11, v6, v7);
        v4 = (__int64 *)v4[1];
        if ( v163 == v4 )
          break;
      }
      else
      {
        *((_DWORD *)v10 + 15) = v11 + 1;
        *v12 = v5;
        ++v10[5];
        v4 = (__int64 *)v4[1];
        if ( v163 == v4 )
          break;
      }
    }
  }
LABEL_23:
  v17 = sub_2649F60((__int64)a1, a2);
  if ( (_BYTE)v17 )
  {
    v147 = a2 + 3;
    v153 = (__int64 *)a2[4];
    if ( a2 + 3 == v153 )
    {
LABEL_80:
      v17 = v167;
      goto LABEL_81;
    }
    while ( 1 )
    {
      v18 = (__int64)(v153 - 7);
      if ( !v153 )
        v18 = 0;
      v149 = v18;
      v19 = v18;
      if ( sub_B2FC80(v18) )
        goto LABEL_55;
      v20 = (__int64 *)&v209;
      v21 = sub_BD5D20(v19);
      v210 = v22;
      v209 = (unsigned __int64)v21;
      if ( sub_C931B0((__int64 *)&v209, (_WORD *)qword_4FF3200, qword_4FF3208, 0) != -1 )
        goto LABEL_55;
      sub_1049690(v172, v19);
      v190 = &v168;
      v23 = (__int64)a2;
      v195 = &v181;
      v191 = &v187;
      v196 = &v167;
      v174[2] = &v187;
      v197 = (unsigned int *)&v169;
      v187 = v189;
      v174[0] = &v190;
      v24 = *a1;
      v188 = 0x400000000LL;
      v194 = v172;
      v174[3] = v172;
      v168 = 0;
      v169 = 0;
      v192 = v19;
      v193 = a2;
      v174[1] = a2;
      v25 = sub_2642C30(v19, (__int64)a2, v24, 0) & 0xFFFFFFFFFFFFFFF8LL;
      v156 = v25;
      if ( !v25 )
        goto LABEL_51;
      v26 = (const void *)a2[21];
      v27 = a2[22];
      v28 = *(__int64 **)(v25 + 24);
      v155 = *(char **)(v25 + 32);
      v29 = (v155 - (char *)v28) >> 5;
      v30 = (v155 - (char *)v28) >> 3;
      if ( v29 > 0 )
      {
        v164 = &v28[4 * v29];
        while ( 1 )
        {
          if ( v27 == *(_QWORD *)(*v28 + 32) )
          {
            if ( !v27 )
              break;
            v23 = (__int64)v26;
            if ( !memcmp(*(const void **)(*v28 + 24), v26, v27) )
              break;
          }
          v31 = v28[1];
          v32 = v28 + 1;
          if ( v27 == *(_QWORD *)(v31 + 32) )
          {
            if ( !v27 )
              goto LABEL_74;
            v23 = (__int64)v26;
            if ( !memcmp(*(const void **)(v31 + 24), v26, v27) )
              goto LABEL_74;
          }
          v33 = v28[2];
          v32 = v28 + 2;
          if ( v27 == *(_QWORD *)(v33 + 32) )
          {
            if ( !v27 )
              goto LABEL_74;
            v23 = (__int64)v26;
            if ( !memcmp(*(const void **)(v33 + 24), v26, v27) )
              goto LABEL_74;
          }
          v34 = v28[3];
          v35 = v28 + 3;
          if ( v27 == *(_QWORD *)(v34 + 32) )
          {
            if ( !v27 || (v23 = (__int64)v26, v36 = memcmp(*(const void **)(v34 + 24), v26, v27), v35 = v28 + 3, !v36) )
            {
              v32 = v35;
LABEL_74:
              v28 = v32;
              break;
            }
          }
          v28 += 4;
          if ( v28 == v164 )
          {
            v30 = (v155 - (char *)v28) >> 3;
            goto LABEL_245;
          }
        }
LABEL_45:
        if ( v155 != (char *)v28 )
        {
          v148 = *v28;
          if ( *v28 )
            goto LABEL_47;
        }
        goto LABEL_248;
      }
LABEL_245:
      if ( v30 != 2 )
      {
        if ( v30 != 3 )
        {
          if ( v30 != 1 )
            goto LABEL_248;
          goto LABEL_267;
        }
        if ( v27 == *(_QWORD *)(*v28 + 32) )
        {
          if ( !v27 )
            goto LABEL_45;
          v23 = (__int64)v26;
          if ( !memcmp(*(const void **)(*v28 + 24), v26, v27) )
            goto LABEL_45;
        }
        ++v28;
      }
      if ( v27 == *(_QWORD *)(*v28 + 32) )
      {
        if ( !v27 )
          goto LABEL_45;
        v23 = (__int64)v26;
        if ( !memcmp(*(const void **)(*v28 + 24), v26, v27) )
          goto LABEL_45;
      }
      ++v28;
LABEL_267:
      if ( v27 == *(_QWORD *)(*v28 + 32) )
      {
        if ( !v27 )
          goto LABEL_45;
        v23 = (__int64)v26;
        if ( !memcmp(*(const void **)(*v28 + 24), v26, v27) )
          goto LABEL_45;
      }
LABEL_248:
      v23 = (__int64)"thinlto_src_module";
      v125 = sub_B91CC0(v149, "thinlto_src_module", 0x12u);
      v126 = *(_BYTE *)(v125 - 16);
      if ( (v126 & 2) != 0 )
        v127 = *(_BYTE ***)(v125 - 32);
      else
        v127 = (_BYTE **)(v125 - 8LL * ((v126 >> 2) & 0xF) - 16);
      v128 = *v127;
      if ( **v127 )
        v128 = 0;
      v129 = sub_B91420((__int64)v128);
      v131 = v130;
      v132 = *(__int64 **)(v156 + 32);
      v133 = *(__int64 **)(v156 + 24);
      if ( v132 != v133 )
      {
        v134 = (const void *)v129;
        do
        {
          v135 = *v133;
          if ( *(_QWORD *)(*v133 + 32) == v131 )
          {
            if ( !v131 || (v23 = (__int64)v134, !memcmp(*(const void **)(v135 + 24), v134, v131)) )
            {
              v148 = v135;
              v20 = (__int64 *)&v209;
              goto LABEL_47;
            }
          }
          ++v133;
        }
        while ( v132 != v133 );
        v20 = (__int64 *)&v209;
      }
      v148 = 0;
LABEL_47:
      if ( *(_DWORD *)(v148 + 8) )
      {
        v37 = *(_QWORD **)(v148 + 104);
        v154 = v37;
        if ( v37 )
        {
          if ( *v37 != v37[1] || (sub_26467C0(v148), v38) )
          {
            v170 = sub_26467C0(v148);
            v154 = (_QWORD *)*v154;
            goto LABEL_85;
          }
        }
        else
        {
          sub_26467C0(v148);
          if ( v124 )
          {
            v170 = sub_26467C0(v148);
LABEL_85:
            v175 = 0;
            v176 = 0;
            v177 = 0;
            v178 = 0;
            v48 = sub_26467C0(v148);
            v52 = 16 * v51;
            v53 = 17 * v51;
            v54 = v48 + 8 * v53;
            while ( 2 )
            {
              v55 = *(_QWORD **)(v148 + 96);
              if ( v55 )
LABEL_87:
                v55 = (_QWORD *)*v55;
              if ( (_QWORD *)v54 == v55 )
              {
LABEL_100:
                v198 = v200;
                v199 = 0x100000000LL;
                v144 = *(_QWORD *)(v149 + 80);
                if ( v149 + 72 == v144 )
                {
                  v74 = v200;
                  v75 = 0;
                }
                else
                {
                  do
                  {
                    if ( !v144 )
                      BUG();
                    if ( v144 + 24 != *(_QWORD *)(v144 + 32) )
                    {
                      v166 = *(_QWORD *)(v144 + 32);
                      v61 = v20;
                      do
                      {
                        if ( !v166 )
                          BUG();
                        v162 = 0;
                        if ( (unsigned __int8)(*(_BYTE *)(v166 - 24) - 34) <= 0x33u )
                        {
                          v62 = v166 - 24;
                          if ( ((0x8000000000041uLL >> (*(_BYTE *)(v166 - 24) - 34)) & 1) == 0 )
                            v62 = 0;
                          v162 = v62;
                        }
                        if ( sub_D78580(v162, v23) )
                        {
                          v63 = *(_QWORD *)(v162 - 32);
                          if ( !v63 )
                            BUG();
                          if ( *(_BYTE *)v63 || *(_QWORD *)(v63 + 24) != *(_QWORD *)(v162 + 80) )
                          {
                            v63 = (__int64)sub_BD3990(*(unsigned __int8 **)(v162 - 32), v23);
                            if ( *(_BYTE *)v63 )
                            {
                              if ( *(_BYTE *)v63 != 1 || (v63 = sub_B325F0(v63), *(_BYTE *)v63) )
                                v63 = 0;
                            }
                          }
                          v64 = 0;
                          if ( (*(_BYTE *)(v166 - 17) & 0x20) != 0 )
                            v64 = sub_B91C10(v166 - 24, 35);
                          v171 = v64;
                          if ( (*(_BYTE *)(v166 - 17) & 0x20) != 0 )
                          {
                            v23 = (__int64)"memprof";
                            v65 = sub_B91C10(v166 - 24, 34);
                            v209 = *(_QWORD *)(v162 + 72);
                            if ( (unsigned __int8)sub_A747A0(v61, "memprof", 7u) )
                              goto LABEL_170;
                            if ( v65 )
                            {
                              v66 = *(_BYTE *)(v65 - 16);
                              if ( (v66 & 2) != 0 )
                              {
                                v67 = *(__int64 **)(v65 - 32);
                                v68 = *(unsigned int *)(v65 - 24);
                              }
                              else
                              {
                                v68 = (*(_WORD *)(v65 - 16) >> 6) & 0xF;
                                v67 = (__int64 *)(v65 - 8LL * ((v66 >> 2) & 0xF) - 16);
                              }
                              for ( i = &v67[v68]; i != v67; ++v67 )
                              {
                                v69 = 0;
                                v201[0] = sub_10390E0(*v67);
                                v70.m128i_i64[0] = sub_2647370(v201, &v171);
                                v203 = v70;
                                sub_1039B70(v61, v201[0], 1);
                                if ( v203.m128i_i64[1] != v210 )
                                  v69 = sub_1039BF0((__int64)&v203) == 0;
                                a3 = _mm_loadu_si128(&v203);
                                for ( j = a3; ; j.m128i_i64[1] += 8 )
                                {
                                  v23 = v201[0];
                                  sub_1039B70(v61, v201[0], 1);
                                  if ( j.m128i_i64[1] == v210 )
                                    break;
                                  if ( sub_1039BF0((__int64)&j) != v69 )
                                    v69 = sub_1039BF0((__int64)&j);
                                }
                              }
                              v71 = v154[1];
                              v72 = v71;
                              if ( (_DWORD)v71 != 1 && !*v190 )
                              {
                                sub_264B5B0((__int64)v61, v192, v71, (__int64)v193, v194, v195, a3);
                                sub_265C280((__int64)v191, (__int64)v61);
                                v23 = v209 + 8LL * (unsigned int)v210;
                                sub_2649CB0(v209, v23);
                                if ( (char **)v209 != &v211 )
                                  _libc_free(v209);
                                *v196 = 1;
                                *v190 = 1;
                                *v197 = v72;
                                v71 = v154[1];
                              }
                              if ( v71 == 1 )
                              {
                                v73 = (_QWORD *)*v154;
                                if ( *(_BYTE *)*v154 != 2 )
                                {
LABEL_137:
                                  v154 += 14;
                                  goto LABEL_138;
                                }
                              }
                              else
                              {
                                if ( !v71 )
                                  goto LABEL_137;
                                v73 = (_QWORD *)*v154;
                              }
                              v158 = (__int64)v61;
                              v89 = 0;
                              v90 = 0;
                              while ( 2 )
                              {
                                v23 = *((unsigned __int8 *)v73 + v90);
                                if ( !(_BYTE)v23 )
                                  goto LABEL_187;
                                sub_10391D0((__int64)&v179, v23);
                                v91 = v179;
                                v92 = v180;
                                v93 = (_QWORD *)sub_B2BE50(v149);
                                v150 = sub_A78730(v93, "memprof", 7u, v91, v92);
                                if ( !v89 )
                                {
                                  v94 = v162;
                                  goto LABEL_177;
                                }
                                v100 = *(_QWORD *)&v187[8 * v89 - 8];
                                j.m128i_i64[1] = 2;
                                v206 = 0;
                                v207 = v162;
                                if ( v162 != -8192 && v162 != -4096 )
                                {
                                  v145 = v100;
                                  sub_BD73F0((__int64)&j.m128i_i64[1]);
                                  v100 = v145;
                                }
                                v208[0] = v100;
                                v146 = v100;
                                j.m128i_i64[0] = (__int64)&unk_49DD7B0;
                                v101 = sub_F9E960(v100, (__int64)&j, v201);
                                v102 = v146;
                                if ( v101 )
                                {
                                  v103 = v207;
                                  v104 = (_QWORD *)(v201[0] + 40);
LABEL_194:
                                  j.m128i_i64[0] = (__int64)&unk_49DB368;
                                  if ( v103 != 0 && v103 != -4096 && v103 != -8192 )
                                    sub_BD60C0(&j.m128i_i64[1]);
                                  v94 = v104[2];
LABEL_177:
                                  v95 = (__int64 *)sub_BD5C60(v94);
                                  *(_QWORD *)(v94 + 72) = sub_A7B440((__int64 *)(v94 + 72), v95, -1, v150);
                                  sub_B174A0(
                                    v158,
                                    (__int64)"memprof-context-disambiguation",
                                    (__int64)"MemprofAttribute",
                                    16,
                                    v94);
                                  sub_B16080((__int64)v201, "AllocationCall", 14, (unsigned __int8 *)v94);
                                  v151 = sub_2647050(v158, (__int64)v201);
                                  sub_B18290(v151, " in clone ", 0xAu);
                                  v96 = (unsigned __int8 *)sub_B43CB0(v94);
                                  sub_B16080((__int64)&v203, "Caller", 6, v96);
                                  v152 = sub_23FD640(v151, (__int64)&v203);
                                  sub_B18290(v152, " marked with memprof allocation attribute ", 0x2Au);
                                  sub_B16430((__int64)&j, "Attribute", 9u, v179, v180);
                                  v23 = sub_23FD640(v152, (__int64)&j);
                                  sub_1049740(v172, v23);
                                  sub_2240A30(v208);
                                  sub_2240A30((unsigned __int64 *)&j);
                                  sub_2240A30(v204);
                                  sub_2240A30((unsigned __int64 *)&v203);
                                  sub_2240A30(v202);
                                  sub_2240A30((unsigned __int64 *)v201);
                                  v97 = v214;
                                  v209 = (unsigned __int64)&unk_49D9D40;
                                  v98 = (unsigned __int64 *)&v214[80 * (unsigned int)v215];
                                  if ( v214 != (char *)v98 )
                                  {
                                    do
                                    {
                                      v98 -= 10;
                                      v99 = v98[4];
                                      if ( (unsigned __int64 *)v99 != v98 + 6 )
                                      {
                                        v23 = v98[6] + 1;
                                        j_j___libc_free_0(v99);
                                      }
                                      if ( (unsigned __int64 *)*v98 != v98 + 2 )
                                      {
                                        v23 = v98[2] + 1;
                                        j_j___libc_free_0(*v98);
                                      }
                                    }
                                    while ( v97 != (char *)v98 );
                                    v98 = (unsigned __int64 *)v214;
                                  }
                                  if ( v98 != (unsigned __int64 *)v216 )
                                    _libc_free((unsigned __int64)v98);
                                  sub_2240A30((unsigned __int64 *)&v179);
LABEL_187:
                                  v90 = ++v89;
                                  if ( (unsigned __int64)v89 >= v154[1] )
                                  {
                                    v61 = (__int64 *)v158;
                                    v154 += 14;
                                    goto LABEL_138;
                                  }
                                  v73 = (_QWORD *)*v154;
                                  continue;
                                }
                                break;
                              }
                              v203.m128i_i64[0] = v201[0];
                              v105 = *(_DWORD *)(v146 + 24);
                              v106 = *(_DWORD *)(v146 + 16);
                              ++*(_QWORD *)v146;
                              v107 = v106 + 1;
                              if ( 4 * v107 >= 3 * v105 )
                              {
                                sub_CF32C0(v146, 2 * v105);
                              }
                              else
                              {
                                if ( v105 - *(_DWORD *)(v146 + 20) - v107 > v105 >> 3 )
                                {
LABEL_202:
                                  *(_DWORD *)(v102 + 16) = v107;
                                  v108 = v203.m128i_i64[0];
                                  v210 = 2;
                                  v211 = 0;
                                  v212 = -4096;
                                  v213[0] = 0;
                                  if ( *(_QWORD *)(v203.m128i_i64[0] + 24) != -4096 )
                                    --*(_DWORD *)(v102 + 20);
                                  v209 = (unsigned __int64)&unk_49DB368;
                                  sub_D68D70(&v210);
                                  v109 = *(_QWORD *)(v108 + 24);
                                  v103 = v207;
                                  if ( v109 != v207 )
                                  {
                                    v110 = (unsigned __int64 *)(v108 + 8);
                                    if ( v109 != 0 && v109 != -4096 && v109 != -8192 )
                                    {
                                      sub_BD60C0((_QWORD *)(v108 + 8));
                                      v103 = v207;
                                      v110 = (unsigned __int64 *)(v108 + 8);
                                    }
                                    *(_QWORD *)(v108 + 24) = v103;
                                    if ( v103 == 0 || v103 == -4096 || v103 == -8192 )
                                    {
                                      v103 = v207;
                                    }
                                    else
                                    {
                                      sub_BD6050(v110, j.m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL);
                                      v103 = v207;
                                    }
                                  }
                                  v111 = v208[0];
                                  v104 = (_QWORD *)(v108 + 40);
                                  *v104 = 6;
                                  v104[1] = 0;
                                  *(v104 - 1) = v111;
                                  v104[2] = 0;
                                  goto LABEL_194;
                                }
                                sub_CF32C0(v146, v105);
                              }
                              sub_F9E960(v146, (__int64)&j, &v203);
                              v102 = v146;
                              v107 = *(_DWORD *)(v146 + 16) + 1;
                              goto LABEL_202;
                            }
                          }
                          else
                          {
                            v23 = (__int64)"memprof";
                            v209 = *(_QWORD *)(v162 + 72);
                            if ( (unsigned __int8)sub_A747A0(v61, "memprof", 7u) )
                            {
LABEL_170:
                              v23 = 0xFFFFFFFFLL;
                              j.m128i_i64[0] = *(_QWORD *)(v162 + 72);
                              v209 = sub_A747B0(&j, -1, "memprof", 7u);
                              sub_A72240(v61);
                              goto LABEL_138;
                            }
                          }
                          if ( v171 )
                          {
                            if ( v63 )
                            {
                              v23 = v170;
                              v170 += 136;
                              sub_265C490((__int64)v174, v23, v162, v63, a3);
                            }
                            else
                            {
                              v112 = *(__int64 **)(v148 + 96);
                              if ( v112 )
                              {
                                v113 = *v112;
                                v114 = 0xF0F0F0F0F0F0F0F1LL * ((v112[1] - *v112) >> 3);
                              }
                              else
                              {
                                v113 = 0;
                                v114 = 0;
                              }
                              v23 = v162;
                              v115 = sub_264C140((__int64)a1, (_BYTE *)v162, v113, v114, &v170, (__int64)&v198);
                              if ( v115 > 1 && !*v190 )
                              {
                                sub_264B5B0((__int64)v61, v192, v115, (__int64)v193, v194, v195, a3);
                                sub_265C280((__int64)v191, (__int64)v61);
                                v23 = v209 + 8LL * (unsigned int)v210;
                                sub_2649CB0(v209, v23);
                                if ( (char **)v209 != &v211 )
                                  _libc_free(v209);
                                *v196 = 1;
                                *v190 = 1;
                                *v197 = v115;
                              }
                            }
                          }
                          else
                          {
                            v116 = sub_B49220(v162);
                            if ( v63 )
                            {
                              if ( v116 )
                              {
                                v23 = (__int64)a2;
                                v117 = sub_2642C30(v63, (__int64)a2, *a1, v149) & 0xFFFFFFFFFFFFFFF8LL;
                                v118 = v117;
                                if ( v117 )
                                {
                                  if ( v178 )
                                  {
                                    v119 = v178 - 1;
                                    v120 = v117 & (v178 - 1);
                                    v121 = &v176[18 * v120];
                                    v122 = *v121 & 0xFFFFFFFFFFFFFFF8LL;
                                    if ( v118 == v122 )
                                    {
LABEL_224:
                                      v23 = (__int64)(v121 + 1);
                                      sub_265C490((__int64)v174, v23, v162, v63, a3);
                                    }
                                    else
                                    {
                                      v136 = *v121 & 0xFFFFFFFFFFFFFFF8LL;
                                      LODWORD(v137) = v120;
                                      v23 = 1;
                                      while ( v136 != -8 )
                                      {
                                        v137 = v119 & ((_DWORD)v137 + (_DWORD)v23);
                                        v136 = v176[18 * v137] & 0xFFFFFFFFFFFFFFF8LL;
                                        if ( v118 == v136 )
                                        {
                                          v138 = 1;
                                          while ( v122 != -8 )
                                          {
                                            v139 = v138 + 1;
                                            v120 = v119 & (v138 + v120);
                                            v121 = &v176[18 * v120];
                                            v122 = *v121 & 0xFFFFFFFFFFFFFFF8LL;
                                            if ( v118 == v122 )
                                              goto LABEL_224;
                                            v138 = v139;
                                          }
                                          v121 = &v176[18 * v178];
                                          goto LABEL_224;
                                        }
                                        v23 = (unsigned int)(v23 + 1);
                                      }
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
LABEL_138:
                        v166 = *(_QWORD *)(v166 + 8);
                      }
                      while ( v144 + 24 != v166 );
                      v20 = v61;
                    }
                    v144 = *(_QWORD *)(v144 + 8);
                  }
                  while ( v149 + 72 != v144 );
                  v74 = v198;
                  v75 = (unsigned int)v199;
                }
                v76 = (unsigned int)v188;
                v140 = (__int64)v74;
                v141 = v75;
                v77 = sub_26467C0(v148);
                sub_26585D0((__int64)a1, (__int64 **)a2, v77, a3, v78, (__int64)v187, v76, v140, v141, v172);
                v79 = v198;
                v80 = &v198[56 * (unsigned int)v199];
                if ( v198 != v80 )
                {
                  do
                  {
                    v81 = *((_QWORD *)v80 - 6);
                    v80 -= 56;
                    if ( v81 )
                      j_j___libc_free_0(v81);
                  }
                  while ( v79 != v80 );
                  v80 = v198;
                }
                if ( v80 != v200 )
                  _libc_free((unsigned __int64)v80);
                v82 = v178;
                if ( v178 )
                {
                  v83 = v176;
                  v84 = &v176[18 * v178];
                  do
                  {
                    while ( 1 )
                    {
                      if ( (*v83 & 0xFFFFFFFFFFFFFFF0LL) != 0xFFFFFFFFFFFFFFF0LL )
                      {
                        v85 = v83[10];
                        if ( (_QWORD *)v85 != v83 + 12 )
                          _libc_free(v85);
                        v86 = v83[2];
                        if ( (_QWORD *)v86 != v83 + 4 )
                          break;
                      }
                      v83 += 18;
                      if ( v84 == v83 )
                        goto LABEL_157;
                    }
                    _libc_free(v86);
                    v83 += 18;
                  }
                  while ( v84 != v83 );
LABEL_157:
                  v82 = v178;
                }
                sub_C7D6A0((__int64)v176, 144 * v82, 8);
                goto LABEL_51;
              }
LABEL_89:
              v56 = *(_DWORD *)(v54 - 56);
              v54 -= 136LL;
              if ( v56 )
                goto LABEL_100;
              v209 = *(_QWORD *)v54;
              v57 = *(_QWORD *)v54;
              v211 = (char *)v213;
              v210 = v57;
              v212 = 0xC00000000LL;
              if ( *(_DWORD *)(v54 + 16) )
                sub_263F620((__int64)&v211, v54 + 8, v53, v52, v49, v50);
              v214 = v216;
              v215 = 0xC00000000LL;
              v58 = *(unsigned int *)(v54 + 80);
              if ( !(_DWORD)v58 )
              {
                v23 = v178;
                if ( v178 )
                  goto LABEL_94;
LABEL_160:
                ++v175;
                j.m128i_i64[0] = 0;
LABEL_161:
                LODWORD(v23) = 2 * v23;
                goto LABEL_162;
              }
              sub_263F620((__int64)&v214, v54 + 72, v53, v58, v49, v50);
              v23 = v178;
              if ( !v178 )
                goto LABEL_160;
LABEL_94:
              v49 = (unsigned int)(v23 - 1);
              v53 = v209 & 0xFFFFFFFFFFFFFFF8LL;
              v52 = v209 & 0xFFFFFFF8 & (unsigned int)(v23 - 1);
              v59 = &v176[18 * v52];
              v60 = *v59 & 0xFFFFFFFFFFFFFFF8LL;
              if ( (v209 & 0xFFFFFFFFFFFFFFF8LL) != v60 )
              {
                v123 = 1;
                v87 = 0;
                while ( v60 != -8 )
                {
                  if ( !v87 && v60 == -16 )
                    v87 = v59;
                  v50 = (unsigned int)(v123 + 1);
                  v52 = (unsigned int)v49 & (v123 + (_DWORD)v52);
                  v59 = &v176[18 * (unsigned int)v52];
                  v60 = *v59 & 0xFFFFFFFFFFFFFFF8LL;
                  if ( v53 == v60 )
                    goto LABEL_95;
                  ++v123;
                }
                if ( !v87 )
                  v87 = v59;
                ++v175;
                v88 = v177 + 1;
                j.m128i_i64[0] = (__int64)v87;
                if ( 4 * ((int)v177 + 1) >= (unsigned int)(3 * v23) )
                  goto LABEL_161;
                v52 = (unsigned int)v23 >> 3;
                if ( (int)v23 - HIDWORD(v177) - v88 <= (unsigned int)v52 )
                {
LABEL_162:
                  sub_2659D40((__int64)&v175, v23);
                  v23 = (__int64)&v209;
                  sub_264AC00((__int64)&v175, (__int64 *)&v209, &j);
                  v87 = (unsigned __int64 *)j.m128i_i64[0];
                  v88 = v177 + 1;
                }
                LODWORD(v177) = v88;
                if ( (*v87 & 0xFFFFFFFFFFFFFFF8LL) != 0xFFFFFFFFFFFFFFF8LL )
                  --HIDWORD(v177);
                *v87 = v209;
                v87[1] = v210;
                v87[2] = (unsigned __int64)(v87 + 4);
                v87[3] = 0xC00000000LL;
                v53 = (unsigned int)v212;
                if ( (_DWORD)v212 )
                {
                  v23 = (__int64)&v211;
                  sub_263F700((__int64)(v87 + 2), &v211, (unsigned int)v212, v52, v49, v50);
                }
                v87[10] = (unsigned __int64)(v87 + 12);
                v87[11] = 0xC00000000LL;
                if ( (_DWORD)v215 )
                {
                  v23 = (__int64)&v214;
                  sub_263F700((__int64)(v87 + 10), &v214, v53, v52, v49, v50);
                }
              }
LABEL_95:
              if ( v214 != v216 )
                _libc_free((unsigned __int64)v214);
              if ( v211 == (char *)v213 )
                continue;
              break;
            }
            _libc_free((unsigned __int64)v211);
            v55 = *(_QWORD **)(v148 + 96);
            if ( v55 )
              goto LABEL_87;
            if ( !v54 )
              goto LABEL_100;
            goto LABEL_89;
          }
        }
      }
LABEL_51:
      sub_2649CB0((__int64)v187, (__int64)&v187[8 * (unsigned int)v188]);
      if ( v187 != v189 )
        _libc_free((unsigned __int64)v187);
      v39 = v173;
      if ( v173 )
      {
        sub_FDC110(v173);
        j_j___libc_free_0((unsigned __int64)v39);
      }
LABEL_55:
      v153 = (__int64 *)v153[1];
      if ( v147 == v153 )
      {
        v165 = a2[4];
        if ( v153 != (__int64 *)v165 )
        {
          v40 = 0x8000000000041LL;
          do
          {
            v41 = v165 - 56;
            if ( !v165 )
              v41 = 0;
            if ( !sub_B2FC80(v41) )
            {
              v42 = sub_BD5D20(v41);
              v210 = v43;
              v209 = (unsigned __int64)v42;
              if ( sub_C931B0((__int64 *)&v209, (_WORD *)qword_4FF3200, qword_4FF3208, 0) == -1 )
              {
                v44 = *(_QWORD *)(v41 + 80);
                for ( k = v41 + 72; k != v44; v44 = *(_QWORD *)(v44 + 8) )
                {
                  if ( !v44 )
                    BUG();
                  for ( m = *(_QWORD *)(v44 + 32); v44 + 24 != m; m = *(_QWORD *)(m + 8) )
                  {
                    if ( !m )
                      BUG();
                    v46 = *(unsigned __int8 *)(m - 24);
                    if ( (unsigned __int8)(v46 - 34) <= 0x33u && _bittest64(&v40, (unsigned int)(v46 - 34)) )
                    {
                      sub_B99FD0(m - 24, 0x22u, 0);
                      sub_B99FD0(m - 24, 0x23u, 0);
                    }
                  }
                }
              }
            }
            v165 = *(_QWORD *)(v165 + 8);
          }
          while ( v153 != (__int64 *)v165 );
        }
        goto LABEL_80;
      }
    }
  }
LABEL_81:
  sub_2641900((unsigned __int64)v183);
  return v17;
}
