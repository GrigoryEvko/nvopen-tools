// Function: sub_F429C0
// Address: 0xf429c0
//
__int64 __fastcall sub_F429C0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  unsigned __int64 v7; // rax
  int v8; // ecx
  _BYTE *v9; // rdi
  unsigned int v10; // r15d
  _QWORD *v11; // r13
  __int64 v12; // r14
  unsigned __int64 v13; // rax
  int v14; // eax
  unsigned int v15; // eax
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rbx
  __int64 v19; // rdx
  __int64 v20; // rax
  char v21; // dh
  __int64 *v22; // r12
  __int64 v23; // rbx
  unsigned __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // r15
  _QWORD *v27; // rax
  _QWORD *v28; // rdx
  char v29; // cl
  _QWORD *v30; // rbx
  _QWORD *v31; // r14
  unsigned __int64 v32; // rax
  int v33; // edx
  __int64 v34; // rdi
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rdx
  bool v38; // cf
  __int64 v39; // rax
  __int16 v40; // dx
  __int64 v41; // rax
  char v42; // dl
  _QWORD *v43; // r14
  __int64 k; // r15
  __int64 v45; // r12
  __int64 v46; // rbx
  __int64 v47; // rax
  unsigned int v48; // esi
  __int64 v49; // rax
  __int64 v50; // r13
  __int64 v51; // rsi
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // r15
  int v55; // eax
  int v56; // eax
  unsigned int v57; // edx
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rdx
  __int64 v61; // rax
  __int64 v62; // rsi
  __int64 v63; // r15
  _QWORD *v64; // r10
  int v65; // eax
  int v66; // eax
  unsigned int v67; // edx
  __int64 v68; // rax
  __int64 v69; // rdx
  __int64 v70; // rdx
  int v71; // eax
  int v72; // eax
  unsigned int v73; // edx
  __int64 v74; // rax
  __int64 v75; // rdx
  __int64 v76; // rdx
  __int64 v77; // rax
  unsigned __int64 v78; // rdx
  __int64 v79; // r13
  unsigned int v80; // r15d
  int v81; // r12d
  __int64 v82; // rbx
  _QWORD *v83; // rax
  __int64 v84; // rdx
  __int64 v85; // rdi
  __int64 v86; // rdx
  _QWORD *v87; // rdx
  int v88; // r11d
  unsigned int v89; // ecx
  __int64 *v90; // rdi
  unsigned __int64 *v91; // rdx
  int v92; // eax
  __int64 v93; // rax
  unsigned __int64 v94; // rdx
  unsigned __int64 v95; // rdx
  __int64 v96; // rax
  __int64 *v97; // rbx
  __int64 *v98; // r14
  unsigned int v99; // eax
  __int64 *v100; // rcx
  __int64 v101; // rdi
  unsigned int v102; // edx
  __int64 *v103; // r10
  __int64 v104; // rdi
  int v105; // eax
  unsigned int v107; // eax
  _QWORD *v108; // r12
  __int64 v109; // rax
  _QWORD *v110; // r13
  __int64 v111; // rdx
  __int64 v112; // rax
  int v113; // r11d
  __int64 *v114; // rcx
  unsigned int v115; // edx
  __int64 v116; // rdi
  unsigned __int64 v117; // rax
  __int64 v118; // rdx
  __int64 v119; // rax
  _QWORD *v120; // rbx
  _QWORD *v121; // r12
  __int64 v122; // rsi
  unsigned __int64 v123; // rax
  unsigned __int64 v124; // r15
  __int64 v125; // rdi
  unsigned int v126; // eax
  __int64 v127; // r8
  __int64 v128; // r9
  unsigned __int64 v129; // rdx
  __int64 v130; // rdi
  int v131; // r14d
  unsigned int i; // r12d
  unsigned int v133; // eax
  __int64 v134; // r8
  __int64 v135; // rdx
  __int64 v136; // r9
  int v137; // eax
  _DWORD *v138; // rdx
  __int64 v139; // rax
  unsigned int v140; // ecx
  unsigned __int64 v141; // r11
  int v142; // edi
  int v143; // ecx
  unsigned __int64 *v144; // rdi
  unsigned int v145; // r14d
  unsigned __int64 v146; // rax
  unsigned int v147; // eax
  __int64 v151; // [rsp+30h] [rbp-2C0h]
  __int64 v152; // [rsp+38h] [rbp-2B8h]
  __int64 v153; // [rsp+40h] [rbp-2B0h]
  __int64 v154; // [rsp+48h] [rbp-2A8h]
  __int64 *v155; // [rsp+50h] [rbp-2A0h]
  bool v156; // [rsp+5Eh] [rbp-292h]
  char v157; // [rsp+5Fh] [rbp-291h]
  __int64 v158; // [rsp+60h] [rbp-290h]
  unsigned int v159; // [rsp+78h] [rbp-278h]
  unsigned __int8 v160; // [rsp+78h] [rbp-278h]
  __int64 v161; // [rsp+78h] [rbp-278h]
  _QWORD *v162; // [rsp+78h] [rbp-278h]
  int v163; // [rsp+78h] [rbp-278h]
  __int64 *v164; // [rsp+80h] [rbp-270h]
  unsigned int v165; // [rsp+88h] [rbp-268h]
  unsigned int v166; // [rsp+88h] [rbp-268h]
  unsigned __int64 v167; // [rsp+90h] [rbp-260h]
  __int64 v168; // [rsp+90h] [rbp-260h]
  __int64 v169; // [rsp+90h] [rbp-260h]
  _QWORD *v170; // [rsp+98h] [rbp-258h]
  __int64 v171; // [rsp+98h] [rbp-258h]
  _QWORD *v172; // [rsp+A0h] [rbp-250h]
  __int64 v173; // [rsp+A0h] [rbp-250h]
  __int64 v174; // [rsp+A8h] [rbp-248h]
  __int64 *v175; // [rsp+A8h] [rbp-248h]
  _BYTE *v176; // [rsp+B0h] [rbp-240h] BYREF
  __int64 v177; // [rsp+B8h] [rbp-238h]
  _BYTE v178[16]; // [rsp+C0h] [rbp-230h] BYREF
  void *v179; // [rsp+D0h] [rbp-220h]
  _QWORD v180[2]; // [rsp+D8h] [rbp-218h] BYREF
  __int64 v181; // [rsp+E8h] [rbp-208h]
  __int64 v182; // [rsp+F0h] [rbp-200h]
  char *v183; // [rsp+100h] [rbp-1F0h] BYREF
  __int64 v184; // [rsp+108h] [rbp-1E8h] BYREF
  __int64 v185; // [rsp+110h] [rbp-1E0h]
  __int64 v186; // [rsp+118h] [rbp-1D8h]
  __int64 j; // [rsp+120h] [rbp-1D0h]
  char *v188; // [rsp+130h] [rbp-1C0h] BYREF
  _QWORD *v189; // [rsp+138h] [rbp-1B8h]
  __int64 v190; // [rsp+140h] [rbp-1B0h]
  unsigned int v191; // [rsp+148h] [rbp-1A8h]
  char v192; // [rsp+150h] [rbp-1A0h]
  char v193; // [rsp+151h] [rbp-19Fh]
  _QWORD *v194; // [rsp+158h] [rbp-198h]
  unsigned int v195; // [rsp+168h] [rbp-188h]
  char v196; // [rsp+170h] [rbp-180h]
  _BYTE *v197; // [rsp+180h] [rbp-170h] BYREF
  __int64 v198; // [rsp+188h] [rbp-168h]
  _BYTE v199[128]; // [rsp+190h] [rbp-160h] BYREF
  __int64 v200; // [rsp+210h] [rbp-E0h] BYREF
  __int64 v201; // [rsp+218h] [rbp-D8h]
  __int64 v202; // [rsp+220h] [rbp-D0h]
  __int64 v203; // [rsp+228h] [rbp-C8h]
  _BYTE *v204; // [rsp+230h] [rbp-C0h] BYREF
  __int64 v205; // [rsp+238h] [rbp-B8h]
  _BYTE v206[176]; // [rsp+240h] [rbp-B0h] BYREF

  v6 = *(_QWORD *)(a1 + 80);
  v204 = v206;
  v205 = 0x1000000000LL;
  v157 = a2;
  v200 = 0;
  v201 = 0;
  v202 = 0;
  v203 = 0;
  v174 = a1 + 72;
  if ( v6 == a1 + 72 )
  {
    v10 = 0;
    goto LABEL_157;
  }
  do
  {
    if ( !v6 )
      BUG();
    v7 = *(_QWORD *)(v6 + 24) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v7 == v6 + 24 )
      goto LABEL_9;
    if ( !v7 )
LABEL_294:
      BUG();
    v8 = *(unsigned __int8 *)(v7 - 24);
    if ( (unsigned int)(v8 - 30) > 0xA )
LABEL_9:
      BUG();
    if ( (_BYTE)v8 == 33 )
    {
      v79 = v7 - 24;
      v80 = 0;
      v81 = sub_B46E30(v7 - 24);
      if ( v81 )
      {
        v173 = v6;
        while ( 1 )
        {
          v82 = sub_B46EC0(v79, v80);
          if ( (_DWORD)v202 )
            break;
          v83 = v204;
          v84 = 8LL * (unsigned int)v205;
          a2 = (unsigned __int64)&v204[v84];
          v85 = v84 >> 3;
          v86 = v84 >> 5;
          if ( !v86 )
            goto LABEL_136;
          v87 = &v204[32 * v86];
          do
          {
            if ( v82 == *v83 )
              goto LABEL_120;
            if ( v82 == v83[1] )
            {
              ++v83;
              goto LABEL_120;
            }
            if ( v82 == v83[2] )
            {
              v83 += 2;
              goto LABEL_120;
            }
            if ( v82 == v83[3] )
            {
              v83 += 3;
              goto LABEL_120;
            }
            v83 += 4;
          }
          while ( v87 != v83 );
          v85 = (__int64)(a2 - (_QWORD)v83) >> 3;
LABEL_136:
          switch ( v85 )
          {
            case 2LL:
              goto LABEL_160;
            case 3LL:
              if ( v82 != *v83 )
              {
                ++v83;
LABEL_160:
                if ( v82 != *v83 )
                {
                  ++v83;
LABEL_162:
                  if ( v82 != *v83 )
                  {
                    v95 = (unsigned int)v205 + 1LL;
                    if ( v95 > HIDWORD(v205) )
                      goto LABEL_164;
                    goto LABEL_140;
                  }
                }
              }
LABEL_120:
              if ( (_QWORD *)a2 == v83 )
                break;
              goto LABEL_121;
            case 1LL:
              goto LABEL_162;
          }
          v95 = (unsigned int)v205 + 1LL;
          if ( v95 <= HIDWORD(v205) )
            goto LABEL_140;
LABEL_164:
          sub_C8D5F0((__int64)&v204, v206, v95, 8u, a5, a6);
          a2 = (unsigned __int64)&v204[8 * (unsigned int)v205];
LABEL_140:
          *(_QWORD *)a2 = v82;
          v96 = (unsigned int)(v205 + 1);
          LODWORD(v205) = v96;
          if ( (unsigned int)v96 > 0x10 )
          {
            v97 = (__int64 *)v204;
            v98 = (__int64 *)&v204[8 * v96];
            while ( 1 )
            {
              a2 = (unsigned int)v203;
              if ( !(_DWORD)v203 )
                break;
              a6 = (unsigned int)(v203 - 1);
              a5 = v201;
              v99 = a6 & (((unsigned int)*v97 >> 9) ^ ((unsigned int)*v97 >> 4));
              v100 = (__int64 *)(v201 + 8LL * v99);
              v101 = *v100;
              if ( *v97 != *v100 )
              {
                v113 = 1;
                v103 = 0;
                while ( v101 != -4096 )
                {
                  if ( v103 || v101 != -8192 )
                    v100 = v103;
                  v99 = a6 & (v113 + v99);
                  v101 = *(_QWORD *)(v201 + 8LL * v99);
                  if ( *v97 == v101 )
                    goto LABEL_143;
                  ++v113;
                  v103 = v100;
                  v100 = (__int64 *)(v201 + 8LL * v99);
                }
                if ( !v103 )
                  v103 = v100;
                ++v200;
                v105 = v202 + 1;
                if ( 4 * ((int)v202 + 1) < (unsigned int)(3 * v203) )
                {
                  if ( (int)v203 - HIDWORD(v202) - v105 <= (unsigned int)v203 >> 3 )
                  {
                    sub_CF28B0((__int64)&v200, v203);
                    if ( !(_DWORD)v203 )
                      goto LABEL_293;
                    a5 = *v97;
                    a6 = v201;
                    v114 = 0;
                    a2 = 1;
                    v115 = (v203 - 1) & (((unsigned int)*v97 >> 9) ^ ((unsigned int)*v97 >> 4));
                    v103 = (__int64 *)(v201 + 8LL * v115);
                    v116 = *v103;
                    v105 = v202 + 1;
                    if ( *v103 != *v97 )
                    {
                      while ( v116 != -4096 )
                      {
                        if ( !v114 && v116 == -8192 )
                          v114 = v103;
                        v115 = (v203 - 1) & (a2 + v115);
                        v166 = a2 + 1;
                        a2 = v115;
                        v103 = (__int64 *)(v201 + 8LL * v115);
                        v116 = *v103;
                        if ( a5 == *v103 )
                          goto LABEL_148;
                        a2 = v166;
                      }
LABEL_204:
                      if ( v114 )
                        v103 = v114;
                    }
                  }
LABEL_148:
                  LODWORD(v202) = v105;
                  if ( *v103 != -4096 )
                    --HIDWORD(v202);
                  *v103 = *v97;
                  goto LABEL_143;
                }
LABEL_146:
                a2 = (unsigned int)(2 * v203);
                sub_CF28B0((__int64)&v200, a2);
                if ( !(_DWORD)v203 )
                  goto LABEL_293;
                a5 = *v97;
                a6 = v201;
                v102 = (v203 - 1) & (((unsigned int)*v97 >> 9) ^ ((unsigned int)*v97 >> 4));
                v103 = (__int64 *)(v201 + 8LL * v102);
                v104 = *v103;
                v105 = v202 + 1;
                if ( *v97 != *v103 )
                {
                  a2 = 1;
                  v114 = 0;
                  while ( v104 != -4096 )
                  {
                    if ( !v114 && v104 == -8192 )
                      v114 = v103;
                    v102 = (v203 - 1) & (a2 + v102);
                    v165 = a2 + 1;
                    a2 = v102;
                    v103 = (__int64 *)(v201 + 8LL * v102);
                    v104 = *v103;
                    if ( a5 == *v103 )
                      goto LABEL_148;
                    a2 = v165;
                  }
                  goto LABEL_204;
                }
                goto LABEL_148;
              }
LABEL_143:
              if ( v98 == ++v97 )
                goto LABEL_121;
            }
            ++v200;
            goto LABEL_146;
          }
LABEL_121:
          if ( v81 == ++v80 )
          {
            v6 = v173;
            goto LABEL_4;
          }
        }
        a2 = (unsigned int)v203;
        if ( (_DWORD)v203 )
        {
          a6 = v201;
          v88 = 1;
          v89 = (v203 - 1) & (((unsigned int)v82 >> 9) ^ ((unsigned int)v82 >> 4));
          v90 = (__int64 *)(v201 + 8LL * v89);
          v91 = 0;
          a5 = *v90;
          if ( v82 == *v90 )
            goto LABEL_121;
          while ( a5 != -4096 )
          {
            if ( !v91 && a5 == -8192 )
              v91 = (unsigned __int64 *)v90;
            v89 = (v203 - 1) & (v88 + v89);
            v90 = (__int64 *)(v201 + 8LL * v89);
            a5 = *v90;
            if ( v82 == *v90 )
              goto LABEL_121;
            ++v88;
          }
          if ( !v91 )
            v91 = (unsigned __int64 *)v90;
          v92 = v202 + 1;
          ++v200;
          if ( 4 * ((int)v202 + 1) < (unsigned int)(3 * v203) )
          {
            if ( (int)v203 - HIDWORD(v202) - v92 <= (unsigned int)v203 >> 3 )
            {
              sub_CF28B0((__int64)&v200, v203);
              if ( !(_DWORD)v203 )
              {
LABEL_293:
                LODWORD(v202) = v202 + 1;
                BUG();
              }
              a5 = (unsigned int)(v203 - 1);
              a6 = v201;
              v143 = 1;
              v144 = 0;
              v145 = a5 & (((unsigned int)v82 >> 9) ^ ((unsigned int)v82 >> 4));
              v91 = (unsigned __int64 *)(v201 + 8LL * v145);
              a2 = *v91;
              v92 = v202 + 1;
              if ( v82 != *v91 )
              {
                while ( a2 != -4096 )
                {
                  if ( !v144 && a2 == -8192 )
                    v144 = v91;
                  v145 = a5 & (v143 + v145);
                  v91 = (unsigned __int64 *)(v201 + 8LL * v145);
                  a2 = *v91;
                  if ( v82 == *v91 )
                    goto LABEL_130;
                  ++v143;
                }
                if ( v144 )
                  v91 = v144;
              }
            }
            goto LABEL_130;
          }
        }
        else
        {
          ++v200;
        }
        a2 = (unsigned int)(2 * v203);
        sub_CF28B0((__int64)&v200, a2);
        if ( !(_DWORD)v203 )
          goto LABEL_293;
        a5 = (unsigned int)(v203 - 1);
        a6 = v201;
        v140 = a5 & (((unsigned int)v82 >> 9) ^ ((unsigned int)v82 >> 4));
        v91 = (unsigned __int64 *)(v201 + 8LL * v140);
        v141 = *v91;
        v92 = v202 + 1;
        if ( v82 != *v91 )
        {
          v142 = 1;
          a2 = 0;
          while ( v141 != -4096 )
          {
            if ( v141 == -8192 && !a2 )
              a2 = (unsigned __int64)v91;
            v140 = a5 & (v142 + v140);
            v91 = (unsigned __int64 *)(v201 + 8LL * v140);
            v141 = *v91;
            if ( v82 == *v91 )
              goto LABEL_130;
            ++v142;
          }
          if ( a2 )
            v91 = (unsigned __int64 *)a2;
        }
LABEL_130:
        LODWORD(v202) = v92;
        if ( *v91 != -4096 )
          --HIDWORD(v202);
        *v91 = v82;
        v93 = (unsigned int)v205;
        v94 = (unsigned int)v205 + 1LL;
        if ( v94 > HIDWORD(v205) )
        {
          a2 = (unsigned __int64)v206;
          sub_C8D5F0((__int64)&v204, v206, v94, 8u, a5, a6);
          v93 = (unsigned int)v205;
        }
        *(_QWORD *)&v204[8 * v93] = v82;
        LODWORD(v205) = v205 + 1;
        goto LABEL_121;
      }
    }
LABEL_4:
    v6 = *(_QWORD *)(v6 + 8);
  }
  while ( v174 != v6 );
  v9 = v204;
  if ( !(_DWORD)v205 )
  {
    v10 = 0;
    goto LABEL_155;
  }
  v175 = (__int64 *)v204;
  v155 = (__int64 *)&v204[8 * (unsigned int)v205];
  v10 = 0;
  v156 = a4 != 0 && a3 != 0;
  while ( 2 )
  {
    v172 = (_QWORD *)*v175;
    if ( !v157 || (v16 = sub_AA5930(*v175), v16 != v17) )
    {
      v197 = v199;
      v198 = 0x1000000000LL;
      v18 = v172[2];
      if ( v18 )
      {
        while ( 1 )
        {
          v19 = *(_QWORD *)(v18 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v19 - 30) <= 0xAu )
            break;
          v18 = *(_QWORD *)(v18 + 8);
          if ( !v18 )
            goto LABEL_20;
        }
        v11 = 0;
        while ( 1 )
        {
          v12 = *(_QWORD *)(v19 + 40);
          v13 = *(_QWORD *)(v12 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v13 == v12 + 48 )
            goto LABEL_294;
          if ( !v13 )
            goto LABEL_294;
          v14 = *(unsigned __int8 *)(v13 - 24);
          if ( (unsigned int)(v14 - 30) > 0xA )
            goto LABEL_294;
          v15 = v14 - 29;
          if ( v15 > 3 )
          {
            if ( v15 != 4 || v11 )
              goto LABEL_18;
            v11 = *(_QWORD **)(v19 + 40);
          }
          else
          {
            if ( v15 <= 1 )
              goto LABEL_18;
            v77 = (unsigned int)v198;
            v78 = (unsigned int)v198 + 1LL;
            if ( v78 > HIDWORD(v198) )
            {
              a2 = (unsigned __int64)v199;
              sub_C8D5F0((__int64)&v197, v199, v78, 8u, a5, a6);
              v77 = (unsigned int)v198;
            }
            *(_QWORD *)&v197[8 * v77] = v12;
            LODWORD(v198) = v198 + 1;
          }
          v18 = *(_QWORD *)(v18 + 8);
          if ( !v18 )
            break;
          while ( 1 )
          {
            v19 = *(_QWORD *)(v18 + 24);
            if ( (unsigned __int8)(*(_BYTE *)v19 - 30) <= 0xAu )
              break;
            v18 = *(_QWORD *)(v18 + 8);
            if ( !v18 )
              goto LABEL_33;
          }
        }
LABEL_33:
        if ( !v11 || !(_DWORD)v198 )
          goto LABEL_18;
        v20 = sub_AA4FF0((__int64)v172);
        v22 = (__int64 *)v20;
        if ( !v20 )
          goto LABEL_294;
        if ( (v23 = 1,
              BYTE1(v23) = v21,
              v24 = (unsigned int)*(unsigned __int8 *)(v20 - 24) - 39,
              (unsigned int)v24 <= 0x38)
          && (v25 = 0x100060000000001LL, _bittest64(&v25, v24))
          || sub_AA5E90((__int64)v172) )
        {
LABEL_18:
          if ( v197 != v199 )
            _libc_free(v197, a2);
        }
        else
        {
          v176 = v178;
          v177 = 0x400000000LL;
          if ( !v156 )
          {
            v193 = 1;
            v188 = ".split";
            v192 = 3;
            v26 = sub_AA8550(v172, v22, v23, (__int64)&v188, 0);
            goto LABEL_41;
          }
          v123 = v172[6] & 0xFFFFFFFFFFFFFFF8LL;
          v124 = v123;
          if ( v172 + 6 != (_QWORD *)v123 )
          {
            if ( !v123 )
              goto LABEL_294;
            v125 = v123 - 24;
            if ( (unsigned int)*(unsigned __int8 *)(v123 - 24) - 30 > 0xA )
              v125 = 0;
            v126 = sub_B46E30(v125);
            v129 = v126;
            if ( v126 <= 4 )
              goto LABEL_225;
            goto LABEL_258;
          }
          v147 = sub_B46E30(0);
          v129 = v147;
          if ( v147 > 4 )
          {
LABEL_258:
            sub_C8D5F0((__int64)&v176, v178, v129, 4u, v127, v128);
            v146 = v172[6] & 0xFFFFFFFFFFFFFFF8LL;
            v124 = v146;
            if ( v172 + 6 == (_QWORD *)v146 )
              goto LABEL_262;
            if ( !v146 )
              goto LABEL_294;
LABEL_225:
            v130 = v124 - 24;
            if ( (unsigned int)*(unsigned __int8 *)(v124 - 24) - 30 > 0xA )
              goto LABEL_262;
          }
          else
          {
LABEL_262:
            v130 = 0;
          }
          v131 = sub_B46E30(v130);
          if ( v131 )
          {
            v164 = v22;
            for ( i = 0; i != v131; ++i )
            {
              v133 = sub_FF0300(a3, v172, i);
              v135 = (unsigned int)v177;
              v136 = v133;
              v137 = v177;
              if ( (unsigned int)v177 >= (unsigned __int64)HIDWORD(v177) )
              {
                if ( HIDWORD(v177) < (unsigned __int64)(unsigned int)v177 + 1 )
                {
                  v163 = v136;
                  sub_C8D5F0((__int64)&v176, v178, (unsigned int)v177 + 1LL, 4u, v134, v136);
                  v135 = (unsigned int)v177;
                  LODWORD(v136) = v163;
                }
                *(_DWORD *)&v176[4 * v135] = v136;
                LODWORD(v177) = v177 + 1;
              }
              else
              {
                v138 = &v176[4 * (unsigned int)v177];
                if ( v138 )
                {
                  *v138 = v136;
                  v137 = v177;
                }
                LODWORD(v177) = v137 + 1;
              }
            }
            v22 = v164;
          }
          sub_FF0C10(a3, v172);
          v193 = 1;
          v188 = ".split";
          v192 = 3;
          v26 = sub_AA8550(v172, v22, v23, (__int64)&v188, 0);
          sub_FF6650(a3, v26, &v176);
          v139 = sub_FDD860(a4, v172);
          sub_FE1040(a4, v26, v139);
LABEL_41:
          v188 = 0;
          v191 = 128;
          if ( v172 == v11 )
            v11 = (_QWORD *)v26;
          v27 = (_QWORD *)sub_C7D670(0x2000, 8);
          v190 = 0;
          v189 = v27;
          v184 = 2;
          v185 = 0;
          v28 = &v27[8 * (unsigned __int64)v191];
          v186 = -4096;
          for ( j = 0; v28 != v27; v27 += 8 )
          {
            if ( v27 )
            {
              v29 = v184;
              v27[2] = 0;
              v27[3] = -4096;
              *v27 = &unk_49DD7B0;
              v27[1] = v29 & 6;
              v27[4] = j;
            }
          }
          v183 = ".clone";
          v196 = 0;
          LOWORD(j) = 259;
          v154 = sub_F4B360(v172, &v188, &v183, a1, 0);
          if ( &v197[8 * (unsigned int)v198] == v197 )
          {
            v167 = 0;
          }
          else
          {
            v170 = &v197[8 * (unsigned int)v198];
            v30 = v197;
            v167 = 0;
            do
            {
              while ( 1 )
              {
                v31 = (_QWORD *)*v30;
                if ( v172 == (_QWORD *)*v30 )
                  v31 = (_QWORD *)v26;
                v32 = v31[6] & 0xFFFFFFFFFFFFFFF8LL;
                if ( (_QWORD *)v32 == v31 + 6 )
                {
                  v34 = 0;
                }
                else
                {
                  if ( !v32 )
                    goto LABEL_294;
                  v33 = *(unsigned __int8 *)(v32 - 24);
                  v34 = 0;
                  v35 = v32 - 24;
                  if ( (unsigned int)(v33 - 30) < 0xB )
                    v34 = v35;
                }
                sub_BD2ED0(v34, (__int64)v172, v154);
                if ( v156 )
                  break;
                if ( v170 == ++v30 )
                  goto LABEL_60;
              }
              v159 = sub_FF0430(a3, v31, v154);
              v183 = (char *)sub_FDD860(a4, v31);
              v36 = sub_1098D20(&v183, v159);
              v37 = -1;
              v38 = __CFADD__(v167, v36);
              v39 = v167 + v36;
              if ( !v38 )
                v37 = v39;
              ++v30;
              v167 = v37;
            }
            while ( v170 != v30 );
          }
LABEL_60:
          if ( v156 )
          {
            sub_FE1040(a4, v154, v167);
            v117 = sub_FDD860(a4, v172);
            v118 = v117 - v167;
            if ( v117 <= v167 )
              v118 = 0;
            sub_FE1040(a4, v172, v118);
          }
          v168 = v172[7];
          v153 = sub_AA4FF0((__int64)v172);
          v171 = *(_QWORD *)(v154 + 56);
          v152 = sub_AA5190(v26);
          if ( v152 )
          {
            LOBYTE(v41) = v40;
            v42 = HIBYTE(v40);
          }
          else
          {
            v42 = 0;
            LOBYTE(v41) = 0;
          }
          v41 = (unsigned __int8)v41;
          BYTE1(v41) = v42;
          v151 = v41;
          if ( v168 != v153 )
          {
            v160 = 1;
            v43 = v11;
            for ( k = v168; ; k = v169 )
            {
              v45 = k - 24;
              v46 = v171 - 24;
              v47 = 0;
              if ( !v171 )
                v46 = 0;
              if ( !k )
                v45 = 0;
              if ( (*(_DWORD *)(v46 + 4) & 0x7FFFFFF) != 0 )
              {
                while ( 1 )
                {
                  v48 = v47;
                  if ( v43 == *(_QWORD **)(*(_QWORD *)(v46 - 8) + 32LL * *(unsigned int *)(v46 + 72) + 8 * v47) )
                    break;
                  if ( (*(_DWORD *)(v46 + 4) & 0x7FFFFFF) == (_DWORD)++v47 )
                    goto LABEL_166;
                }
              }
              else
              {
LABEL_166:
                v48 = -1;
              }
              sub_B48BF0(v46, v48, 1);
              v171 = *(_QWORD *)(v171 + 8);
              v169 = *(_QWORD *)(k + 8);
              v183 = "ind";
              LOWORD(j) = 259;
              v158 = *(_QWORD *)(v45 + 8);
              v49 = sub_BD2DA0(80);
              v50 = v49;
              if ( v49 )
              {
                sub_B44260(v49, v158, 55, 0x8000000u, k, v160);
                *(_DWORD *)(v50 + 72) = 1;
                sub_BD6B50((unsigned __int8 *)v50, (const char **)&v183);
                sub_BD2A10(v50, *(_DWORD *)(v50 + 72), 1);
              }
              v51 = *(_QWORD *)(v45 - 8);
              if ( (*(_DWORD *)(v45 + 4) & 0x7FFFFFF) != 0 )
              {
                v52 = 0;
                while ( v43 != *(_QWORD **)(v51 + 32LL * *(unsigned int *)(v45 + 72) + 8 * v52) )
                {
                  if ( (*(_DWORD *)(v45 + 4) & 0x7FFFFFF) == (_DWORD)++v52 )
                    goto LABEL_165;
                }
                v53 = 32 * v52;
              }
              else
              {
LABEL_165:
                v53 = 0x1FFFFFFFE0LL;
              }
              v54 = *(_QWORD *)(v51 + v53);
              v55 = *(_DWORD *)(v50 + 4) & 0x7FFFFFF;
              if ( v55 == *(_DWORD *)(v50 + 72) )
              {
                sub_B48D90(v50);
                v55 = *(_DWORD *)(v50 + 4) & 0x7FFFFFF;
              }
              v56 = (v55 + 1) & 0x7FFFFFF;
              v57 = v56 | *(_DWORD *)(v50 + 4) & 0xF8000000;
              v58 = *(_QWORD *)(v50 - 8) + 32LL * (unsigned int)(v56 - 1);
              *(_DWORD *)(v50 + 4) = v57;
              if ( *(_QWORD *)v58 )
              {
                v59 = *(_QWORD *)(v58 + 8);
                **(_QWORD **)(v58 + 16) = v59;
                if ( v59 )
                  *(_QWORD *)(v59 + 16) = *(_QWORD *)(v58 + 16);
              }
              *(_QWORD *)v58 = v54;
              if ( v54 )
              {
                v60 = *(_QWORD *)(v54 + 16);
                *(_QWORD *)(v58 + 8) = v60;
                if ( v60 )
                  *(_QWORD *)(v60 + 16) = v58 + 8;
                *(_QWORD *)(v58 + 16) = v54 + 16;
                *(_QWORD *)(v54 + 16) = v58;
              }
              *(_QWORD *)(*(_QWORD *)(v50 - 8)
                        + 32LL * *(unsigned int *)(v50 + 72)
                        + 8LL * ((*(_DWORD *)(v50 + 4) & 0x7FFFFFFu) - 1)) = v43;
              v183 = "merge";
              LOWORD(j) = 259;
              v161 = *(_QWORD *)(v45 + 8);
              v61 = sub_BD2DA0(80);
              v62 = v161;
              v63 = v61;
              if ( v61 )
              {
                v162 = (_QWORD *)v61;
                sub_B44260(v61, v62, 55, 0x8000000u, 0, 0);
                *(_DWORD *)(v63 + 72) = 2;
                sub_BD6B50((unsigned __int8 *)v63, (const char **)&v183);
                sub_BD2A10(v63, *(_DWORD *)(v63 + 72), 1);
                v64 = v162;
              }
              else
              {
                v64 = 0;
              }
              sub_B44220(v64, v152, v151);
              v65 = *(_DWORD *)(v63 + 4) & 0x7FFFFFF;
              if ( v65 == *(_DWORD *)(v63 + 72) )
              {
                sub_B48D90(v63);
                v65 = *(_DWORD *)(v63 + 4) & 0x7FFFFFF;
              }
              v66 = (v65 + 1) & 0x7FFFFFF;
              v67 = v66 | *(_DWORD *)(v63 + 4) & 0xF8000000;
              v68 = *(_QWORD *)(v63 - 8) + 32LL * (unsigned int)(v66 - 1);
              *(_DWORD *)(v63 + 4) = v67;
              if ( *(_QWORD *)v68 )
              {
                v69 = *(_QWORD *)(v68 + 8);
                **(_QWORD **)(v68 + 16) = v69;
                if ( v69 )
                  *(_QWORD *)(v69 + 16) = *(_QWORD *)(v68 + 16);
              }
              *(_QWORD *)v68 = v50;
              v70 = *(_QWORD *)(v50 + 16);
              *(_QWORD *)(v68 + 8) = v70;
              if ( v70 )
                *(_QWORD *)(v70 + 16) = v68 + 8;
              *(_QWORD *)(v68 + 16) = v50 + 16;
              *(_QWORD *)(v50 + 16) = v68;
              *(_QWORD *)(*(_QWORD *)(v63 - 8)
                        + 32LL * *(unsigned int *)(v63 + 72)
                        + 8LL * ((*(_DWORD *)(v63 + 4) & 0x7FFFFFFu) - 1)) = v172;
              v71 = *(_DWORD *)(v63 + 4) & 0x7FFFFFF;
              if ( v71 == *(_DWORD *)(v63 + 72) )
              {
                sub_B48D90(v63);
                v71 = *(_DWORD *)(v63 + 4) & 0x7FFFFFF;
              }
              v72 = (v71 + 1) & 0x7FFFFFF;
              v73 = v72 | *(_DWORD *)(v63 + 4) & 0xF8000000;
              v74 = *(_QWORD *)(v63 - 8) + 32LL * (unsigned int)(v72 - 1);
              *(_DWORD *)(v63 + 4) = v73;
              if ( *(_QWORD *)v74 )
              {
                v75 = *(_QWORD *)(v74 + 8);
                **(_QWORD **)(v74 + 16) = v75;
                if ( v75 )
                  *(_QWORD *)(v75 + 16) = *(_QWORD *)(v74 + 16);
              }
              *(_QWORD *)v74 = v46;
              v76 = *(_QWORD *)(v46 + 16);
              *(_QWORD *)(v74 + 8) = v76;
              if ( v76 )
                *(_QWORD *)(v76 + 16) = v74 + 8;
              *(_QWORD *)(v74 + 16) = v46 + 16;
              *(_QWORD *)(v46 + 16) = v74;
              *(_QWORD *)(*(_QWORD *)(v63 - 8)
                        + 32LL * *(unsigned int *)(v63 + 72)
                        + 8LL * ((*(_DWORD *)(v63 + 4) & 0x7FFFFFFu) - 1)) = v154;
              sub_BD84D0(v45, v63);
              sub_B43D60((_QWORD *)v45);
              v160 = 0;
              if ( v169 == v153 )
                break;
            }
          }
          if ( v196 )
          {
            v119 = v195;
            v196 = 0;
            if ( v195 )
            {
              v120 = v194;
              v121 = &v194[2 * v195];
              do
              {
                if ( *v120 != -8192 && *v120 != -4096 )
                {
                  v122 = v120[1];
                  if ( v122 )
                    sub_B91220((__int64)(v120 + 1), v122);
                }
                v120 += 2;
              }
              while ( v121 != v120 );
              v119 = v195;
            }
            sub_C7D6A0((__int64)v194, 16 * v119, 8);
          }
          v107 = v191;
          if ( v191 )
          {
            v108 = v189;
            v180[0] = 2;
            v180[1] = 0;
            v109 = -4096;
            v110 = &v189[8 * (unsigned __int64)v191];
            v181 = -4096;
            v179 = &unk_49DD7B0;
            v182 = 0;
            v184 = 2;
            v185 = 0;
            v186 = -8192;
            v183 = (char *)&unk_49DD7B0;
            j = 0;
            while ( 1 )
            {
              v111 = v108[3];
              if ( v109 != v111 )
              {
                v109 = v186;
                if ( v111 != v186 )
                {
                  v112 = v108[7];
                  if ( v112 != 0 && v112 != -4096 && v112 != -8192 )
                  {
                    sub_BD60C0(v108 + 5);
                    v111 = v108[3];
                  }
                  v109 = v111;
                }
              }
              *v108 = &unk_49DB368;
              if ( v109 != 0 && v109 != -4096 && v109 != -8192 )
                sub_BD60C0(v108 + 1);
              v108 += 8;
              if ( v110 == v108 )
                break;
              v109 = v181;
            }
            v183 = (char *)&unk_49DB368;
            if ( v186 != -4096 && v186 != 0 && v186 != -8192 )
              sub_BD60C0(&v184);
            v179 = &unk_49DB368;
            if ( v181 != 0 && v181 != -4096 && v181 != -8192 )
              sub_BD60C0(v180);
            v107 = v191;
          }
          a2 = (unsigned __int64)v107 << 6;
          sub_C7D6A0((__int64)v189, a2, 8);
          if ( v176 != v178 )
            _libc_free(v176, a2);
          if ( v197 != v199 )
            _libc_free(v197, a2);
          v10 = 1;
        }
      }
    }
LABEL_20:
    if ( v155 != ++v175 )
      continue;
    break;
  }
  v9 = v204;
LABEL_155:
  if ( v9 != v206 )
    _libc_free(v9, a2);
LABEL_157:
  sub_C7D6A0(v201, 8LL * (unsigned int)v203, 8);
  return v10;
}
