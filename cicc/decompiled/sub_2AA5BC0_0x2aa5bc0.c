// Function: sub_2AA5BC0
// Address: 0x2aa5bc0
//
_QWORD *__fastcall sub_2AA5BC0(_QWORD *a1, int *a2, __int64 a3, __int64 a4, _QWORD *a5)
{
  __int64 v9; // rax
  int v10; // r13d
  unsigned __int64 v11; // rsi
  char *v12; // rax
  char *v13; // r8
  char *v14; // rdi
  __int64 v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // rax
  _DWORD *v18; // rdi
  __int64 v19; // rcx
  __int64 v20; // rdx
  int v21; // r15d
  unsigned __int64 v22; // rsi
  char *v23; // r8
  char *v24; // rdx
  char *v25; // rdi
  __int64 v26; // rcx
  __int64 v27; // rax
  __int64 v28; // rax
  _DWORD *v29; // rdi
  __int64 v30; // rcx
  __int64 v31; // rdx
  _QWORD *v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r13
  __int64 v37; // rdi
  __int64 *v38; // r13
  __int64 v39; // r12
  __int64 v40; // rbx
  __int64 m; // rdi
  __int64 *v42; // r13
  __int64 *v43; // r12
  __int64 *v44; // rdx
  __int64 v45; // r15
  __int64 *v46; // r15
  __int64 *n; // r14
  __int64 v48; // r12
  __int64 v49; // rbx
  __int64 *v50; // rax
  __int64 v51; // rsi
  _QWORD *v52; // rcx
  _QWORD *v53; // rdx
  _BYTE *v54; // rbx
  __int64 ii; // r13
  __int64 v56; // rsi
  _QWORD *v57; // rcx
  _QWORD *v58; // rax
  bool v59; // zf
  __int64 v60; // r15
  __int64 v61; // rax
  _BYTE *v62; // rcx
  __int64 *j; // r14
  __int64 v64; // r13
  __int64 v65; // rbx
  __int64 k; // r12
  __int64 v67; // rsi
  _QWORD *v68; // rdx
  _QWORD *v69; // rax
  __int64 v70; // rbx
  __int64 v71; // r12
  unsigned __int64 v72; // rax
  unsigned __int64 v73; // r13
  _BYTE *v74; // r15
  unsigned __int64 v75; // rax
  __int64 v76; // rdx
  __int64 v77; // rax
  __int64 v78; // rax
  __int64 v79; // r15
  unsigned __int64 v80; // rax
  unsigned __int64 v81; // r13
  _BYTE *v82; // r15
  unsigned __int64 v83; // rax
  __int64 v84; // rdx
  __int64 v85; // rsi
  __int64 v86; // rdx
  __int64 v87; // rcx
  _BYTE *v88; // r13
  __int64 v89; // r15
  __int64 v90; // rdx
  __int64 v91; // rcx
  __int64 v92; // rsi
  __int64 v93; // rdx
  __int64 v94; // rcx
  __int64 v95; // rax
  _BYTE *v96; // rdx
  __int64 v97; // rax
  __int64 v98; // rax
  __int64 v99; // rax
  _BYTE *v100; // r14
  __int64 v101; // r13
  __int64 v102; // rdx
  __int64 v103; // r12
  __int64 v104; // rbx
  __int64 v105; // rax
  __int64 v106; // rax
  __int64 *v107; // rax
  _QWORD *v108; // rbx
  unsigned __int64 v109; // rax
  __int64 v110; // rcx
  unsigned __int64 v111; // rax
  _BYTE *v112; // rdi
  __int64 v113; // r12
  __int64 v114; // r13
  int v115; // eax
  __int64 v116; // rcx
  __int64 v117; // rdx
  __int64 v118; // rcx
  __int64 v119; // r15
  __int64 v120; // r14
  __int64 v121; // r13
  _QWORD *v122; // r12
  __int64 v123; // rbx
  _QWORD *v124; // rbx
  __int64 v125; // r12
  __int64 v126; // r13
  __int64 v127; // r12
  unsigned int v128; // r13d
  __int64 *v129; // rax
  __int64 v130; // rax
  int v131; // edx
  __int64 v132; // rcx
  char v133; // bl
  __int64 v134; // r15
  unsigned __int8 v135; // al
  char v136; // al
  __int64 v137; // rdx
  __int64 v138; // rt0
  __int64 v139; // rax
  __int64 v140; // rax
  __int64 *v141; // rsi
  char v142; // al
  __int64 v143; // rcx
  __int64 v144; // rdx
  __int64 *v145; // rsi
  char v146; // al
  __int64 v147; // r8
  __int64 v148; // rax
  unsigned __int64 v149; // rax
  __int64 v150; // rdx
  __int64 v151; // rcx
  unsigned __int64 v152; // rax
  __int64 v153; // rdx
  __int64 v154; // rcx
  __int64 v155; // rdx
  __int64 v156; // rcx
  __int64 v157; // rdx
  __int64 v158; // rcx
  __int64 v159; // rdx
  __int64 v160; // rcx
  __int128 v161; // [rsp-18h] [rbp-258h]
  __int64 v162; // [rsp+0h] [rbp-240h]
  __int64 v163; // [rsp+8h] [rbp-238h]
  __int64 v164; // [rsp+8h] [rbp-238h]
  __int64 v165; // [rsp+10h] [rbp-230h]
  __int64 v166; // [rsp+18h] [rbp-228h]
  __int64 v167; // [rsp+18h] [rbp-228h]
  __int64 v168; // [rsp+20h] [rbp-220h]
  __int64 v169; // [rsp+20h] [rbp-220h]
  __int64 v170; // [rsp+28h] [rbp-218h]
  __int64 v171; // [rsp+28h] [rbp-218h]
  __int64 v172; // [rsp+28h] [rbp-218h]
  __int64 i; // [rsp+30h] [rbp-210h]
  __int64 v174; // [rsp+30h] [rbp-210h]
  __int64 v175; // [rsp+38h] [rbp-208h]
  __int64 v176; // [rsp+38h] [rbp-208h]
  __int64 v177; // [rsp+38h] [rbp-208h]
  __int64 v178; // [rsp+38h] [rbp-208h]
  __int64 v179; // [rsp+40h] [rbp-200h]
  __int64 v180; // [rsp+40h] [rbp-200h]
  _BYTE *v181; // [rsp+40h] [rbp-200h]
  __int64 v182; // [rsp+40h] [rbp-200h]
  __int64 v183; // [rsp+48h] [rbp-1F8h]
  __int64 v184; // [rsp+48h] [rbp-1F8h]
  __int64 v185; // [rsp+48h] [rbp-1F8h]
  _QWORD *v186; // [rsp+48h] [rbp-1F8h]
  __int64 v187; // [rsp+48h] [rbp-1F8h]
  __int64 v188; // [rsp+48h] [rbp-1F8h]
  _QWORD *v189; // [rsp+50h] [rbp-1F0h]
  _QWORD *v190; // [rsp+50h] [rbp-1F0h]
  _QWORD *v191; // [rsp+50h] [rbp-1F0h]
  _QWORD *v192; // [rsp+50h] [rbp-1F0h]
  __int64 *v193; // [rsp+50h] [rbp-1F0h]
  __int64 v194; // [rsp+50h] [rbp-1F0h]
  unsigned __int64 v195; // [rsp+50h] [rbp-1F0h]
  __int64 v196; // [rsp+58h] [rbp-1E8h]
  _QWORD *v197; // [rsp+58h] [rbp-1E8h]
  _QWORD *v198; // [rsp+58h] [rbp-1E8h]
  __int64 v199; // [rsp+58h] [rbp-1E8h]
  __int64 v200; // [rsp+58h] [rbp-1E8h]
  __int64 v201; // [rsp+58h] [rbp-1E8h]
  __int64 v202; // [rsp+58h] [rbp-1E8h]
  __int64 v203; // [rsp+58h] [rbp-1E8h]
  __int64 v204; // [rsp+68h] [rbp-1D8h] BYREF
  __int64 v205; // [rsp+70h] [rbp-1D0h] BYREF
  __int64 v206; // [rsp+78h] [rbp-1C8h] BYREF
  _BYTE *v207; // [rsp+80h] [rbp-1C0h] BYREF
  __int64 v208; // [rsp+88h] [rbp-1B8h] BYREF
  _BYTE *v209[2]; // [rsp+90h] [rbp-1B0h] BYREF
  __int64 v210; // [rsp+A0h] [rbp-1A0h] BYREF
  __int64 v211; // [rsp+A8h] [rbp-198h]
  _BYTE **v212; // [rsp+B0h] [rbp-190h]
  __int64 *v213; // [rsp+B8h] [rbp-188h]
  __int64 v214; // [rsp+C0h] [rbp-180h]
  __int64 *v215; // [rsp+D0h] [rbp-170h] BYREF
  __int64 v216; // [rsp+D8h] [rbp-168h]
  _QWORD v217[2]; // [rsp+E0h] [rbp-160h] BYREF
  __int64 v218; // [rsp+F0h] [rbp-150h]
  _DWORD v219[2]; // [rsp+110h] [rbp-130h] BYREF
  __int64 v220; // [rsp+118h] [rbp-128h]
  __int64 v221; // [rsp+120h] [rbp-120h]
  __int64 v222; // [rsp+128h] [rbp-118h]
  _QWORD *v223; // [rsp+130h] [rbp-110h]
  __int64 v224; // [rsp+138h] [rbp-108h]
  __int64 v225; // [rsp+140h] [rbp-100h]
  __int64 v226; // [rsp+148h] [rbp-F8h]
  __int64 v227; // [rsp+150h] [rbp-F0h]
  __int64 v228; // [rsp+158h] [rbp-E8h]
  __int64 v229; // [rsp+160h] [rbp-E0h]
  __int64 v230; // [rsp+170h] [rbp-D0h] BYREF
  _QWORD *v231[2]; // [rsp+178h] [rbp-C8h] BYREF
  char *v232; // [rsp+188h] [rbp-B8h]
  char v233; // [rsp+198h] [rbp-A8h] BYREF
  char *v234; // [rsp+1B8h] [rbp-88h]
  char v235; // [rsp+1C8h] [rbp-78h] BYREF

  if ( (_BYTE)qword_500CCC8 )
    goto LABEL_36;
  v9 = sub_AA4E30(**(_QWORD **)(a3 + 32));
  v10 = *a2;
  v196 = v9;
  v189 = sub_C52410();
  v11 = sub_C959E0();
  v12 = (char *)v189[2];
  v13 = (char *)(v189 + 1);
  if ( v12 )
  {
    v14 = (char *)(v189 + 1);
    do
    {
      while ( 1 )
      {
        v15 = *((_QWORD *)v12 + 2);
        v16 = *((_QWORD *)v12 + 3);
        if ( v11 <= *((_QWORD *)v12 + 4) )
          break;
        v12 = (char *)*((_QWORD *)v12 + 3);
        if ( !v16 )
          goto LABEL_7;
      }
      v14 = v12;
      v12 = (char *)*((_QWORD *)v12 + 2);
    }
    while ( v15 );
LABEL_7:
    if ( v13 != v14 && v11 >= *((_QWORD *)v14 + 4) )
      v13 = v14;
  }
  v190 = v13;
  if ( v13 != (char *)sub_C52410() + 8 )
  {
    v17 = v190[7];
    if ( v17 )
    {
      v18 = v190 + 6;
      do
      {
        while ( 1 )
        {
          v19 = *(_QWORD *)(v17 + 16);
          v20 = *(_QWORD *)(v17 + 24);
          if ( *(_DWORD *)(v17 + 32) >= dword_500C9E8 )
            break;
          v17 = *(_QWORD *)(v17 + 24);
          if ( !v20 )
            goto LABEL_16;
        }
        v18 = (_DWORD *)v17;
        v17 = *(_QWORD *)(v17 + 16);
      }
      while ( v19 );
LABEL_16:
      if ( v190 + 6 != (_QWORD *)v18 && dword_500C9E8 >= v18[8] && v18[9] )
        v10 = dword_500CA68;
    }
  }
  v21 = a2[1];
  v191 = sub_C52410();
  v22 = sub_C959E0();
  v23 = (char *)(v191 + 1);
  v24 = (char *)v191[2];
  if ( v24 )
  {
    v25 = (char *)(v191 + 1);
    do
    {
      while ( 1 )
      {
        v26 = *((_QWORD *)v24 + 2);
        v27 = *((_QWORD *)v24 + 3);
        if ( v22 <= *((_QWORD *)v24 + 4) )
          break;
        v24 = (char *)*((_QWORD *)v24 + 3);
        if ( !v27 )
          goto LABEL_23;
      }
      v25 = v24;
      v24 = (char *)*((_QWORD *)v24 + 2);
    }
    while ( v26 );
LABEL_23:
    if ( v23 != v25 && v22 >= *((_QWORD *)v25 + 4) )
      v23 = v25;
  }
  v192 = v23;
  if ( v23 != (char *)sub_C52410() + 8 )
  {
    v28 = v192[7];
    if ( v28 )
    {
      v29 = v192 + 6;
      do
      {
        while ( 1 )
        {
          v30 = *(_QWORD *)(v28 + 16);
          v31 = *(_QWORD *)(v28 + 24);
          if ( *(_DWORD *)(v28 + 32) >= dword_500C828 )
            break;
          v28 = *(_QWORD *)(v28 + 24);
          if ( !v31 )
            goto LABEL_32;
        }
        v29 = (_DWORD *)v28;
        v28 = *(_QWORD *)(v28 + 16);
      }
      while ( v30 );
LABEL_32:
      if ( v192 + 6 != (_QWORD *)v29 && dword_500C828 >= v29[8] && v29[9] )
        v21 = qword_500C8A8;
    }
  }
  v32 = (_QWORD *)a5[6];
  v33 = a5[3];
  v219[0] = v10;
  v34 = a5[2];
  v219[1] = v21;
  v223 = v32;
  v221 = v34;
  v222 = v33;
  v224 = v196;
  v225 = 0;
  v226 = 0;
  v227 = 0;
  v228 = 0;
  v229 = 0;
  v220 = a3;
  if ( (_BYTE)qword_500CCC8 )
    goto LABEL_36;
  v35 = *(_QWORD *)(**(_QWORD **)(a3 + 32) + 72LL);
  if ( (unsigned __int8)sub_B2D610(v35, 47)
    || (unsigned __int8)sub_B2D610(v35, 18)
    || (unsigned __int8)sub_B2D610(v35, 30)
    || !sub_D4B130(a3) )
  {
    goto LABEL_36;
  }
  if ( (unsigned __int8)sub_DFE610((__int64)v223) )
  {
    v230 = sub_DFB4D0(v223);
    if ( BYTE4(v230) )
    {
      if ( !(_BYTE)qword_500C988 )
      {
        v37 = *(_QWORD *)(**(_QWORD **)(v220 + 32) + 16LL);
        for ( i = **(_QWORD **)(v220 + 32); v37; v37 = *(_QWORD *)(v37 + 8) )
        {
          if ( (unsigned __int8)(**(_BYTE **)(v37 + 24) - 30) <= 0xAu )
            break;
        }
        if ( (unsigned int)sub_2A9D270(v37, 0, v220) == 1 )
        {
          v38 = *(__int64 **)(v220 + 32);
          v193 = *(__int64 **)(v220 + 40);
          if ( (unsigned int)(v193 - v38) == 2 )
          {
            v39 = *(_QWORD *)(i + 56);
            if ( !v39 )
              BUG();
            if ( *(_BYTE *)(v39 - 24) == 84
              && (*(_DWORD *)(v39 - 20) & 0x7FFFFFF) == 2
              && sub_AA6A60(*v38) <= 4
              && sub_AA6A60(v38[1]) <= 7 )
            {
              v50 = *(__int64 **)(v39 - 32);
              v51 = v50[4 * *(unsigned int *)(v39 + 48)];
              if ( *(_BYTE *)(v220 + 84) )
              {
                v52 = *(_QWORD **)(v220 + 64);
                v53 = &v52[*(unsigned int *)(v220 + 76)];
                while ( v53 != v52 )
                {
                  if ( v51 == *v52 )
                    goto LABEL_99;
                  ++v52;
                }
              }
              else
              {
                v59 = sub_C8CA60(v220 + 56, v51) == 0;
                v50 = *(__int64 **)(v39 - 32);
                if ( !v59 )
                {
LABEL_99:
                  v54 = (_BYTE *)*v50;
                  if ( *(_BYTE *)*v50 <= 0x1Cu )
                    goto LABEL_55;
                  v176 = v50[4];
                  goto LABEL_101;
                }
              }
              v54 = (_BYTE *)v50[4];
              if ( *v54 <= 0x1Cu )
                goto LABEL_55;
              v176 = *v50;
LABEL_101:
              if ( sub_BCAC40(*((_QWORD *)v54 + 1), 32) )
              {
                v231[0] = 0;
                v60 = v39 - 24;
                v230 = v39 - 24;
                if ( *v54 == 42
                  && ((v61 = *((_QWORD *)v54 - 8)) != 0
                   && v60 == v61
                   && (unsigned __int8)sub_993A50(v231, *((_QWORD *)v54 - 4))
                   || *((_QWORD *)v54 - 4) == v230 && (unsigned __int8)sub_993A50(v231, *((_QWORD *)v54 - 8))) )
                {
                  v184 = v39;
                  v62 = v54;
                  v198 = a1;
                  for ( j = v38; v193 != j; ++j )
                  {
                    v64 = *(_QWORD *)(*j + 56);
                    v65 = *j + 48;
                    while ( v65 != v64 )
                    {
                      if ( !v64 )
                        BUG();
                      if ( v60 != v64 - 24 && (_BYTE *)(v64 - 24) != v62 )
                      {
                        for ( k = *(_QWORD *)(v64 - 8); k; k = *(_QWORD *)(k + 8) )
                        {
                          v67 = *(_QWORD *)(*(_QWORD *)(k + 24) + 40LL);
                          if ( *(_BYTE *)(v220 + 84) )
                          {
                            v68 = *(_QWORD **)(v220 + 64);
                            v69 = &v68[*(unsigned int *)(v220 + 76)];
                            if ( v68 == v69 )
                              goto LABEL_185;
                            while ( v67 != *v68 )
                            {
                              if ( v69 == ++v68 )
                                goto LABEL_185;
                            }
                          }
                          else
                          {
                            v181 = v62;
                            v107 = sub_C8CA60(v220 + 56, v67);
                            v62 = v181;
                            if ( !v107 )
                            {
LABEL_185:
                              a1 = v198;
                              goto LABEL_55;
                            }
                          }
                        }
                      }
                      v64 = *(_QWORD *)(v64 + 8);
                    }
                  }
                  v70 = (__int64)v62;
                  v230 = 32;
                  v71 = v184;
                  a1 = v198;
                  v72 = sub_986580(i);
                  v73 = v72;
                  if ( *(_BYTE *)v72 == 31 && (*(_DWORD *)(v72 + 4) & 0x7FFFFFF) == 3 )
                  {
                    v74 = *(_BYTE **)(v72 - 96);
                    if ( *v74 == 82 )
                    {
                      v75 = sub_B53900(*(_QWORD *)(v72 - 96));
                      v215 = (__int64 *)sub_B53630(v75, v230);
                      LODWORD(v216) = v76;
                      if ( (_BYTE)v76 )
                      {
                        v77 = *((_QWORD *)v74 - 8);
                        if ( v77 )
                        {
                          if ( v77 == v70 )
                          {
                            v195 = *((_QWORD *)v74 - 4);
                            if ( v195 )
                            {
                              v199 = *(_QWORD *)(v73 - 32);
                              if ( v199 )
                              {
                                v78 = *(_QWORD *)(v73 - 64);
                                v165 = v78;
                                v79 = v78;
                                if ( v78 )
                                {
                                  if ( (unsigned __int8)sub_B19060(v220 + 56, v78, v76, v220) )
                                  {
                                    v230 = 32;
                                    v80 = sub_986580(v79);
                                    v81 = v80;
                                    if ( *(_BYTE *)v80 == 31 && (*(_DWORD *)(v80 + 4) & 0x7FFFFFF) == 3 )
                                    {
                                      v82 = *(_BYTE **)(v80 - 96);
                                      if ( *v82 == 82 )
                                      {
                                        v83 = sub_B53900(*(_QWORD *)(v80 - 96));
                                        v215 = (__int64 *)sub_B53630(v83, v230);
                                        LODWORD(v216) = v84;
                                        if ( (_BYTE)v84 )
                                        {
                                          v185 = *((_QWORD *)v82 - 8);
                                          if ( v185 )
                                          {
                                            v170 = *((_QWORD *)v82 - 4);
                                            if ( v170 )
                                            {
                                              v85 = *(_QWORD *)(v81 - 32);
                                              if ( v85 )
                                              {
                                                v180 = *(_QWORD *)(v81 - 64);
                                                if ( v180 )
                                                {
                                                  if ( (unsigned __int8)sub_B19060(v220 + 56, v85, v84, v220) )
                                                  {
                                                    v230 = (__int64)v209;
                                                    if ( *(_BYTE *)v185 == 61 )
                                                    {
                                                      if ( (unsigned __int8)sub_2A9D110((_QWORD **)&v230, v185) )
                                                      {
                                                        v215 = &v210;
                                                        if ( *(_BYTE *)v170 == 61 )
                                                        {
                                                          if ( (unsigned __int8)sub_2A9D110(&v215, v170) )
                                                          {
                                                            if ( sub_2A9D660(v185) && sub_2A9D660(v170) )
                                                            {
                                                              v88 = v209[0];
                                                              v89 = v210;
                                                              if ( *v209[0] == 63 && *(_BYTE *)v210 == 63 )
                                                              {
                                                                v166 = *(_QWORD *)&v209[0][-32
                                                                                         * (*((_DWORD *)v209[0] + 1)
                                                                                          & 0x7FFFFFF)];
                                                                v168 = *(_QWORD *)(v210
                                                                                 - 32LL
                                                                                 * (*(_DWORD *)(v210 + 4) & 0x7FFFFFF));
                                                                if ( (unsigned __int8)sub_D48480(v220, v166, v86, v87) )
                                                                {
                                                                  if ( (unsigned __int8)sub_D48480(v220, v168, v90, v91) )
                                                                  {
                                                                    if ( sub_BCAC40(*((_QWORD *)v88 + 10), 8)
                                                                      && sub_BCAC40(*(_QWORD *)(v89 + 80), 8)
                                                                      && sub_BCAC40(*(_QWORD *)(v185 + 8), 8)
                                                                      && sub_BCAC40(*(_QWORD *)(v170 + 8), 8)
                                                                      && v166 != v168 )
                                                                    {
                                                                      v92 = *((_DWORD *)v88 + 1) & 0x7FFFFFF;
                                                                      v93 = (unsigned int)(v92 - 1);
                                                                      if ( (unsigned int)v93 <= 1 )
                                                                      {
                                                                        v94 = *(_DWORD *)(v89 + 4) & 0x7FFFFFF;
                                                                        v95 = (unsigned int)(v94 - 1);
                                                                        if ( (unsigned int)v95 <= 1 )
                                                                        {
                                                                          v96 = *(_BYTE **)&v88[32 * (v93 - v92)];
                                                                          if ( v96 == *(_BYTE **)(v89 + 32 * (v95 - v94))
                                                                            && *v96 == 68 )
                                                                          {
                                                                            v97 = *((_QWORD *)v96 - 4);
                                                                            if ( v97 )
                                                                            {
                                                                              if ( v97 == v70 )
                                                                              {
                                                                                v98 = *(_QWORD *)(v71 - 8);
                                                                                if ( v98 )
                                                                                {
                                                                                  if ( !*(_QWORD *)(v98 + 8) )
                                                                                  {
                                                                                    if ( v199 != v180 )
                                                                                    {
LABEL_167:
                                                                                      sub_2AA00B0(
                                                                                        (__int64)v219,
                                                                                        (__int64)v88,
                                                                                        v89,
                                                                                        v195,
                                                                                        v70,
                                                                                        v176,
                                                                                        v180,
                                                                                        v199);
LABEL_168:
                                                                                      memset(a1, 0, 0x60u);
                                                                                      *((_BYTE *)a1 + 28) = 1;
                                                                                      a1[1] = a1 + 4;
                                                                                      *((_DWORD *)a1 + 4) = 2;
                                                                                      a1[7] = a1 + 10;
                                                                                      *((_DWORD *)a1 + 16) = 2;
                                                                                      *((_BYTE *)a1 + 76) = 1;
                                                                                      return a1;
                                                                                    }
                                                                                    v99 = sub_AA5930(v199);
                                                                                    v186 = a1;
                                                                                    v100 = v88;
                                                                                    v101 = v70;
                                                                                    v171 = v102;
                                                                                    v103 = v99;
                                                                                    while ( 1 )
                                                                                    {
                                                                                      if ( v171 == v103 )
                                                                                      {
                                                                                        v70 = v101;
                                                                                        v88 = v100;
                                                                                        a1 = v186;
                                                                                        goto LABEL_167;
                                                                                      }
                                                                                      v104 = sub_F0A930(v103, i);
                                                                                      v105 = sub_F0A930(v103, v165);
                                                                                      if ( v104 != v105
                                                                                        && (v104 != v101 && v104 != v195
                                                                                         || v105 != v101) )
                                                                                      {
                                                                                        break;
                                                                                      }
                                                                                      if ( !v103 )
                                                                                        BUG();
                                                                                      v106 = *(_QWORD *)(v103 + 32);
                                                                                      if ( !v106 )
                                                                                        BUG();
                                                                                      v103 = v106 - 24;
                                                                                      if ( *(_BYTE *)(v106 - 24) != 84 )
                                                                                        v103 = 0;
                                                                                    }
                                                                                    a1 = v186;
                                                                                  }
                                                                                }
                                                                              }
                                                                            }
                                                                          }
                                                                        }
                                                                      }
                                                                    }
                                                                  }
                                                                }
                                                              }
                                                            }
                                                          }
                                                        }
                                                      }
                                                    }
                                                  }
                                                }
                                              }
                                            }
                                          }
                                        }
                                      }
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
LABEL_55:
  if ( !(unsigned __int8)sub_DFE610((__int64)v223) )
    goto LABEL_36;
  v230 = sub_DFB4D0(v223);
  if ( !BYTE4(v230) || (_BYTE)qword_500C7C8 )
    goto LABEL_36;
  v194 = **(_QWORD **)(v220 + 32);
  v40 = sub_AA48A0(v194);
  for ( m = *(_QWORD *)(**(_QWORD **)(v220 + 32) + 16LL); m; m = *(_QWORD *)(m + 8) )
  {
    if ( (unsigned __int8)(**(_BYTE **)(m + 24) - 30) <= 0xAu )
      break;
  }
  if ( (unsigned int)sub_2A9D270(m, 0, v220) != 1 )
    goto LABEL_36;
  v42 = *(__int64 **)(v220 + 40);
  v43 = *(__int64 **)(v220 + 32);
  if ( (unsigned int)(v42 - v43) != 4 )
    goto LABEL_36;
  v44 = *(__int64 **)(v220 + 8);
  if ( *(_QWORD *)(v220 + 16) - (_QWORD)v44 != 8 )
    goto LABEL_36;
  v45 = *(_QWORD *)(v194 + 56);
  if ( !v45 )
    BUG();
  if ( *(_BYTE *)(v45 - 24) != 84 )
    goto LABEL_36;
  if ( (*(_DWORD *)(v45 - 20) & 0x7FFFFFF) != 2 )
    goto LABEL_36;
  v183 = *v44;
  if ( sub_AA6A60(*v43) > 3 || sub_AA6A60(v43[1]) > 4 || sub_AA6A60(v43[2]) > 3 || sub_AA6A60(v43[3]) > 3 )
    goto LABEL_36;
  v179 = v40;
  v174 = v45 - 24;
  v175 = v45;
  v46 = v42;
  v197 = a1;
  for ( n = v43; v46 != n; ++n )
  {
    v48 = *(_QWORD *)(*n + 56);
    v49 = *n + 48;
    while ( v49 != v48 )
    {
      if ( !v48 )
        BUG();
      if ( v174 != v48 - 24 )
      {
        for ( ii = *(_QWORD *)(v48 - 8); ii; ii = *(_QWORD *)(ii + 8) )
        {
          v56 = *(_QWORD *)(*(_QWORD *)(ii + 24) + 40LL);
          if ( *(_BYTE *)(v220 + 84) )
          {
            v57 = *(_QWORD **)(v220 + 64);
            v58 = &v57[*(unsigned int *)(v220 + 76)];
            while ( v58 != v57 )
            {
              if ( v56 == *v57 )
                goto LABEL_233;
              ++v57;
            }
            goto LABEL_203;
          }
          if ( !sub_C8CA60(v220 + 56, v56) )
            goto LABEL_203;
LABEL_233:
          ;
        }
      }
      v48 = *(_QWORD *)(v48 + 8);
    }
  }
  v108 = (_QWORD *)v179;
  a1 = v197;
  v109 = sub_986580(v194);
  if ( *(_BYTE *)v109 != 31 )
    goto LABEL_36;
  if ( (*(_DWORD *)(v109 + 4) & 0x7FFFFFF) != 1 )
    goto LABEL_36;
  v172 = *(_QWORD *)(v109 - 32);
  v182 = v183 + 56;
  if ( !(unsigned __int8)sub_B19060(v183 + 56, v172, (*(_DWORD *)(v109 + 4) & 0x7FFFFFFu) - 1, v110) )
    goto LABEL_36;
  v111 = sub_986580(v172);
  if ( *(_BYTE *)v111 != 31 )
    goto LABEL_36;
  if ( (*(_DWORD *)(v111 + 4) & 0x7FFFFFF) != 3 )
    goto LABEL_36;
  v112 = *(_BYTE **)(v111 - 96);
  if ( *v112 != 82 )
    goto LABEL_36;
  v113 = *((_QWORD *)v112 - 8);
  v200 = v111;
  if ( !v113 )
    goto LABEL_36;
  v114 = *((_QWORD *)v112 - 4);
  if ( !v114 )
    goto LABEL_36;
  v115 = sub_B53900((__int64)v112);
  v169 = *(_QWORD *)(v200 - 32);
  if ( !v169 )
    goto LABEL_36;
  v116 = *(_QWORD *)(v200 - 64);
  v167 = v116;
  if ( !v116 || v115 != 32 || !(unsigned __int8)sub_B19060(v182, v116, v200, v116) )
    goto LABEL_36;
  v119 = *(_QWORD *)(v175 - 8);
  v197 = a1;
  v120 = v114;
  v121 = v113;
  v122 = v108;
  while ( v119 )
  {
    v123 = *(_QWORD *)(v119 + 24);
    if ( !(unsigned __int8)sub_B19060(v220 + 56, *(_QWORD *)(v123 + 40), v117, v118)
      && (*(_BYTE *)v123 != 84 || *(_QWORD *)(v123 + 40) != v169) )
    {
LABEL_203:
      a1 = v197;
      goto LABEL_36;
    }
    v119 = *(_QWORD *)(v119 + 8);
  }
  v124 = v122;
  v125 = v121;
  v126 = v120;
  v230 = (__int64)&v204;
  a1 = v197;
  if ( *(_BYTE *)v125 != 61 )
    goto LABEL_36;
  if ( !(unsigned __int8)sub_2A9D110((_QWORD **)&v230, v125) )
    goto LABEL_36;
  v215 = &v205;
  if ( *(_BYTE *)v126 != 61 )
    goto LABEL_36;
  if ( !(unsigned __int8)sub_2A9D110(&v215, v126) )
    goto LABEL_36;
  if ( !sub_2A9D660(v125) )
    goto LABEL_36;
  if ( !sub_2A9D660(v126) )
    goto LABEL_36;
  v127 = *(_QWORD *)(v125 + 8);
  if ( *(_BYTE *)(v127 + 8) != 12 || v127 != *(_QWORD *)(v126 + 8) )
    goto LABEL_36;
  v128 = 0x80u / (*(_DWORD *)(v127 + 8) >> 8);
  v187 = sub_BCDE10((__int64 *)v127, v128);
  v201 = sub_BCDA70((__int64 *)v127, v128);
  v129 = (__int64 *)sub_BCB2A0(v124);
  v218 = sub_BCDE10(v129, v128);
  v216 = 0x600000003LL;
  v217[1] = v201;
  *((_QWORD *)&v161 + 1) = 1;
  *(_QWORD *)&v161 = 0;
  v215 = v217;
  v217[0] = v187;
  sub_DF8CB0((__int64)&v230, 162, v218, (char *)v217, 3, 0, 0, v161);
  v130 = sub_DFD690((__int64)v223, (__int64)&v230);
  if ( v131 )
    v133 = v131 > 0;
  else
    v133 = v130 > 4;
  if ( v133 )
    goto LABEL_231;
  v134 = v205;
  v135 = *(_BYTE *)v205;
  if ( *(_BYTE *)v204 != 84 )
  {
    if ( v135 > 0x1Cu && v135 != 84 )
      goto LABEL_220;
LABEL_231:
    v133 = 0;
    goto LABEL_220;
  }
  if ( v135 <= 0x1Cu )
    goto LABEL_231;
  if ( v135 != 84 )
    goto LABEL_220;
  if ( (*(_DWORD *)(v204 + 4) & 0x7FFFFFF) != 2 || (*(_DWORD *)(v205 + 4) & 0x7FFFFFF) != 2 )
    goto LABEL_231;
  v202 = v204;
  v136 = sub_B19060(v182, *(_QWORD *)(v204 + 40), v204, v132);
  v137 = v202;
  if ( v136 )
  {
    v138 = v134;
    v134 = v202;
    v137 = v138;
  }
  v139 = *(_QWORD *)(v194 + 56);
  if ( !v139 )
    goto LABEL_231;
  if ( v137 != v139 - 24 )
    goto LABEL_231;
  v140 = *(_QWORD *)(v172 + 56);
  if ( !v140 || v134 != v140 - 24 )
    goto LABEL_231;
  v141 = *(__int64 **)(v137 - 8);
  v177 = v137;
  v188 = *v141;
  v203 = v141[4];
  v142 = sub_B19060(v220 + 56, v141[4 * *(unsigned int *)(v137 + 72)], v137, v203);
  v144 = v177;
  if ( v142 )
  {
    v143 = v203;
    v203 = v188;
    v188 = v143;
  }
  v145 = *(__int64 **)(v134 - 8);
  v162 = v177;
  v178 = *v145;
  v163 = v145[4];
  v146 = sub_B19060(v182, v145[4 * *(unsigned int *)(v134 + 72)], v144, v143);
  v147 = v163;
  if ( v146 )
  {
    v148 = v178;
    v178 = v163;
    v147 = v148;
  }
  v164 = v147;
  v211 = v162;
  v210 = 0;
  if ( (unsigned __int8)sub_2A9FE50((__int64)&v210, v203) )
  {
    v209[1] = (_BYTE *)v134;
    v209[0] = 0;
    if ( (unsigned __int8)sub_2A9FE50((__int64)v209, v164) )
    {
      if ( v127 != *(_QWORD *)(v203 + 80) || v127 != *(_QWORD *)(v164 + 80) )
        goto LABEL_231;
      v210 = 32;
      v212 = &v207;
      v213 = &v206;
      v211 = v164;
      v214 = v172;
      v149 = sub_986580(v167);
      if ( sub_2AA0010((__int64)&v210, v149) )
      {
        if ( (unsigned __int8)sub_B19060(v220 + 56, v206, v150, v151) )
        {
          v210 = 32;
          v212 = v209;
          v211 = v203;
          v213 = &v208;
          v214 = v194;
          v152 = sub_986580(v206);
          if ( sub_2AA0010((__int64)&v210, v152) )
          {
            if ( (unsigned __int8)sub_D48480(v220, v188, v153, v154) )
            {
              if ( (unsigned __int8)sub_D48480(v220, (__int64)v209[0], v155, v156) )
              {
                if ( (unsigned __int8)sub_D48480(v220, v178, v157, v158) )
                {
                  v133 = sub_D48480(v220, (__int64)v207, v159, v160);
                  if ( v133 )
                  {
                    sub_2AA3190((__int64)v219, v174, v128, (__int64 *)v127, v169, v208, v188, v209[0], v178, v207);
                    goto LABEL_220;
                  }
                }
              }
            }
          }
          goto LABEL_231;
        }
      }
    }
  }
LABEL_220:
  if ( v234 != &v235 )
    _libc_free((unsigned __int64)v234);
  if ( v232 != &v233 )
    _libc_free((unsigned __int64)v232);
  if ( v215 != v217 )
    _libc_free((unsigned __int64)v215);
  if ( v133 )
    goto LABEL_168;
LABEL_36:
  a1[6] = 0;
  a1[1] = a1 + 4;
  a1[7] = a1 + 10;
  a1[2] = 0x100000002LL;
  a1[8] = 2;
  *((_DWORD *)a1 + 18) = 0;
  *((_BYTE *)a1 + 76) = 1;
  *((_DWORD *)a1 + 6) = 0;
  *((_BYTE *)a1 + 28) = 1;
  a1[4] = &qword_4F82400;
  *a1 = 1;
  return a1;
}
