// Function: sub_839D30
// Address: 0x839d30
//
__int64 __fastcall sub_839D30(
        __int64 a1,
        __m128i *a2,
        int a3,
        int a4,
        int a5,
        unsigned int a6,
        int a7,
        int a8,
        unsigned int a9,
        __int64 a10,
        char *a11,
        __m128i *a12)
{
  __int64 v14; // r13
  bool v15; // al
  bool v16; // r12
  bool v17; // r12
  __m128i *v18; // rdi
  unsigned int v19; // eax
  char v20; // r14
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  int v25; // r10d
  char v26; // bl
  __int64 v27; // rax
  __m128i *v28; // r12
  unsigned int v29; // eax
  __int64 v30; // r12
  __int64 v31; // rdi
  __int32 v32; // eax
  __int64 v34; // rax
  bool v35; // bl
  __int64 *v36; // rdi
  _BOOL4 v37; // ecx
  __int64 v38; // rdi
  __int64 v39; // rdi
  __int64 v40; // rax
  char v41; // al
  __int64 v42; // rax
  int v43; // eax
  __int8 v44; // dl
  __m128i *v45; // rax
  _BYTE *v46; // r8
  char v47; // al
  char v48; // al
  int v49; // eax
  __int64 v50; // rax
  __int64 v51; // rax
  char v52; // al
  __int16 v53; // ax
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // r8
  __int64 v57; // r9
  __int64 v58; // r12
  _QWORD *v59; // rdi
  char n; // dl
  __int64 v61; // rax
  char v62; // al
  int v63; // eax
  __m128i *v64; // r14
  __int64 v65; // r12
  bool v66; // sf
  __int64 v67; // rcx
  __int64 v68; // r8
  __m128i *v69; // r11
  int v70; // r10d
  int v71; // r14d
  __int64 v72; // rsi
  __int64 v73; // rax
  unsigned int v74; // edx
  int v75; // eax
  char v76; // al
  __int64 v77; // rax
  char ii; // dl
  __int64 v79; // rdx
  __int64 v80; // rcx
  __int64 v81; // r8
  __int64 v82; // r9
  _QWORD *v83; // rdx
  __int64 v84; // rcx
  __int64 v85; // r8
  __int64 v86; // r9
  _BYTE *v87; // rsi
  __int64 v88; // rax
  __m128i v89; // xmm1
  __m128i v90; // xmm2
  __m128i v91; // xmm3
  __m128i v92; // xmm4
  __m128i v93; // xmm5
  __m128i v94; // xmm6
  __m128i v95; // xmm7
  __m128i v96; // xmm0
  __m128i *v97; // r14
  __m128i *v98; // rdi
  __int64 v99; // r12
  _DWORD *v100; // rax
  __m128i *v101; // rax
  _DWORD *v102; // rax
  __int64 v103; // rax
  char m; // dl
  _BYTE *v105; // rax
  int v106; // r14d
  int v107; // eax
  unsigned int v108; // eax
  __m128i v109; // xmm2
  __m128i v110; // xmm3
  __m128i v111; // xmm4
  __m128i v112; // xmm5
  __m128i v113; // xmm6
  __m128i v114; // xmm7
  __m128i v115; // xmm1
  __m128i v116; // xmm2
  __m128i v117; // xmm3
  __m128i v118; // xmm4
  __m128i v119; // xmm5
  __m128i v120; // xmm6
  __int64 v121; // rdx
  int v122; // r12d
  int v123; // ecx
  int v124; // r8d
  __m128i *j; // rax
  __int64 v126; // r14
  __int64 v127; // rax
  char jj; // dl
  __int64 v129; // r14
  _QWORD *v130; // rax
  _BYTE *v131; // rsi
  __int64 v132; // rax
  __int64 *v133; // rax
  _BYTE *v134; // rsi
  _BYTE *v135; // rax
  __int64 v136; // r12
  int v137; // eax
  int v138; // r9d
  char v139; // dl
  __int64 v140; // rax
  __m128i *v141; // rcx
  _BOOL8 v142; // rsi
  __m128i *v143; // rcx
  _BOOL8 v144; // rsi
  __int64 v145; // r12
  __int64 v146; // rdi
  __int64 v147; // rax
  __int64 v148; // rdi
  __int64 v149; // rax
  __int64 v150; // rax
  int v151; // r14d
  _BYTE *v152; // rsi
  __int64 v153; // rax
  __int64 v154; // rdi
  __m128i *v155; // rsi
  __m128i *v156; // rdi
  bool v157; // zf
  __m128i *v158; // rsi
  __m128i *v159; // rdi
  __m128i *k; // r14
  int v161; // edx
  char v162; // al
  __int64 v163; // rdi
  _DWORD *v164; // r12
  char v165; // al
  __int64 v166; // rdx
  int *v167; // rax
  __int64 v168; // rdx
  _QWORD *v169; // rax
  __m128i *i; // rax
  __int8 v171; // dl
  _DWORD *v172; // rax
  __m128i *v173; // rax
  __int64 v174; // [rsp-10h] [rbp-410h]
  __int64 v175; // [rsp-10h] [rbp-410h]
  __int64 v176; // [rsp-8h] [rbp-408h]
  __int64 v177; // [rsp-8h] [rbp-408h]
  _BOOL4 v178; // [rsp+Ch] [rbp-3F4h]
  int v179; // [rsp+10h] [rbp-3F0h]
  _BOOL4 v180; // [rsp+14h] [rbp-3ECh]
  _BOOL4 v181; // [rsp+18h] [rbp-3E8h]
  unsigned int v182; // [rsp+1Ch] [rbp-3E4h]
  char v183; // [rsp+20h] [rbp-3E0h]
  char v184; // [rsp+24h] [rbp-3DCh]
  __int64 *v185; // [rsp+28h] [rbp-3D8h]
  _BOOL4 v186; // [rsp+30h] [rbp-3D0h]
  bool v187; // [rsp+30h] [rbp-3D0h]
  unsigned __int8 v188; // [rsp+30h] [rbp-3D0h]
  char v189; // [rsp+30h] [rbp-3D0h]
  unsigned int v190; // [rsp+34h] [rbp-3CCh]
  int v191; // [rsp+38h] [rbp-3C8h]
  int v192; // [rsp+38h] [rbp-3C8h]
  unsigned __int8 v193; // [rsp+38h] [rbp-3C8h]
  unsigned int v194; // [rsp+3Ch] [rbp-3C4h]
  __int64 v196; // [rsp+48h] [rbp-3B8h]
  int v197; // [rsp+50h] [rbp-3B0h]
  int v198; // [rsp+50h] [rbp-3B0h]
  bool v199; // [rsp+58h] [rbp-3A8h]
  __int64 v200; // [rsp+58h] [rbp-3A8h]
  __int64 v201; // [rsp+58h] [rbp-3A8h]
  int v203; // [rsp+60h] [rbp-3A0h]
  int v205; // [rsp+64h] [rbp-39Ch]
  __m128i *v206; // [rsp+68h] [rbp-398h] BYREF
  int v207; // [rsp+70h] [rbp-390h] BYREF
  unsigned int v208; // [rsp+74h] [rbp-38Ch] BYREF
  int v210; // [rsp+7Ch] [rbp-384h] BYREF
  __int64 v211; // [rsp+80h] [rbp-380h] BYREF
  _BYTE *v212; // [rsp+88h] [rbp-378h] BYREF
  __int64 *v213; // [rsp+90h] [rbp-370h] BYREF
  __int64 v214; // [rsp+98h] [rbp-368h] BYREF
  _BYTE v215[112]; // [rsp+A0h] [rbp-360h] BYREF
  __m128i v216; // [rsp+110h] [rbp-2F0h] BYREF
  __m128i v217; // [rsp+120h] [rbp-2E0h] BYREF
  __m128i v218; // [rsp+130h] [rbp-2D0h] BYREF
  __m128i v219; // [rsp+140h] [rbp-2C0h] BYREF
  _BYTE v220[12]; // [rsp+150h] [rbp-2B0h] BYREF
  __int128 v221; // [rsp+15Ch] [rbp-2A4h] BYREF
  __m128i v222; // [rsp+170h] [rbp-290h] BYREF
  __m128i v223; // [rsp+180h] [rbp-280h] BYREF
  __m128i v224; // [rsp+190h] [rbp-270h] BYREF
  __m128i v225; // [rsp+1A0h] [rbp-260h] BYREF
  __m128i v226; // [rsp+1B0h] [rbp-250h] BYREF
  __m128i v227; // [rsp+1C0h] [rbp-240h] BYREF
  __m128i v228; // [rsp+1D0h] [rbp-230h] BYREF
  __m128i v229; // [rsp+1E0h] [rbp-220h] BYREF
  __m128i v230; // [rsp+1F0h] [rbp-210h] BYREF
  __m128i v231; // [rsp+200h] [rbp-200h] BYREF
  __m128i v232; // [rsp+210h] [rbp-1F0h] BYREF
  __m128i v233; // [rsp+220h] [rbp-1E0h] BYREF
  __m128i v234; // [rsp+230h] [rbp-1D0h] BYREF
  __m128i v235; // [rsp+240h] [rbp-1C0h] BYREF
  __m128i v236; // [rsp+250h] [rbp-1B0h] BYREF
  __m128i v237; // [rsp+260h] [rbp-1A0h] BYREF
  __m128i v238[2]; // [rsp+270h] [rbp-190h] BYREF
  char v239; // [rsp+298h] [rbp-168h]

  v206 = a2;
  v14 = (__int64)a11;
  v211 = 0;
  v212 = 0;
  v207 = 0;
  v194 = sub_8D3A70(a2);
  v213 = 0;
  v185 = (__int64 *)sub_6E1A20(a1);
  v196 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 && *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 3 )
    v196 = sub_6BBB10((_QWORD *)a1);
  *(_QWORD *)a1 = 0;
  if ( (*(_BYTE *)(a1 + 9) & 8) != 0 )
  {
    if ( a11 )
      goto LABEL_9;
    v180 = 0;
    v17 = 0;
    v199 = 1;
    a4 = 1;
LABEL_134:
    if ( a12 )
    {
LABEL_13:
      v14 = 0;
      sub_82D850((__int64)a12);
      a10 = 0;
LABEL_14:
      v18 = v206;
      v205 = 0;
      v181 = 0;
      a12[2].m128i_i64[0] = (__int64)v206;
      v183 = 0;
      v184 = 0;
      goto LABEL_15;
    }
    v14 = 0;
    v183 = 0;
    v184 = 0;
    v18 = v206;
    v181 = *(char *)(qword_4D03C50 + 18LL) >= 0;
    v49 = 1;
    v205 = 1;
    if ( !a8 )
      v49 = a7;
    a7 = v49;
    goto LABEL_15;
  }
  if ( !a11 )
  {
    v17 = a4 == 0 && a5 != 0;
    v199 = a4 != 0;
    v180 = v17;
    goto LABEL_134;
  }
  if ( a11[40] < 0 )
  {
LABEL_9:
    v199 = 1;
    v15 = 0;
    a4 = 1;
    goto LABEL_10;
  }
  v199 = a4 != 0;
  v15 = a4 == 0;
LABEL_10:
  v16 = 1;
  if ( (a11[41] & 1) == 0 )
    v16 = a5 != 0;
  v17 = v15 && v16;
  v180 = v17;
  if ( a12 )
    goto LABEL_13;
  v41 = a11[40];
  if ( (v41 & 0x60) == 0x60 )
  {
    sub_82D850((__int64)v215);
    if ( (a11[43] & 0x20) != 0 )
      *(_BYTE *)(qword_4D03C50 + 21LL) |= 0x10u;
    a12 = (__m128i *)v215;
    goto LABEL_14;
  }
  if ( (v41 & 0x20) != 0 )
  {
    v181 = 0;
    v184 = *(_BYTE *)(qword_4D03C50 + 18LL) >> 7;
    v183 = *(_BYTE *)(qword_4D03C50 + 19LL) & 1;
    v53 = *(_WORD *)(qword_4D03C50 + 18LL) & 0xFE7F;
    LOBYTE(v53) = v53 | 0x80;
    *(_WORD *)(qword_4D03C50 + 18LL) = v53;
    if ( (a11[43] & 0x20) == 0 )
    {
LABEL_92:
      v205 = 1;
      v18 = v206;
      goto LABEL_15;
    }
  }
  else
  {
    if ( (v41 & 0x40) != 0 )
      goto LABEL_549;
    v181 = 1;
    v183 = 0;
    v184 = 0;
    if ( (a11[43] & 0x20) == 0 )
      goto LABEL_92;
  }
  *(_BYTE *)(qword_4D03C50 + 21LL) |= 0x10u;
  v18 = v206;
  v205 = 1;
LABEL_15:
  v19 = a6;
  BYTE1(v19) = BYTE1(a6) | 4;
  if ( !a3 )
    v19 = a6;
  v190 = v19;
  if ( dword_4F077C4 == 2 )
  {
    if ( (unsigned int)sub_8D23B0(v18) )
      sub_8AE000(v18);
    v18 = v206;
  }
  v20 = *(_BYTE *)(a1 + 8);
  v191 = 0;
  v186 = v20 == 1;
  if ( v20 == 1 )
  {
    v43 = sub_8D3BB0(v18);
    v18 = v206;
    if ( !v43 )
    {
      v44 = v206[8].m128i_i8[12];
      if ( v44 == 12 )
      {
        v45 = v206;
        do
        {
          v45 = (__m128i *)v45[10].m128i_i64[0];
          v44 = v45[8].m128i_i8[12];
        }
        while ( v44 == 12 );
      }
      if ( v44 )
      {
        v18 = v206;
        v191 = sub_8DD3B0(v206);
        if ( !v191 )
          goto LABEL_19;
      }
      if ( (*(_BYTE *)(a1 + 9) & 4) == 0 )
      {
        v191 = *(_BYTE *)(qword_4D03C50 + 16LL) <= 3u;
        goto LABEL_19;
      }
    }
    v25 = sub_8D32E0(v18);
    v179 = a6 & 8;
    v178 = (a6 & 0x8008) != 0;
    v182 = a6 & 0x2000;
    if ( *(_BYTE *)(a1 + 8) )
    {
      v191 = 1;
LABEL_25:
      v27 = *(_QWORD *)(a1 + 24);
      v28 = 0;
      v200 = v27;
      if ( v27 )
      {
        v28 = *(__m128i **)v27;
        if ( *(_QWORD *)v27 )
        {
          v28 = 0;
          if ( dword_4F077BC )
          {
            if ( qword_4F077A8 > 0x9EFBu )
            {
              if ( (unsigned int)sub_8D2B50(v206) )
                goto LABEL_275;
              if ( !dword_4F077BC )
                goto LABEL_30;
            }
            v28 = 0;
            if ( !(unsigned int)sub_8D2B80(v206) )
              goto LABEL_30;
LABEL_275:
            v191 = 1;
            v28 = 0;
          }
        }
        else if ( !*(_QWORD *)(v27 + 16) && !*(_BYTE *)(v27 + 8) )
        {
          v28 = *(__m128i **)(*(_QWORD *)(v27 + 24) + 8LL);
        }
      }
LABEL_30:
      if ( (*(_BYTE *)(a1 + 9) & 0x10) != 0 && !v194 )
      {
        v197 = dword_4F077BC | sub_8DD3B0(v206);
        if ( v197 )
        {
          v197 = 0;
        }
        else
        {
          v62 = 5;
          if ( dword_4D04964 )
            v62 = unk_4F07471;
          if ( a12 )
          {
            v197 = sub_67D3C0((int *)0x93D, v62, v185);
          }
          else if ( !v14
                 || (v121 = *(_QWORD *)(v14 + 16), (*(_BYTE *)(v121 + 127) & 8) == 0)
                 || (*(_BYTE *)(v121 + 125) & 3) == 0 )
          {
            sub_6E5C80(v62, 0x93Du, v185);
          }
        }
        if ( !(unsigned int)sub_8D23B0(v206) || (unsigned int)sub_8D23E0(v206) )
        {
          v21 = 0;
          if ( !v191 )
            goto LABEL_192;
          goto LABEL_163;
        }
      }
      else
      {
        v29 = sub_8D23B0(v206);
        if ( !v29 || (v29 = sub_8D23E0(v206), (v197 = v29) != 0) )
        {
          v197 = 0;
          goto LABEL_34;
        }
      }
      if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 4) == 0 || (v29 = sub_8DD3B0(v206)) == 0 )
      {
        v26 = v205;
        if ( !a12 )
        {
          if ( (unsigned int)sub_6E5430() )
            sub_685360(0x942u, v185, (__int64)v206);
          goto LABEL_201;
        }
        goto LABEL_165;
      }
LABEL_34:
      v23 = v194;
      LOBYTE(v29) = v194 != 0;
      v21 = v29;
      v187 = v194 != 0 && v28 != 0;
      if ( v187 )
      {
        if ( ((_DWORD)qword_4F077B4 || (unsigned int)sub_8D3BB0(v206) || (unsigned int)sub_8D3C40(v206))
          && (unsigned int)sub_8DF8D0(v206, v28) )
        {
          v26 = v205 & 1;
          if ( a10 )
          {
            sub_839D30(v200, (_DWORD)v206, a3, a4, v180, v190, a7, a8, a9, (__int64)&v216, v14, (__int64)a12);
            a8 = 0;
            v200 = 0;
            v192 = 0;
            v186 = 1;
            a7 = 0;
            goto LABEL_43;
          }
          sub_839D30(v200, (_DWORD)v206, a3, a4, v180, v190, a7, a8, a9, 0, v14, (__int64)a12);
          if ( v205 )
          {
            v26 = v194 != 0 && v28 != 0;
            v186 = v205;
            v200 = 0;
            v212 = *(_BYTE **)v14;
            v192 = 0;
            v211 = *(_QWORD *)(v14 + 8);
            a8 = 0;
            a7 = 0;
            goto LABEL_43;
          }
LABEL_552:
          v200 = 0;
          v26 = 0;
          v192 = 0;
          a8 = 0;
          a7 = 0;
          v186 = 1;
          goto LABEL_43;
        }
        if ( !v191 )
          goto LABEL_191;
      }
      else
      {
        if ( !v191 )
        {
LABEL_192:
          v193 = v21;
          v63 = sub_8DD3B0(v206);
          v21 = v193;
          v26 = v205 & 1;
          if ( v63 || (v188 = v193, (v192 = sub_6E1B90((_QWORD *)v200)) != 0) )
          {
            if ( !a12 )
            {
              sub_832E80((_QWORD *)v200);
              sub_6C55E0(0, 1u, v200, 0, 0, &v211);
              if ( !(unsigned int)sub_8D3D40(v206) || *(_BYTE *)(v211 + 48) != 5 )
                goto LABEL_342;
              v132 = *(_QWORD *)(v211 + 64);
              if ( v132 )
              {
                v200 = *(_QWORD *)(v132 + 16);
                if ( v200 )
                  goto LABEL_342;
                if ( *(_BYTE *)(v132 + 24) != 2 )
                  goto LABEL_343;
              }
              v238[0].m128i_i64[0] = (__int64)sub_724DC0();
              v200 = v211;
              v133 = (__int64 *)sub_6EC670((__int64)v206, v211, a9, HIBYTE(v190) & 1);
              v134 = (_BYTE *)v238[0].m128i_i64[0];
              sub_70FD90(v133, v238[0].m128i_i64[0]);
              v135 = (_BYTE *)sub_724E50(v238[0].m128i_i64, v134);
              v211 = 0;
              v212 = v135;
              v192 = 1;
              v186 = 1;
              goto LABEL_43;
            }
            if ( !v182 )
            {
              v200 = 0;
              v192 = 1;
              a12->m128i_i32[2] = 4;
              v186 = 1;
              goto LABEL_43;
            }
          }
          else
          {
            v97 = v206;
            v157 = (v188 & (v200 == 0)) == 0;
            v98 = v206;
            v189 = v188 & (v200 == 0);
            if ( v157 )
            {
              if ( !v194 )
              {
LABEL_311:
                if ( v200 )
                {
                  if ( *(_QWORD *)v200 )
                    goto LABEL_313;
                  if ( !(unsigned int)sub_8D32E0(v98)
                    || v28
                    && ((v150 = sub_8D46C0(v206), (unsigned int)sub_8DF8D0(v150, v28)) || (unsigned int)sub_8DBE70(v28)) )
                  {
                    if ( *(_BYTE *)(v200 + 8) == 1 )
                    {
                      if ( a12 )
                      {
                        v151 = (int)v206;
                        v168 = v200;
                        v197 = 1;
                      }
                      else
                      {
                        v151 = (int)v206;
                        for ( i = v206; ; i = (__m128i *)i[10].m128i_i64[0] )
                        {
                          v171 = i[8].m128i_i8[12];
                          if ( v171 != 12 )
                            break;
                        }
                        if ( v171 )
                        {
                          v172 = (_DWORD *)sub_6E1A20(v200);
                          sub_6E5D20(7, 0x928u, v172, (__int64)v206);
                          v151 = (int)v206;
                        }
                        v168 = v200;
                      }
                      while ( *(_BYTE *)(v168 + 8) == 1 )
                      {
                        v169 = *(_QWORD **)(v168 + 24);
                        if ( !v169 || *v169 || v169[2] )
                          break;
                        v168 = *(_QWORD *)(v168 + 24);
                      }
                      LODWORD(v200) = v168;
                    }
                    else
                    {
                      v151 = (int)v206;
                    }
                  }
                  else
                  {
                    v151 = (int)v206;
                    v98 = v206;
                    if ( *(_BYTE *)(v200 + 8) || *(_BYTE *)(*(_QWORD *)(v200 + 24) + 24LL) != 3 )
                    {
LABEL_313:
                      if ( !(unsigned int)sub_8D32E0(v98) )
                      {
                        if ( !a12 )
                        {
                          if ( dword_4F077C0 )
                          {
                            v99 = *(_QWORD *)v200;
                            if ( *(_QWORD *)v200 && *(_BYTE *)(v99 + 8) == 3 )
                              v99 = sub_6BBB10((_QWORD *)v200);
                            *(_QWORD *)v200 = 0;
                            v100 = (_DWORD *)sub_6E1A20(v99);
                            sub_69D070(0x48Au, v100);
                            sub_6E6470(v99);
                            sub_6E1990((_QWORD *)v99);
                            if ( a10 )
                            {
                              sub_839D30(a1, (_DWORD)v206, a3, 0, 0, v190, a7, a8, 0, (__int64)&v216, v14, 0);
                              v200 = 0;
                              v192 = 0;
                              v186 = 1;
                            }
                            else
                            {
                              sub_839D30(a1, (_DWORD)v206, a3, 0, 0, v190, a7, a8, 0, 0, v14, 0);
                              v200 = 0;
                              v192 = 0;
                              v212 = *(_BYTE **)v14;
                              v186 = 1;
                              v211 = *(_QWORD *)(v14 + 8);
                            }
                            goto LABEL_43;
                          }
                          sub_6E6470(v200);
                          sub_832E80((_QWORD *)v200);
                          v163 = *(_QWORD *)v200;
                          if ( *(_QWORD *)v200 && *(_BYTE *)(v163 + 8) == 3 )
                            v163 = sub_6BBB10((_QWORD *)v200);
                          v164 = (_DWORD *)sub_6E1A20(v163);
                          if ( (unsigned int)sub_6E5430() )
                            sub_6851C0(0x92u, v164);
LABEL_201:
                          sub_6E6260(&v216);
                          v200 = 0;
                          v192 = 1;
                          v186 = 1;
                          goto LABEL_43;
                        }
                        goto LABEL_165;
                      }
LABEL_412:
                      v136 = sub_8D46C0(v206);
                      if ( !(unsigned int)sub_8D3A70(v136)
                        && (unsigned int)sub_8D3070(v206)
                        && (*(_BYTE *)(v136 + 140) & 0xFB) == 8 )
                      {
                        v137 = sub_8D4C10(v136, dword_4F077C4 != 2) & 1;
                      }
                      else
                      {
                        v137 = 0;
                      }
                      v138 = v190 & 0x2018 | 0x800;
                      if ( a12 )
                      {
                        sub_839D30(a1, v136, 0, 1, 0, v138, 1, 1, v137, 0, 0, (__int64)a12);
                        if ( a12->m128i_i32[2] == 7
                          || (unsigned int)sub_8D3070(v206)
                          && ((*(_BYTE *)(v136 + 140) & 0xFB) != 8 || (sub_8D4C10(v136, dword_4F077C4 != 2) & 1) == 0) )
                        {
                          v186 = 0;
                          v200 = 0;
                          v197 = 1;
                        }
                        else
                        {
                          v186 = 0;
                          v200 = 0;
                          v161 = a12[5].m128i_u8[4];
                          a12[2].m128i_i64[0] = (__int64)v206;
                          v162 = 2 * v161;
                          v21 = v161 & 0xFFFFFFF9;
                          a12[5].m128i_i8[4] = v21 | v162 & 4;
                        }
                      }
                      else
                      {
                        sub_839D30(a1, v136, 0, 1, 0, v138, 1, 1, v137, (__int64)&v216, 0, 0);
                        if ( (unsigned int)sub_8D23E0(v136) && v217.m128i_i8[0] )
                        {
                          v165 = *(_BYTE *)(v216.m128i_i64[0] + 140);
                          if ( v165 == 12 )
                          {
                            v166 = v216.m128i_i64[0];
                            do
                            {
                              v166 = *(_QWORD *)(v166 + 160);
                              v165 = *(_BYTE *)(v166 + 140);
                            }
                            while ( v165 == 12 );
                          }
                          if ( v165 )
                          {
                            v173 = sub_73CA70((const __m128i *)v216.m128i_i64[0], v136);
                            v206 = (__m128i *)sub_72D750(v173, (__int64)v206);
                          }
                        }
                        sub_842520(&v216, v206, 0, v178, v190, 144);
                        v200 = 0;
                        v186 = 0;
                      }
                      goto LABEL_43;
                    }
                  }
                  v190 |= 0x100000u;
                  if ( a10 )
                  {
                    sub_839D30(v200, v151, a3, a4, v180, v190, a7, a8, a9, (__int64)&v216, v14, (__int64)a12);
                    a8 = 0;
                    v200 = 0;
                    a7 = 0;
                    v186 = 1;
                    goto LABEL_43;
                  }
                  sub_839D30(v200, v151, a3, a4, v180, v190, a7, a8, a9, 0, v14, (__int64)a12);
                  if ( v205 )
                  {
                    v186 = v205;
                    v26 = 1;
                    v200 = 0;
                    v212 = *(_BYTE **)v14;
                    a8 = 0;
                    v211 = *(_QWORD *)(v14 + 8);
                    a7 = 0;
                    goto LABEL_43;
                  }
                  goto LABEL_552;
                }
                if ( (unsigned int)sub_8D32E0(v98) )
                  goto LABEL_412;
                v143 = (__m128i *)(a1 + 32);
                v144 = a3 == 0;
                if ( v14 && (*(_BYTE *)(v14 + 40) & 0x20) != 0 )
                {
                  sub_8326B0((__int64)v206, (__m128i *)v144, v205, v143, 0, &v210, &v211, &v212, (_BYTE *)v14, &v207);
                  if ( a12 )
                    goto LABEL_450;
LABEL_343:
                  v192 = 1;
                  v186 = 1;
                  goto LABEL_43;
                }
                if ( a12 )
                {
                  sub_8326B0((__int64)v206, (__m128i *)v144, v205, v143, 0, &v210, &v211, &v212, (_BYTE *)v14, &v207);
LABEL_450:
                  if ( !v207 )
                  {
                    v192 = 1;
                    v186 = 1;
                    a12->m128i_i32[2] = 0;
                    goto LABEL_43;
                  }
                  goto LABEL_166;
                }
                sub_8326B0((__int64)v206, (__m128i *)v144, v205, v143, 0, &v210, &v211, &v212, (_BYTE *)v14, 0);
LABEL_342:
                v200 = 0;
                goto LABEL_343;
              }
              goto LABEL_376;
            }
            if ( !(unsigned int)sub_8D5940(v206, 0, 0) )
            {
              v97 = v206;
LABEL_376:
              if ( (unsigned int)sub_828BC0((__int64)v97, &v214) )
              {
                v122 = v190 & 0x5201;
                if ( v205 )
                  sub_832E80((_QWORD *)v200);
                v123 = HIBYTE(v190) & 1;
                if ( v14 )
                {
                  v124 = v122 | 0x200;
                  if ( (*(_BYTE *)(v14 + 40) & 2) == 0 )
                    v124 = v190 & 0x5201;
                  sub_847910(a1, v214, (_DWORD)v206, v123, v124, (unsigned int)&v211, 0, (__int64)a12);
                  v24 = v175;
                  if ( !a12 )
                  {
                    v200 = v211;
                    if ( !v211 )
                    {
                      sub_6E6260(&v216);
                      v192 = 1;
                      v186 = 1;
                      goto LABEL_43;
                    }
                  }
                }
                else
                {
                  sub_847910(a1, v214, (_DWORD)v206, v123, v122, 0, (__int64)&v216, (__int64)a12);
                  v23 = v177;
                }
                goto LABEL_342;
              }
              v98 = v206;
              for ( j = v206; j[8].m128i_i8[12] == 12; j = (__m128i *)j[10].m128i_i64[0] )
                ;
              v126 = *(_QWORD *)(*(_QWORD *)(j->m128i_i64[0] + 96) + 8LL);
              if ( !v126 )
                goto LABEL_311;
              v210 = 0;
              if ( !a12 )
              {
                v145 = 0;
                if ( v14 && (*(_BYTE *)(v14 + 43) & 8) != 0 )
                  v145 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(qword_4F04C50 + 32LL) + 40LL) + 32LL);
                sub_832E80((_QWORD *)v200);
                v146 = v200;
                while ( v146 )
                {
                  v147 = *(_QWORD *)v146;
                  *(_BYTE *)(v146 + 9) |= 8u;
                  if ( !v147 )
                    break;
                  if ( *(_BYTE *)(v147 + 8) == 3 )
                    v146 = sub_6BBB10((_QWORD *)v146);
                  else
                    v146 = v147;
                }
                sub_6C5750(
                  v126,
                  (__int64)v185,
                  v145,
                  0,
                  a7,
                  1,
                  0,
                  v190,
                  0,
                  1u,
                  v200,
                  a1,
                  0,
                  &v210,
                  &v208,
                  0,
                  0,
                  0,
                  &v211,
                  0,
                  0);
                v148 = v200;
                while ( v148 )
                {
                  v149 = *(_QWORD *)v148;
                  *(_BYTE *)(v148 + 9) &= ~8u;
                  if ( !v149 )
                    break;
                  if ( *(_BYTE *)(v149 + 8) == 3 )
                    v148 = sub_6BBB10((_QWORD *)v148);
                  else
                    v148 = v149;
                }
                v21 = v208;
                v192 = v208 == 0;
                if ( !v211 )
                  sub_6E6260(&v216);
                goto LABEL_392;
              }
              if ( v182 )
                goto LABEL_390;
              if ( (unsigned int)sub_836C50(
                                   0,
                                   a1,
                                   v206,
                                   1u,
                                   0,
                                   a3 == 0,
                                   0,
                                   0,
                                   v190,
                                   (__int64)v238,
                                   0,
                                   (unsigned int *)&v216,
                                   0) )
              {
                v157 = v216.m128i_i32[0] == 0;
                v22 = 12;
                v158 = v238;
                a12->m128i_i32[2] = 4;
                v159 = a12 + 3;
                while ( v22 )
                {
                  v159->m128i_i32[0] = v158->m128i_i32[0];
                  v158 = (__m128i *)((char *)v158 + 4);
                  v159 = (__m128i *)((char *)v159 + 4);
                  --v22;
                }
                if ( v157 )
                {
                  v21 = v238[0].m128i_i64[0];
                  if ( v238[0].m128i_i64[0] )
                  {
                    v210 = *(_BYTE *)(v238[0].m128i_i64[0] + 193) >> 7;
                    if ( (*(_BYTE *)(v238[0].m128i_i64[0] + 194) & 0x10) == 0 )
                    {
                      if ( v28 )
                      {
                        for ( k = v206; k[8].m128i_i8[12] == 12; k = (__m128i *)k[10].m128i_i64[0] )
                          ;
                        while ( v28[8].m128i_i8[12] == 12 )
                          v28 = (__m128i *)v28[10].m128i_i64[0];
                        if ( v28 == k || (unsigned int)sub_8D97D0(v28, k, 0, 0, v23) )
                        {
                          a12->m128i_i32[2] = 0;
                        }
                        else if ( sub_8D5CE0(v28, k) )
                        {
                          a12->m128i_i32[2] = 2;
                        }
                      }
                    }
                  }
                }
                goto LABEL_391;
              }
              if ( !v216.m128i_i32[0] )
              {
LABEL_390:
                a12->m128i_i32[2] = 7;
              }
              else
              {
                v22 = 12;
                v155 = v238;
                a12->m128i_i32[2] = 4;
                v156 = a12 + 3;
                while ( v22 )
                {
                  v156->m128i_i32[0] = v155->m128i_i32[0];
                  v155 = (__m128i *)((char *)v155 + 4);
                  v156 = (__m128i *)((char *)v156 + 4);
                  --v22;
                }
              }
LABEL_391:
              v192 = 1;
LABEL_392:
              if ( !a3 && v210 )
              {
                if ( a12 )
                {
                  v197 = 1;
                }
                else if ( (unsigned int)sub_6E5430() )
                {
                  sub_6851C0(0x91Eu, v185);
                }
              }
              a7 = 0;
              v200 = 0;
              v186 = 1;
              goto LABEL_43;
            }
            if ( (unsigned int)sub_828BC0((__int64)v206, &v214)
              || (LOBYTE(v21) = v182 != 0, (v189 = v182 != 0 && a12 != 0) == 0) )
            {
              v141 = (__m128i *)(a1 + 32);
              v142 = a3 == 0;
              if ( v14 && (*(_BYTE *)(v14 + 40) & 0x20) != 0 )
              {
                sub_8326B0(
                  (__int64)v206,
                  (__m128i *)v142,
                  v205,
                  v141,
                  v238[0].m128i_i64,
                  &v210,
                  &v211,
                  &v212,
                  (_BYTE *)v14,
                  &v207);
                if ( !v207 )
                {
                  if ( !a12 )
                    goto LABEL_342;
LABEL_442:
                  if ( v189 )
                  {
                    a12[5].m128i_i8[5] |= 0x80u;
                    a12->m128i_i32[2] = 0;
                  }
                  else
                  {
                    a12->m128i_i32[2] = 4;
                    a12[3].m128i_i64[0] = v238[0].m128i_i64[0];
                  }
                  goto LABEL_342;
                }
              }
              else if ( a12 )
              {
                sub_8326B0(
                  (__int64)v206,
                  (__m128i *)v142,
                  v205,
                  v141,
                  v238[0].m128i_i64,
                  &v210,
                  &v211,
                  &v212,
                  (_BYTE *)v14,
                  &v207);
                if ( !v207 )
                  goto LABEL_442;
              }
              else
              {
                v167 = &v207;
                if ( *(char *)(qword_4D03C50 + 18LL) >= 0 )
                  v167 = 0;
                sub_8326B0(
                  (__int64)v206,
                  (__m128i *)v142,
                  v205,
                  v141,
                  v238[0].m128i_i64,
                  &v210,
                  &v211,
                  &v212,
                  (_BYTE *)v14,
                  v167);
                if ( !v207 )
                  goto LABEL_342;
              }
              v197 = 1;
              goto LABEL_342;
            }
          }
LABEL_165:
          v200 = 0;
LABEL_166:
          v192 = 1;
          v197 = 1;
          v186 = 1;
          goto LABEL_43;
        }
        if ( !v28 || !v194 )
          goto LABEL_163;
        v187 = 1;
      }
      if ( (unsigned int)sub_8DD3B0(v28) )
      {
LABEL_191:
        LOBYTE(v21) = v187;
        goto LABEL_192;
      }
LABEL_163:
      v26 = v205;
      if ( !a12 || (v22 = v182) == 0 )
      {
        v64 = (__m128i *)v14;
        if ( !v14 )
        {
          sub_6E6990((__int64)v238);
          if ( !v181 )
            v239 |= 0x20u;
          if ( !v205 )
            v239 |= 0x40u;
          v64 = v238;
          if ( v179 )
            v239 |= 8u;
        }
        if ( a12 )
        {
          v65 = (__int64)a12;
          if ( !(unsigned int)sub_8D3410(v206)
            && (!dword_4F077BC || qword_4F077A8 <= 0x9EFBu || !(unsigned int)sub_8D2B50(v206)) )
          {
            v65 = 0;
          }
          sub_638440((__int64 *)a1, (__int64 *)&v206, v64, v65, a7);
          v66 = v64[2].m128i_i8[10] < 0;
          v212 = (_BYTE *)v64->m128i_i64[0];
          v211 = v64->m128i_i64[1];
          if ( v66 )
            *(_BYTE *)(qword_4D03C50 + 19LL) |= 0x20u;
          if ( (v64[2].m128i_i8[9] & 2) != 0 )
          {
            v197 = 1;
          }
          else if ( v65 )
          {
            if ( a12->m128i_i32[2] == 7 )
              a12->m128i_i32[2] = 0;
          }
          else
          {
            a12->m128i_i32[2] = 4;
          }
        }
        else
        {
          sub_638440((__int64 *)a1, (__int64 *)&v206, v64, 0, a7);
          v66 = v64[2].m128i_i8[10] < 0;
          v212 = (_BYTE *)v64->m128i_i64[0];
          v211 = v64->m128i_i64[1];
          if ( v66 )
            *(_BYTE *)(qword_4D03C50 + 19LL) |= 0x20u;
          if ( (v64[2].m128i_i8[9] & 2) != 0 )
          {
            v211 = 0;
            v212 = 0;
            sub_6E6260(&v216);
          }
        }
        v200 = 0;
        v192 = 1;
        v186 = 1;
        a7 = 0;
        goto LABEL_43;
      }
      goto LABEL_165;
    }
LABEL_40:
    v26 = v205;
    if ( a12 )
    {
      if ( v199 && (unsigned int)sub_827F00(*(_QWORD *)(a1 + 24) + 8LL, (__int64)v206, 0, 0, 0, 0) )
      {
        v200 = 0;
        v192 = 1;
        v197 = 1;
      }
      else
      {
        sub_838020(*(_QWORD *)(a1 + 24) + 8LL, 0, v206, 0, v182 == 0, 0, a12);
        v21 = v176;
        v200 = 0;
        v192 = 1;
        v197 = 0;
      }
      goto LABEL_43;
    }
    v203 = v25;
    v238[0].m128i_i32[0] = 0;
    sub_6E6610((_QWORD *)a1, &v216, 0);
    v69 = &v216;
    v70 = v203;
    if ( v199 )
    {
      v74 = unk_4D041E4;
      if ( unk_4D041E4 )
        v74 = (v190 >> 20) & 1;
      v75 = sub_827F00((__int64)&v216, (__int64)v206, v74, 1, 0, v238);
      v69 = &v216;
      v70 = v203;
      if ( v75 )
      {
        if ( !v181 )
          sub_6E50A0();
        sub_6E6840((__int64)&v216);
        v192 = 1;
        goto LABEL_230;
      }
    }
    if ( v194 )
    {
      if ( !a3 )
        goto LABEL_363;
      if ( dword_4F077C4 == 2 && unk_4F07778 > 201702 && v217.m128i_i8[1] == 2 )
      {
        v106 = (int)v206;
        v198 = v70;
        if ( (__m128i *)v216.m128i_i64[0] == v206 )
        {
LABEL_364:
          sub_8470D0((unsigned int)&v216, v106, a7, v190, 144, (unsigned int)&v208, (__int64)&v211);
          v23 = v174;
          v24 = v176;
          if ( v211 )
          {
            v108 = v208;
            if ( v208 )
            {
              if ( !(unsigned int)sub_8319F0((__int64)&v216, &v213) || v213[7] != v211 )
                v213 = 0;
              v108 = v208;
            }
          }
          else
          {
            if ( v181 )
            {
              v139 = *(_BYTE *)(v216.m128i_i64[0] + 140);
              if ( v139 == 12 )
              {
                v140 = v216.m128i_i64[0];
                do
                {
                  v140 = *(_QWORD *)(v140 + 160);
                  v139 = *(_BYTE *)(v140 + 140);
                }
                while ( v139 == 12 );
              }
              if ( v139 )
                sub_6861A0(0x19Fu, &v220[4], v216.m128i_i64[0], (__int64)v206);
            }
            sub_6E6840((__int64)&v216);
            v108 = v208;
          }
          a7 = 0;
          v192 = v108 == 0;
LABEL_230:
          v200 = 0;
          v197 = 0;
          goto LABEL_43;
        }
        v107 = sub_8D97D0(v206, v216.m128i_i64[0], 32, v67, v68);
        v69 = &v216;
        v70 = v198;
        if ( v107 )
        {
LABEL_363:
          v106 = (int)v206;
          goto LABEL_364;
        }
      }
    }
    v71 = (int)v206;
    if ( v70 )
    {
      if ( v179 )
        sub_6FAB30(&v216, (__int64)v206, 1u, (v190 & 0x1000000) == 0, 0);
      else
        sub_842520(&v216, v206, 0, v178, v190, 144);
      v192 = 1;
    }
    else
    {
      if ( v217.m128i_i8[0] )
      {
        v103 = v216.m128i_i64[0];
        for ( m = *(_BYTE *)(v216.m128i_i64[0] + 140); m == 12; m = *(_BYTE *)(v103 + 140) )
          v103 = *(_QWORD *)(v103 + 160);
        if ( m && (v17 || v238[0].m128i_i32[0]) )
        {
          sub_827F00((__int64)&v216, (__int64)v206, 0, 0, 1, 0);
          v71 = (int)v206;
          LODWORD(v69) = (unsigned int)&v216;
        }
      }
      sub_843C40((_DWORD)v69, v71, 0, 0, a3 == 0, v190, 144);
      v21 = v176;
      v192 = 1;
    }
    goto LABEL_230;
  }
LABEL_19:
  v25 = sub_8D32E0(v18);
  if ( (*(_BYTE *)(a1 + 9) & 4) == 0 || (v197 = v25 | v191) != 0 )
  {
    v179 = a6 & 8;
    v178 = (a6 & 0x8008) != 0;
    v182 = a6 & 0x2000;
    if ( *(_BYTE *)(a1 + 8) )
    {
      if ( v20 == 1 )
        goto LABEL_25;
LABEL_549:
      sub_721090();
    }
    goto LABEL_40;
  }
  if ( a12 )
  {
    v200 = 0;
    v192 = 1;
    v197 = 1;
    v26 = v205 & 1;
  }
  else
  {
    v58 = *(_QWORD *)(a1 + 24);
    v59 = (_QWORD *)v58;
    if ( v205 )
    {
      sub_6E6470(v58);
      v59 = (_QWORD *)v58;
    }
LABEL_172:
    for ( n = *((_BYTE *)v59 + 8); n != 2; v59 = (_QWORD *)*v59 )
    {
      if ( !n )
      {
        v61 = v59[3];
        if ( *(_BYTE *)(v61 + 24) == 2 && *(_BYTE *)(v61 + 325) == 13 )
          break;
      }
      if ( !*v59 )
        BUG();
      n = *(_BYTE *)(*v59 + 8LL);
      if ( n == 3 )
      {
        v59 = (_QWORD *)sub_6BBB10(v59);
        goto LABEL_172;
      }
    }
    if ( (unsigned int)sub_6E5430() )
    {
      v102 = (_DWORD *)sub_6E1A20((__int64)v59);
      sub_685360(0x935u, v102, (__int64)v206);
    }
    sub_6E6260(&v216);
    v200 = 0;
    v192 = 1;
    v26 = v205 & 1;
  }
LABEL_43:
  if ( word_4D04898 || !v205 )
    goto LABEL_45;
  if ( *(_BYTE *)(qword_4D03C50 + 16LL) <= 3u )
  {
    if ( !v211 )
    {
      v30 = v200;
      if ( !v212 )
      {
        sub_6E6B60(&v216, 0, v21, v22, v23, v24);
        if ( v217.m128i_i8[0] == 1 )
        {
          if ( dword_4F04C44 != -1
            || (v83 = qword_4F04C68, (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) != 0) )
          {
            sub_6F4B70(&v216, 0, (__int64)v83, v84, v85, v86);
          }
        }
        v87 = sub_724DC0();
        v238[0].m128i_i64[0] = (__int64)v87;
        sub_6F4950(&v216, (__int64)v87);
        v88 = sub_724E50(v238[0].m128i_i64, v87);
        v30 = v200;
        v212 = (_BYTE *)v88;
        *(_QWORD *)(v88 + 64) = *(_QWORD *)&v220[4];
        *(_QWORD *)(v88 + 112) = v221;
      }
      goto LABEL_109;
    }
    if ( *(_BYTE *)(v211 + 48) == 2 )
    {
      v212 = *(_BYTE **)(v211 + 56);
      if ( !sub_7307F0(v211) )
      {
        v30 = v200;
        if ( *(_BYTE *)(qword_4D03C50 + 16LL) )
        {
          v101 = sub_740190((__int64)v212, 0, 0);
          sub_72F900(v211, v101);
          sub_71AAB0((__int64)v212, v211);
          v30 = v211;
        }
        goto LABEL_108;
      }
    }
    else
    {
      if ( (unsigned int)sub_6E5430() )
        sub_6851C0(0x1Cu, v185);
      v212 = sub_72C9A0();
    }
    v30 = v200;
LABEL_108:
    v211 = 0;
LABEL_109:
    a8 = sub_6E4B40();
    if ( a8 )
    {
      a8 = 0;
      v212 = sub_72C9A0();
    }
    goto LABEL_46;
  }
  if ( !(unsigned int)sub_6E4B50() )
  {
LABEL_45:
    v30 = v200;
    goto LABEL_46;
  }
  v31 = v211;
  v30 = v200;
  if ( v211 )
    goto LABEL_47;
  if ( v212 )
  {
    if ( v200 || !v26 )
    {
      v30 = v200;
      v35 = v200 != 0;
LABEL_65:
      if ( a10 )
      {
LABEL_66:
        if ( v31 )
          goto LABEL_67;
        v46 = v212;
LABEL_232:
        if ( !v46 )
          goto LABEL_303;
        goto LABEL_233;
      }
      goto LABEL_123;
    }
    v46 = v212;
    if ( v192 )
      goto LABEL_147;
    goto LABEL_253;
  }
  sub_6E6B60(&v216, 0, v54, v55, v56, v57);
LABEL_46:
  v31 = v211;
  if ( !v211 )
  {
LABEL_61:
    if ( v30 || !v26 )
      goto LABEL_63;
    goto LABEL_145;
  }
LABEL_47:
  if ( *(_BYTE *)(v31 + 48) != 2 || v14 && (*(_BYTE *)(v14 + 40) & 8) != 0 )
  {
    if ( v30 || !v26 )
      goto LABEL_63;
    goto LABEL_50;
  }
  if ( !(unsigned int)sub_8D3410(*(_QWORD *)(*(_QWORD *)(v31 + 56) + 128LL)) )
  {
    v34 = v211;
    v211 = 0;
    v212 = *(_BYTE **)(v34 + 56);
    goto LABEL_61;
  }
  if ( v30 || !v26 )
    goto LABEL_63;
  if ( v211 )
  {
LABEL_50:
    if ( !v205 )
      goto LABEL_51;
    v35 = 0;
    v30 = 0;
    v31 = v211;
    if ( a10 )
      goto LABEL_66;
LABEL_123:
    v46 = v212;
    goto LABEL_124;
  }
LABEL_145:
  v46 = v212;
  if ( !v212 )
  {
    if ( !v205 )
      goto LABEL_51;
    if ( !a10 )
    {
      v76 = *(_BYTE *)(v14 + 40);
      v31 = v211;
      v30 = 0;
      v35 = 0;
      *(_QWORD *)v14 = 0;
      *(_QWORD *)(v14 + 8) = 0;
      v48 = v76 & 8;
      goto LABEL_284;
    }
    goto LABEL_254;
  }
  if ( !v192 )
  {
    if ( !v205 )
      goto LABEL_51;
LABEL_253:
    if ( !a10 )
    {
      v31 = v211;
      v30 = 0;
      v35 = 0;
LABEL_124:
      v47 = *(_BYTE *)(v14 + 40);
      *(_QWORD *)v14 = 0;
      *(_QWORD *)(v14 + 8) = 0;
      v48 = v47 & 8;
      if ( v46 )
        goto LABEL_125;
LABEL_284:
      if ( v31 )
        goto LABEL_239;
      if ( !v48 )
      {
        if ( v217.m128i_i8[0] == 2 )
        {
          v131 = sub_724DC0();
          v238[0].m128i_i64[0] = (__int64)v131;
          sub_6F4950(&v216, (__int64)v131);
          v46 = (_BYTE *)sub_724E50(v238[0].m128i_i64, v131);
          *((_QWORD *)v46 + 8) = *(_QWORD *)&v220[4];
          *((_QWORD *)v46 + 14) = v221;
          goto LABEL_126;
        }
        if ( !v217.m128i_i8[0] )
          goto LABEL_351;
        v77 = v216.m128i_i64[0];
        for ( ii = *(_BYTE *)(v216.m128i_i64[0] + 140); ii == 12; ii = *(_BYTE *)(v77 + 140) )
          v77 = *(_QWORD *)(v77 + 160);
        if ( !ii )
        {
LABEL_351:
          if ( *(_BYTE *)(qword_4D03C50 + 16LL) <= 3u )
          {
            v105 = sub_72C9A0();
            *(_QWORD *)v14 = v105;
            v46 = v105;
            goto LABEL_247;
          }
        }
      }
      if ( !(unsigned int)sub_831A40((__int64)&v216, a7 == 0, &v214, &v211) )
      {
        if ( v200 )
        {
LABEL_280:
          v211 = v200;
          v31 = v200;
          goto LABEL_239;
        }
        if ( v212 )
          goto LABEL_402;
        if ( v217.m128i_i8[0] == 2 )
        {
          v211 = sub_6EAFA0(2u);
          v152 = sub_724DC0();
          v238[0].m128i_i64[0] = (__int64)v152;
          sub_6F4950(&v216, (__int64)v152);
          v153 = sub_724E50(v238[0].m128i_i64, v152);
          v154 = v211;
          *(_QWORD *)(v153 + 64) = *(_QWORD *)&v220[4];
          *(_QWORD *)(v153 + 112) = v221;
          sub_72F900(v154, (_BYTE *)v153);
          v31 = v211;
LABEL_239:
          *(_QWORD *)(v14 + 8) = v31;
          if ( a7 && v194 )
          {
            LODWORD(v72) = (_DWORD)v206;
            if ( (*(_BYTE *)(v14 + 43) & 8) != 0 )
              v72 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 216) + 40LL) + 32LL);
            v73 = sub_6EB2F0((int)v206, v72, (int)v185, 0);
            if ( v73 )
            {
              *(_QWORD *)(v211 + 16) = v73;
              if ( (*(_BYTE *)(qword_4D03C50 + 17LL) & 2) != 0 )
                *(_BYTE *)(v73 + 193) |= 0x40u;
            }
          }
          v46 = *(_BYTE **)v14;
LABEL_247:
          if ( !v46 )
          {
            if ( sub_730800(*(_QWORD *)(v14 + 8)) )
              *(_BYTE *)(v14 + 41) |= 2u;
            if ( *(char *)(*(_QWORD *)(v14 + 8) + 50LL) < 0 )
              goto LABEL_130;
            goto LABEL_77;
          }
LABEL_127:
          if ( !v46[173] )
            *(_BYTE *)(v14 + 41) |= 2u;
          if ( (v46[170] & 0x40) != 0 )
LABEL_130:
            *(_BYTE *)(v14 + 41) |= 0x10u;
LABEL_77:
          v38 = v211;
          if ( !v211 && v35 )
          {
            v211 = v30;
            if ( (v205 & v192) == 0 )
              goto LABEL_51;
            v38 = v30;
            if ( (v190 & 0x1000000) == 0 )
              goto LABEL_82;
          }
          else
          {
            if ( (v205 & v192) == 0 || !v211 )
              goto LABEL_51;
            if ( (v190 & 0x1000000) == 0 )
              goto LABEL_82;
          }
          *(_BYTE *)(v38 + 50) |= 0x10u;
          v50 = sub_730290(v38);
          *(_BYTE *)(v50 + 50) |= 0x10u;
LABEL_82:
          if ( v186 )
          {
            if ( !v14 || (*(_BYTE *)(v14 + 43) & 0x20) == 0 )
            {
              v39 = v211;
              *(_BYTE *)(v211 + 50) |= 0x40u;
              v40 = sub_730290(v39);
              *(_BYTE *)(v40 + 50) |= 0x40u;
            }
            if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 2) != 0 )
              sub_6EA060(v211, a1);
          }
          goto LABEL_51;
        }
        v211 = sub_6EAFA0(3u);
        v201 = v211;
        *(_QWORD *)(v201 + 56) = sub_6F6F40(&v216, 0, v79, v80, v81, v82);
      }
      v31 = v211;
      goto LABEL_239;
    }
LABEL_254:
    v35 = 0;
    v30 = 0;
    goto LABEL_232;
  }
LABEL_147:
  v51 = *((_QWORD *)v46 + 18);
  if ( !v51 || *(_BYTE *)(v51 + 24) != 5 )
  {
    if ( !v205 )
      goto LABEL_51;
    v35 = 0;
    v30 = 0;
    if ( a10 )
    {
LABEL_233:
      if ( !a8
        && ((*((_QWORD *)v46 + 21) & 0xFF0002000000LL) != 0xA0000000000LL || *(_BYTE *)(qword_4D03C50 + 16LL) <= 3u) )
      {
        if ( (v190 & 0x1000000) != 0 )
          v46[168] |= 0x20u;
        sub_6E6A50((__int64)v46, a10);
        goto LABEL_76;
      }
      if ( v200 )
      {
        v211 = v200;
        v31 = v200;
        v212 = 0;
        goto LABEL_67;
      }
      v211 = sub_6EAFA0(2u);
      sub_72F900(v211, v212);
      v31 = v211;
      v212 = 0;
      if ( v211 )
      {
LABEL_67:
        if ( v194 && a7 )
          sub_6EB360(v31, (__int64)v206, (int)v206, v185);
        v36 = v213;
        if ( v213 && v213[7] == v211 )
        {
          sub_6EB510((__int64)v213);
        }
        else
        {
          v37 = 1;
          if ( (v190 & 0x1000000) == 0 )
            v37 = (*(_BYTE *)(v211 + 50) & 0x10) != 0;
          v36 = (__int64 *)sub_6EC670((__int64)v206, v211, a9, v37);
        }
        sub_6E7170(v36, a10);
LABEL_76:
        *(_QWORD *)(a10 + 68) = *v185;
        *(_QWORD *)(a10 + 76) = *(_QWORD *)sub_6E1A60(a1);
        goto LABEL_77;
      }
LABEL_303:
      if ( a8 )
      {
        if ( (unsigned int)sub_8319F0((__int64)&v216, &v214) )
        {
          if ( !v192 && !sub_7307F0(*(_QWORD *)(v214 + 56)) )
            sub_844770(&v216, a9);
        }
        else if ( v217.m128i_i8[0] )
        {
          v127 = v216.m128i_i64[0];
          for ( jj = *(_BYTE *)(v216.m128i_i64[0] + 140); jj == 12; jj = *(_BYTE *)(v127 + 140) )
            v127 = *(_QWORD *)(v127 + 160);
          if ( jj )
          {
            sub_68F8E0(v238, &v216);
            sub_8283A0((__int64)&v216, 0, a9, 0);
            sub_6E4BC0((__int64)&v216, (__int64)v238);
          }
        }
        if ( (unsigned int)sub_8319F0((__int64)&v216, &v214) )
          v211 = *(_QWORD *)(v214 + 56);
      }
      v89 = _mm_loadu_si128(&v217);
      v90 = _mm_loadu_si128(&v218);
      *(__m128i *)a10 = _mm_loadu_si128(&v216);
      v91 = _mm_loadu_si128(&v219);
      v92 = _mm_loadu_si128((const __m128i *)v220);
      v93 = _mm_loadu_si128((const __m128i *)((char *)&v221 + 4));
      v94 = _mm_loadu_si128(&v222);
      *(__m128i *)(a10 + 16) = v89;
      v95 = _mm_loadu_si128(&v223);
      v96 = _mm_loadu_si128(&v224);
      *(__m128i *)(a10 + 32) = v90;
      *(__m128i *)(a10 + 48) = v91;
      *(__m128i *)(a10 + 64) = v92;
      *(__m128i *)(a10 + 80) = v93;
      *(__m128i *)(a10 + 96) = v94;
      *(__m128i *)(a10 + 112) = v95;
      *(__m128i *)(a10 + 128) = v96;
      if ( v217.m128i_i8[0] == 2 )
      {
        v109 = _mm_loadu_si128(&v226);
        v110 = _mm_loadu_si128(&v227);
        v111 = _mm_loadu_si128(&v228);
        v112 = _mm_loadu_si128(&v229);
        *(__m128i *)(a10 + 144) = _mm_loadu_si128(&v225);
        v113 = _mm_loadu_si128(&v230);
        v114 = _mm_loadu_si128(&v231);
        *(__m128i *)(a10 + 160) = v109;
        v115 = _mm_loadu_si128(&v232);
        v116 = _mm_loadu_si128(&v233);
        *(__m128i *)(a10 + 176) = v110;
        *(__m128i *)(a10 + 192) = v111;
        v117 = _mm_loadu_si128(&v234);
        v118 = _mm_loadu_si128(&v235);
        *(__m128i *)(a10 + 208) = v112;
        v119 = _mm_loadu_si128(&v236);
        *(__m128i *)(a10 + 224) = v113;
        v120 = _mm_loadu_si128(&v237);
        *(__m128i *)(a10 + 240) = v114;
        *(__m128i *)(a10 + 256) = v115;
        *(__m128i *)(a10 + 272) = v116;
        *(__m128i *)(a10 + 288) = v117;
        *(__m128i *)(a10 + 304) = v118;
        *(__m128i *)(a10 + 320) = v119;
        *(__m128i *)(a10 + 336) = v120;
      }
      else if ( v217.m128i_i8[0] == 5 || v217.m128i_i8[0] == 1 )
      {
        *(_QWORD *)(a10 + 144) = v225.m128i_i64[0];
      }
      goto LABEL_76;
    }
    v52 = *(_BYTE *)(v14 + 40);
    *(_QWORD *)v14 = 0;
    *(_QWORD *)(v14 + 8) = 0;
    v31 = v211;
    v48 = v52 & 8;
LABEL_125:
    if ( !v48 )
    {
LABEL_126:
      *(_QWORD *)v14 = v46;
      goto LABEL_127;
    }
    if ( v31 )
      goto LABEL_239;
    if ( v200 )
      goto LABEL_280;
LABEL_402:
    v129 = sub_6EAFA0(3u);
    v211 = v129;
    v130 = sub_730690((__int64)v212);
    v31 = v211;
    *(_QWORD *)(v129 + 56) = v130;
    goto LABEL_239;
  }
  v30 = *(_QWORD *)(v51 + 56);
LABEL_63:
  v35 = v30 != 0;
  if ( v205 )
  {
    v31 = v211;
    goto LABEL_65;
  }
  if ( !v211 && v30 )
    v211 = v30;
LABEL_51:
  if ( a12 )
  {
    if ( v197 )
      a12->m128i_i32[2] = 7;
    if ( v14 )
    {
      v32 = a12->m128i_i32[2];
      *(_QWORD *)v14 = 0;
      *(_QWORD *)(v14 + 8) = 0;
      *(_BYTE *)(v14 + 41) = (2 * ((unsigned int)(v32 - 6) <= 1)) | *(_BYTE *)(v14 + 41) & 0xFD;
    }
  }
  else if ( v14 )
  {
    if ( (*(_BYTE *)(v14 + 40) & 0x20) != 0 )
    {
      v42 = qword_4D03C50;
      *(_BYTE *)(qword_4D03C50 + 18LL) = (v184 << 7) | *(_BYTE *)(qword_4D03C50 + 18LL) & 0x7F;
      if ( (*(_BYTE *)(v42 + 19) & 1) != 0 )
        *(_BYTE *)(v14 + 41) |= 2u;
      *(_BYTE *)(v42 + 19) = v183 | *(_BYTE *)(v42 + 19) & 0xFE;
    }
  }
  else if ( v197 && qword_4D03C50 && *(char *)(qword_4D03C50 + 18LL) < 0 )
  {
    sub_6E50A0();
  }
  *(_QWORD *)a1 = v196;
  return v196;
}
