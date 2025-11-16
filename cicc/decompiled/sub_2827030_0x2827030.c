// Function: sub_2827030
// Address: 0x2827030
//
__int64 __fastcall sub_2827030(__int64 *a1)
{
  __int64 v1; // r13
  __int64 v2; // rdx
  unsigned int v3; // r15d
  __int64 *v5; // rbx
  __int64 i; // rdi
  __int64 v7; // r12
  __int64 v8; // r15
  unsigned __int64 v9; // rax
  unsigned __int8 *v10; // rcx
  __int64 v11; // r14
  __int64 v12; // rcx
  __int64 *v13; // rdx
  __int64 v14; // r14
  char v15; // al
  __int64 v16; // rdx
  unsigned __int8 v17; // al
  int v18; // r14d
  __int64 v19; // rax
  unsigned __int64 v20; // rsi
  int v21; // eax
  __int64 v22; // r12
  __int64 v23; // rsi
  __int64 v24; // rsi
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // r12
  unsigned int *v28; // rax
  int v29; // ecx
  unsigned int *v30; // rdx
  __int64 v31; // rax
  int v32; // edx
  bool v33; // r12
  _QWORD *v34; // rdx
  const char *v35; // rax
  __int64 **v36; // r14
  __int64 v37; // rdx
  unsigned int v38; // eax
  unsigned __int8 *v39; // r14
  __int64 v40; // rsi
  const char *v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rdi
  const char *v46; // rax
  unsigned __int8 *v47; // r14
  __int64 v48; // rdx
  __int64 v49; // rdi
  const char *v50; // rax
  _BYTE *v51; // rax
  __int64 v52; // rdx
  __int64 v53; // rdi
  const char *v54; // rax
  __int64 v55; // r12
  __int64 v56; // rax
  char v57; // dh
  __int16 v58; // cx
  __int64 v59; // r8
  char v60; // al
  const char *v61; // rax
  __int64 v62; // rdx
  _BYTE *v63; // rax
  __int64 v64; // rax
  __int64 v65; // rdx
  __int64 v66; // rdi
  const char *v67; // rax
  unsigned __int8 *v68; // r14
  unsigned __int64 v69; // rax
  int v70; // edx
  __int64 v71; // rsi
  __int64 v72; // rax
  _QWORD *v73; // rax
  __int64 v74; // r13
  __int64 v75; // r14
  unsigned int *v76; // r14
  unsigned int *v77; // rbx
  __int64 v78; // rdx
  unsigned int v79; // esi
  unsigned __int64 v80; // rax
  int v81; // edx
  _QWORD *v82; // rdi
  _QWORD *v83; // rax
  __int64 v84; // r13
  int v85; // eax
  int v86; // eax
  unsigned int v87; // edx
  __int64 v88; // rax
  __int64 v89; // rdx
  __int64 v90; // rdx
  int v91; // eax
  int v92; // eax
  unsigned int v93; // edx
  __int64 v94; // rax
  __int64 v95; // rdx
  __int64 v96; // rdx
  __int64 v97; // rdx
  int v98; // ecx
  int v99; // eax
  _QWORD *v100; // rdi
  __int64 *v101; // rax
  __int64 v102; // rsi
  __int64 v103; // rdx
  unsigned int *v104; // r12
  unsigned int *v105; // rbx
  __int64 v106; // rdx
  unsigned int v107; // esi
  __int64 v108; // rdx
  unsigned int *v109; // r12
  unsigned int *v110; // rbx
  __int64 v111; // rdx
  unsigned int v112; // esi
  __int64 v113; // rdx
  unsigned int *v114; // r12
  unsigned int *v115; // rbx
  __int64 v116; // rdx
  unsigned int v117; // esi
  __int64 v118; // rdx
  unsigned int *v119; // r14
  unsigned int *v120; // rbx
  __int64 v121; // rdx
  unsigned int v122; // esi
  unsigned __int64 v123; // rsi
  bool v124; // al
  _BYTE *v125; // rcx
  _BYTE *v126; // r9
  char v127; // al
  __int64 v128; // r9
  unsigned __int8 *v129; // rax
  __int64 v130; // rax
  bool v131; // al
  __int64 v132; // r9
  bool v133; // al
  __int64 *v134; // rax
  _BYTE *v135; // rcx
  __int64 v136; // rsi
  __int64 v137; // rdx
  _BYTE *v138; // rax
  char v139; // cl
  unsigned int v140; // r14d
  unsigned int v141; // ebx
  __int64 v142; // rax
  __int64 *v143; // rax
  int v144; // [rsp+8h] [rbp-5E8h]
  unsigned __int8 *v145; // [rsp+10h] [rbp-5E0h]
  __int64 *v146; // [rsp+18h] [rbp-5D8h]
  __int64 v147; // [rsp+20h] [rbp-5D0h]
  __int64 *v148; // [rsp+20h] [rbp-5D0h]
  __int64 *v149; // [rsp+20h] [rbp-5D0h]
  __int64 *v150; // [rsp+20h] [rbp-5D0h]
  __int64 v151; // [rsp+28h] [rbp-5C8h]
  int v152; // [rsp+30h] [rbp-5C0h]
  unsigned __int8 *v153; // [rsp+30h] [rbp-5C0h]
  __int64 v154; // [rsp+40h] [rbp-5B0h]
  _BYTE *v155; // [rsp+60h] [rbp-590h]
  __int64 v156; // [rsp+60h] [rbp-590h]
  _BYTE *v157; // [rsp+60h] [rbp-590h]
  __int64 v158; // [rsp+60h] [rbp-590h]
  __int64 v159; // [rsp+60h] [rbp-590h]
  __int64 v160; // [rsp+68h] [rbp-588h]
  _BOOL4 v161; // [rsp+68h] [rbp-588h]
  __int64 v162; // [rsp+68h] [rbp-588h]
  unsigned __int8 *v163; // [rsp+68h] [rbp-588h]
  __int64 v164; // [rsp+68h] [rbp-588h]
  __int64 *v165; // [rsp+68h] [rbp-588h]
  bool v166; // [rsp+68h] [rbp-588h]
  __int64 v167; // [rsp+70h] [rbp-580h]
  char v168; // [rsp+70h] [rbp-580h]
  unsigned __int8 *v169; // [rsp+70h] [rbp-580h]
  __int64 v170; // [rsp+70h] [rbp-580h]
  __int64 v171; // [rsp+78h] [rbp-578h]
  __int64 v172; // [rsp+78h] [rbp-578h]
  _BYTE *v173; // [rsp+78h] [rbp-578h]
  unsigned __int8 *v174; // [rsp+78h] [rbp-578h]
  __int64 v175; // [rsp+78h] [rbp-578h]
  bool v176; // [rsp+78h] [rbp-578h]
  char v177; // [rsp+78h] [rbp-578h]
  int v178; // [rsp+78h] [rbp-578h]
  __int64 v179; // [rsp+80h] [rbp-570h]
  __int64 *v180; // [rsp+80h] [rbp-570h]
  _BYTE *v181; // [rsp+88h] [rbp-568h]
  __int64 *v182; // [rsp+88h] [rbp-568h]
  int v183; // [rsp+88h] [rbp-568h]
  __int64 v184; // [rsp+88h] [rbp-568h]
  _BYTE *v185; // [rsp+88h] [rbp-568h]
  __int64 v186; // [rsp+88h] [rbp-568h]
  int v187; // [rsp+88h] [rbp-568h]
  unsigned __int8 *v188; // [rsp+90h] [rbp-560h]
  __int64 *v189; // [rsp+98h] [rbp-558h]
  __int64 *v190; // [rsp+98h] [rbp-558h]
  __int64 **v191; // [rsp+A8h] [rbp-548h] BYREF
  __int64 **v192; // [rsp+B0h] [rbp-540h] BYREF
  __int64 v193; // [rsp+B8h] [rbp-538h]
  __int64 v194; // [rsp+C0h] [rbp-530h] BYREF
  _BYTE *v195; // [rsp+C8h] [rbp-528h]
  const char *v196; // [rsp+D0h] [rbp-520h]
  __int16 v197; // [rsp+E0h] [rbp-510h]
  const char *v198; // [rsp+F0h] [rbp-500h] BYREF
  __int64 v199; // [rsp+F8h] [rbp-4F8h]
  char *v200; // [rsp+100h] [rbp-4F0h]
  __int16 v201; // [rsp+110h] [rbp-4E0h]
  unsigned int *v202; // [rsp+120h] [rbp-4D0h] BYREF
  __int64 v203; // [rsp+128h] [rbp-4C8h]
  _BYTE v204[32]; // [rsp+130h] [rbp-4C0h] BYREF
  __int64 v205; // [rsp+150h] [rbp-4A0h]
  __int64 v206; // [rsp+158h] [rbp-498h]
  __int64 v207; // [rsp+160h] [rbp-490h]
  __int64 *v208; // [rsp+168h] [rbp-488h]
  void **v209; // [rsp+170h] [rbp-480h]
  void **v210; // [rsp+178h] [rbp-478h]
  __int64 v211; // [rsp+180h] [rbp-470h]
  int v212; // [rsp+188h] [rbp-468h]
  __int16 v213; // [rsp+18Ch] [rbp-464h]
  char v214; // [rsp+18Eh] [rbp-462h]
  __int64 v215; // [rsp+190h] [rbp-460h]
  __int64 v216; // [rsp+198h] [rbp-458h]
  void *v217; // [rsp+1A0h] [rbp-450h] BYREF
  void *v218; // [rsp+1A8h] [rbp-448h] BYREF
  _BYTE v219[24]; // [rsp+1B0h] [rbp-440h] BYREF
  char *v220; // [rsp+1C8h] [rbp-428h]
  char v221; // [rsp+1D8h] [rbp-418h] BYREF
  char *v222; // [rsp+1F8h] [rbp-3F8h]
  char v223; // [rsp+208h] [rbp-3E8h] BYREF
  __int64 v224; // [rsp+250h] [rbp-3A0h] BYREF
  _QWORD *v225[3]; // [rsp+258h] [rbp-398h] BYREF
  __int16 v226; // [rsp+270h] [rbp-380h]
  char v227[408]; // [rsp+458h] [rbp-198h] BYREF

  v1 = *a1;
  v2 = *(_QWORD *)(*a1 + 32);
  if ( (unsigned int)((*(_QWORD *)(*a1 + 40) - v2) >> 3) != 1 )
    return 0;
  v5 = a1;
  v189 = (__int64 *)a1[4];
  for ( i = *(_QWORD *)(*(_QWORD *)v2 + 16LL); i; i = *(_QWORD *)(i + 8) )
  {
    if ( (unsigned __int8)(**(_BYTE **)(i + 24) - 30) <= 0xAu )
      break;
  }
  if ( (unsigned int)sub_281E010(i, 0, v1) != 1 )
    return 0;
  v7 = **(_QWORD **)(v1 + 32);
  v8 = sub_D4B130(v1);
  v9 = *(_QWORD *)(v7 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v9 == v7 + 48 )
    goto LABEL_218;
  if ( !v9 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v9 - 24) - 30 > 0xA )
LABEL_218:
    BUG();
  if ( *(_BYTE *)(v9 - 24) != 31 )
    return 0;
  if ( (*(_DWORD *)(v9 - 20) & 0x7FFFFFF) != 3 )
    return 0;
  v10 = *(unsigned __int8 **)(v9 - 120);
  v145 = v10;
  if ( *v10 <= 0x1Cu )
    return 0;
  v160 = *(_QWORD *)(v9 - 56);
  if ( !v160 )
    return 0;
  v167 = *(_QWORD *)(v9 - 88);
  if ( !v167 )
    return 0;
  if ( *v10 != 82 )
    return 0;
  v188 = (unsigned __int8 *)*((_QWORD *)v10 - 8);
  if ( *v188 <= 0x1Cu )
    return 0;
  v11 = *((_QWORD *)v10 - 4);
  if ( *(_BYTE *)v11 > 0x15u )
    return 0;
  if ( !sub_AC30F0(*((_QWORD *)v10 - 4)) )
  {
    if ( *(_BYTE *)v11 == 17 )
    {
      if ( *(_DWORD *)(v11 + 32) <= 0x40u )
      {
        v124 = *(_QWORD *)(v11 + 24) == 0;
      }
      else
      {
        v183 = *(_DWORD *)(v11 + 32);
        v124 = v183 == (unsigned int)sub_C444A0(v11 + 24);
      }
    }
    else
    {
      v137 = *(_QWORD *)(v11 + 8);
      v186 = v137;
      if ( (unsigned int)*(unsigned __int8 *)(v137 + 8) - 17 > 1 )
        return 0;
      v138 = sub_AD7630(v11, 0, v137);
      v139 = 0;
      if ( !v138 || *v138 != 17 )
      {
        if ( *(_BYTE *)(v186 + 8) != 17 )
          return 0;
        v180 = v5;
        v141 = 0;
        v187 = *(_DWORD *)(v186 + 32);
        while ( v187 != v141 )
        {
          v177 = v139;
          v142 = sub_AD69F0((unsigned __int8 *)v11, v141);
          if ( !v142 )
            return 0;
          v139 = v177;
          if ( *(_BYTE *)v142 != 13 )
          {
            if ( *(_BYTE *)v142 != 17 )
              return 0;
            if ( *(_DWORD *)(v142 + 32) <= 0x40u )
            {
              if ( *(_QWORD *)(v142 + 24) )
                return 0;
            }
            else
            {
              v178 = *(_DWORD *)(v142 + 32);
              if ( v178 != (unsigned int)sub_C444A0(v142 + 24) )
                return 0;
            }
            v139 = 1;
          }
          ++v141;
        }
        v5 = v180;
        if ( !v139 )
          return 0;
        goto LABEL_19;
      }
      v140 = *((_DWORD *)v138 + 8);
      if ( v140 <= 0x40 )
        v124 = *((_QWORD *)v138 + 3) == 0;
      else
        v124 = v140 == (unsigned int)sub_C444A0((__int64)(v138 + 24));
    }
    if ( v124 )
      goto LABEL_19;
    return 0;
  }
LABEL_19:
  v144 = sub_B53900((__int64)v145);
  if ( (unsigned int)(v144 - 32) > 1 || (unsigned int)*v188 - 54 > 2 )
    return 0;
  if ( (v188[7] & 0x40) != 0 )
  {
    v13 = (__int64 *)*((_QWORD *)v188 - 1);
  }
  else
  {
    v12 = (__int64)&v188[-32 * (*((_DWORD *)v188 + 1) & 0x7FFFFFF)];
    v13 = (__int64 *)v12;
  }
  v14 = *v13;
  v171 = *v13;
  v15 = sub_D48480(v1, *v13, (__int64)v13, v12);
  if ( !v14 || !v15 )
    return 0;
  v16 = (v188[7] & 0x40) != 0 ? *((_QWORD *)v188 - 1) : (__int64)&v188[-32 * (*((_DWORD *)v188 + 1) & 0x7FFFFFF)];
  v179 = *(_QWORD *)(v16 + 32);
  v17 = *(_BYTE *)v179;
  if ( *(_BYTE *)v179 <= 0x1Cu )
    return 0;
  v18 = 2 * (*v188 == 54) + 65;
  if ( v17 == 42 )
  {
    v125 = *(_BYTE **)(v179 - 64);
    v126 = *(_BYTE **)(v179 - 32);
    v184 = (__int64)v125;
    if ( *v125 <= 0x1Cu )
    {
      if ( *v126 <= 0x1Cu )
        goto LABEL_30;
    }
    else
    {
      if ( v126 )
      {
        v156 = *(_QWORD *)(v179 - 32);
        v127 = sub_D48480(v1, (__int64)v126, v16, (__int64)v125);
        v128 = v156;
        if ( v127 )
          goto LABEL_175;
      }
      else
      {
        sub_D48480(v1, 0, v16, (__int64)v125);
      }
      v129 = (unsigned __int8 *)v179;
      v126 = *(_BYTE **)(v179 - 32);
      if ( *v126 <= 0x1Cu )
      {
LABEL_178:
        v17 = *v129;
        goto LABEL_29;
      }
      v184 = *(_QWORD *)(v179 - 64);
      if ( !v184 )
      {
        sub_D48480(v1, 0, v16, (__int64)v125);
        v17 = *(_BYTE *)v179;
        goto LABEL_29;
      }
    }
    v157 = v126;
    if ( !(unsigned __int8)sub_D48480(v1, v184, v16, (__int64)v125) )
    {
LABEL_177:
      v129 = (unsigned __int8 *)v179;
      goto LABEL_178;
    }
    v130 = v184;
    v184 = (__int64)v157;
    v128 = v130;
LABEL_175:
    v158 = v128;
    v131 = sub_B44900(v179);
    v132 = v158;
    if ( v131 || (v133 = sub_B448F0(v179), v132 = v158, v133) )
    {
      v134 = sub_DD8400((__int64)v189, v132);
      v146 = sub_DCAF50(v189, (__int64)v134, 0);
      v179 = v184;
      goto LABEL_31;
    }
    goto LABEL_177;
  }
LABEL_29:
  if ( v17 != 44 )
    goto LABEL_30;
  v135 = *(_BYTE **)(v179 - 64);
  v185 = v135;
  if ( *v135 <= 0x1Cu )
    goto LABEL_30;
  v136 = *(_QWORD *)(v179 - 32);
  if ( !v136 )
  {
    sub_D48480(v1, 0, v16, (__int64)v135);
LABEL_30:
    v146 = sub_DA2C50((__int64)v189, *(_QWORD *)(v179 + 8), 0, 0);
    goto LABEL_31;
  }
  v159 = *(_QWORD *)(v179 - 32);
  if ( !(unsigned __int8)sub_D48480(v1, v136, v16, (__int64)v135) || !sub_B44900(v179) )
    goto LABEL_30;
  v146 = sub_DD8400((__int64)v189, v159);
  v179 = (__int64)v185;
LABEL_31:
  if ( *(_BYTE *)v179 != 84 )
    return 0;
  if ( v7 != *(_QWORD *)(v179 + 40) )
    return 0;
  v181 = (_BYTE *)sub_F0A930(v179, v7);
  if ( *v181 <= 0x1Cu )
    return 0;
  v224 = v179;
  v225[0] = 0;
  v155 = (_BYTE *)sub_F0A930(v179, v8);
  if ( *v181 != 42 )
    return 0;
  v19 = *((_QWORD *)v181 - 8);
  if ( v19 != v179 )
    return 0;
  if ( !v19 )
    return 0;
  v3 = sub_993A50(v225, *((_QWORD *)v181 - 4));
  if ( !(_BYTE)v3 )
    return 0;
  if ( v144 != 32 )
  {
    sub_B52870(33);
    v167 = v160;
  }
  if ( v7 != v167 )
    return 0;
  if ( *v188 == 56 && !(unsigned __int8)sub_D4A4C0(v1) )
  {
    v143 = sub_DD8400((__int64)v189, v171);
    if ( !(unsigned __int8)sub_DBED40((__int64)v189, (__int64)v143) )
      return 0;
  }
  v182 = **(__int64 ***)(*v5 + 32);
  v154 = sub_D4B130(*v5);
  v151 = sub_D47470(*v5);
  v20 = *(_QWORD *)(v154 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v20 == v154 + 48 )
  {
    v22 = 0;
  }
  else
  {
    if ( !v20 )
      BUG();
    v21 = *(unsigned __int8 *)(v20 - 24);
    v22 = 0;
    v23 = v20 - 24;
    if ( (unsigned int)(v21 - 30) < 0xB )
      v22 = v23;
  }
  v208 = (__int64 *)sub_BD5C60(v22);
  v209 = &v217;
  v210 = &v218;
  v202 = (unsigned int *)v204;
  v217 = &unk_49DA100;
  v203 = 0x200000000LL;
  v213 = 512;
  v218 = &unk_49DA0B0;
  v211 = 0;
  v212 = 0;
  v214 = 7;
  v215 = 0;
  v216 = 0;
  v205 = 0;
  v206 = 0;
  LOWORD(v207) = 0;
  sub_D5F1F0((__int64)&v202, v22);
  v24 = *(_QWORD *)(v179 + 48);
  v224 = v24;
  if ( !v24 || (sub_B96E90((__int64)&v224, v24, 1), (v27 = v224) == 0) )
  {
    sub_93FB40((__int64)&v202, 0);
    v27 = v224;
    goto LABEL_123;
  }
  v28 = v202;
  v29 = v203;
  v30 = &v202[4 * (unsigned int)v203];
  if ( v202 == v30 )
  {
LABEL_125:
    if ( (unsigned int)v203 >= (unsigned __int64)HIDWORD(v203) )
    {
      v123 = (unsigned int)v203 + 1LL;
      if ( HIDWORD(v203) < v123 )
      {
        sub_C8D5F0((__int64)&v202, v204, v123, 0x10u, v25, v26);
        v30 = &v202[4 * (unsigned int)v203];
      }
      *(_QWORD *)v30 = 0;
      *((_QWORD *)v30 + 1) = v27;
      v27 = v224;
      LODWORD(v203) = v203 + 1;
    }
    else
    {
      if ( v30 )
      {
        *v30 = 0;
        *((_QWORD *)v30 + 1) = v27;
        v29 = v203;
        v27 = v224;
      }
      LODWORD(v203) = v29 + 1;
    }
LABEL_123:
    if ( !v27 )
      goto LABEL_54;
    goto LABEL_53;
  }
  while ( 1 )
  {
    v26 = *v28;
    if ( !(_DWORD)v26 )
      break;
    v28 += 4;
    if ( v30 == v28 )
      goto LABEL_125;
  }
  *((_QWORD *)v28 + 1) = v224;
LABEL_53:
  sub_B91220((__int64)&v224, v27);
LABEL_54:
  v191 = *(__int64 ***)(v171 + 8);
  v152 = sub_BCB060((__int64)v191);
  v224 = sub_ACADE0(v191);
  v225[0] = (_QWORD *)sub_ACD720(v208);
  sub_DF8D10((__int64)v219, v18, (__int64)v191, (char *)&v224, 2);
  v31 = sub_DFD690(v5[6], (__int64)v219);
  if ( v32 )
    v33 = v32 > 0;
  else
    v33 = v31 > 1;
  if ( v33 )
  {
    v3 = 0;
  }
  else
  {
    v161 = 0;
    if ( !*((_WORD *)v146 + 12) )
    {
      v33 = sub_D968A0((__int64)v146);
      v161 = v33;
    }
    BYTE4(v194) = 0;
    v224 = (__int64)sub_BD5D20(v171);
    v226 = 773;
    v225[0] = v34;
    v225[1] = ".numleadingzeros";
    v198 = (const char *)v171;
    v199 = sub_ACD720(v208);
    v147 = sub_B33D10((__int64)&v202, v18, (__int64)&v191, 1, (int)&v198, 2, v194, (__int64)&v224);
    v168 = v152 != 2;
    v35 = sub_BD5D20(v171);
    v36 = v191;
    v199 = v37;
    v201 = 773;
    v200 = ".numactivebits";
    v198 = v35;
    v38 = sub_BCB060((__int64)v191);
    v172 = sub_AD64C0((__int64)v36, v38, 0);
    v39 = (unsigned __int8 *)(*((__int64 (__fastcall **)(void **, __int64, __int64, __int64, __int64, bool))*v209 + 4))(
                               v209,
                               15,
                               v172,
                               v147,
                               1,
                               v152 != 2);
    if ( !v39 )
    {
      v226 = 257;
      v39 = (unsigned __int8 *)sub_B504D0(15, v172, v147, (__int64)&v224, 0, 0);
      (*((void (__fastcall **)(void **, unsigned __int8 *, const char **, __int64, __int64))*v210 + 2))(
        v210,
        v39,
        &v198,
        v206,
        v207);
      v108 = 4LL * (unsigned int)v203;
      if ( v202 != &v202[v108] )
      {
        v176 = v33;
        v109 = &v202[v108];
        v148 = v5;
        v110 = v202;
        do
        {
          v111 = *((_QWORD *)v110 + 1);
          v112 = *v110;
          v110 += 4;
          sub_B99FD0((__int64)v39, v112, v111);
        }
        while ( v109 != v110 );
        v33 = v176;
        v5 = v148;
      }
      sub_B447F0(v39, 1);
      if ( v152 != 2 )
        sub_B44850(v39, 1);
    }
    sub_27C1C30((__int64)&v224, (__int64 *)v5[4], v5[7], (__int64)"loop-idiom", 1);
    v40 = v206;
    if ( v206 )
      v40 = v206 - 24;
    sub_D5F1F0((__int64)v227, v40);
    v173 = sub_F8DB50(&v224, (__int64)v146, 0);
    v41 = sub_BD5D20((__int64)v39);
    v201 = 773;
    v198 = v41;
    v199 = v42;
    v200 = ".offset";
    v194 = sub_929C50(&v202, v39, v173, (__int64)&v198, v161, 1);
    BYTE4(v193) = 0;
    v195 = v155;
    v198 = "iv.final";
    v201 = 259;
    v192 = v191;
    v43 = sub_B33D10((__int64)&v202, 0x149u, (__int64)&v192, 1, (int)&v194, 2, v193, (__int64)&v198);
    v44 = 14;
    v174 = (unsigned __int8 *)v43;
    v45 = **(_QWORD **)(*v5 + 32);
    v46 = "<unnamed loop>";
    if ( v45 && (*(_BYTE *)(v45 + 7) & 0x10) != 0 )
      v46 = sub_BD5D20(v45);
    v194 = (__int64)v46;
    v197 = 773;
    v195 = (_BYTE *)v44;
    v196 = ".backedgetakencount";
    v47 = (unsigned __int8 *)(*((__int64 (__fastcall **)(void **, __int64, unsigned __int8 *, _BYTE *, _BOOL4, __int64))*v209
                              + 4))(
                               v209,
                               15,
                               v174,
                               v155,
                               v161,
                               1);
    if ( !v47 )
    {
      v201 = 257;
      v47 = (unsigned __int8 *)sub_B504D0(15, (__int64)v174, (__int64)v155, (__int64)&v198, 0, 0);
      (*((void (__fastcall **)(void **, unsigned __int8 *, __int64 *, __int64, __int64))*v210 + 2))(
        v210,
        v47,
        &v194,
        v206,
        v207);
      v113 = 4LL * (unsigned int)v203;
      if ( v202 != &v202[v113] )
      {
        v166 = v33;
        v114 = &v202[v113];
        v149 = v5;
        v115 = v202;
        do
        {
          v116 = *((_QWORD *)v115 + 1);
          v117 = *v115;
          v115 += 4;
          sub_B99FD0((__int64)v47, v117, v116);
        }
        while ( v114 != v115 );
        v33 = v166;
        v5 = v149;
      }
      if ( v33 )
        sub_B447F0(v47, 1);
      sub_B44850(v47, 1);
    }
    v48 = 14;
    v49 = **(_QWORD **)(*v5 + 32);
    v50 = "<unnamed loop>";
    if ( v49 && (*(_BYTE *)(v49 + 7) & 0x10) != 0 )
      v50 = sub_BD5D20(v49);
    v198 = v50;
    v199 = v48;
    v201 = 773;
    v200 = ".tripcount";
    v51 = (_BYTE *)sub_AD64C0((__int64)v191, 1, 0);
    v162 = sub_929C50(&v202, v47, v51, (__int64)&v198, 1u, v168);
    sub_BD7E80((unsigned __int8 *)v179, v174, v182);
    sub_A88F30((__int64)&v202, (__int64)v182, v182[7], 1);
    v52 = 14;
    v53 = **(_QWORD **)(*v5 + 32);
    v54 = "<unnamed loop>";
    if ( v53 && (*(_BYTE *)(v53 + 7) & 0x10) != 0 )
      v54 = sub_BD5D20(v53);
    v198 = v54;
    v199 = v52;
    v201 = 773;
    v200 = ".iv";
    v55 = sub_D5C860((__int64 *)&v202, (__int64)v191, 2, (__int64)&v198);
    v56 = sub_AA4FF0((__int64)v182);
    LOBYTE(v58) = 1;
    v59 = v56;
    v60 = 0;
    if ( v59 )
      v60 = v57;
    HIBYTE(v58) = v60;
    sub_A88F30((__int64)&v202, (__int64)v182, v59, v58);
    v61 = sub_BD5D20(v55);
    v201 = 773;
    v198 = v61;
    v199 = v62;
    v200 = ".next";
    v63 = (_BYTE *)sub_AD64C0((__int64)v191, 1, 0);
    v64 = sub_929C50(&v202, (_BYTE *)v55, v63, (__int64)&v198, 1u, v168);
    v65 = 14;
    v175 = v64;
    v66 = **(_QWORD **)(*v5 + 32);
    v67 = "<unnamed loop>";
    if ( v66 && (*(_BYTE *)(v66 + 7) & 0x10) != 0 )
      v67 = sub_BD5D20(v66);
    v194 = (__int64)v67;
    v195 = (_BYTE *)v65;
    v197 = 773;
    v196 = ".ivcheck";
    v68 = (unsigned __int8 *)(*((__int64 (__fastcall **)(void **, __int64, __int64, __int64))*v209 + 7))(
                               v209,
                               32,
                               v175,
                               v162);
    if ( !v68 )
    {
      v201 = 257;
      v68 = (unsigned __int8 *)sub_BD2C40(72, unk_3F10FD0);
      if ( v68 )
      {
        v97 = *(_QWORD *)(v175 + 8);
        v98 = *(unsigned __int8 *)(v97 + 8);
        if ( (unsigned int)(v98 - 17) > 1 )
        {
          v102 = sub_BCB2A0(*(_QWORD **)v97);
        }
        else
        {
          v99 = *(_DWORD *)(v97 + 32);
          v100 = *(_QWORD **)v97;
          BYTE4(v193) = (_BYTE)v98 == 18;
          LODWORD(v193) = v99;
          v101 = (__int64 *)sub_BCB2A0(v100);
          v102 = sub_BCE1B0(v101, v193);
        }
        sub_B523C0((__int64)v68, v102, 53, 32, v175, v162, (__int64)&v198, 0, 0, 0);
      }
      (*((void (__fastcall **)(void **, unsigned __int8 *, __int64 *, __int64, __int64))*v210 + 2))(
        v210,
        v68,
        &v194,
        v206,
        v207);
      v103 = 4LL * (unsigned int)v203;
      if ( v202 != &v202[v103] )
      {
        v170 = v55;
        v104 = &v202[v103];
        v165 = v5;
        v105 = v202;
        do
        {
          v106 = *((_QWORD *)v105 + 1);
          v107 = *v105;
          v105 += 4;
          sub_B99FD0((__int64)v68, v107, v106);
        }
        while ( v104 != v105 );
        v55 = v170;
        v5 = v165;
      }
    }
    v169 = v68;
    if ( v144 != 32 )
    {
      v197 = 257;
      v164 = sub_AD62B0(*((_QWORD *)v68 + 1));
      v169 = (unsigned __int8 *)(*((__int64 (__fastcall **)(void **, __int64, unsigned __int8 *, __int64))*v209 + 2))(
                                  v209,
                                  30,
                                  v68,
                                  v164);
      if ( !v169 )
      {
        v201 = 257;
        v169 = (unsigned __int8 *)sub_B504D0(30, (__int64)v68, v164, (__int64)&v198, 0, 0);
        (*((void (__fastcall **)(void **, unsigned __int8 *, __int64 *, __int64, __int64))*v210 + 2))(
          v210,
          v169,
          &v194,
          v206,
          v207);
        v118 = 4LL * (unsigned int)v203;
        if ( v202 != &v202[v118] )
        {
          v153 = v68;
          v119 = &v202[v118];
          v150 = v5;
          v120 = v202;
          do
          {
            v121 = *((_QWORD *)v120 + 1);
            v122 = *v120;
            v120 += 4;
            sub_B99FD0((__int64)v169, v122, v121);
          }
          while ( v119 != v120 );
          v68 = v153;
          v5 = v150;
        }
      }
      sub_BD6B90(v169, v145);
    }
    v201 = 257;
    v163 = (unsigned __int8 *)sub_929C50(&v202, (_BYTE *)v55, v155, (__int64)&v198, 0, 1);
    sub_BD6B90(v163, (unsigned __int8 *)v179);
    v69 = v182[6] & 0xFFFFFFFFFFFFFFF8LL;
    if ( v182 + 6 == (__int64 *)v69 )
    {
      v71 = 0;
    }
    else
    {
      if ( !v69 )
        BUG();
      v70 = *(unsigned __int8 *)(v69 - 24);
      v71 = 0;
      v72 = v69 - 24;
      if ( (unsigned int)(v70 - 30) < 0xB )
        v71 = v72;
    }
    sub_D5F1F0((__int64)&v202, v71);
    v201 = 257;
    v73 = sub_BD2C40(72, 3u);
    v74 = (__int64)v73;
    if ( v73 )
      sub_B4C9A0((__int64)v73, v151, (__int64)v182, (__int64)v68, 3u, 0, 0, 0);
    (*((void (__fastcall **)(void **, __int64, const char **, __int64, __int64))*v210 + 2))(
      v210,
      v74,
      &v198,
      v206,
      v207);
    v75 = 4LL * (unsigned int)v203;
    if ( v202 != &v202[v75] )
    {
      v190 = v5;
      v76 = &v202[v75];
      v77 = v202;
      do
      {
        v78 = *((_QWORD *)v77 + 1);
        v79 = *v77;
        v77 += 4;
        sub_B99FD0(v74, v79, v78);
      }
      while ( v76 != v77 );
      v5 = v190;
    }
    v80 = v182[6] & 0xFFFFFFFFFFFFFFF8LL;
    if ( (__int64 *)v80 == v182 + 6 )
    {
      v82 = 0;
    }
    else
    {
      if ( !v80 )
        BUG();
      v81 = *(unsigned __int8 *)(v80 - 24);
      v82 = 0;
      v83 = (_QWORD *)(v80 - 24);
      if ( (unsigned int)(v81 - 30) < 0xB )
        v82 = v83;
    }
    sub_B43D60(v82);
    v84 = sub_AD64C0((__int64)v191, 0, 0);
    v85 = *(_DWORD *)(v55 + 4) & 0x7FFFFFF;
    if ( v85 == *(_DWORD *)(v55 + 72) )
    {
      sub_B48D90(v55);
      v85 = *(_DWORD *)(v55 + 4) & 0x7FFFFFF;
    }
    v86 = (v85 + 1) & 0x7FFFFFF;
    v87 = v86 | *(_DWORD *)(v55 + 4) & 0xF8000000;
    v88 = *(_QWORD *)(v55 - 8) + 32LL * (unsigned int)(v86 - 1);
    *(_DWORD *)(v55 + 4) = v87;
    if ( *(_QWORD *)v88 )
    {
      v89 = *(_QWORD *)(v88 + 8);
      **(_QWORD **)(v88 + 16) = v89;
      if ( v89 )
        *(_QWORD *)(v89 + 16) = *(_QWORD *)(v88 + 16);
    }
    *(_QWORD *)v88 = v84;
    if ( v84 )
    {
      v90 = *(_QWORD *)(v84 + 16);
      *(_QWORD *)(v88 + 8) = v90;
      if ( v90 )
        *(_QWORD *)(v90 + 16) = v88 + 8;
      *(_QWORD *)(v88 + 16) = v84 + 16;
      *(_QWORD *)(v84 + 16) = v88;
    }
    *(_QWORD *)(*(_QWORD *)(v55 - 8)
              + 32LL * *(unsigned int *)(v55 + 72)
              + 8LL * ((*(_DWORD *)(v55 + 4) & 0x7FFFFFFu) - 1)) = v154;
    v91 = *(_DWORD *)(v55 + 4) & 0x7FFFFFF;
    if ( v91 == *(_DWORD *)(v55 + 72) )
    {
      sub_B48D90(v55);
      v91 = *(_DWORD *)(v55 + 4) & 0x7FFFFFF;
    }
    v92 = (v91 + 1) & 0x7FFFFFF;
    v93 = v92 | *(_DWORD *)(v55 + 4) & 0xF8000000;
    v94 = *(_QWORD *)(v55 - 8) + 32LL * (unsigned int)(v92 - 1);
    *(_DWORD *)(v55 + 4) = v93;
    if ( *(_QWORD *)v94 )
    {
      v95 = *(_QWORD *)(v94 + 8);
      **(_QWORD **)(v94 + 16) = v95;
      if ( v95 )
        *(_QWORD *)(v95 + 16) = *(_QWORD *)(v94 + 16);
    }
    *(_QWORD *)v94 = v175;
    if ( v175 )
    {
      v96 = *(_QWORD *)(v175 + 16);
      *(_QWORD *)(v94 + 8) = v96;
      if ( v96 )
        *(_QWORD *)(v96 + 16) = v94 + 8;
      *(_QWORD *)(v94 + 16) = v175 + 16;
      *(_QWORD *)(v175 + 16) = v94;
    }
    *(_QWORD *)(*(_QWORD *)(v55 - 8)
              + 32LL * *(unsigned int *)(v55 + 72)
              + 8LL * ((*(_DWORD *)(v55 + 4) & 0x7FFFFFFu) - 1)) = v182;
    sub_DAC210(v5[4], *v5);
    sub_BD84D0(v179, (__int64)v163);
    sub_B43D60((_QWORD *)v179);
    sub_BD84D0((__int64)v145, (__int64)v169);
    sub_B43D60(v145);
    sub_27C20B0((__int64)&v224);
  }
  if ( v222 != &v223 )
    _libc_free((unsigned __int64)v222);
  if ( v220 != &v221 )
    _libc_free((unsigned __int64)v220);
  nullsub_61();
  v217 = &unk_49DA100;
  nullsub_63();
  if ( v202 != (unsigned int *)v204 )
    _libc_free((unsigned __int64)v202);
  return v3;
}
