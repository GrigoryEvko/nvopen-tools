// Function: sub_1A57210
// Address: 0x1a57210
//
void __fastcall sub_1A57210(__int64 a1, char **a2, unsigned __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char **v7; // rbx
  __int64 v8; // rax
  __int64 v9; // r14
  int v10; // r8d
  int v11; // r9d
  unsigned __int64 *v12; // rax
  char *v13; // r15
  char v14; // al
  __int64 *v15; // rdx
  __int64 v16; // rcx
  unsigned int v17; // eax
  unsigned int v18; // esi
  __int64 v19; // rax
  unsigned __int64 *v20; // rax
  __int64 *v21; // r15
  __int64 *v22; // r10
  int v23; // edx
  unsigned int v24; // esi
  char *v25; // rdi
  __int64 v26; // r8
  __int64 v27; // rax
  unsigned int v28; // esi
  unsigned int v29; // edx
  unsigned int v30; // r8d
  __int64 v31; // r12
  _QWORD *v32; // rax
  __int64 v33; // rdx
  unsigned int i; // r12d
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // r13
  _QWORD *v38; // rax
  char *v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rcx
  int v42; // r8d
  int v43; // r9d
  __int64 *v44; // r8
  __int64 v45; // r9
  __int64 *v46; // r13
  __int64 v47; // r12
  char *v48; // r15
  __int64 v49; // rax
  char *v50; // r12
  _BYTE *v51; // rsi
  char *v52; // rsi
  _QWORD *v53; // rax
  int v54; // esi
  __int64 *v55; // rcx
  int v56; // r8d
  unsigned int v57; // eax
  __int64 v58; // rdi
  __int64 v59; // rdx
  __int64 *v60; // r12
  __int64 v61; // r15
  __int64 *v62; // r13
  __int64 *v63; // r12
  __int64 v64; // r15
  unsigned int v65; // r13d
  __int64 v66; // rbx
  int v67; // ecx
  _QWORD *v68; // rsi
  unsigned int v69; // edx
  __int64 *v70; // rax
  __int64 v71; // rdi
  __int64 v72; // r13
  __int64 v73; // rax
  unsigned int v74; // r12d
  __int64 v75; // rdx
  __int64 v76; // rax
  __int64 v77; // rbx
  _QWORD *v78; // rax
  __int64 **v79; // r12
  void **v80; // rdi
  __int64 **v81; // rbx
  __int64 (__fastcall *v82)(__int64); // rax
  __int64 *v83; // rax
  __int64 *v84; // rdi
  _QWORD *v85; // rcx
  int v86; // edx
  __int64 v87; // rsi
  unsigned int v88; // edi
  __int64 *v89; // rax
  __int64 v90; // r8
  __int64 v91; // rdi
  __int64 **v92; // r12
  __int64 (__fastcall *v93)(__int64); // rax
  void **v94; // rdi
  __int64 **v95; // rbx
  __int64 *v96; // rdi
  __int64 *v97; // rbx
  _QWORD *v98; // rdi
  int v99; // esi
  unsigned int v100; // ecx
  __int64 *v101; // rax
  __int64 v102; // r9
  __int64 v103; // rsi
  int v104; // r8d
  int v105; // r9d
  __int64 v106; // r12
  __int64 v107; // rax
  __int64 v108; // r12
  __int64 v109; // r15
  _QWORD *v110; // rax
  __int64 v111; // r12
  _BYTE *v112; // rdx
  unsigned int v113; // esi
  _QWORD *v114; // rdi
  int v115; // esi
  unsigned int v116; // ecx
  __int64 *v117; // rdx
  __int64 v118; // rax
  unsigned int v119; // edx
  unsigned int v120; // ecx
  unsigned int v121; // eax
  __int64 v122; // rcx
  int v123; // r10d
  int v124; // eax
  int v125; // eax
  int v126; // r9d
  int v127; // r11d
  char *v128; // r9
  __int64 *v129; // r12
  size_t v130; // rdx
  __int64 v131; // rbx
  unsigned __int64 v132; // rax
  __int64 *v133; // r13
  __int64 *v134; // r14
  __int64 v135; // rbx
  __int64 v136; // rdx
  __int64 *v137; // r12
  int v138; // eax
  int v139; // edx
  __int64 v142; // [rsp+40h] [rbp-560h]
  __int64 v143; // [rsp+40h] [rbp-560h]
  __int64 v144; // [rsp+50h] [rbp-550h]
  _QWORD *v145; // [rsp+50h] [rbp-550h]
  __int64 v147; // [rsp+60h] [rbp-540h]
  __int64 *v148; // [rsp+60h] [rbp-540h]
  __int64 *j; // [rsp+60h] [rbp-540h]
  __int64 *v150; // [rsp+60h] [rbp-540h]
  __int64 *v151; // [rsp+60h] [rbp-540h]
  char *v152; // [rsp+68h] [rbp-538h]
  __int64 *k; // [rsp+68h] [rbp-538h]
  char **v155; // [rsp+78h] [rbp-528h]
  __int64 *v156; // [rsp+78h] [rbp-528h]
  char *v157; // [rsp+80h] [rbp-520h] BYREF
  __int64 v158; // [rsp+88h] [rbp-518h] BYREF
  void *src; // [rsp+90h] [rbp-510h] BYREF
  __int64 v160; // [rsp+98h] [rbp-508h]
  _BYTE v161[32]; // [rsp+A0h] [rbp-500h] BYREF
  __int64 *v162; // [rsp+C0h] [rbp-4E0h] BYREF
  __int64 v163; // [rsp+C8h] [rbp-4D8h]
  _BYTE v164[32]; // [rsp+D0h] [rbp-4D0h] BYREF
  __int64 *v165; // [rsp+F0h] [rbp-4B0h] BYREF
  char *v166; // [rsp+F8h] [rbp-4A8h]
  unsigned __int64 v167; // [rsp+100h] [rbp-4A0h]
  __int64 *v168; // [rsp+108h] [rbp-498h]
  void **v169; // [rsp+110h] [rbp-490h]
  void **p_src; // [rsp+118h] [rbp-488h]
  __int64 *v171; // [rsp+120h] [rbp-480h] BYREF
  __int64 v172; // [rsp+128h] [rbp-478h]
  __int64 (__fastcall *v173)(__int64); // [rsp+130h] [rbp-470h]
  __int64 v174; // [rsp+138h] [rbp-468h]
  __int64 (__fastcall *v175)(__int64 *); // [rsp+140h] [rbp-460h]
  __int64 v176; // [rsp+148h] [rbp-458h]
  _BYTE *v177; // [rsp+150h] [rbp-450h] BYREF
  __int64 v178; // [rsp+158h] [rbp-448h]
  _BYTE v179[128]; // [rsp+160h] [rbp-440h] BYREF
  __int64 v180; // [rsp+1E0h] [rbp-3C0h] BYREF
  _BYTE *v181; // [rsp+1E8h] [rbp-3B8h]
  _BYTE *v182; // [rsp+1F0h] [rbp-3B0h]
  __int64 v183; // [rsp+1F8h] [rbp-3A8h]
  int v184; // [rsp+200h] [rbp-3A0h]
  _BYTE v185[136]; // [rsp+208h] [rbp-398h] BYREF
  char *v186; // [rsp+290h] [rbp-310h] BYREF
  _BYTE *v187; // [rsp+298h] [rbp-308h]
  _BYTE *v188; // [rsp+2A0h] [rbp-300h]
  __int64 v189; // [rsp+2A8h] [rbp-2F8h]
  int v190; // [rsp+2B0h] [rbp-2F0h]
  _BYTE v191[136]; // [rsp+2B8h] [rbp-2E8h] BYREF
  __int64 v192; // [rsp+340h] [rbp-260h] BYREF
  __int64 v193; // [rsp+348h] [rbp-258h]
  _QWORD *v194; // [rsp+350h] [rbp-250h] BYREF
  unsigned int v195; // [rsp+358h] [rbp-248h]
  __int64 *v196; // [rsp+450h] [rbp-150h] BYREF
  __int64 v197; // [rsp+458h] [rbp-148h]
  __int64 v198; // [rsp+460h] [rbp-140h] BYREF
  unsigned int v199; // [rsp+468h] [rbp-138h]
  __int64 *v200; // [rsp+4E0h] [rbp-C0h] BYREF
  __int64 v201; // [rsp+4E8h] [rbp-B8h]
  _BYTE v202[176]; // [rsp+4F0h] [rbp-B0h] BYREF

  v7 = a2;
  v157 = 0;
  v8 = sub_13FC520(a1);
  v9 = **(_QWORD **)(a1 + 32);
  v192 = v8;
  sub_1A51850((unsigned __int64 *)&v196, a4, &v192);
  v158 = v198;
  sub_1455FA0((__int64)&v196);
  v192 = v9;
  sub_1A51850((unsigned __int64 *)&v196, a4, &v192);
  v142 = v198;
  sub_1455FA0((__int64)&v196);
  v192 = 0;
  src = v161;
  v160 = 0x400000000LL;
  v12 = (unsigned __int64 *)&v194;
  v193 = 1;
  do
  {
    *v12 = -8;
    v12 += 2;
  }
  while ( v12 != (unsigned __int64 *)&v196 );
  if ( a3 > 4 )
    sub_16CD150((__int64)&src, v161, a3, 8, v10, v11);
  v147 = 0;
  v155 = &a2[a3];
  if ( v155 != a2 )
  {
    while ( 1 )
    {
      v13 = *v7;
      v186 = *v7;
      sub_1A51850((unsigned __int64 *)&v196, a4, (__int64 *)&v186);
      v180 = v198;
      sub_1455FA0((__int64)&v196);
      if ( !v180 )
        goto LABEL_17;
      v144 = sub_13AE450(a5, (__int64)v13);
      if ( !v144 )
        goto LABEL_17;
      v14 = sub_1A542D0((__int64)&v192, &v180, &v196);
      v15 = v196;
      v16 = v144;
      if ( !v14 )
        break;
LABEL_15:
      v15[1] = v16;
      v145 = (_QWORD *)v16;
      sub_15CDD90((__int64)&src, &v180);
      if ( !v147 )
        goto LABEL_69;
      if ( v145 != (_QWORD *)v147 )
      {
        v53 = v145;
        while ( 1 )
        {
          v53 = (_QWORD *)*v53;
          if ( (_QWORD *)v147 == v53 )
            break;
          if ( !v53 )
            goto LABEL_17;
        }
LABEL_69:
        v147 = (__int64)v145;
      }
LABEL_17:
      if ( v155 == ++v7 )
        goto LABEL_18;
    }
    ++v192;
    v17 = ((unsigned int)v193 >> 1) + 1;
    if ( (v193 & 1) != 0 )
    {
      v18 = 16;
      if ( 4 * v17 < 0x30 )
      {
LABEL_11:
        if ( v18 - (v17 + HIDWORD(v193)) > v18 >> 3 )
        {
LABEL_12:
          LODWORD(v193) = v193 & 1 | (2 * v17);
          if ( *v15 != -8 )
            --HIDWORD(v193);
          v19 = v180;
          v15[1] = 0;
          *v15 = v19;
          goto LABEL_15;
        }
LABEL_206:
        sub_1A54BC0((__int64)&v192, v18);
        sub_1A542D0((__int64)&v192, &v180, &v196);
        v15 = v196;
        v16 = v144;
        v17 = ((unsigned int)v193 >> 1) + 1;
        goto LABEL_12;
      }
    }
    else
    {
      v18 = v195;
      if ( 3 * v195 > 4 * v17 )
        goto LABEL_11;
    }
    v18 *= 2;
    goto LABEL_206;
  }
LABEL_18:
  v20 = (unsigned __int64 *)&v198;
  v196 = 0;
  v197 = 1;
  do
    *v20++ = -8;
  while ( v20 != (unsigned __int64 *)&v200 );
  v200 = (__int64 *)v202;
  v201 = 0x1000000000LL;
  v21 = *(__int64 **)(a1 + 32);
  v156 = *(__int64 **)(a1 + 40);
  if ( v156 != v21 )
  {
    while ( 1 )
    {
      v180 = *v21;
      sub_1A51850((unsigned __int64 *)&v186, a4, &v180);
      v177 = v188;
      sub_1455FA0((__int64)&v186);
      v27 = (__int64)v177;
      if ( !v177 )
        goto LABEL_24;
      if ( (v197 & 1) != 0 )
      {
        v22 = &v198;
        v23 = 15;
      }
      else
      {
        v28 = v199;
        v22 = (__int64 *)v198;
        v23 = v199 - 1;
        if ( !v199 )
        {
          v29 = v197;
          v196 = (__int64 *)((char *)v196 + 1);
          v25 = 0;
          v30 = ((unsigned int)v197 >> 1) + 1;
          goto LABEL_29;
        }
      }
      v24 = v23 & (((unsigned int)v177 >> 9) ^ ((unsigned int)v177 >> 4));
      v25 = (char *)&v22[v24];
      v26 = *(_QWORD *)v25;
      if ( v177 != *(_BYTE **)v25 )
      {
        v127 = 1;
        v128 = 0;
        while ( v26 != -8 )
        {
          if ( v26 == -16 && !v128 )
            v128 = v25;
          v24 = v23 & (v127 + v24);
          v25 = (char *)&v22[v24];
          v26 = *(_QWORD *)v25;
          if ( v177 == *(_BYTE **)v25 )
            goto LABEL_24;
          ++v127;
        }
        v29 = v197;
        if ( v128 )
          v25 = v128;
        v196 = (__int64 *)((char *)v196 + 1);
        v30 = ((unsigned int)v197 >> 1) + 1;
        if ( (v197 & 1) == 0 )
        {
          v28 = v199;
LABEL_29:
          if ( 4 * v30 >= 3 * v28 )
            goto LABEL_213;
          goto LABEL_30;
        }
        v28 = 16;
        if ( 4 * v30 >= 0x30 )
        {
LABEL_213:
          v28 *= 2;
LABEL_214:
          sub_19B89E0((__int64)&v196, v28);
          sub_1A54690((__int64)&v196, (__int64 *)&v177, &v186);
          v25 = v186;
          v27 = (__int64)v177;
          v29 = v197;
          goto LABEL_31;
        }
LABEL_30:
        if ( v28 - HIDWORD(v197) - v30 <= v28 >> 3 )
          goto LABEL_214;
LABEL_31:
        LODWORD(v197) = (2 * (v29 >> 1) + 2) | v29 & 1;
        if ( *(_QWORD *)v25 != -8 )
          --HIDWORD(v197);
        *(_QWORD *)v25 = v27;
        ++v21;
        sub_15CDD90((__int64)&v200, &v177);
        if ( v156 == v21 )
          break;
      }
      else
      {
LABEL_24:
        if ( v156 == ++v21 )
          break;
      }
    }
  }
  v180 = 0;
  v177 = v179;
  v178 = 0x1000000000LL;
  v181 = v185;
  v182 = v185;
  v183 = 16;
  v184 = 0;
  v31 = *(_QWORD *)(v142 + 8);
  if ( !v31 )
    goto LABEL_64;
  while ( 1 )
  {
    v32 = sub_1648700(v31);
    if ( (unsigned __int8)(*((_BYTE *)v32 + 16) - 25) <= 9u )
      break;
    v31 = *(_QWORD *)(v31 + 8);
    if ( !v31 )
      goto LABEL_64;
  }
LABEL_38:
  v33 = v32[5];
  v171 = (__int64 *)v33;
  if ( v158 != v33 )
  {
    sub_1953970((__int64)&v186, (__int64)&v180, v33);
    if ( (_BYTE)v190 )
    {
      if ( v171 != (__int64 *)v142 )
        sub_15CDD90((__int64)&v177, &v171);
    }
  }
  while ( 1 )
  {
    v31 = *(_QWORD *)(v31 + 8);
    if ( !v31 )
      break;
    v32 = sub_1648700(v31);
    if ( (unsigned __int8)(*((_BYTE *)v32 + 16) - 25) <= 9u )
      goto LABEL_38;
  }
  if ( v184 == HIDWORD(v183) )
  {
LABEL_64:
    v186 = 0;
    v187 = v191;
    v188 = v191;
    v189 = 16;
    v190 = 0;
  }
  else
  {
    sub_1953970((__int64)&v186, (__int64)&v180, v142);
    for ( i = v178; (_DWORD)v178; i = v178 )
    {
      while ( 1 )
      {
        v35 = i--;
        v36 = *(_QWORD *)&v177[8 * v35 - 8];
        LODWORD(v178) = i;
        v37 = *(_QWORD *)(v36 + 8);
        if ( v37 )
          break;
LABEL_48:
        if ( !i )
          goto LABEL_49;
      }
      while ( 1 )
      {
        v38 = sub_1648700(v37);
        if ( (unsigned __int8)(*((_BYTE *)v38 + 16) - 25) <= 9u )
          break;
        v37 = *(_QWORD *)(v37 + 8);
        if ( !v37 )
          goto LABEL_48;
      }
LABEL_76:
      v59 = v38[5];
      v171 = (__int64 *)v59;
      if ( (v197 & 1) != 0 )
      {
        v54 = 15;
        v55 = &v198;
      }
      else
      {
        v55 = (__int64 *)v198;
        v54 = v199 - 1;
        if ( !v199 )
          goto LABEL_74;
      }
      v56 = 1;
      v57 = v54 & (((unsigned int)v59 >> 9) ^ ((unsigned int)v59 >> 4));
      v58 = v55[v57];
      if ( v59 != v58 )
      {
        while ( 1 )
        {
          if ( v58 == -8 )
            goto LABEL_74;
          v57 = v54 & (v56 + v57);
          v58 = v55[v57];
          if ( v59 == v58 )
            break;
          ++v56;
        }
      }
      sub_1953970((__int64)&v186, (__int64)&v180, v59);
      if ( (_BYTE)v190 )
        sub_15CDD90((__int64)&v177, &v171);
LABEL_74:
      while ( 1 )
      {
        v37 = *(_QWORD *)(v37 + 8);
        if ( !v37 )
          break;
        v38 = sub_1648700(v37);
        if ( (unsigned __int8)(*((_BYTE *)v38 + 16) - 25) <= 9u )
          goto LABEL_76;
      }
    }
LABEL_49:
    v39 = (char *)sub_194ACF0(a5);
    v157 = v39;
    if ( v147 )
    {
      sub_1400330(v147, v158, a5);
      v186 = v157;
      *(_QWORD *)v157 = v147;
      sub_1A541E0(v147 + 8, &v186);
    }
    else
    {
      v186 = v39;
      sub_1A541E0(a5 + 32, &v186);
    }
    sub_1A51A10(a6, &v157, v40, v41, v42, v43);
    sub_13FC0C0((__int64)(v157 + 32), (unsigned int)(HIDWORD(v183) - v184));
    v46 = *(__int64 **)(a1 + 32);
    v148 = *(__int64 **)(a1 + 40);
    if ( v46 != v148 )
    {
      while ( 1 )
      {
        v47 = *v46;
        v171 = (__int64 *)*v46;
        sub_1A51850((unsigned __int64 *)&v186, a4, (__int64 *)&v171);
        v48 = v188;
        if ( !v188 )
          break;
        sub_1455FA0((__int64)&v186);
        if ( sub_183E920((__int64)&v180, (__int64)v48) )
        {
          v49 = sub_13AE450(a5, v47);
          v50 = v157;
          if ( a1 != v49 )
          {
            while ( v50 )
            {
              v186 = v48;
              v51 = (_BYTE *)*((_QWORD *)v50 + 5);
              if ( v51 == *((_BYTE **)v50 + 6) )
              {
                sub_1292090((__int64)(v50 + 32), v51, &v186);
                v52 = v186;
              }
              else
              {
                if ( v51 )
                {
                  *(_QWORD *)v51 = v48;
                  v51 = (_BYTE *)*((_QWORD *)v50 + 5);
                }
                *((_QWORD *)v50 + 5) = v51 + 8;
                v52 = v48;
              }
              sub_1412190((__int64)(v50 + 56), (__int64)v52);
              v50 = *(char **)v50;
            }
            goto LABEL_82;
          }
          ++v46;
          sub_1400330((__int64)v157, (__int64)v48, a5);
          if ( v148 == v46 )
            goto LABEL_83;
        }
        else
        {
LABEL_82:
          if ( v148 == ++v46 )
            goto LABEL_83;
        }
      }
      sub_1455FA0((__int64)&v186);
      goto LABEL_82;
    }
LABEL_83:
    v60 = *(__int64 **)(a1 + 8);
    for ( j = *(__int64 **)(a1 + 16); j != v60; ++v60 )
    {
      v61 = *v60;
      v171 = **(__int64 ***)(*v60 + 32);
      sub_1A51850((unsigned __int64 *)&v186, a4, (__int64 *)&v171);
      if ( v188 )
      {
        v143 = (__int64)v188;
        sub_1455FA0((__int64)&v186);
        if ( sub_183E920((__int64)&v180, v143) )
          sub_1A56560(v61, (__int64)v157, a4, a5);
      }
      else
      {
        sub_1455FA0((__int64)&v186);
      }
    }
    v186 = 0;
    v187 = v191;
    v188 = v191;
    v189 = 16;
    v190 = 0;
    if ( v184 != HIDWORD(v183) )
      goto LABEL_89;
  }
  sub_1953970((__int64)&v171, (__int64)&v186, v158);
LABEL_89:
  v62 = v200;
  v63 = &v200[(unsigned int)v201];
  if ( v63 != v200 )
  {
    do
    {
      while ( 1 )
      {
        v64 = *v62;
        if ( !sub_183E920((__int64)&v180, *v62) )
          break;
        if ( v63 == ++v62 )
          goto LABEL_94;
      }
      ++v62;
      sub_1953970((__int64)&v171, (__int64)&v186, v64);
    }
    while ( v63 != v62 );
  }
LABEL_94:
  v65 = v160;
  v162 = (__int64 *)v164;
  v163 = 0x400000000LL;
  if ( (_DWORD)v160 )
  {
    v129 = (__int64 *)v164;
    v130 = 8LL * (unsigned int)v160;
    if ( (unsigned int)v160 <= 4
      || (sub_16CD150((__int64)&v162, v164, (unsigned int)v160, 8, (int)v44, v45),
          v129 = v162,
          (v130 = 8LL * (unsigned int)v160) != 0) )
    {
      memcpy(v129, src, v130);
      v129 = v162;
    }
    v131 = v65;
    LODWORD(v163) = v65;
    v151 = &v129[v131];
    _BitScanReverse64(&v132, (v131 * 8) >> 3);
    sub_1A4F870(v129, &v129[v131], 2LL * (int)(63 - (v132 ^ 0x3F)), (__int64)&v192);
    if ( (unsigned __int64)v131 <= 16 )
    {
      sub_1A4FC40(v129, v151, (__int64)&v192);
    }
    else
    {
      v133 = v129 + 16;
      sub_1A4FC40(v129, v129 + 16, (__int64)&v192);
      if ( &v129[v131] != v129 + 16 )
      {
        do
        {
          v134 = v133;
          v171 = &v192;
          v135 = *v133;
          while ( 1 )
          {
            v136 = *(v134 - 1);
            v137 = v134--;
            if ( !sub_1A4F560((__int64 *)&v171, v135, v136) )
              break;
            v134[1] = *v134;
          }
          *v137 = v135;
          ++v133;
        }
        while ( v151 != v133 );
      }
    }
  }
  if ( HIDWORD(v189) != v190 )
  {
    while ( 1 )
    {
      if ( !(_DWORD)v163 )
        goto LABEL_111;
      v66 = v162[(unsigned int)v163 - 1];
      LODWORD(v163) = v163 - 1;
      if ( (v193 & 1) != 0 )
        break;
      v68 = v194;
      v72 = 0;
      if ( v195 )
      {
        v67 = v195 - 1;
LABEL_99:
        v69 = v67 & (((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4));
        v70 = &v68[2 * v69];
        v71 = *v70;
        if ( v66 == *v70 )
        {
LABEL_100:
          v72 = v70[1];
        }
        else
        {
          v124 = 1;
          while ( v71 != -8 )
          {
            LODWORD(v44) = v124 + 1;
            v69 = v67 & (v124 + v69);
            v70 = &v68[2 * v69];
            v71 = *v70;
            if ( v66 == *v70 )
              goto LABEL_100;
            v124 = (int)v44;
          }
          v72 = 0;
        }
      }
      v73 = (unsigned int)v178;
      if ( (unsigned int)v178 >= HIDWORD(v178) )
      {
        sub_16CD150((__int64)&v177, v179, 0, 8, (int)v44, v45);
        v73 = (unsigned int)v178;
      }
      *(_QWORD *)&v177[8 * v73] = v66;
      v74 = v178 + 1;
      LODWORD(v178) = v178 + 1;
      do
      {
        while ( 1 )
        {
          v75 = v74--;
          v76 = *(_QWORD *)&v177[8 * v75 - 8];
          LODWORD(v178) = v74;
          if ( v158 != v76 )
          {
            v77 = *(_QWORD *)(v76 + 8);
            if ( v77 )
              break;
          }
LABEL_104:
          if ( !v74 )
            goto LABEL_110;
        }
        do
        {
          v78 = sub_1648700(v77);
          if ( (unsigned __int8)(*((_BYTE *)v78 + 16) - 25) <= 9u )
          {
            while ( 1 )
            {
              v111 = v78[5];
              v110 = v187;
              if ( v188 == v187 )
              {
                v112 = &v187[8 * HIDWORD(v189)];
                if ( v187 == v112 )
                {
LABEL_171:
                  v110 = &v187[8 * HIDWORD(v189)];
                }
                else
                {
                  while ( v111 != *v110 )
                  {
                    if ( v112 == (_BYTE *)++v110 )
                      goto LABEL_171;
                  }
                }
              }
              else
              {
                v110 = sub_16CC9F0((__int64)&v186, v111);
                if ( v111 == *v110 )
                {
                  if ( v188 == v187 )
                    v112 = &v188[8 * HIDWORD(v189)];
                  else
                    v112 = &v188[8 * (unsigned int)v189];
                }
                else
                {
                  if ( v188 != v187 )
                    goto LABEL_153;
                  v110 = &v188[8 * HIDWORD(v189)];
                  v112 = v110;
                }
              }
              if ( v112 != (_BYTE *)v110 )
              {
                *v110 = -2;
                ++v190;
                v171 = (__int64 *)v111;
                v172 = v72;
                if ( (v193 & 1) == 0 )
                {
                  v113 = v195;
                  v114 = v194;
                  if ( v195 )
                  {
                    v115 = v195 - 1;
                    goto LABEL_164;
                  }
                  v119 = v193;
                  ++v192;
                  v44 = 0;
                  v120 = ((unsigned int)v193 >> 1) + 1;
LABEL_174:
                  if ( 4 * v120 < 3 * v113 )
                    goto LABEL_175;
LABEL_186:
                  v113 *= 2;
                  goto LABEL_187;
                }
                v115 = 15;
                v114 = &v194;
LABEL_164:
                v116 = v115 & (((unsigned int)v111 >> 9) ^ ((unsigned int)v111 >> 4));
                v117 = &v114[2 * v116];
                v45 = *v117;
                if ( v111 == *v117 )
                {
LABEL_165:
                  v118 = (unsigned int)v178;
                  if ( (unsigned int)v178 < HIDWORD(v178) )
                    goto LABEL_166;
LABEL_179:
                  sub_16CD150((__int64)&v177, v179, 0, 8, (int)v44, v45);
                  v118 = (unsigned int)v178;
                }
                else
                {
                  v123 = 1;
                  v44 = 0;
                  while ( v45 != -8 )
                  {
                    if ( v45 != -16 || v44 )
                      v117 = v44;
                    LODWORD(v44) = v123 + 1;
                    v116 = v115 & (v123 + v116);
                    v45 = v114[2 * v116];
                    if ( v111 == v45 )
                      goto LABEL_165;
                    ++v123;
                    v44 = v117;
                    v117 = &v114[2 * v116];
                  }
                  if ( !v44 )
                    v44 = v117;
                  v119 = v193;
                  ++v192;
                  v120 = ((unsigned int)v193 >> 1) + 1;
                  if ( (v193 & 1) == 0 )
                  {
                    v113 = v195;
                    goto LABEL_174;
                  }
                  v113 = 16;
                  if ( 4 * v120 >= 0x30 )
                    goto LABEL_186;
LABEL_175:
                  v121 = v113 - HIDWORD(v193) - v120;
                  v122 = v111;
                  if ( v121 <= v113 >> 3 )
                  {
LABEL_187:
                    sub_1A54BC0((__int64)&v192, v113);
                    sub_1A542D0((__int64)&v192, (__int64 *)&v171, &v165);
                    v44 = v165;
                    v122 = (__int64)v171;
                    v119 = v193;
                  }
                  LODWORD(v193) = (2 * (v119 >> 1) + 2) | v119 & 1;
                  if ( *v44 != -8 )
                    --HIDWORD(v193);
                  *v44 = v122;
                  v44[1] = v172;
                  v118 = (unsigned int)v178;
                  if ( (unsigned int)v178 >= HIDWORD(v178) )
                    goto LABEL_179;
                }
LABEL_166:
                *(_QWORD *)&v177[8 * v118] = v111;
                LODWORD(v178) = v178 + 1;
                v77 = *(_QWORD *)(v77 + 8);
                if ( !v77 )
                {
LABEL_167:
                  v74 = v178;
                  goto LABEL_104;
                }
                goto LABEL_154;
              }
              do
              {
LABEL_153:
                v77 = *(_QWORD *)(v77 + 8);
                if ( !v77 )
                  goto LABEL_167;
LABEL_154:
                v78 = sub_1648700(v77);
              }
              while ( (unsigned __int8)(*((_BYTE *)v78 + 16) - 25) > 9u );
            }
          }
          v77 = *(_QWORD *)(v77 + 8);
        }
        while ( v77 );
      }
      while ( v74 );
LABEL_110:
      if ( HIDWORD(v189) == v190 )
        goto LABEL_111;
    }
    v67 = 15;
    v68 = &v194;
    goto LABEL_99;
  }
LABEL_111:
  v165 = (__int64 *)src;
  v152 = (char *)src + 8 * (unsigned int)v160;
  v167 = (unsigned __int64)v200;
  v169 = (void **)&v158;
  v150 = &v200[(unsigned int)v201];
  v166 = v152;
  v168 = v150;
  p_src = &src;
  do
  {
    v79 = &v171;
    v174 = 0;
    v176 = 0;
    v80 = (void **)&v165;
    v173 = sub_1A4EB50;
    v81 = &v171;
    v175 = sub_1A4EB70;
    v82 = sub_1A4EB30;
    if ( ((unsigned __int8)sub_1A4EB30 & 1) == 0 )
      goto LABEL_114;
    while ( 1 )
    {
      v82 = *(__int64 (__fastcall **)(__int64))((char *)v82 + (_QWORD)*v80 - 1);
LABEL_114:
      v83 = (__int64 *)v82((__int64)v80);
      if ( v83 )
        break;
      while ( 1 )
      {
        v84 = v81[3];
        v82 = (__int64 (__fastcall *)(__int64))v81[2];
        v79 += 2;
        v81 = v79;
        v80 = (void **)((char *)&v165 + (_QWORD)v84);
        if ( ((unsigned __int8)v82 & 1) != 0 )
          break;
        v83 = (__int64 *)v82((__int64)v80);
        if ( v83 )
          goto LABEL_117;
      }
    }
LABEL_117:
    if ( (v193 & 1) != 0 )
    {
      v85 = &v194;
      v86 = 15;
    }
    else
    {
      v85 = v194;
      if ( !v195 )
        goto LABEL_122;
      v86 = v195 - 1;
    }
    v87 = *v83;
    v88 = v86 & (((unsigned int)*v83 >> 9) ^ ((unsigned int)*v83 >> 4));
    v89 = &v85[2 * v88];
    v90 = *v89;
    if ( v87 == *v89 )
    {
LABEL_120:
      v91 = v89[1];
      if ( v91 )
        sub_1400330(v91, v87, a5);
    }
    else
    {
      v125 = 1;
      while ( v90 != -8 )
      {
        v126 = v125 + 1;
        v88 = v86 & (v125 + v88);
        v89 = &v85[2 * v88];
        v90 = *v89;
        if ( v87 == *v89 )
          goto LABEL_120;
        v125 = v126;
      }
    }
LABEL_122:
    v92 = &v171;
    v93 = sub_1A4EAA0;
    v173 = sub_1A4EAD0;
    v174 = 0;
    v94 = (void **)&v165;
    v175 = sub_1A4EB00;
    v95 = &v171;
    v176 = 0;
    if ( ((unsigned __int8)sub_1A4EAA0 & 1) == 0 )
      goto LABEL_124;
    while ( 1 )
    {
      v93 = *(__int64 (__fastcall **)(__int64))((char *)v93 + (_QWORD)*v94 - 1);
LABEL_124:
      if ( (unsigned __int8)v93((__int64)v94) )
        break;
      while ( 1 )
      {
        v96 = v95[3];
        v93 = (__int64 (__fastcall *)(__int64))v95[2];
        v92 += 2;
        v95 = v92;
        v94 = (void **)((char *)&v165 + (_QWORD)v96);
        if ( ((unsigned __int8)v93 & 1) != 0 )
          break;
        if ( (unsigned __int8)v93((__int64)v94) )
          goto LABEL_127;
      }
    }
LABEL_127:
    ;
  }
  while ( v169 != &src
       || p_src != &src
       || v150 != (__int64 *)v167
       || v150 != v168
       || v152 != (char *)v165
       || v152 != v166 );
  v97 = *(__int64 **)(a1 + 8);
  for ( k = *(__int64 **)(a1 + 16); k != v97; ++v97 )
  {
    v108 = *v97;
    v165 = **(__int64 ***)(*v97 + 32);
    sub_1A51850((unsigned __int64 *)&v171, a4, (__int64 *)&v165);
    v109 = (__int64)v173;
    if ( v173 )
    {
      sub_1455FA0((__int64)&v171);
      if ( !sub_183E920((__int64)&v180, v109) )
      {
        if ( (v193 & 1) != 0 )
        {
          v98 = &v194;
          v99 = 15;
LABEL_136:
          v100 = v99 & (((unsigned int)v109 >> 9) ^ ((unsigned int)v109 >> 4));
          v101 = &v98[2 * v100];
          v102 = *v101;
          if ( v109 == *v101 )
          {
LABEL_137:
            v103 = v101[1];
          }
          else
          {
            v138 = 1;
            while ( v102 != -8 )
            {
              v139 = v138 + 1;
              v100 = v99 & (v138 + v100);
              v101 = &v98[2 * v100];
              v102 = *v101;
              if ( v109 == *v101 )
                goto LABEL_137;
              v138 = v139;
            }
            v103 = 0;
          }
        }
        else
        {
          v98 = v194;
          v103 = 0;
          if ( v195 )
          {
            v99 = v195 - 1;
            goto LABEL_136;
          }
        }
        v106 = sub_1A56560(v108, v103, a4, a5);
        v107 = *(unsigned int *)(a6 + 8);
        if ( (unsigned int)v107 >= *(_DWORD *)(a6 + 12) )
        {
          sub_16CD150(a6, (const void *)(a6 + 16), 0, 8, v104, v105);
          v107 = *(unsigned int *)(a6 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a6 + 8 * v107) = v106;
        ++*(_DWORD *)(a6 + 8);
      }
    }
    else
    {
      sub_1455FA0((__int64)&v171);
    }
  }
  if ( v162 != (__int64 *)v164 )
    _libc_free((unsigned __int64)v162);
  if ( v188 != v187 )
    _libc_free((unsigned __int64)v188);
  if ( v182 != v181 )
    _libc_free((unsigned __int64)v182);
  if ( v177 != v179 )
    _libc_free((unsigned __int64)v177);
  if ( v200 != (__int64 *)v202 )
    _libc_free((unsigned __int64)v200);
  if ( (v197 & 1) == 0 )
    j___libc_free_0(v198);
  if ( (v193 & 1) == 0 )
    j___libc_free_0(v194);
  if ( src != v161 )
    _libc_free((unsigned __int64)src);
}
