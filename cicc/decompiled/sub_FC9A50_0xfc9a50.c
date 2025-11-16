// Function: sub_FC9A50
// Address: 0xfc9a50
//
_QWORD *__fastcall sub_FC9A50(__int64 *a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 **v6; // rax
  _QWORD *v7; // rbx
  bool v8; // cl
  __int64 *v9; // rax
  _QWORD *v10; // r10
  int v11; // r11d
  _QWORD *v12; // rdi
  unsigned int v13; // esi
  unsigned __int64 v14; // r8
  __int64 v15; // rdx
  __int64 v16; // r12
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 *v19; // r13
  __int64 v20; // rbx
  __int64 v21; // rax
  __int64 v22; // rdx
  _QWORD *v23; // rdi
  int v24; // esi
  unsigned int v25; // eax
  int v26; // r10d
  __int64 *v27; // rcx
  unsigned int v28; // edi
  unsigned int v29; // eax
  __int64 v30; // rsi
  unsigned __int8 v31; // al
  __int64 v32; // rbx
  __int64 v33; // rax
  __int64 v34; // rcx
  unsigned __int64 v35; // rdx
  const __m128i *v36; // rbx
  __m128i *v37; // rax
  _QWORD *v38; // rdi
  _QWORD *v39; // r8
  int v40; // edx
  unsigned int v41; // eax
  int v42; // r10d
  __int64 *v43; // rdi
  int v44; // edx
  __int64 *v45; // rdi
  __int64 *v46; // rax
  char v47; // dl
  __int64 v48; // rdx
  unsigned __int64 v49; // rdx
  __int64 v50; // rbx
  __int64 *v51; // r13
  int v52; // esi
  _QWORD *v53; // r8
  unsigned int v54; // eax
  _QWORD *v55; // rdi
  __int64 v56; // r9
  _BYTE *v57; // r14
  __int64 v58; // rbx
  unsigned int v59; // esi
  unsigned int v60; // edx
  int v61; // edx
  unsigned int v62; // eax
  __int64 v63; // r8
  int v64; // r10d
  __int64 *v65; // rdi
  __int64 *v66; // rbx
  __int64 *v67; // r12
  __int64 **v68; // r12
  __int64 **v69; // rbx
  __int64 v70; // rdi
  unsigned int v72; // eax
  __int64 *v73; // rcx
  unsigned int v74; // edi
  unsigned int v75; // r8d
  unsigned __int8 v76; // al
  __int64 *v77; // rax
  __int64 v78; // rdx
  __int64 v79; // rbx
  __int64 *v80; // rdi
  __int64 v81; // rdx
  __int64 v82; // rbx
  char v83; // r8
  __int64 *v84; // r12
  __int64 v85; // r9
  int v86; // edi
  _QWORD *v87; // rdx
  unsigned int v88; // ecx
  _QWORD *v89; // rsi
  __int64 v90; // r10
  _QWORD *v91; // rdi
  __int64 v92; // r10
  int v93; // r9d
  int v94; // esi
  unsigned int v95; // ecx
  _QWORD *v96; // r11
  __int64 v97; // rbx
  __int64 v98; // rsi
  _QWORD *v99; // r8
  int v100; // eax
  unsigned int v101; // edx
  _QWORD *v102; // rcx
  __int64 v103; // r9
  __m128i *v104; // rax
  __m128i *v105; // r13
  unsigned __int8 v106; // al
  bool v107; // dl
  unsigned int v108; // r14d
  __int64 v109; // rbx
  __int8 *v110; // rax
  __int64 v111; // r14
  unsigned __int8 *v112; // rax
  __int64 v113; // rdx
  bool v114; // zf
  unsigned __int8 *v115; // rdx
  int v116; // r9d
  unsigned int v117; // eax
  __int64 v118; // rcx
  __int64 v119; // r10
  int v120; // r11d
  __int64 v121; // r10
  unsigned int v122; // ecx
  _QWORD *v123; // r11
  __int64 v124; // rbx
  int v125; // r11d
  __int64 v126; // r10
  unsigned int v127; // ecx
  _QWORD *v128; // r9
  __int64 v129; // r11
  int v130; // r9d
  char v131; // dl
  __int64 v132; // r8
  int v133; // edi
  _QWORD *v134; // r9
  unsigned int v135; // ecx
  _QWORD *v136; // rsi
  __int64 v137; // r10
  __int64 v138; // rcx
  __int64 v139; // rcx
  __int64 v140; // rsi
  int v141; // esi
  int v142; // ebx
  int v143; // r10d
  int v144; // esi
  _QWORD *v145; // rdi
  unsigned int v146; // edx
  __int64 v147; // r8
  int v148; // r9d
  __int64 *v149; // rax
  int v150; // esi
  _QWORD *v151; // rdi
  unsigned int v152; // edx
  __int64 v153; // r8
  int v154; // r9d
  int v155; // r11d
  __int64 v156; // rdi
  __int64 v157; // r8
  __int64 v158; // rsi
  _QWORD *v159; // r9
  int v160; // esi
  unsigned int v161; // ecx
  _QWORD *v162; // rdi
  __int64 v163; // r10
  __int64 v164; // rcx
  __int64 v165; // r8
  int v166; // edi
  _QWORD *v167; // r9
  unsigned int v168; // ecx
  _QWORD *v169; // rsi
  __int64 v170; // r10
  __int64 v171; // rcx
  __int64 v172; // rdi
  unsigned int v173; // edi
  unsigned __int64 *v174; // rax
  unsigned int v175; // eax
  unsigned __int64 *v176; // rdx
  unsigned int v177; // ecx
  unsigned int v178; // edi
  _QWORD *v179; // rax
  __int64 v180; // rax
  __m128i *v181; // rax
  _QWORD *v182; // rsi
  __m128i *v183; // rbx
  __int64 v184; // rdx
  __int64 v185; // rcx
  __int64 v186; // r8
  __int64 v187; // r9
  __int64 v188; // rax
  unsigned __int64 v189; // rdx
  __int64 *v190; // rbx
  __int64 *v191; // r12
  __int64 v192; // rdi
  _QWORD *v193; // rax
  unsigned int v194; // r9d
  unsigned int v195; // eax
  __int64 *v196; // rdi
  unsigned int v197; // ecx
  unsigned int v198; // r8d
  __int64 v199; // r8
  __int64 v200; // rdi
  __int64 v201; // rdx
  __int64 v202; // r8
  __int64 v203; // rcx
  __int64 v204; // rcx
  int v205; // ecx
  char *v206; // rbx
  int v207; // r10d
  int v208; // r11d
  __int64 v209; // rsi
  _QWORD *v210; // r9
  int v211; // r8d
  unsigned int v212; // edx
  __int64 v213; // rsi
  int v214; // ecx
  __int64 *v215; // rax
  unsigned int v216; // edx
  int v217; // r10d
  _QWORD *v218; // r8
  int v219; // esi
  unsigned int v220; // edx
  __int64 v221; // rcx
  int v222; // eax
  __int64 *v223; // r9
  int v224; // r10d
  unsigned int v225; // edx
  __int64 v226; // rsi
  int v227; // esi
  int v228; // esi
  __int64 v229; // rdi
  int v230; // r13d
  int v231; // edi
  int v232; // r8d
  unsigned int v233; // ecx
  int v234; // edi
  unsigned __int64 *v235; // rax
  unsigned int v236; // ecx
  int v237; // edi
  int v238; // r11d
  __int64 v239; // rsi
  int v240; // r11d
  __int64 v241; // rsi
  int v242; // r11d
  __int64 v243; // rdi
  __int64 v244; // [rsp+18h] [rbp-5E8h]
  _QWORD *v245; // [rsp+28h] [rbp-5D8h]
  __int64 *v247; // [rsp+40h] [rbp-5C0h]
  int v248; // [rsp+40h] [rbp-5C0h]
  int v249; // [rsp+40h] [rbp-5C0h]
  char v250; // [rsp+48h] [rbp-5B8h]
  __m128i *v251; // [rsp+48h] [rbp-5B8h]
  char v252; // [rsp+50h] [rbp-5B0h]
  __int64 v253; // [rsp+50h] [rbp-5B0h]
  __int64 *v254; // [rsp+60h] [rbp-5A0h]
  __int64 *v255; // [rsp+60h] [rbp-5A0h]
  __int64 *v256; // [rsp+68h] [rbp-598h]
  __int64 v257; // [rsp+68h] [rbp-598h]
  __m128i *v258; // [rsp+70h] [rbp-590h] BYREF
  unsigned __int8 *v259; // [rsp+78h] [rbp-588h] BYREF
  unsigned __int8 *v260; // [rsp+80h] [rbp-580h] BYREF
  __int64 v261; // [rsp+88h] [rbp-578h]
  char v262; // [rsp+90h] [rbp-570h]
  unsigned __int64 v263; // [rsp+A0h] [rbp-560h] BYREF
  __int64 v264; // [rsp+A8h] [rbp-558h]
  _QWORD v265[2]; // [rsp+B0h] [rbp-550h] BYREF
  char v266; // [rsp+C0h] [rbp-540h]
  __int64 v267; // [rsp+230h] [rbp-3D0h] BYREF
  __int64 v268; // [rsp+238h] [rbp-3C8h]
  _QWORD *v269; // [rsp+240h] [rbp-3C0h] BYREF
  unsigned int v270; // [rsp+248h] [rbp-3B8h]
  __int64 *v271; // [rsp+540h] [rbp-C0h] BYREF
  __int64 v272; // [rsp+548h] [rbp-B8h]
  _BYTE v273[176]; // [rsp+550h] [rbp-B0h] BYREF

  v6 = &v269;
  v7 = a2;
  v267 = 0;
  v268 = 1;
  do
  {
    *v6 = (__int64 *)-4096LL;
    v6 += 3;
  }
  while ( v6 != &v271 );
  HIDWORD(v264) = 16;
  v271 = (__int64 *)v273;
  v272 = 0x1000000000LL;
  v263 = (unsigned __int64)v265;
  v8 = (*(_BYTE *)(a2 - 2) & 2) != 0;
  if ( (*(_BYTE *)(a2 - 2) & 2) != 0 )
    v9 = (__int64 *)*(a2 - 4);
  else
    v9 = &a2[-((*((_BYTE *)a2 - 16) >> 2) & 0xF) - 2];
  v265[0] = a2;
  v265[1] = v9;
  v266 = 0;
  LODWORD(v264) = 1;
  v245 = a2;
  if ( (v268 & 1) != 0 )
  {
    v10 = &v269;
    v11 = 31;
  }
  else
  {
    v30 = v270;
    v10 = v269;
    v11 = v270 - 1;
    if ( !v270 )
    {
      v175 = v268;
      ++v267;
      v176 = 0;
      v177 = ((unsigned int)v268 >> 1) + 1;
      goto LABEL_268;
    }
  }
  v12 = v265;
  v13 = v11 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
  v14 = (unsigned __int64)&v10[3 * v13];
  v15 = 1;
  a6 = *(_QWORD *)v14;
  if ( v7 != *(_QWORD **)v14 )
  {
    v230 = 1;
    v176 = 0;
    while ( a6 != -4096 )
    {
      if ( !v176 && a6 == -8192 )
        v176 = (unsigned __int64 *)v14;
      v13 = v11 & (v230 + v13);
      v14 = (unsigned __int64)&v10[3 * v13];
      a6 = *(_QWORD *)v14;
      if ( v7 == *(_QWORD **)v14 )
      {
        v12 = v265;
        v15 = 1;
        goto LABEL_8;
      }
      ++v230;
    }
    v175 = v268;
    v178 = 96;
    v30 = 32;
    if ( !v176 )
      v176 = (unsigned __int64 *)v14;
    ++v267;
    v177 = ((unsigned int)v268 >> 1) + 1;
    if ( (v268 & 1) != 0 )
    {
LABEL_269:
      v14 = 4 * v177;
      if ( (unsigned int)v14 < v178 )
      {
        if ( (_DWORD)v30 - HIDWORD(v268) - v177 > (unsigned int)v30 >> 3 )
        {
LABEL_271:
          v34 = 2 * (v175 >> 1) + 2;
          LODWORD(v268) = v34 | v175 & 1;
          if ( *v176 != -4096 )
            --HIDWORD(v268);
          *v176 = (unsigned __int64)v7;
          v176[1] = 0xFFFFFFFF00000000LL;
          v176[2] = 0;
          v15 = (unsigned int)v264;
          if ( !(_DWORD)v264 )
          {
            if ( (_QWORD *)v263 != v265 )
              _libc_free(v263, v30);
LABEL_85:
            v65 = v271;
            v66 = &v271[(unsigned int)v272];
            if ( v66 != v271 )
            {
              v67 = v271;
              do
              {
                v30 = *v67++;
                sub_FC80D0(*a1, v30, v30);
              }
              while ( v66 != v67 );
              v65 = v271;
            }
            goto LABEL_89;
          }
          v12 = (_QWORD *)v263;
          v179 = (_QWORD *)(v263 + 24LL * (unsigned int)v264 - 24);
          v7 = (_QWORD *)*v179;
          v9 = (__int64 *)v179[1];
          v8 = (*(_BYTE *)(v7 - 2) & 2) != 0;
          goto LABEL_8;
        }
        sub_FC7090((__int64)&v267, v30);
        if ( (v268 & 1) != 0 )
        {
          a6 = (__int64)&v269;
          v30 = 31;
LABEL_426:
          v236 = v30 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
          v176 = (unsigned __int64 *)(a6 + 24LL * v236);
          v175 = v268;
          v14 = *v176;
          if ( v7 == (_QWORD *)*v176 )
            goto LABEL_271;
          v237 = 1;
          v235 = 0;
          while ( v14 != -4096 )
          {
            if ( v14 == -8192 && !v235 )
              v235 = v176;
            v236 = v30 & (v237 + v236);
            v176 = (unsigned __int64 *)(a6 + 24LL * v236);
            v14 = *v176;
            if ( v7 == (_QWORD *)*v176 )
              goto LABEL_423;
            ++v237;
          }
          goto LABEL_421;
        }
        a6 = (__int64)v269;
        if ( v270 )
        {
          v30 = v270 - 1;
          goto LABEL_426;
        }
LABEL_494:
        LODWORD(v268) = (2 * ((unsigned int)v268 >> 1) + 2) | v268 & 1;
        BUG();
      }
      sub_FC7090((__int64)&v267, 2 * v30);
      if ( (v268 & 1) != 0 )
      {
        a6 = (__int64)&v269;
        v30 = 31;
      }
      else
      {
        a6 = (__int64)v269;
        if ( !v270 )
          goto LABEL_494;
        v30 = v270 - 1;
      }
      v233 = v30 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v176 = (unsigned __int64 *)(a6 + 24LL * v233);
      v175 = v268;
      v14 = *v176;
      if ( v7 == (_QWORD *)*v176 )
        goto LABEL_271;
      v234 = 1;
      v235 = 0;
      while ( v14 != -4096 )
      {
        if ( v14 == -8192 && !v235 )
          v235 = v176;
        v233 = v30 & (v234 + v233);
        v176 = (unsigned __int64 *)(a6 + 24LL * v233);
        v14 = *v176;
        if ( v7 == (_QWORD *)*v176 )
          goto LABEL_423;
        ++v234;
      }
LABEL_421:
      if ( v235 )
        v176 = v235;
LABEL_423:
      v175 = v268;
      goto LABEL_271;
    }
    v30 = v270;
LABEL_268:
    v178 = 3 * v30;
    goto LABEL_269;
  }
LABEL_8:
  v250 = 0;
  v16 = (__int64)&v12[3 * v15 - 3];
  if ( !v8 )
    goto LABEL_33;
  while ( 2 )
  {
    v17 = *(v7 - 4);
    v18 = *((unsigned int *)v7 - 6);
LABEL_10:
    v19 = (__int64 *)(v17 + 8 * v18);
    if ( v19 == v9 )
    {
LABEL_47:
      v30 = v268 & 1;
      if ( (v268 & 1) != 0 )
      {
        v14 = (unsigned __int64)&v269;
        v44 = 31;
      }
      else
      {
        v60 = v270;
        v14 = (unsigned __int64)v269;
        if ( !v270 )
        {
          v173 = v268;
          ++v267;
          v174 = 0;
          v14 = ((unsigned int)v268 >> 1) + 1;
          goto LABEL_260;
        }
        v44 = v270 - 1;
      }
      v34 = v44 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v45 = (__int64 *)(v14 + 24 * v34);
      a6 = *v45;
      if ( v7 == (_QWORD *)*v45 )
      {
LABEL_50:
        v46 = v45 + 1;
        goto LABEL_51;
      }
      v208 = 1;
      v174 = 0;
      while ( a6 != -4096 )
      {
        if ( a6 == -8192 && !v174 )
          v174 = (unsigned __int64 *)v45;
        v34 = v44 & (unsigned int)(v34 + v208);
        v45 = (__int64 *)(v14 + 24 * v34);
        a6 = *v45;
        if ( (_QWORD *)*v45 == v7 )
          goto LABEL_50;
        ++v208;
      }
      a6 = 96;
      v60 = 32;
      if ( !v174 )
        v174 = (unsigned __int64 *)v45;
      v173 = v268;
      ++v267;
      v14 = ((unsigned int)v268 >> 1) + 1;
      if ( (_BYTE)v30 )
      {
LABEL_261:
        if ( 4 * (int)v14 < (unsigned int)a6 )
        {
          v34 = v60 - HIDWORD(v268) - (unsigned int)v14;
          v30 = v60 >> 3;
          if ( (unsigned int)v34 > (unsigned int)v30 )
          {
LABEL_263:
            LODWORD(v268) = (2 * (v173 >> 1) + 2) | v173 & 1;
            if ( *v174 != -4096 )
              --HIDWORD(v268);
            *v174 = (unsigned __int64)v7;
            v46 = (__int64 *)(v174 + 1);
            *v46 = 0xFFFFFFFF00000000LL;
            v46[1] = 0;
LABEL_51:
            v47 = *(_BYTE *)(v16 + 16);
            v250 |= v47;
            *(_BYTE *)v46 = v47;
            v48 = (unsigned int)v272;
            *((_DWORD *)v46 + 1) = v272;
            v49 = v48 + 1;
            v50 = *(_QWORD *)v16;
            if ( v49 > HIDWORD(v272) )
            {
              v30 = (__int64)v273;
              sub_C8D5F0((__int64)&v271, v273, v49, 8u, v14, a6);
            }
            v271[(unsigned int)v272] = v50;
            LODWORD(v272) = v272 + 1;
            v38 = (_QWORD *)v263;
            v15 = (unsigned int)(v264 - 1);
            LODWORD(v264) = v15;
            if ( (_DWORD)v15 )
              goto LABEL_32;
            break;
          }
          sub_FC7090((__int64)&v267, v60);
          if ( (v268 & 1) != 0 )
          {
            a6 = (__int64)&v269;
            v34 = 31;
LABEL_374:
            v173 = v268;
            v224 = 1;
            v30 = 0;
            v225 = v34 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
            v174 = (unsigned __int64 *)(a6 + 24LL * v225);
            v14 = *v174;
            if ( (_QWORD *)*v174 == v7 )
              goto LABEL_263;
            while ( v14 != -4096 )
            {
              if ( !v30 && v14 == -8192 )
                v30 = (__int64)v174;
              v225 = v34 & (v224 + v225);
              v174 = (unsigned __int64 *)(a6 + 24LL * v225);
              v14 = *v174;
              if ( (_QWORD *)*v174 == v7 )
                goto LABEL_363;
              ++v224;
            }
            goto LABEL_361;
          }
          a6 = (__int64)v269;
          if ( v270 )
          {
            v34 = v270 - 1;
            goto LABEL_374;
          }
LABEL_491:
          LODWORD(v268) = (2 * ((unsigned int)v268 >> 1) + 2) | v268 & 1;
          BUG();
        }
        v30 = 2 * v60;
        sub_FC7090((__int64)&v267, v30);
        if ( (v268 & 1) != 0 )
        {
          a6 = (__int64)&v269;
          v34 = 31;
        }
        else
        {
          a6 = (__int64)v269;
          if ( !v270 )
            goto LABEL_491;
          v34 = v270 - 1;
        }
        v173 = v268;
        v216 = v34 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v174 = (unsigned __int64 *)(a6 + 24LL * v216);
        v14 = *v174;
        if ( (_QWORD *)*v174 == v7 )
          goto LABEL_263;
        v217 = 1;
        v30 = 0;
        while ( v14 != -4096 )
        {
          if ( v14 == -8192 && !v30 )
            v30 = (__int64)v174;
          v216 = v34 & (v217 + v216);
          v174 = (unsigned __int64 *)(a6 + 24LL * v216);
          v14 = *v174;
          if ( (_QWORD *)*v174 == v7 )
            goto LABEL_363;
          ++v217;
        }
LABEL_361:
        if ( v30 )
          v174 = (unsigned __int64 *)v30;
LABEL_363:
        v173 = v268;
        goto LABEL_263;
      }
      v60 = v270;
LABEL_260:
      a6 = 3 * v60;
      goto LABEL_261;
    }
    while ( 1 )
    {
      *(_QWORD *)(v16 + 8) = v9 + 1;
      v20 = *v9;
      v21 = sub_FC99C0(a1, *v9, (__int64)(v9 + 1), v17, v14, a6);
      v261 = v22;
      v260 = (unsigned __int8 *)v21;
      if ( (_BYTE)v22 )
      {
        *(_BYTE *)(v16 + 16) |= v260 != (unsigned __int8 *)v20;
        goto LABEL_13;
      }
      if ( (v268 & 1) != 0 )
      {
        v23 = &v269;
        v24 = 31;
      }
      else
      {
        v30 = v270;
        v23 = v269;
        if ( !v270 )
        {
          v28 = v268;
          ++v267;
          v27 = 0;
          v29 = ((unsigned int)v268 >> 1) + 1;
          goto LABEL_37;
        }
        v24 = v270 - 1;
      }
      v25 = v24 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
      v17 = 3LL * v25;
      a6 = (__int64)&v23[3 * v25];
      v14 = *(_QWORD *)a6;
      if ( v20 != *(_QWORD *)a6 )
        break;
LABEL_13:
      v9 = *(__int64 **)(v16 + 8);
      if ( v9 == v19 )
        goto LABEL_46;
    }
    v26 = 1;
    v27 = 0;
    while ( v14 != -4096 )
    {
      if ( v14 != -8192 || v27 )
        a6 = (__int64)v27;
      v25 = v24 & (v26 + v25);
      v17 = 3LL * v25;
      v14 = v23[3 * v25];
      if ( v20 == v14 )
        goto LABEL_13;
      v27 = (__int64 *)a6;
      ++v26;
      a6 = (__int64)&v23[3 * v25];
    }
    v28 = v268;
    if ( !v27 )
      v27 = (__int64 *)a6;
    ++v267;
    v29 = ((unsigned int)v268 >> 1) + 1;
    if ( (v268 & 1) != 0 )
    {
      v30 = 32;
      if ( 4 * v29 < 0x60 )
        goto LABEL_24;
LABEL_38:
      sub_FC7090((__int64)&v267, 2 * v30);
      if ( (v268 & 1) != 0 )
      {
        v39 = &v269;
        v40 = 31;
        goto LABEL_40;
      }
      v39 = v269;
      if ( v270 )
      {
        v40 = v270 - 1;
LABEL_40:
        v28 = v268;
        v41 = v40 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
        v27 = &v39[3 * v41];
        v30 = *v27;
        if ( v20 != *v27 )
        {
          v42 = 1;
          v43 = 0;
          while ( v30 != -4096 )
          {
            if ( !v43 && v30 == -8192 )
              v43 = v27;
            a6 = (unsigned int)(v42 + 1);
            v41 = v40 & (v42 + v41);
            v27 = &v39[3 * v41];
            v30 = *v27;
            if ( v20 == *v27 )
              goto LABEL_45;
            ++v42;
          }
LABEL_43:
          if ( v43 )
            v27 = v43;
LABEL_45:
          v28 = v268;
        }
        goto LABEL_25;
      }
LABEL_493:
      LODWORD(v268) = (2 * ((unsigned int)v268 >> 1) + 2) | v268 & 1;
      BUG();
    }
    v30 = v270;
LABEL_37:
    if ( 4 * v29 >= 3 * (int)v30 )
      goto LABEL_38;
LABEL_24:
    if ( (_DWORD)v30 - HIDWORD(v268) - v29 > (unsigned int)v30 >> 3 )
      goto LABEL_25;
    sub_FC7090((__int64)&v267, v30);
    if ( (v268 & 1) != 0 )
    {
      v30 = (__int64)&v269;
      v61 = 31;
    }
    else
    {
      v30 = (__int64)v269;
      if ( !v270 )
        goto LABEL_493;
      v61 = v270 - 1;
    }
    v28 = v268;
    v62 = v61 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
    v27 = (__int64 *)(v30 + 24LL * v62);
    v63 = *v27;
    if ( v20 != *v27 )
    {
      v64 = 1;
      v43 = 0;
      while ( v63 != -4096 )
      {
        if ( v63 == -8192 && !v43 )
          v43 = v27;
        a6 = (unsigned int)(v64 + 1);
        v62 = v61 & (v64 + v62);
        v27 = (__int64 *)(v30 + 24LL * v62);
        v63 = *v27;
        if ( v20 == *v27 )
          goto LABEL_45;
        ++v64;
      }
      goto LABEL_43;
    }
LABEL_25:
    LODWORD(v268) = (2 * (v28 >> 1) + 2) | v28 & 1;
    if ( *v27 != -4096 )
      --HIDWORD(v268);
    *v27 = v20;
    *((_BYTE *)v27 + 8) = 0;
    *((_DWORD *)v27 + 3) = -1;
    v27[2] = 0;
    if ( !v20 )
    {
LABEL_46:
      v7 = *(_QWORD **)v16;
      goto LABEL_47;
    }
    v31 = *(_BYTE *)(v20 - 16);
    v260 = (unsigned __int8 *)v20;
    if ( (v31 & 2) != 0 )
      v32 = *(_QWORD *)(v20 - 32);
    else
      v32 = -16 - 8LL * ((v31 >> 2) & 0xF) + v20;
    v33 = (unsigned int)v264;
    v34 = HIDWORD(v264);
    v261 = v32;
    v262 = 0;
    v35 = v263;
    v14 = (unsigned int)v264 + 1LL;
    v36 = (const __m128i *)&v260;
    if ( v14 > HIDWORD(v264) )
    {
      if ( v263 > (unsigned __int64)&v260 || (unsigned __int64)&v260 >= v263 + 24LL * (unsigned int)v264 )
      {
        v30 = (__int64)v265;
        sub_C8D5F0((__int64)&v263, v265, (unsigned int)v264 + 1LL, 0x18u, v14, a6);
        v35 = v263;
        v33 = (unsigned int)v264;
        v36 = (const __m128i *)&v260;
      }
      else
      {
        v30 = (__int64)v265;
        v206 = (char *)&v260 - v263;
        sub_C8D5F0((__int64)&v263, v265, (unsigned int)v264 + 1LL, 0x18u, v14, a6);
        v35 = v263;
        v33 = (unsigned int)v264;
        v36 = (const __m128i *)&v206[v263];
      }
    }
    v37 = (__m128i *)(v35 + 24 * v33);
    *v37 = _mm_loadu_si128(v36);
    v37[1].m128i_i64[0] = v36[1].m128i_i64[0];
    v38 = (_QWORD *)v263;
    v15 = (unsigned int)(v264 + 1);
    LODWORD(v264) = v15;
    if ( (_DWORD)v15 )
    {
LABEL_32:
      v16 = (__int64)&v38[3 * v15 - 3];
      v7 = *(_QWORD **)v16;
      v9 = *(__int64 **)(v16 + 8);
      if ( (*(_BYTE *)(*(_QWORD *)v16 - 16LL) & 2) != 0 )
        continue;
LABEL_33:
      v18 = (*((_WORD *)v7 - 8) >> 6) & 0xF;
      v17 = (__int64)v7 - 16 - 8LL * ((*((_BYTE *)v7 - 16) >> 2) & 0xF);
      goto LABEL_10;
    }
    break;
  }
  if ( v38 != v265 )
    _libc_free(v38, v30);
  if ( !v250 )
    goto LABEL_85;
  while ( 2 )
  {
    v256 = &v271[(unsigned int)v272];
    if ( v271 == v256 )
      goto LABEL_297;
    v252 = 0;
    v51 = v271;
    while ( 2 )
    {
      while ( 2 )
      {
        v58 = *v51;
        if ( (v268 & 1) != 0 )
        {
          v52 = 31;
          v53 = &v269;
          goto LABEL_60;
        }
        v59 = v270;
        v53 = v269;
        if ( !v270 )
        {
          v72 = v268;
          ++v267;
          v73 = 0;
          v74 = ((unsigned int)v268 >> 1) + 1;
          goto LABEL_102;
        }
        v52 = v270 - 1;
LABEL_60:
        v54 = v52 & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
        v55 = &v53[3 * v54];
        v56 = *v55;
        if ( v58 != *v55 )
        {
          v143 = 1;
          v73 = 0;
          while ( v56 != -4096 )
          {
            if ( v56 == -8192 && !v73 )
              v73 = v55;
            v54 = v52 & (v143 + v54);
            v55 = &v53[3 * v54];
            v56 = *v55;
            if ( v58 == *v55 )
              goto LABEL_61;
            ++v143;
          }
          v72 = v268;
          v75 = 96;
          v59 = 32;
          if ( !v73 )
            v73 = v55;
          ++v267;
          v74 = ((unsigned int)v268 >> 1) + 1;
          if ( (v268 & 1) != 0 )
          {
LABEL_103:
            if ( v75 <= 4 * v74 )
            {
              sub_FC7090((__int64)&v267, 2 * v59);
              if ( (v268 & 1) != 0 )
              {
                v144 = 31;
                v145 = &v269;
              }
              else
              {
                v145 = v269;
                if ( !v270 )
                  goto LABEL_492;
                v144 = v270 - 1;
              }
              v146 = v144 & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
              v73 = &v145[3 * v146];
              v72 = v268;
              v147 = *v73;
              if ( *v73 == v58 )
                goto LABEL_105;
              v148 = 1;
              v149 = 0;
              while ( v147 != -4096 )
              {
                if ( v147 == -8192 && !v149 )
                  v149 = v73;
                v146 = v144 & (v148 + v146);
                v73 = &v145[3 * v146];
                v147 = *v73;
                if ( v58 == *v73 )
                  goto LABEL_220;
                ++v148;
              }
            }
            else
            {
              if ( v59 - HIDWORD(v268) - v74 > v59 >> 3 )
                goto LABEL_105;
              sub_FC7090((__int64)&v267, v59);
              if ( (v268 & 1) != 0 )
              {
                v150 = 31;
                v151 = &v269;
              }
              else
              {
                v151 = v269;
                if ( !v270 )
                {
LABEL_492:
                  LODWORD(v268) = (2 * ((unsigned int)v268 >> 1) + 2) | v268 & 1;
                  BUG();
                }
                v150 = v270 - 1;
              }
              v152 = v150 & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
              v73 = &v151[3 * v152];
              v72 = v268;
              v153 = *v73;
              if ( v58 == *v73 )
                goto LABEL_105;
              v154 = 1;
              v149 = 0;
              while ( v153 != -4096 )
              {
                if ( !v149 && v153 == -8192 )
                  v149 = v73;
                v152 = v150 & (v154 + v152);
                v73 = &v151[3 * v152];
                v153 = *v73;
                if ( v58 == *v73 )
                  goto LABEL_220;
                ++v154;
              }
            }
            if ( v149 )
              v73 = v149;
LABEL_220:
            v72 = v268;
LABEL_105:
            LODWORD(v268) = (2 * (v72 >> 1) + 2) | v72 & 1;
            if ( *v73 != -4096 )
              --HIDWORD(v268);
            *v73 = v58;
            v57 = v73 + 1;
            v73[1] = 0xFFFFFFFF00000000LL;
            v73[2] = 0;
            goto LABEL_108;
          }
          v59 = v270;
LABEL_102:
          v75 = 3 * v59;
          goto LABEL_103;
        }
LABEL_61:
        v57 = v55 + 1;
        if ( *((_BYTE *)v55 + 8) )
          goto LABEL_62;
LABEL_108:
        v76 = *(_BYTE *)(v58 - 16);
        if ( (v76 & 2) != 0 )
        {
          v77 = *(__int64 **)(v58 - 32);
          v78 = *(unsigned int *)(v58 - 24);
        }
        else
        {
          v78 = (*(_WORD *)(v58 - 16) >> 6) & 0xF;
          v77 = (__int64 *)(v58 - 8LL * ((v76 >> 2) & 0xF) - 16);
        }
        v79 = 8 * v78;
        v80 = &v77[v78];
        v81 = (8 * v78) >> 3;
        v82 = v79 >> 5;
        v254 = v80;
        if ( v82 )
        {
          v83 = v268 & 1;
          v84 = &v77[4 * v82];
          while ( 1 )
          {
            v85 = *v77;
            if ( v83 )
            {
              v86 = 31;
              v87 = &v269;
            }
            else
            {
              v139 = v270;
              v87 = v269;
              if ( !v270 )
                goto LABEL_194;
              v86 = v270 - 1;
            }
            v88 = v86 & (((unsigned int)v85 >> 9) ^ ((unsigned int)v85 >> 4));
            v89 = &v87[3 * v88];
            v90 = *v89;
            if ( *v89 == v85 )
              goto LABEL_115;
            v141 = 1;
            while ( v90 != -4096 )
            {
              v155 = v141 + 1;
              v88 = v86 & (v141 + v88);
              v89 = &v87[3 * v88];
              v90 = *v89;
              if ( v85 == *v89 )
                goto LABEL_115;
              v141 = v155;
            }
            if ( v83 )
            {
              v140 = 96;
              goto LABEL_195;
            }
            v139 = v270;
LABEL_194:
            v140 = 3 * v139;
LABEL_195:
            v89 = &v87[v140];
LABEL_115:
            if ( v83 )
            {
              v91 = v87 + 96;
              if ( v87 + 96 != v89 && *((_BYTE *)v89 + 8) )
                goto LABEL_123;
              v92 = v77[1];
              v93 = 32;
LABEL_119:
              v94 = v93 - 1;
              v95 = (v93 - 1) & (((unsigned int)v92 >> 9) ^ ((unsigned int)v92 >> 4));
              v96 = &v87[3 * v95];
              v97 = *v96;
              if ( *v96 == v92 )
              {
LABEL_120:
                if ( v96 != v91 && *((_BYTE *)v96 + 8) )
                {
                  ++v77;
                  goto LABEL_123;
                }
              }
              else
              {
                v120 = 1;
                while ( v97 != -4096 )
                {
                  v95 = v94 & (v120 + v95);
                  v248 = v120 + 1;
                  v96 = &v87[3 * v95];
                  v97 = *v96;
                  if ( *v96 == v92 )
                    goto LABEL_120;
                  v120 = v248;
                }
              }
              v121 = v77[2];
              if ( v83 )
                goto LABEL_155;
              goto LABEL_186;
            }
            v93 = v270;
            v91 = &v87[3 * v270];
            if ( v89 != v91 && *((_BYTE *)v89 + 8) )
              goto LABEL_123;
            v92 = v77[1];
            if ( v270 )
              goto LABEL_119;
            v121 = v77[2];
LABEL_186:
            if ( !v270 )
            {
              v126 = v77[3];
              goto LABEL_191;
            }
            v94 = v93 - 1;
LABEL_155:
            v122 = v94 & (((unsigned int)v121 >> 9) ^ ((unsigned int)v121 >> 4));
            v123 = &v87[3 * v122];
            v124 = *v123;
            if ( *v123 == v121 )
            {
LABEL_156:
              if ( v123 != v91 && *((_BYTE *)v123 + 8) )
              {
                v77 += 2;
                goto LABEL_123;
              }
            }
            else
            {
              v125 = 1;
              while ( v124 != -4096 )
              {
                v122 = v94 & (v125 + v122);
                v249 = v125 + 1;
                v123 = &v87[3 * v122];
                v124 = *v123;
                if ( *v123 == v121 )
                  goto LABEL_156;
                v125 = v249;
              }
            }
            v126 = v77[3];
            if ( v83 )
              goto LABEL_162;
LABEL_191:
            if ( v270 )
            {
              v94 = v93 - 1;
LABEL_162:
              v127 = v94 & (((unsigned int)v126 >> 9) ^ ((unsigned int)v126 >> 4));
              v128 = &v87[3 * v127];
              v129 = *v128;
              if ( v126 == *v128 )
              {
LABEL_163:
                if ( v128 != v91 && *((_BYTE *)v128 + 8) )
                {
                  v77 += 3;
                  goto LABEL_123;
                }
              }
              else
              {
                v130 = 1;
                while ( v129 != -4096 )
                {
                  v142 = v130 + 1;
                  v127 = v94 & (v130 + v127);
                  v128 = &v87[3 * v127];
                  v129 = *v128;
                  if ( *v128 == v126 )
                    goto LABEL_163;
                  v130 = v142;
                }
              }
            }
            v77 += 4;
            if ( v84 == v77 )
            {
              v81 = v254 - v77;
              break;
            }
          }
        }
        if ( v81 == 2 )
        {
          v131 = v268 & 1;
LABEL_248:
          v165 = *v77;
          if ( v131 )
          {
            v166 = 31;
            v167 = &v269;
          }
          else
          {
            v172 = v270;
            v167 = v269;
            if ( !v270 )
              goto LABEL_382;
            v166 = v270 - 1;
          }
          v168 = v166 & (((unsigned int)v165 >> 9) ^ ((unsigned int)v165 >> 4));
          v169 = &v167[3 * v168];
          v170 = *v169;
          if ( *v169 == v165 )
            goto LABEL_251;
          v228 = 1;
          while ( v170 != -4096 )
          {
            v240 = v228 + 1;
            v241 = v166 & (v168 + v228);
            v168 = v241;
            v169 = &v167[3 * v241];
            v170 = *v169;
            if ( v165 == *v169 )
              goto LABEL_251;
            v228 = v240;
          }
          if ( v131 )
          {
            v226 = 96;
            goto LABEL_383;
          }
          v172 = v270;
LABEL_382:
          v226 = 3 * v172;
LABEL_383:
          v169 = &v167[v226];
LABEL_251:
          v171 = 96;
          if ( !v131 )
            v171 = 3LL * v270;
          if ( v169 != &v167[v171] && *((_BYTE *)v169 + 8) )
            goto LABEL_123;
          ++v77;
LABEL_174:
          v132 = *v77;
          if ( v131 )
          {
            v133 = 31;
            v134 = &v269;
          }
          else
          {
            v156 = v270;
            v134 = v269;
            if ( !v270 )
              goto LABEL_347;
            v133 = v270 - 1;
          }
          v135 = v133 & (((unsigned int)v132 >> 9) ^ ((unsigned int)v132 >> 4));
          v136 = &v134[3 * v135];
          v137 = *v136;
          if ( *v136 == v132 )
            goto LABEL_177;
          v227 = 1;
          while ( v137 != -4096 )
          {
            v238 = v227 + 1;
            v239 = v133 & (v135 + v227);
            v135 = v239;
            v136 = &v134[3 * v239];
            v137 = *v136;
            if ( v132 == *v136 )
              goto LABEL_177;
            v227 = v238;
          }
          if ( v131 )
          {
            v209 = 96;
            goto LABEL_348;
          }
          v156 = v270;
LABEL_347:
          v209 = 3 * v156;
LABEL_348:
          v136 = &v134[v209];
LABEL_177:
          v138 = 96;
          if ( !v131 )
            v138 = 3LL * v270;
          if ( v136 == &v134[v138] || !*((_BYTE *)v136 + 8) )
            goto LABEL_62;
          goto LABEL_123;
        }
        if ( v81 != 3 )
        {
          if ( v81 == 1 )
          {
            v131 = v268 & 1;
            goto LABEL_174;
          }
LABEL_62:
          if ( v256 == ++v51 )
            goto LABEL_125;
          continue;
        }
        break;
      }
      v157 = *v77;
      v131 = v268 & 1;
      if ( (v268 & 1) == 0 )
      {
        v158 = v270;
        v159 = v269;
        if ( v270 )
        {
          v160 = v270 - 1;
          goto LABEL_242;
        }
LABEL_399:
        v229 = 3 * v158;
        goto LABEL_400;
      }
      v160 = 31;
      v159 = &v269;
LABEL_242:
      v161 = v160 & (((unsigned int)v157 >> 9) ^ ((unsigned int)v157 >> 4));
      v162 = &v159[3 * v161];
      v163 = *v162;
      if ( v157 != *v162 )
      {
        v231 = 1;
        while ( v163 != -4096 )
        {
          v242 = v231 + 1;
          v243 = v160 & (v161 + v231);
          v161 = v243;
          v162 = &v159[3 * v243];
          v163 = *v162;
          if ( v157 == *v162 )
            goto LABEL_243;
          v231 = v242;
        }
        if ( !v131 )
        {
          v158 = v270;
          goto LABEL_399;
        }
        v229 = 96;
LABEL_400:
        v162 = &v159[v229];
      }
LABEL_243:
      v164 = 96;
      if ( !v131 )
        v164 = 3LL * v270;
      if ( v162 == &v159[v164] || !*((_BYTE *)v162 + 8) )
      {
        ++v77;
        goto LABEL_248;
      }
LABEL_123:
      if ( v254 == v77 )
        goto LABEL_62;
      *v57 = 1;
      ++v51;
      v252 = v250;
      if ( v256 != v51 )
        continue;
      break;
    }
LABEL_125:
    if ( v252 )
      continue;
    break;
  }
  v263 = (unsigned __int64)v265;
  v264 = 0x1000000000LL;
  v247 = &v271[(unsigned int)v272];
  if ( v247 != v271 )
  {
    v255 = v271;
    while ( 1 )
    {
      v253 = *v255;
      v98 = v268 & 1;
      if ( (v268 & 1) != 0 )
      {
        v99 = &v269;
        v100 = 31;
      }
      else
      {
        v194 = v270;
        v99 = v269;
        v100 = v270 - 1;
        if ( !v270 )
        {
          v195 = v268;
          ++v267;
          v196 = 0;
          v197 = ((unsigned int)v268 >> 1) + 1;
          goto LABEL_300;
        }
      }
      v101 = v100 & (((unsigned int)v253 >> 9) ^ ((unsigned int)v253 >> 4));
      v102 = &v99[3 * v101];
      v103 = *v102;
      if ( *v102 != v253 )
        break;
LABEL_131:
      if ( *((_BYTE *)v102 + 8) )
      {
        v104 = (__m128i *)v102[2];
        v251 = v104;
        if ( v104 )
        {
          v258 = (__m128i *)v102[2];
          v105 = v104;
          v102[2] = 0;
        }
        else
        {
          v98 = *v255;
          sub_B9C990(&v258, v253);
          v105 = v258;
        }
        v106 = v105[-1].m128i_u8[0];
        v107 = (v106 & 2) != 0;
        if ( (v106 & 2) != 0 )
          v108 = v105[-2].m128i_u32[2];
        else
          v108 = ((unsigned __int16)v105[-1].m128i_i16[0] >> 6) & 0xF;
        if ( v108 )
        {
          v257 = v108;
          v109 = 0;
          while ( 2 )
          {
            if ( v107 )
              v110 = (__int8 *)v105[-2].m128i_i64[0];
            else
              v110 = &v105->m128i_i8[-16 - 8LL * ((v106 >> 2) & 0xF)];
            v111 = *(_QWORD *)&v110[8 * v109];
            if ( !v111 )
              goto LABEL_148;
            v98 = *(_QWORD *)&v110[8 * v109];
            v112 = (unsigned __int8 *)sub_FC6E30((__int64)a1, (_QWORD *)v98);
            v261 = v113;
            v114 = (_BYTE)v113 == 0;
            v260 = v112;
            v115 = v112;
            if ( !v114 )
            {
LABEL_146:
              if ( (unsigned __int8 *)v111 != v115 )
              {
                v98 = (unsigned int)v109;
                sub_BA6610(v105, v109, v115);
              }
LABEL_148:
              if ( v257 == ++v109 )
              {
                v105 = v258;
                goto LABEL_282;
              }
              v106 = v105[-1].m128i_u8[0];
              v107 = (v106 & 2) != 0;
              continue;
            }
            break;
          }
          v115 = (unsigned __int8 *)v111;
          if ( (v268 & 1) != 0 )
          {
            v98 = (__int64)&v269;
            v116 = 31;
            goto LABEL_143;
          }
          v180 = v270;
          v98 = (__int64)v269;
          if ( v270 )
          {
            v116 = v270 - 1;
LABEL_143:
            v117 = v116 & (((unsigned int)v111 >> 9) ^ ((unsigned int)v111 >> 4));
            v118 = v98 + 24LL * v117;
            v119 = *(_QWORD *)v118;
            if ( *(_QWORD *)v118 == v111 )
            {
LABEL_144:
              if ( *(_BYTE *)(v118 + 8) )
              {
                v115 = *(unsigned __int8 **)(v118 + 16);
                if ( !v115 )
                {
                  v98 = v111;
                  v244 = v118;
                  sub_B9C990(&v259, v111);
                  v115 = v259;
                  v259 = 0;
                  v200 = *(_QWORD *)(v244 + 16);
                  *(_QWORD *)(v244 + 16) = v115;
                  if ( v200 )
                  {
                    sub_BA65D0(v200, v111, (__int64)v115, v244, v199);
                    v203 = v244;
                    if ( v259 )
                    {
                      sub_BA65D0((__int64)v259, v111, v201, v244, v202);
                      v203 = v244;
                    }
                    v115 = *(unsigned __int8 **)(v203 + 16);
                  }
                }
              }
              goto LABEL_146;
            }
            v205 = 1;
            while ( v119 != -4096 )
            {
              v232 = v205 + 1;
              v117 = v116 & (v117 + v205);
              v118 = v98 + 24LL * v117;
              v119 = *(_QWORD *)v118;
              if ( v111 == *(_QWORD *)v118 )
                goto LABEL_144;
              v205 = v232;
            }
            if ( (v268 & 1) != 0 )
            {
              v204 = 768;
              goto LABEL_314;
            }
            v180 = v270;
          }
          v204 = 24 * v180;
LABEL_314:
          v118 = v98 + v204;
          goto LABEL_144;
        }
LABEL_282:
        v258 = 0;
        v181 = sub_BA6670(v105, v98);
        v182 = (_QWORD *)v253;
        v183 = v181;
        sub_FC80D0(*a1, v253, (__int64)v181);
        if ( v251 )
        {
          v188 = (unsigned int)v264;
          v185 = HIDWORD(v264);
          v189 = (unsigned int)v264 + 1LL;
          if ( v189 > HIDWORD(v264) )
          {
            v182 = v265;
            sub_C8D5F0((__int64)&v263, v265, v189, 8u, v186, v187);
            v188 = (unsigned int)v264;
          }
          v184 = v263;
          *(_QWORD *)(v263 + 8 * v188) = v183;
          LODWORD(v264) = v264 + 1;
        }
        if ( v258 )
          sub_BA65D0((__int64)v258, (__int64)v182, v184, v185, v186);
        goto LABEL_288;
      }
LABEL_306:
      v182 = (_QWORD *)v253;
      sub_FC80D0(*a1, v253, v253);
LABEL_288:
      if ( v247 == ++v255 )
      {
        v190 = (__int64 *)v263;
        v191 = (__int64 *)(v263 + 8LL * (unsigned int)v264);
        if ( (__int64 *)v263 != v191 )
        {
          do
          {
            v192 = *v190;
            if ( (*(_BYTE *)(*v190 + 1) & 0x7F) == 2 || *(_DWORD *)(v192 - 8) )
              sub_B931A0(v192, (__int64)v182, v184, v185, v186);
            ++v190;
          }
          while ( v191 != v190 );
          v191 = (__int64 *)v263;
        }
        if ( v191 != v265 )
          _libc_free(v191, v182);
        goto LABEL_297;
      }
    }
    v207 = 1;
    v196 = 0;
    while ( v103 != -4096 )
    {
      if ( v103 == -8192 && !v196 )
        v196 = v102;
      v101 = v100 & (v207 + v101);
      v102 = &v99[3 * v101];
      v103 = *v102;
      if ( v253 == *v102 )
        goto LABEL_131;
      ++v207;
    }
    v195 = v268;
    v198 = 96;
    v194 = 32;
    if ( !v196 )
      v196 = v102;
    ++v267;
    v197 = ((unsigned int)v268 >> 1) + 1;
    if ( !(_BYTE)v98 )
    {
      v194 = v270;
LABEL_300:
      v198 = 3 * v194;
    }
    if ( v198 <= 4 * v197 )
    {
      sub_FC7090((__int64)&v267, 2 * v194);
      if ( (v268 & 1) != 0 )
      {
        v210 = &v269;
        v211 = 31;
      }
      else
      {
        v210 = v269;
        if ( !v270 )
        {
LABEL_495:
          LODWORD(v268) = (2 * ((unsigned int)v268 >> 1) + 2) | v268 & 1;
          BUG();
        }
        v211 = v270 - 1;
      }
      v212 = v211 & (((unsigned int)v253 >> 9) ^ ((unsigned int)v253 >> 4));
      v196 = &v210[3 * v212];
      v195 = v268;
      v213 = *v196;
      if ( v253 != *v196 )
      {
        v214 = 1;
        v215 = 0;
        while ( v213 != -4096 )
        {
          if ( !v215 && v213 == -8192 )
            v215 = v196;
          v212 = v211 & (v214 + v212);
          v196 = &v210[3 * v212];
          v213 = *v196;
          if ( v253 == *v196 )
            goto LABEL_371;
          ++v214;
        }
        if ( v215 )
        {
          v196 = v215;
          v195 = v268;
        }
        else
        {
LABEL_371:
          v195 = v268;
        }
      }
    }
    else if ( v194 - HIDWORD(v268) - v197 <= v194 >> 3 )
    {
      sub_FC7090((__int64)&v267, v194);
      if ( (v268 & 1) != 0 )
      {
        v218 = &v269;
        v219 = 31;
      }
      else
      {
        v218 = v269;
        if ( !v270 )
          goto LABEL_495;
        v219 = v270 - 1;
      }
      v220 = v219 & (((unsigned int)v253 >> 9) ^ ((unsigned int)v253 >> 4));
      v196 = &v218[3 * v220];
      v195 = v268;
      v221 = *v196;
      if ( v253 != *v196 )
      {
        v222 = 1;
        v223 = 0;
        while ( v221 != -4096 )
        {
          if ( !v223 && v221 == -8192 )
            v223 = v196;
          v220 = v219 & (v222 + v220);
          v196 = &v218[3 * v220];
          v221 = *v196;
          if ( v253 == *v196 )
            goto LABEL_371;
          ++v222;
        }
        if ( v223 )
          v196 = v223;
        goto LABEL_371;
      }
    }
    LODWORD(v268) = (2 * (v195 >> 1) + 2) | v195 & 1;
    if ( *v196 != -4096 )
      --HIDWORD(v268);
    v196[2] = 0;
    *v196 = v253;
    v196[1] = 0xFFFFFFFF00000000LL;
    goto LABEL_306;
  }
LABEL_297:
  v30 = (__int64)v245;
  v193 = sub_FC6E30((__int64)a1, v245);
  v65 = v271;
  v263 = (unsigned __int64)v193;
  v264 = v15;
  v245 = v193;
LABEL_89:
  if ( v65 != (__int64 *)v273 )
    _libc_free(v65, v30);
  if ( (v268 & 1) != 0 )
  {
    v69 = &v271;
    v68 = &v269;
    goto LABEL_94;
  }
  v15 = v270;
  v68 = (__int64 **)v269;
  v30 = 24LL * v270;
  if ( !v270 )
    goto LABEL_278;
  v69 = (__int64 **)((char *)v269 + v30);
  if ( (_QWORD *)((char *)v269 + v30) == v269 )
    goto LABEL_278;
  do
  {
LABEL_94:
    if ( *v68 != (__int64 *)-8192LL && *v68 != (__int64 *)-4096LL )
    {
      v70 = (__int64)v68[2];
      if ( v70 )
        sub_BA65D0(v70, v30, v15, v34, v14);
    }
    v68 += 3;
  }
  while ( v69 != v68 );
  if ( (v268 & 1) == 0 )
  {
    v68 = (__int64 **)v269;
    v30 = 24LL * v270;
LABEL_278:
    sub_C7D6A0((__int64)v68, v30, 8);
  }
  return v245;
}
