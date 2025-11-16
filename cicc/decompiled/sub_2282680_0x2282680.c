// Function: sub_2282680
// Address: 0x2282680
//
__int64 __fastcall sub_2282680(__int64 a1, __int64 **a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rcx
  _QWORD *v8; // r12
  _QWORD *v9; // rax
  _QWORD *v10; // r15
  _QWORD *i; // r14
  __int64 v12; // rsi
  __int64 v13; // r13
  _QWORD *v14; // rsi
  __int64 v15; // rdi
  int v16; // r9d
  unsigned int v17; // edx
  __int64 *v18; // rax
  __int64 v19; // r10
  __int64 v20; // rdx
  __int64 v21; // r13
  _QWORD *v22; // rax
  __int64 v23; // rax
  __int64 v24; // rcx
  char *v25; // rdx
  _QWORD *v26; // r12
  _QWORD *v27; // rax
  _QWORD *v28; // r9
  _QWORD *v29; // r15
  _QWORD *v30; // r14
  __int64 v31; // rax
  __int64 v32; // r9
  __int64 v33; // r13
  __int64 *v34; // r12
  __int64 *v35; // r13
  _BYTE *v36; // rbx
  char *v37; // rax
  _QWORD *v38; // rax
  __int64 v39; // rax
  unsigned __int64 v40; // rdx
  __int64 v41; // rbx
  __int64 v42; // rcx
  unsigned __int64 v43; // r8
  __int64 v44; // r9
  __int64 *v45; // r12
  __int64 *j; // r15
  __int64 v47; // rdx
  __int64 *v48; // r12
  __int64 *k; // r15
  __int64 v50; // rdx
  __int64 v51; // rdx
  __int64 *v52; // r12
  __int64 *v53; // rbx
  __int64 v54; // r14
  size_t v55; // rdx
  size_t *v56; // rax
  size_t *v57; // r13
  size_t *v58; // r12
  size_t v59; // rsi
  int v60; // ecx
  __int64 v61; // rdi
  int v62; // ecx
  unsigned int v63; // edx
  __int64 *v64; // rax
  __int64 v65; // r10
  __int64 v66; // rax
  unsigned __int64 v67; // rdx
  size_t *v68; // rax
  __int64 v69; // rcx
  __int64 *v70; // r13
  __int64 v71; // rax
  int v72; // r14d
  __int64 *v73; // r12
  int v74; // esi
  __int64 v75; // r9
  unsigned int v76; // edx
  unsigned __int64 v77; // rax
  __int64 v78; // r11
  unsigned __int64 *v79; // r14
  __int64 v80; // rdx
  int v81; // ecx
  __int64 v82; // rsi
  int v83; // ecx
  unsigned int v84; // edx
  __int64 *v85; // rax
  __int64 v86; // rdi
  __int64 *v87; // rax
  __int64 *v88; // r14
  __int64 *v89; // r13
  __int64 *v90; // r12
  __int64 v91; // rdx
  int v92; // ecx
  __int64 v93; // r8
  __int64 v94; // rsi
  int v95; // ecx
  unsigned int v96; // edx
  __int64 *v97; // rax
  __int64 v98; // r10
  __int64 **v99; // r15
  __int64 v100; // rdx
  __int64 *v101; // r14
  __int64 *v102; // r12
  __int64 v103; // rax
  __int64 *v104; // r12
  __int64 v105; // r14
  int v106; // esi
  __int64 v107; // rdx
  __int64 v108; // rdi
  int v109; // esi
  unsigned int v110; // ecx
  __int64 *v111; // rax
  __int64 v112; // r8
  __int64 *v113; // r15
  char v114; // r13
  int v115; // r13d
  __int64 *v116; // r9
  int v117; // esi
  unsigned int v118; // ecx
  __int64 *v119; // rax
  __int64 v120; // r10
  __int64 v121; // rdx
  __int64 v122; // rcx
  __int64 v123; // rcx
  __int64 *v124; // rsi
  int v125; // r9d
  unsigned int v126; // edx
  __int64 *v127; // rax
  __int64 v128; // r11
  __int64 v129; // r15
  __int64 v130; // rdx
  __int64 v131; // r15
  __int64 v132; // rax
  __int64 v133; // r14
  __int64 v134; // r12
  __int64 v135; // rdx
  __int64 v136; // rdi
  __int64 v137; // r12
  unsigned int v139; // edx
  unsigned __int64 v140; // rax
  __int64 v141; // r11
  __int64 v142; // r9
  unsigned int v143; // edx
  unsigned __int64 v144; // rax
  __int64 v145; // r11
  __int64 v146; // rdx
  unsigned __int64 v147; // rax
  __int64 v148; // r11
  __int64 v149; // rdx
  char v150; // dl
  int v151; // esi
  __int64 v152; // rdi
  int v153; // esi
  unsigned int v154; // edx
  __int64 *v155; // rax
  __int64 v156; // r9
  __int64 v157; // r13
  __int64 v158; // rdx
  __int64 v159; // rcx
  __int64 v160; // r9
  __int64 v161; // rax
  unsigned __int64 v162; // rdx
  __int64 v163; // rax
  unsigned int v164; // esi
  int ii; // eax
  int v166; // r10d
  int n; // eax
  int v168; // r10d
  int kk; // eax
  int v170; // r10d
  int jj; // eax
  int v172; // r10d
  int nn; // eax
  int v174; // edi
  int v175; // eax
  int v176; // r9d
  __int64 v177; // rax
  int mm; // eax
  __int64 v179; // rax
  int m; // eax
  int v181; // r9d
  __int64 v182; // rax
  __int64 v183; // rax
  __int64 v184; // rax
  __int64 v185; // rax
  int v186; // eax
  int v187; // eax
  int v188; // eax
  __int64 v189; // rax
  unsigned __int64 v190; // rdx
  unsigned __int64 *v191; // rdi
  __int64 v192; // rax
  bool v193; // zf
  int v194; // r10d
  int v195; // r11d
  int v196; // ecx
  __int64 v197; // rax
  int v198; // eax
  int v199; // ecx
  __int64 v201; // [rsp+18h] [rbp-4C8h]
  _QWORD *v202; // [rsp+18h] [rbp-4C8h]
  __int64 v204; // [rsp+48h] [rbp-498h]
  __int64 v206; // [rsp+58h] [rbp-488h]
  __int64 v207; // [rsp+60h] [rbp-480h]
  _QWORD *v209; // [rsp+70h] [rbp-470h]
  __int64 v210; // [rsp+70h] [rbp-470h]
  __int64 *v211; // [rsp+70h] [rbp-470h]
  unsigned __int64 v212; // [rsp+70h] [rbp-470h]
  __int64 v213; // [rsp+70h] [rbp-470h]
  _QWORD *dest; // [rsp+78h] [rbp-468h]
  _QWORD *desta; // [rsp+78h] [rbp-468h]
  __int64 *destb; // [rsp+78h] [rbp-468h]
  char v217; // [rsp+8Fh] [rbp-451h] BYREF
  __int64 **v218; // [rsp+90h] [rbp-450h] BYREF
  __int64 *v219; // [rsp+98h] [rbp-448h] BYREF
  _QWORD *v220; // [rsp+A0h] [rbp-440h] BYREF
  _QWORD *v221; // [rsp+A8h] [rbp-438h]
  _QWORD *v222; // [rsp+B0h] [rbp-430h]
  __int16 v223; // [rsp+B8h] [rbp-428h]
  _QWORD *v224; // [rsp+C0h] [rbp-420h] BYREF
  _QWORD *v225; // [rsp+C8h] [rbp-418h]
  _QWORD *v226; // [rsp+D0h] [rbp-410h]
  __int16 v227; // [rsp+D8h] [rbp-408h]
  _QWORD v228[6]; // [rsp+E0h] [rbp-400h] BYREF
  unsigned __int64 v229; // [rsp+110h] [rbp-3D0h] BYREF
  __int64 v230; // [rsp+118h] [rbp-3C8h]
  _QWORD v231[4]; // [rsp+120h] [rbp-3C0h] BYREF
  __int64 v232; // [rsp+140h] [rbp-3A0h] BYREF
  __int64 v233; // [rsp+148h] [rbp-398h]
  __int64 v234; // [rsp+150h] [rbp-390h]
  __int64 v235; // [rsp+158h] [rbp-388h]
  __int64 *v236; // [rsp+160h] [rbp-380h]
  __int64 v237; // [rsp+168h] [rbp-378h]
  _BYTE v238[32]; // [rsp+170h] [rbp-370h] BYREF
  __int64 v239; // [rsp+190h] [rbp-350h] BYREF
  __int64 v240; // [rsp+198h] [rbp-348h]
  __int64 v241; // [rsp+1A0h] [rbp-340h]
  __int64 v242; // [rsp+1A8h] [rbp-338h]
  __int64 *v243; // [rsp+1B0h] [rbp-330h]
  __int64 v244; // [rsp+1B8h] [rbp-328h]
  _BYTE v245[32]; // [rsp+1C0h] [rbp-320h] BYREF
  __int64 v246; // [rsp+1E0h] [rbp-300h] BYREF
  __int64 v247; // [rsp+1E8h] [rbp-2F8h]
  __int64 v248; // [rsp+1F0h] [rbp-2F0h]
  __int64 v249; // [rsp+1F8h] [rbp-2E8h]
  __int64 *v250; // [rsp+200h] [rbp-2E0h]
  __int64 v251; // [rsp+208h] [rbp-2D8h]
  _BYTE v252[32]; // [rsp+210h] [rbp-2D0h] BYREF
  __int64 v253; // [rsp+230h] [rbp-2B0h] BYREF
  __int64 v254; // [rsp+238h] [rbp-2A8h]
  __int64 v255; // [rsp+240h] [rbp-2A0h]
  __int64 v256; // [rsp+248h] [rbp-298h]
  __int64 *v257; // [rsp+250h] [rbp-290h]
  __int64 v258; // [rsp+258h] [rbp-288h]
  _BYTE v259[32]; // [rsp+260h] [rbp-280h] BYREF
  __int64 v260; // [rsp+280h] [rbp-260h] BYREF
  unsigned __int64 v261; // [rsp+288h] [rbp-258h] BYREF
  __int64 v262; // [rsp+290h] [rbp-250h]
  _QWORD *v263; // [rsp+298h] [rbp-248h]
  char v264; // [rsp+2A0h] [rbp-240h] BYREF
  __int64 v265; // [rsp+2B0h] [rbp-230h]
  char *v266; // [rsp+2B8h] [rbp-228h]
  __int64 v267; // [rsp+2C0h] [rbp-220h]
  int v268; // [rsp+2C8h] [rbp-218h]
  char v269; // [rsp+2CCh] [rbp-214h]
  char v270; // [rsp+2D0h] [rbp-210h] BYREF
  char *v271; // [rsp+2E0h] [rbp-200h] BYREF
  __int64 v272; // [rsp+2E8h] [rbp-1F8h]
  _BYTE v273[128]; // [rsp+2F0h] [rbp-1F0h] BYREF
  __int64 v274; // [rsp+370h] [rbp-170h] BYREF
  char *v275; // [rsp+378h] [rbp-168h]
  __int64 v276; // [rsp+380h] [rbp-160h]
  int v277; // [rsp+388h] [rbp-158h]
  char v278; // [rsp+38Ch] [rbp-154h]
  char v279; // [rsp+390h] [rbp-150h] BYREF
  __int64 v280; // [rsp+410h] [rbp-D0h] BYREF
  char *v281; // [rsp+418h] [rbp-C8h]
  __int64 v282; // [rsp+420h] [rbp-C0h]
  int v283; // [rsp+428h] [rbp-B8h]
  char v284; // [rsp+42Ch] [rbp-B4h]
  char v285; // [rsp+430h] [rbp-B0h] BYREF

  v7 = *(_QWORD *)(a3 + 8);
  v219 = *a2;
  v271 = v273;
  v272 = 0x1000000000LL;
  v275 = &v279;
  v218 = a2;
  v281 = &v285;
  v204 = v7;
  v207 = a5;
  v274 = 0;
  v276 = 16;
  v277 = 0;
  v278 = 1;
  v280 = 0;
  v282 = 16;
  v283 = 0;
  v284 = 1;
  v232 = 0;
  v233 = 0;
  v234 = 0;
  v8 = *(_QWORD **)(v7 + 80);
  v236 = (__int64 *)v238;
  v243 = (__int64 *)v245;
  v237 = 0x400000000LL;
  v244 = 0x400000000LL;
  v250 = (__int64 *)v252;
  v251 = 0x400000000LL;
  v258 = 0x400000000LL;
  v235 = 0;
  v239 = 0;
  v240 = 0;
  v241 = 0;
  v242 = 0;
  v246 = 0;
  v247 = 0;
  v248 = 0;
  v249 = 0;
  v253 = 0;
  v254 = 0;
  v255 = 0;
  v256 = 0;
  v257 = (__int64 *)v259;
  dest = (_QWORD *)(v7 + 72);
  v220 = (_QWORD *)(v7 + 72);
  v221 = v8;
  v222 = 0;
  v223 = 0;
  if ( (_QWORD *)(v7 + 72) != v8 )
  {
    if ( v8 )
    {
      v9 = (_QWORD *)v8[4];
      LOBYTE(v223) = 1;
      v222 = v9;
      sub_227B8F0((__int64)&v220);
      v10 = v220;
      v8 = v221;
      i = v222;
      goto LABEL_4;
    }
    goto LABEL_295;
  }
  v10 = (_QWORD *)(v7 + 72);
  i = 0;
LABEL_4:
  while ( v8 != dest )
  {
    if ( !i )
      BUG();
LABEL_6:
    if ( (unsigned __int8)(*((_BYTE *)i - 24) - 34) > 0x33u )
      goto LABEL_19;
    v12 = 0x8000000000041LL;
    if ( !_bittest64(&v12, (unsigned int)*((unsigned __int8 *)i - 24) - 34) )
      goto LABEL_19;
    v13 = *(i - 7);
    if ( v13 && !*(_BYTE *)v13 && *(_QWORD *)(v13 + 24) == i[7] )
    {
      sub_AE6EC0((__int64)&v274, *(i - 7));
      if ( !v150 || sub_B2FC80(v13) )
        goto LABEL_19;
      v151 = *(_DWORD *)(a1 + 120);
      v152 = *(_QWORD *)(a1 + 104);
      if ( v151 )
      {
        v153 = v151 - 1;
        v154 = v153 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v155 = (__int64 *)(v152 + 16LL * v154);
        v156 = *v155;
        if ( *v155 == v13 )
        {
LABEL_190:
          v157 = v155[1];
          goto LABEL_191;
        }
        v198 = 1;
        while ( v156 != -4096 )
        {
          v199 = v198 + 1;
          v154 = v153 & (v198 + v154);
          v155 = (__int64 *)(v152 + 16LL * v154);
          v156 = *v155;
          if ( v13 == *v155 )
            goto LABEL_190;
          v198 = v199;
        }
      }
      v157 = 0;
LABEL_191:
      v260 = v157;
      v202 = sub_227B070(a3 + 24, v157);
      sub_AE6EC0((__int64)&v280, v157);
      a5 = (__int64)v202;
      if ( v202 )
      {
        if ( (*(_BYTE *)v202 & 4) == 0 )
          sub_2281A50((__int64)&v232, &v260, v158, v159, (__int64)&v260, v160);
      }
      else
      {
        sub_2281A50((__int64)&v246, &v260, v158, v159, (__int64)&v260, v160);
      }
      goto LABEL_19;
    }
    v14 = i - 3;
    a5 = *(_BYTE *)(v207 + 144) & 1;
    if ( (*(_BYTE *)(v207 + 144) & 1) != 0 )
    {
      v15 = v207 + 152;
      v16 = 15;
    }
    else
    {
      v23 = *(unsigned int *)(v207 + 160);
      v15 = *(_QWORD *)(v207 + 152);
      if ( !(_DWORD)v23 )
        goto LABEL_248;
      v16 = v23 - 1;
    }
    v17 = v16 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
    v18 = (__int64 *)(v15 + 16LL * v17);
    v19 = *v18;
    if ( v14 == (_QWORD *)*v18 )
      goto LABEL_14;
    v188 = 1;
    while ( v19 != -4096 )
    {
      v196 = v188 + 1;
      v197 = v16 & (v17 + v188);
      v17 = v197;
      v18 = (__int64 *)(v15 + 16 * v197);
      v19 = *v18;
      if ( v14 == (_QWORD *)*v18 )
        goto LABEL_14;
      v188 = v196;
    }
    if ( (_BYTE)a5 )
    {
      v185 = 256;
      goto LABEL_249;
    }
    v23 = *(unsigned int *)(v207 + 160);
LABEL_248:
    v185 = 16 * v23;
LABEL_249:
    v18 = (__int64 *)(v15 + v185);
LABEL_14:
    v20 = 256;
    if ( !(_BYTE)a5 )
      v20 = 16LL * *(unsigned int *)(v207 + 160);
    if ( v18 == (__int64 *)(v15 + v20)
      || (v21 = *(_QWORD *)(v207 + 408) + 32LL * *((unsigned int *)v18 + 2),
          v21 == *(_QWORD *)(v207 + 408) + 32LL * *(unsigned int *)(v207 + 416)) )
    {
      v229 = 6;
      v230 = 0;
      v231[0] = i - 3;
      if ( i == (_QWORD *)-8168LL || i == (_QWORD *)-4072LL )
      {
        v260 = (__int64)(i - 3);
        v261 = 6;
        v262 = 0;
        v263 = i - 3;
      }
      else
      {
        sub_BD73F0((__int64)&v229);
        v261 = 6;
        v262 = 0;
        v260 = (__int64)(i - 3);
        v263 = (_QWORD *)v231[0];
        if ( v231[0] != -8192 && v231[0] != -4096 && v231[0] )
          sub_BD6050(&v261, v229 & 0xFFFFFFFFFFFFFFF8LL);
      }
      sub_2282310(v207 + 136, &v260, &v261);
      sub_D68D70(&v261);
      sub_D68D70(&v229);
      goto LABEL_19;
    }
    if ( !*(_QWORD *)(v21 + 24) )
    {
      v260 = 6;
      v261 = 0;
      v262 = (__int64)(i - 3);
      if ( i == (_QWORD *)-8168LL || i == (_QWORD *)-4072LL )
      {
        v191 = (unsigned __int64 *)(v21 + 8);
      }
      else
      {
        sub_BD73F0((__int64)&v260);
        v182 = *(_QWORD *)(v21 + 24);
        if ( v182 == v262 )
        {
LABEL_234:
          sub_D68D70(&v260);
          goto LABEL_19;
        }
        v191 = (unsigned __int64 *)(v21 + 8);
        if ( v182 != 0 && v182 != -4096 && v182 != -8192 )
          sub_BD60C0(v191);
      }
      v192 = v262;
      v193 = v262 == -4096;
      *(_QWORD *)(v21 + 24) = v262;
      if ( v192 != 0 && !v193 && v192 != -8192 )
      {
        sub_BD6050(v191, v260 & 0xFFFFFFFFFFFFFFF8LL);
        sub_D68D70(&v260);
        goto LABEL_19;
      }
      goto LABEL_234;
    }
LABEL_19:
    for ( i = (_QWORD *)i[1]; ; i = (_QWORD *)v8[4] )
    {
      v22 = v8 - 3;
      if ( !v8 )
        v22 = 0;
      if ( i != v22 + 6 )
        break;
      v8 = (_QWORD *)v8[1];
      if ( v8 == v10 )
        break;
      if ( !v8 )
        goto LABEL_295;
    }
  }
  if ( v10 != v8 && i )
    goto LABEL_6;
  v24 = v204;
  v25 = 0;
  v226 = 0;
  v227 = 0;
  v26 = *(_QWORD **)(v204 + 80);
  v224 = dest;
  v225 = v26;
  if ( dest != v26 )
  {
    if ( v26 )
    {
      v27 = (_QWORD *)v26[4];
      LOBYTE(v227) = 1;
      v226 = v27;
      sub_227B8F0((__int64)&v224);
      v28 = v224;
      v26 = v225;
      v29 = v226;
      goto LABEL_33;
    }
LABEL_295:
    BUG();
  }
  v28 = dest;
  v29 = 0;
LABEL_33:
  v201 = a3;
  v30 = v28;
LABEL_34:
  if ( dest != v26 )
  {
    if ( !v29 )
      BUG();
LABEL_36:
    v31 = 32LL * (*((_DWORD *)v29 - 5) & 0x7FFFFFF);
    if ( (*((_BYTE *)v29 - 17) & 0x40) != 0 )
    {
      v32 = *(v29 - 4);
      v33 = v32 + v31;
    }
    else
    {
      v33 = (__int64)(v29 - 3);
      v32 = (__int64)&v29[v31 / 0xFFFFFFFFFFFFFFF8LL - 3];
    }
    if ( v32 == v33 )
      goto LABEL_48;
    v209 = v26;
    v34 = (__int64 *)v33;
    v35 = (__int64 *)v32;
    while ( 1 )
    {
      v36 = (_BYTE *)*v35;
      if ( *(_BYTE *)*v35 > 0x15u )
        goto LABEL_46;
      if ( !v278 )
        goto LABEL_55;
      v37 = v275;
      v24 = HIDWORD(v276);
      v25 = &v275[8 * HIDWORD(v276)];
      if ( v275 != v25 )
      {
        while ( v36 != *(_BYTE **)v37 )
        {
          v37 += 8;
          if ( v25 == v37 )
            goto LABEL_60;
        }
LABEL_46:
        v35 += 4;
        if ( v34 == v35 )
          goto LABEL_47;
        continue;
      }
LABEL_60:
      if ( HIDWORD(v276) < (unsigned int)v276 )
      {
        ++HIDWORD(v276);
        *(_QWORD *)v25 = v36;
        ++v274;
      }
      else
      {
LABEL_55:
        sub_C8CC70((__int64)&v274, *v35, (__int64)v25, v24, a5, v32);
        if ( !(_BYTE)v25 )
          goto LABEL_46;
      }
      v39 = (unsigned int)v272;
      v24 = HIDWORD(v272);
      v40 = (unsigned int)v272 + 1LL;
      if ( v40 > HIDWORD(v272) )
      {
        sub_C8D5F0((__int64)&v271, v273, v40, 8u, a5, v32);
        v39 = (unsigned int)v272;
      }
      v25 = v271;
      v35 += 4;
      *(_QWORD *)&v271[8 * v39] = v36;
      LODWORD(v272) = v272 + 1;
      if ( v34 == v35 )
      {
LABEL_47:
        v26 = v209;
LABEL_48:
        v29 = (_QWORD *)v29[1];
        v25 = 0;
        while ( 1 )
        {
          v38 = v26 - 3;
          if ( !v26 )
            v38 = 0;
          if ( v29 != v38 + 6 )
            goto LABEL_34;
          v26 = (_QWORD *)v26[1];
          if ( v26 == v30 )
            goto LABEL_34;
          if ( !v26 )
            goto LABEL_295;
          v29 = (_QWORD *)v26[4];
        }
      }
    }
  }
  if ( v30 != dest && v29 )
    goto LABEL_36;
  v41 = v201;
  v228[0] = a1;
  v228[3] = &v253;
  v228[4] = &v239;
  v228[1] = v201;
  v228[2] = &v280;
  sub_D24710((__int64)&v271, (__int64)&v274, (void (__fastcall *)(__int64, __int64))sub_2281F80, (__int64)v228, a5);
  v45 = &v257[(unsigned int)v258];
  for ( j = v257; v45 != j; ++j )
  {
    v47 = *j;
    sub_D25720((__int64)v219, v201, v47);
  }
  v48 = &v250[(unsigned int)v251];
  for ( k = v250; v48 != k; ++k )
  {
    v50 = *k;
    sub_D25720((__int64)v219, v201, v50);
  }
  v51 = *(_QWORD *)(a1 + 640);
  if ( v51 != v51 + 8LL * *(unsigned int *)(a1 + 648) )
  {
    v52 = (__int64 *)(v51 + 8LL * *(unsigned int *)(a1 + 648));
    v53 = *(__int64 **)(a1 + 640);
    do
    {
      while ( 1 )
      {
        v54 = *v53;
        if ( !(unsigned __int8)sub_B19060((__int64)&v274, *v53, v51, v42) )
          break;
        if ( v52 == ++v53 )
          goto LABEL_74;
      }
      ++v53;
      sub_2281E00(v228, v54);
    }
    while ( v52 != v53 );
LABEL_74:
    v41 = v201;
  }
  v55 = *(unsigned int *)(v41 + 32);
  v229 = (unsigned __int64)v231;
  v230 = 0x400000000LL;
  v56 = *(size_t **)(v41 + 24);
  v57 = &v56[v55];
  if ( v56 == v57 )
    goto LABEL_185;
  while ( 1 )
  {
    v55 = *v56;
    v58 = v56;
    v42 = *v56 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v42 )
    {
      if ( *(_QWORD *)v42 )
        break;
    }
    if ( v57 == ++v56 )
      goto LABEL_185;
  }
  if ( v57 == v56 )
  {
LABEL_185:
    v88 = v231;
    desta = v231;
  }
  else
  {
    v59 = *v56;
    do
    {
      if ( !(unsigned __int8)sub_B19060((__int64)&v280, v59 & 0xFFFFFFFFFFFFFFF8LL, v55, v42) )
      {
        v60 = *(_DWORD *)(a1 + 328);
        v61 = *(_QWORD *)(a1 + 312);
        v43 = *v58 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v60 )
          goto LABEL_294;
        v62 = v60 - 1;
        v63 = v62 & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
        v64 = (__int64 *)(v61 + 16LL * v63);
        v65 = *v64;
        if ( v43 != *v64 )
        {
          for ( m = 1; ; m = v181 )
          {
            if ( v65 == -4096 )
              goto LABEL_294;
            v181 = m + 1;
            v63 = v62 & (m + v63);
            v64 = (__int64 *)(v61 + 16LL * v63);
            v65 = *v64;
            if ( v43 == *v64 )
              break;
          }
        }
        v44 = v64[1];
        if ( *(__int64 **)v44 == v219 && (*v58 & 4) != 0 )
        {
          if ( v218 == (__int64 **)v44 )
          {
            v213 = v64[1];
            v189 = sub_D27A10(v219, v41, *v58 & 0xFFFFFFFFFFFFFFF8LL);
            v261 = v190;
            v260 = v189;
            v218 = (__int64 **)sub_2280320((__int64)&v260, a1, v213, a4, (__int64 *)v207);
          }
          else
          {
            sub_D23F80((__int64)v219, v41, *v58 & 0xFFFFFFFFFFFFFFF8LL);
          }
          v43 = *v58 & 0xFFFFFFFFFFFFFFF8LL;
        }
        v66 = (unsigned int)v230;
        v42 = HIDWORD(v230);
        v67 = (unsigned int)v230 + 1LL;
        if ( v67 > HIDWORD(v230) )
        {
          v212 = v43;
          sub_C8D5F0((__int64)&v229, v231, v67, 8u, v43, v44);
          v66 = (unsigned int)v230;
          v43 = v212;
        }
        *(_QWORD *)(v229 + 8 * v66) = v43;
        LODWORD(v230) = v230 + 1;
      }
      v68 = v58 + 1;
      if ( v57 == v58 + 1 )
        break;
      while ( 1 )
      {
        v59 = *v68;
        v58 = v68;
        v55 = *v68 & 0xFFFFFFFFFFFFFFF8LL;
        if ( v55 )
        {
          if ( *(_QWORD *)v55 )
            break;
        }
        if ( v57 == ++v68 )
          goto LABEL_92;
      }
    }
    while ( v68 != v57 );
LABEL_92:
    v55 = (size_t)&v219;
    v262 = v41;
    v261 = (unsigned __int64)&v219;
    v69 = 8LL * (unsigned int)v230;
    desta = (_QWORD *)v229;
    v70 = (__int64 *)(v229 + v69);
    v42 = v69 >> 5;
    v260 = a1;
    v71 = (8LL * (unsigned int)v230) >> 3;
    if ( v42 )
    {
      v72 = *(_DWORD *)(a1 + 328);
      v73 = (__int64 *)v229;
      v43 = *(_QWORD *)(a1 + 312);
      v74 = v72 - 1;
      do
      {
        v75 = *v73;
        if ( !v72 )
          BUG();
        v76 = v74 & (((unsigned int)v75 >> 9) ^ ((unsigned int)v75 >> 4));
        v77 = v43 + 16LL * v76;
        v78 = *(_QWORD *)v77;
        if ( *(_QWORD *)v77 != v75 )
        {
          for ( n = 1; ; n = v168 )
          {
            if ( v78 == -4096 )
              goto LABEL_294;
            v168 = n + 1;
            v76 = v74 & (n + v76);
            v77 = v43 + 16LL * v76;
            v78 = *(_QWORD *)v77;
            if ( v75 == *(_QWORD *)v77 )
              break;
          }
        }
        if ( v219 != **(__int64 ***)(v77 + 8) )
          goto LABEL_97;
        v75 = v73[1];
        v139 = v74 & (((unsigned int)v75 >> 9) ^ ((unsigned int)v75 >> 4));
        v140 = v43 + 16LL * v139;
        v141 = *(_QWORD *)v140;
        if ( v75 != *(_QWORD *)v140 )
        {
          for ( ii = 1; ; ii = v166 )
          {
            if ( v141 == -4096 )
              goto LABEL_294;
            v166 = ii + 1;
            v139 = v74 & (ii + v139);
            v140 = v43 + 16LL * v139;
            v141 = *(_QWORD *)v140;
            if ( *(_QWORD *)v140 == v75 )
              break;
          }
        }
        if ( **(__int64 ***)(v140 + 8) != v219 )
        {
          ++v73;
LABEL_97:
          sub_D23FE0((__int64)v219, v41, v75);
          goto LABEL_98;
        }
        v142 = v73[2];
        v143 = v74 & (((unsigned int)v142 >> 9) ^ ((unsigned int)v142 >> 4));
        v144 = v43 + 16LL * v143;
        v145 = *(_QWORD *)v144;
        if ( v142 != *(_QWORD *)v144 )
        {
          for ( jj = 1; ; jj = v172 )
          {
            if ( v145 == -4096 )
              goto LABEL_294;
            v172 = jj + 1;
            v143 = v74 & (jj + v143);
            v144 = v43 + 16LL * v143;
            v145 = *(_QWORD *)v144;
            if ( *(_QWORD *)v144 == v142 )
              break;
          }
        }
        if ( **(__int64 ***)(v144 + 8) != v219 )
        {
          v146 = v73[2];
          v73 += 2;
          sub_D23FE0((__int64)v219, v41, v146);
          goto LABEL_98;
        }
        v44 = v73[3];
        v55 = v74 & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
        v147 = v43 + 16 * v55;
        v148 = *(_QWORD *)v147;
        if ( v44 != *(_QWORD *)v147 )
        {
          for ( kk = 1; ; kk = v170 )
          {
            if ( v148 == -4096 )
              goto LABEL_294;
            v170 = kk + 1;
            v55 = v74 & (unsigned int)(kk + v55);
            v147 = v43 + 16LL * (unsigned int)v55;
            v148 = *(_QWORD *)v147;
            if ( *(_QWORD *)v147 == v44 )
              break;
          }
        }
        if ( **(__int64 ***)(v147 + 8) != v219 )
        {
          v149 = v73[3];
          v73 += 3;
          sub_D23FE0((__int64)v219, v41, v149);
          goto LABEL_98;
        }
        v73 += 4;
        --v42;
      }
      while ( v42 );
      v71 = v70 - v73;
    }
    else
    {
      v73 = (__int64 *)v229;
    }
    if ( v71 == 2 )
    {
LABEL_175:
      if ( !(unsigned __int8)sub_227A860((__int64)&v260, *v73) )
      {
        ++v73;
        goto LABEL_177;
      }
LABEL_98:
      if ( v70 == v73 )
      {
        v42 = v229;
        desta = (_QWORD *)v229;
        v87 = (__int64 *)(v229 + 8LL * (unsigned int)v230);
        v88 = v87;
        v55 = (char *)v87 - (char *)v70;
      }
      else
      {
        v79 = (unsigned __int64 *)(v73 + 1);
        if ( v73 + 1 != v70 )
        {
          do
          {
            while ( 1 )
            {
              v81 = *(_DWORD *)(a1 + 328);
              v43 = *v79;
              v82 = *(_QWORD *)(a1 + 312);
              if ( !v81 )
                goto LABEL_294;
              v83 = v81 - 1;
              v84 = v83 & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
              v85 = (__int64 *)(v82 + 16LL * v84);
              v86 = *v85;
              if ( *v85 != v43 )
              {
                for ( mm = 1; ; mm = v44 )
                {
                  if ( v86 == -4096 )
                    goto LABEL_294;
                  v44 = (unsigned int)(mm + 1);
                  v179 = v83 & (v84 + mm);
                  v84 = v179;
                  v85 = (__int64 *)(v82 + 16 * v179);
                  v86 = *v85;
                  if ( v43 == *v85 )
                    break;
                }
              }
              if ( *(__int64 **)v85[1] == v219 )
                break;
              v80 = *v79++;
              sub_D23FE0((__int64)v219, v41, v80);
              if ( v70 == (__int64 *)v79 )
                goto LABEL_106;
            }
            ++v79;
            *v73++ = v43;
          }
          while ( v70 != (__int64 *)v79 );
        }
LABEL_106:
        v42 = v229;
        desta = (_QWORD *)v229;
        v87 = (__int64 *)(v229 + 8LL * (unsigned int)v230);
        v55 = (char *)v87 - (char *)v70;
        v88 = (__int64 *)((char *)v73 + (char *)v87 - (char *)v70);
      }
LABEL_107:
      if ( v87 != v70 )
      {
        memmove(v73, v70, v55);
        desta = (_QWORD *)v229;
      }
    }
    else
    {
      if ( v71 == 3 )
      {
        if ( !(unsigned __int8)sub_227A860((__int64)&v260, *v73) )
        {
          ++v73;
          goto LABEL_175;
        }
        goto LABEL_98;
      }
      v88 = v70;
      if ( v71 == 1 )
      {
LABEL_177:
        if ( (unsigned __int8)sub_227A860((__int64)&v260, *v73) )
          goto LABEL_98;
        v73 = v70;
        v42 = v229;
        desta = (_QWORD *)v229;
        v87 = (__int64 *)(v229 + 8LL * (unsigned int)v230);
        v88 = v87;
        v55 = (char *)v87 - (char *)v70;
        goto LABEL_107;
      }
    }
  }
  v89 = v243;
  v90 = &v243[(unsigned int)v244];
  LODWORD(v230) = v88 - desta;
  if ( v243 != v90 )
  {
    do
    {
      while ( 1 )
      {
        v92 = *(_DWORD *)(a1 + 328);
        v93 = *v89;
        v94 = *(_QWORD *)(a1 + 312);
        if ( !v92 )
          goto LABEL_294;
        v95 = v92 - 1;
        v96 = v95 & (((unsigned int)v93 >> 9) ^ ((unsigned int)v93 >> 4));
        v97 = (__int64 *)(v94 + 16LL * v96);
        v98 = *v97;
        if ( v93 != *v97 )
        {
          for ( nn = 1; ; nn = v174 )
          {
            if ( v98 == -4096 )
              goto LABEL_294;
            v174 = nn + 1;
            v96 = v95 & (nn + v96);
            v97 = (__int64 *)(v94 + 16LL * v96);
            v98 = *v97;
            if ( v93 == *v97 )
              break;
          }
        }
        v99 = (__int64 **)v97[1];
        if ( *v99 == v219 )
          break;
        v100 = *v89++;
        sub_D23FC0((__int64)v219, v41, v100);
        if ( v90 == v89 )
          goto LABEL_118;
      }
      v91 = *v89;
      if ( v218 == v99 )
      {
        v161 = sub_D27A10(v219, v41, v91);
        v261 = v162;
        v260 = v161;
        v218 = (__int64 **)sub_2280320((__int64)&v260, a1, (__int64)v99, a4, (__int64 *)v207);
      }
      else
      {
        sub_D23F80((__int64)v219, v41, v91);
      }
      ++v89;
    }
    while ( v90 != v89 );
  }
LABEL_118:
  v101 = v250;
  v102 = &v250[(unsigned int)v251];
  if ( v250 != v102 )
  {
    do
    {
      v103 = *v101++;
      v260 = v103;
      sub_2281A50((__int64)&v232, &v260, v55, v42, v43, v44);
    }
    while ( v102 != v101 );
  }
  v104 = v236;
  v105 = a1;
  destb = &v236[(unsigned int)v237];
  if ( v236 != destb )
  {
    do
    {
      while ( 1 )
      {
        v106 = *(_DWORD *)(v105 + 328);
        v107 = *v104;
        v108 = *(_QWORD *)(v105 + 312);
        if ( !v106 )
          goto LABEL_294;
        v109 = v106 - 1;
        v110 = v109 & (((unsigned int)v107 >> 9) ^ ((unsigned int)v107 >> 4));
        v111 = (__int64 *)(v108 + 16LL * v110);
        v112 = *v111;
        if ( v107 != *v111 )
        {
          v175 = 1;
          while ( v112 != -4096 )
          {
            v176 = v175 + 1;
            v110 = v109 & (v175 + v110);
            v111 = (__int64 *)(v108 + 16LL * v110);
            v112 = *v111;
            if ( v107 == *v111 )
              goto LABEL_126;
            v175 = v176;
          }
LABEL_294:
          BUG();
        }
LABEL_126:
        v113 = (__int64 *)v111[1];
        if ( (__int64 *)*v113 == v219 )
          break;
        sub_D23FA0((__int64)v219, v41, v107);
LABEL_123:
        if ( destb == ++v104 )
          goto LABEL_144;
      }
      v114 = *((_BYTE *)v219 + 64);
      v217 = 0;
      v115 = v114 & 1;
      if ( v115 )
      {
        v116 = v219 + 9;
        v117 = 3;
      }
      else
      {
        v164 = *((_DWORD *)v219 + 20);
        v116 = (__int64 *)v219[9];
        if ( !v164 )
          goto LABEL_236;
        v117 = v164 - 1;
      }
      v118 = v117 & (((unsigned int)v218 >> 9) ^ ((unsigned int)v218 >> 4));
      v119 = &v116[2 * v118];
      v120 = *v119;
      if ( v218 == (__int64 **)*v119 )
        goto LABEL_130;
      v187 = 1;
      while ( v120 != -4096 )
      {
        v195 = v187 + 1;
        v118 = v117 & (v187 + v118);
        v119 = &v116[2 * v118];
        v120 = *v119;
        if ( v218 == (__int64 **)*v119 )
          goto LABEL_130;
        v187 = v195;
      }
      if ( (_BYTE)v115 )
      {
        v183 = 8;
        goto LABEL_237;
      }
      v164 = *((_DWORD *)v219 + 20);
LABEL_236:
      v183 = 2LL * v164;
LABEL_237:
      v119 = &v116[v183];
LABEL_130:
      v210 = 8LL * *((int *)v119 + 2);
      v260 = (__int64)&v217;
      v261 = a4;
      v262 = v207;
      if ( (unsigned __int8)sub_D25FD0(
                              v219,
                              v41,
                              v107,
                              (void (__fastcall *)(__int64, __int64 *, signed __int64))sub_227D140,
                              (__int64)&v260) )
      {
        v218 = (__int64 **)v113;
        if ( v217 )
        {
          v177 = sub_227ED20(a4, &qword_4FDADA8, v113, v105);
          v122 = a6;
          *(_QWORD *)(v177 + 8) = a6;
        }
        v260 = 0;
        v261 = (unsigned __int64)&v264;
        v262 = 2;
        LODWORD(v263) = 0;
        BYTE4(v263) = 1;
        v265 = 0;
        v266 = &v270;
        v267 = 2;
        v268 = 0;
        v269 = 1;
        if ( !(unsigned __int8)sub_B19060((__int64)&v260, (__int64)&qword_4F82400, v121, v122) )
          sub_AE6EC0((__int64)&v260, (__int64)&unk_4F82420);
        sub_227AC60((__int64)&v260, (__int64)&qword_4FDADA8);
        sub_227C930(a4, (__int64)v218, (__int64)&v260, v123);
        sub_227AD40((__int64)&v260);
      }
      if ( (v219[8] & 1) != 0 )
      {
        v124 = v219 + 9;
        v125 = 3;
      }
      else
      {
        v163 = *((unsigned int *)v219 + 20);
        v124 = (__int64 *)v219[9];
        if ( !(_DWORD)v163 )
          goto LABEL_239;
        v125 = v163 - 1;
      }
      v126 = v125 & (((unsigned int)v218 >> 9) ^ ((unsigned int)v218 >> 4));
      v127 = &v124[2 * v126];
      v128 = *v127;
      if ( v218 != (__int64 **)*v127 )
      {
        v186 = 1;
        while ( v128 != -4096 )
        {
          v194 = v186 + 1;
          v126 = v125 & (v186 + v126);
          v127 = &v124[2 * v126];
          v128 = *v127;
          if ( v218 == (__int64 **)*v127 )
            goto LABEL_139;
          v186 = v194;
        }
        if ( (v219[8] & 1) != 0 )
        {
          v184 = 8;
        }
        else
        {
          v163 = *((unsigned int *)v219 + 20);
LABEL_239:
          v184 = 2 * v163;
        }
        v127 = &v124[v184];
      }
LABEL_139:
      v129 = 8LL * *((int *)v127 + 2);
      if ( v210 >= v129 )
        goto LABEL_123;
      sub_22801B0(*(_QWORD *)v207, (__int64 *)&v218);
      v130 = v219[1];
      v131 = v130 + v129;
      v132 = v130 + v210;
      if ( v131 == v130 + v210 )
        goto LABEL_123;
      v206 = v105;
      v133 = v131;
      v211 = v104;
      v134 = v132;
      do
      {
        v135 = *(_QWORD *)(v133 - 8);
        v136 = *(_QWORD *)v207;
        v133 -= 8;
        v260 = v135;
        sub_22801B0(v136, &v260);
      }
      while ( v134 != v133 );
      v105 = v206;
      v104 = v211 + 1;
    }
    while ( destb != v211 + 1 );
  }
LABEL_144:
  v137 = (__int64)v218;
  if ( a2 != v218 )
    *(_QWORD *)(v207 + 16) = v218;
  if ( (_QWORD *)v229 != v231 )
    _libc_free(v229);
  if ( v257 != (__int64 *)v259 )
    _libc_free((unsigned __int64)v257);
  sub_C7D6A0(v254, 8LL * (unsigned int)v256, 8);
  if ( v250 != (__int64 *)v252 )
    _libc_free((unsigned __int64)v250);
  sub_C7D6A0(v247, 8LL * (unsigned int)v249, 8);
  if ( v243 != (__int64 *)v245 )
    _libc_free((unsigned __int64)v243);
  sub_C7D6A0(v240, 8LL * (unsigned int)v242, 8);
  if ( v236 != (__int64 *)v238 )
    _libc_free((unsigned __int64)v236);
  sub_C7D6A0(v233, 8LL * (unsigned int)v235, 8);
  if ( !v284 )
    _libc_free((unsigned __int64)v281);
  if ( !v278 )
    _libc_free((unsigned __int64)v275);
  if ( v271 != v273 )
    _libc_free((unsigned __int64)v271);
  return v137;
}
