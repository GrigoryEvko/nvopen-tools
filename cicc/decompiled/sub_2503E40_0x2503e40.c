// Function: sub_2503E40
// Address: 0x2503e40
//
__int64 __fastcall sub_2503E40(__int64 a1, __int64 a2, unsigned int a3, char a4)
{
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r9
  __int64 v8; // rbx
  __int64 i; // rdx
  __int64 v10; // r12
  __int64 v11; // r13
  char v12; // r15
  __int64 v13; // rbx
  unsigned __int64 v14; // rax
  __int64 v15; // rax
  unsigned __int8 *v16; // r12
  int v17; // eax
  __int64 result; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r13
  __int64 j; // rbx
  __int64 v23; // rdi
  __int64 v24; // rax
  __int64 v25; // r8
  __int64 v26; // r9
  bool v27; // zf
  unsigned __int64 *v28; // rax
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // rax
  __int64 v32; // rbx
  __int64 v33; // r12
  __int64 v34; // rbx
  __int16 v35; // ax
  char v36; // r14
  char v37; // al
  char v38; // al
  __int16 v39; // ax
  __int64 v40; // rax
  _QWORD *v41; // r12
  unsigned int v42; // edx
  __int64 v43; // rsi
  _QWORD *v44; // rbx
  __int64 v45; // r13
  unsigned __int64 v46; // rdi
  __int64 v47; // rsi
  int v48; // eax
  __int64 v49; // r15
  __int64 *v50; // r14
  unsigned __int8 *v51; // r12
  int v52; // eax
  unsigned __int64 v53; // rax
  __int64 v54; // rbx
  __int64 v55; // r13
  int v56; // ebx
  __int64 v57; // rcx
  __int64 v58; // rdx
  __int64 *v59; // rax
  __int64 m128i_i64; // rcx
  unsigned __int8 *v61; // r15
  unsigned int v62; // ebx
  __int64 v63; // r14
  __int64 v64; // rax
  __int64 v65; // r13
  char v66; // al
  __int64 v67; // r14
  __int64 v68; // r12
  __int64 v69; // r14
  __int64 *v70; // rax
  __int64 *v71; // rdx
  __int64 v72; // r14
  const __m128i *v73; // r12
  const __m128i *v74; // r14
  unsigned int v75; // esi
  __int64 v76; // rbx
  int v77; // eax
  const __m128i *v78; // r13
  __int64 *v79; // rax
  __int64 v80; // rax
  const __m128i *v81; // r12
  unsigned __int64 v82; // rbx
  _QWORD *v83; // rsi
  __int64 *v84; // r10
  const __m128i *v85; // rbx
  const __m128i *v86; // r13
  __int64 v87; // r12
  __int64 v88; // rax
  unsigned __int64 v89; // rdx
  __int64 v90; // rax
  __int64 *v91; // rdx
  unsigned __int64 v92; // rax
  __int64 v93; // r13
  __int64 v94; // r12
  unsigned __int8 *v95; // r14
  int v96; // eax
  unsigned __int64 v97; // rax
  __int64 v98; // rax
  __int64 v99; // rdx
  int v100; // esi
  unsigned __int64 v101; // rcx
  __int64 v102; // r9
  unsigned int v103; // edx
  unsigned __int64 *v104; // rax
  unsigned __int64 v105; // rdi
  const __m128i *v106; // r14
  __int64 v107; // r13
  __int64 *v108; // rdx
  const __m128i *v109; // r12
  __int64 *v110; // rdx
  unsigned __int64 v111; // rcx
  int v112; // ebx
  __int64 v113; // r13
  int v114; // ebx
  __int64 v115; // r12
  __int64 *v116; // rax
  __int64 *v117; // rax
  __int64 v118; // rax
  unsigned __int64 v119; // rdx
  __int64 v120; // r15
  __int64 v121; // r14
  unsigned __int8 *v122; // rbx
  __int64 v123; // rax
  __int64 v124; // rax
  __m128i v125; // xmm2
  __m128i v126; // xmm3
  _QWORD *v127; // rdi
  __int64 *v128; // rax
  __m128i v129; // xmm4
  __int64 *v130; // rax
  char v131; // bl
  const __m128i *v132; // rdx
  __int64 *v133; // rax
  unsigned __int64 v134; // rdx
  int v135; // r13d
  unsigned __int64 *v136; // r10
  int v137; // edx
  __int64 v138; // r8
  __int64 *v139; // r14
  __int64 v140; // r13
  __int64 v141; // rbx
  __int64 v142; // rsi
  __int64 v143; // rdx
  __int64 v144; // rcx
  __int64 v145; // rbx
  __m128i *v146; // rdi
  __m128i *v147; // rsi
  __int64 v148; // rcx
  __int64 v149; // r8
  __int64 v150; // r9
  __int64 v151; // rdx
  __int64 v152; // r8
  __int64 v153; // r9
  unsigned __int64 v154; // rcx
  __int64 v155; // rax
  unsigned __int64 v156; // rdx
  const __m128i *v157; // rax
  unsigned __int64 v158; // rdi
  __int64 v159; // rcx
  __m128i *v160; // rsi
  const __m128i *v161; // rcx
  unsigned __int64 v162; // rcx
  __int64 v163; // rax
  unsigned __int64 v164; // rdx
  const __m128i *v165; // rax
  unsigned __int64 v166; // rdi
  __m128i *v167; // rsi
  const __m128i *v168; // rcx
  __int64 v169; // rsi
  __int64 v170; // rdx
  __int64 v171; // rcx
  unsigned __int64 v172; // rax
  char v173; // si
  __int64 v174; // [rsp-10h] [rbp-900h]
  __int64 v175; // [rsp-8h] [rbp-8F8h]
  unsigned __int64 v176; // [rsp+10h] [rbp-8E0h]
  unsigned __int64 v177; // [rsp+10h] [rbp-8E0h]
  unsigned __int64 v178; // [rsp+10h] [rbp-8E0h]
  _QWORD *v179; // [rsp+18h] [rbp-8D8h]
  __int64 v180; // [rsp+20h] [rbp-8D0h]
  char v181; // [rsp+20h] [rbp-8D0h]
  __int64 v182; // [rsp+28h] [rbp-8C8h]
  __int64 v183; // [rsp+38h] [rbp-8B8h]
  __int64 *v184; // [rsp+50h] [rbp-8A0h]
  __int64 *v185; // [rsp+58h] [rbp-898h]
  __int16 v186; // [rsp+60h] [rbp-890h]
  __int64 *v187; // [rsp+68h] [rbp-888h]
  unsigned __int64 v188; // [rsp+80h] [rbp-870h]
  __int64 *v189; // [rsp+90h] [rbp-860h]
  unsigned int v191; // [rsp+A0h] [rbp-850h]
  __int64 v194; // [rsp+A8h] [rbp-848h]
  __int64 v195; // [rsp+A8h] [rbp-848h]
  char v196; // [rsp+B2h] [rbp-83Eh] BYREF
  unsigned __int8 v197; // [rsp+B3h] [rbp-83Dh] BYREF
  unsigned int v198; // [rsp+B4h] [rbp-83Ch] BYREF
  unsigned __int64 v199; // [rsp+B8h] [rbp-838h] BYREF
  unsigned __int64 v200; // [rsp+C0h] [rbp-830h] BYREF
  __int64 v201; // [rsp+C8h] [rbp-828h] BYREF
  __int64 v202[2]; // [rsp+D0h] [rbp-820h] BYREF
  __int64 v203; // [rsp+E0h] [rbp-810h] BYREF
  _QWORD *v204; // [rsp+E8h] [rbp-808h]
  __int64 v205; // [rsp+F0h] [rbp-800h]
  unsigned int v206; // [rsp+F8h] [rbp-7F8h]
  __m128i v207[3]; // [rsp+100h] [rbp-7F0h] BYREF
  _QWORD v208[8]; // [rsp+130h] [rbp-7C0h] BYREF
  __int64 v209; // [rsp+170h] [rbp-780h] BYREF
  __int64 *v210; // [rsp+178h] [rbp-778h]
  __int64 v211; // [rsp+180h] [rbp-770h]
  int v212; // [rsp+188h] [rbp-768h]
  char v213; // [rsp+18Ch] [rbp-764h]
  __int64 v214[4]; // [rsp+190h] [rbp-760h] BYREF
  __m128i v215; // [rsp+1B0h] [rbp-740h] BYREF
  __m128i v216; // [rsp+1C0h] [rbp-730h] BYREF
  __m128i v217[4]; // [rsp+1D0h] [rbp-720h] BYREF
  unsigned __int64 v218; // [rsp+210h] [rbp-6E0h]
  unsigned __int64 v219; // [rsp+218h] [rbp-6D8h]
  __int64 v220; // [rsp+220h] [rbp-6D0h]
  __m128i v221; // [rsp+230h] [rbp-6C0h] BYREF
  __m128i v222; // [rsp+240h] [rbp-6B0h]
  __m128i v223; // [rsp+250h] [rbp-6A0h] BYREF
  char v224; // [rsp+260h] [rbp-690h]
  unsigned __int64 v225; // [rsp+290h] [rbp-660h]
  unsigned __int64 v226; // [rsp+298h] [rbp-658h]
  unsigned __int64 v227; // [rsp+2A0h] [rbp-650h]
  __int64 *v228; // [rsp+2B0h] [rbp-640h] BYREF
  __int64 v229; // [rsp+2B8h] [rbp-638h]
  _BYTE v230[128]; // [rsp+2C0h] [rbp-630h] BYREF
  const __m128i *v231; // [rsp+340h] [rbp-5B0h] BYREF
  __int64 v232; // [rsp+348h] [rbp-5A8h]
  _BYTE v233[128]; // [rsp+350h] [rbp-5A0h] BYREF
  __int64 v234; // [rsp+3D0h] [rbp-520h] BYREF
  __int64 v235; // [rsp+3D8h] [rbp-518h]
  const __m128i *v236; // [rsp+3E0h] [rbp-510h] BYREF
  unsigned int v237; // [rsp+3E8h] [rbp-508h]
  _BYTE *v238; // [rsp+460h] [rbp-490h] BYREF
  __int64 v239; // [rsp+468h] [rbp-488h]
  _BYTE v240[128]; // [rsp+470h] [rbp-480h] BYREF
  __int64 *v241; // [rsp+4F0h] [rbp-400h] BYREF
  __int64 v242; // [rsp+4F8h] [rbp-3F8h]
  _BYTE v243[128]; // [rsp+500h] [rbp-3F0h] BYREF
  __int64 *v244; // [rsp+580h] [rbp-370h] BYREF
  unsigned __int64 v245; // [rsp+588h] [rbp-368h]
  __int64 v246; // [rsp+590h] [rbp-360h] BYREF
  int v247; // [rsp+598h] [rbp-358h]
  char v248; // [rsp+59Ch] [rbp-354h]
  char v249; // [rsp+5A0h] [rbp-350h] BYREF
  unsigned __int64 v250; // [rsp+620h] [rbp-2D0h] BYREF
  __int64 *v251; // [rsp+628h] [rbp-2C8h] BYREF
  __int64 v252; // [rsp+630h] [rbp-2C0h]
  __int64 v253; // [rsp+638h] [rbp-2B8h] BYREF
  unsigned int v254; // [rsp+640h] [rbp-2B0h]
  const __m128i *v255; // [rsp+680h] [rbp-270h]
  const __m128i *v256; // [rsp+688h] [rbp-268h]
  char v257[8]; // [rsp+698h] [rbp-258h] BYREF
  unsigned __int64 v258; // [rsp+6A0h] [rbp-250h]
  char v259; // [rsp+6B4h] [rbp-23Ch]
  const __m128i *v260; // [rsp+6F8h] [rbp-1F8h]
  const __m128i *v261; // [rsp+700h] [rbp-1F0h]
  _QWORD v262[2]; // [rsp+778h] [rbp-178h] BYREF
  char v263; // [rsp+788h] [rbp-168h]
  _BYTE *v264; // [rsp+790h] [rbp-160h]
  __int64 v265; // [rsp+798h] [rbp-158h]
  _BYTE v266[128]; // [rsp+7A0h] [rbp-150h] BYREF
  __int16 v267; // [rsp+820h] [rbp-D0h]
  _QWORD v268[2]; // [rsp+828h] [rbp-C8h] BYREF
  __int64 v269; // [rsp+838h] [rbp-B8h]
  __int64 v270; // [rsp+840h] [rbp-B0h] BYREF
  unsigned int v271; // [rsp+848h] [rbp-A8h]
  char v272; // [rsp+8C0h] [rbp-30h] BYREF

  if ( (unsigned __int8)sub_B2D610(a1, 20) )
    return 0;
  if ( (*(_BYTE *)(a1 + 32) & 0xFu) - 7 > 1 )
    return 0;
  if ( *(_DWORD *)(*(_QWORD *)(a1 + 24) + 8LL) >> 8 )
    return 0;
  v250 = *(_QWORD *)(a1 + 120);
  if ( (unsigned __int8)sub_A74390((__int64 *)&v250, 83, 0) )
    return 0;
  v228 = (__int64 *)v230;
  v229 = 0x1000000000LL;
  if ( (*(_BYTE *)(a1 + 2) & 1) == 0 )
  {
    v8 = *(_QWORD *)(a1 + 96);
    i = 0;
    v10 = v8 + 40LL * *(_QWORD *)(a1 + 104);
    if ( v10 != v8 )
      goto LABEL_9;
    return 0;
  }
  sub_B2C6D0(a1, 83, v5, v6);
  v8 = *(_QWORD *)(a1 + 96);
  v10 = v8 + 40LL * *(_QWORD *)(a1 + 104);
  if ( (*(_BYTE *)(a1 + 2) & 1) != 0 )
  {
    sub_B2C6D0(a1, 83, v19, v20);
    v8 = *(_QWORD *)(a1 + 96);
  }
  for ( i = (unsigned int)v229; v8 != v10; LODWORD(v229) = v229 + 1 )
  {
LABEL_9:
    while ( *(_BYTE *)(*(_QWORD *)(v8 + 8) + 8LL) != 14 )
    {
      v8 += 40;
      if ( v8 == v10 )
        goto LABEL_13;
    }
    if ( i + 1 > (unsigned __int64)HIDWORD(v229) )
    {
      sub_C8D5F0((__int64)&v228, v230, i + 1, 8u, i + 1, v7);
      i = (unsigned int)v229;
    }
    v228[i] = v8;
    v8 += 40;
    i = (unsigned int)(v229 + 1);
  }
LABEL_13:
  if ( !(_DWORD)i )
  {
LABEL_25:
    result = 0;
    goto LABEL_26;
  }
  v11 = *(_QWORD *)(a1 + 16);
  if ( v11 )
  {
    v12 = a4;
    v13 = 0x8000000000041LL;
    do
    {
      v16 = *(unsigned __int8 **)(v11 + 24);
      v17 = *v16;
      if ( (unsigned __int8)v17 <= 0x1Cu )
        goto LABEL_25;
      v14 = (unsigned int)(v17 - 34);
      if ( (unsigned __int8)v14 > 0x33u
        || !_bittest64(&v13, v14)
        || (unsigned __int8 *)v11 != v16 - 32
        || *((_QWORD *)v16 + 10) != *(_QWORD *)(a1 + 24)
        || sub_B49200(*(_QWORD *)(v11 + 24)) )
      {
        goto LABEL_25;
      }
      v15 = sub_B43CB0((__int64)v16);
      v11 = *(_QWORD *)(v11 + 8);
      if ( a1 == v15 )
        v12 = 1;
    }
    while ( v11 );
    a4 = v12;
  }
  v21 = *(_QWORD *)(a1 + 80);
  for ( j = a1 + 72; j != v21; v21 = *(_QWORD *)(v21 + 8) )
  {
    v23 = v21 - 24;
    if ( !v21 )
      v23 = 0;
    if ( sub_AA4E50(v23) )
      goto LABEL_25;
  }
  v183 = sub_B2BEC0(a1);
  v179 = (_QWORD *)(sub_BC1CD0(a2, &unk_4F86540, a1) + 8);
  v204 = 0;
  v205 = 0;
  v187 = (__int64 *)(sub_BC1CD0(a2, &unk_4F89C30, a1) + 8);
  v24 = *(_QWORD *)(a1 + 24);
  v206 = 0;
  v203 = 0;
  v191 = *(_DWORD *)(v24 + 12) - 1;
  v184 = &v228[(unsigned int)v229];
  if ( v184 == v228 )
  {
    v40 = 0;
    v41 = 0;
    v43 = 0;
    goto LABEL_89;
  }
  v189 = v228;
  v182 = a1;
  do
  {
    v188 = *v189;
    if ( (unsigned __int8)sub_B2D720(*v189) )
    {
      v112 = *(_DWORD *)(v188 + 32);
      sub_B2D580(v182, v112, 85);
      sub_B2D3C0(v182, v112, 22);
      v113 = *(_QWORD *)(v182 + 16);
      if ( v113 )
      {
        v114 = v112 + 1;
        do
        {
          v115 = *(_QWORD *)(v113 + 24);
          v116 = (__int64 *)sub_BD5C60(v115);
          *(_QWORD *)(v115 + 72) = sub_A7B980((__int64 *)(v115 + 72), v116, v114, 85);
          v117 = (__int64 *)sub_BD5C60(v115);
          *(_QWORD *)(v115 + 72) = sub_A7A090((__int64 *)(v115 + 72), v117, v114, 22);
          v113 = *(_QWORD *)(v113 + 8);
        }
        while ( v113 );
      }
    }
    v231 = (const __m128i *)v233;
    v232 = 0x400000000LL;
    v198 = a3;
    v27 = *(_QWORD *)(v188 + 16) == 0;
    v199 = v188;
    v196 = a4;
    if ( v27 )
    {
      v244 = &v246;
      v91 = &v246;
      v245 = 0x400000000LL;
      v92 = 0;
      goto LABEL_155;
    }
    v28 = (unsigned __int64 *)&v236;
    v234 = 0;
    v235 = 1;
    do
    {
      *v28 = 0x7FFFFFFFFFFFFFFFLL;
      v28 += 4;
    }
    while ( v28 != (unsigned __int64 *)&v238 );
    v197 = 0;
    v200 = 0;
    LOBYTE(v186) = 0;
    if ( sub_B2BD20(v188) )
      v186 = (unsigned __int16)sub_B2BD00(v199) >> 8;
    v208[0] = v183;
    v208[1] = &v199;
    v208[2] = &v196;
    v208[3] = &v234;
    v208[4] = &v198;
    v208[5] = &v200;
    v208[6] = &v197;
    v31 = v199;
    v32 = *(_QWORD *)(*(_QWORD *)(v199 + 24) + 80LL);
    if ( !v32 )
      BUG();
    v33 = *(_QWORD *)(v32 + 32);
    v34 = v32 + 24;
    if ( v33 != v34 )
    {
      while ( 1 )
      {
        if ( !v33 )
          BUG();
        v38 = *(_BYTE *)(v33 - 24);
        if ( v38 == 61 )
          break;
        if ( v38 == 62 )
        {
          v39 = sub_24FF260((__int64)v208, (unsigned __int8 *)(v33 - 24), *(_QWORD *)(*(_QWORD *)(v33 - 88) + 8LL), 1);
          v36 = v39;
          v37 = HIBYTE(v39);
          goto LABEL_54;
        }
LABEL_56:
        if ( (unsigned __int8)sub_98CD80((char *)(v33 - 24)) )
        {
          v33 = *(_QWORD *)(v33 + 8);
          if ( v34 != v33 )
            continue;
        }
        v31 = v199;
        goto LABEL_91;
      }
      v35 = sub_24FEE50((__int64)v208, (unsigned __int8 *)(v33 - 24), *(_QWORD *)(v33 - 16), 1);
      v36 = v35;
      v37 = HIBYTE(v35);
LABEL_54:
      if ( v37 && !v36 )
        goto LABEL_72;
      goto LABEL_56;
    }
LABEL_91:
    v244 = 0;
    v245 = (unsigned __int64)&v249;
    v238 = v240;
    v242 = 0x1000000000LL;
    v246 = 16;
    v247 = 0;
    v248 = 1;
    v241 = (__int64 *)v243;
    v211 = 4;
    v212 = 0;
    v213 = 1;
    v47 = *(_QWORD *)(v31 + 16);
    v202[0] = (__int64)&v244;
    v239 = 0x1000000000LL;
    v209 = 0;
    v210 = v214;
    v202[1] = (__int64)&v238;
    sub_24FE290(v202, v47, v214, (unsigned __int64)&v238, v29, v30);
    v48 = v239;
    if ( (_DWORD)v239 )
    {
      v49 = 0x8000000000041LL;
      do
      {
        v50 = *(__int64 **)&v238[8 * v48 - 8];
        LODWORD(v239) = v48 - 1;
        v51 = (unsigned __int8 *)v50[3];
        v52 = *v51;
        if ( (unsigned __int8)v52 <= 0x1Cu )
          goto LABEL_63;
        if ( (_BYTE)v52 == 63 )
        {
          if ( !(unsigned __int8)sub_B4DD90(v50[3]) )
            goto LABEL_63;
          sub_24FE290(v202, *((_QWORD *)v51 + 2), v110, v111, v25, v26);
        }
        else if ( (_BYTE)v52 == 61 )
        {
          if ( !(unsigned __int8)sub_24FEE50((__int64)v208, (unsigned __int8 *)v50[3], *((_QWORD *)v51 + 1), 0) )
            goto LABEL_63;
          v118 = (unsigned int)v242;
          v119 = (unsigned int)v242 + 1LL;
          if ( v119 > HIDWORD(v242) )
          {
            sub_C8D5F0((__int64)&v241, v243, v119, 8u, v25, v26);
            v118 = (unsigned int)v242;
          }
          v241[v118] = (__int64)v51;
          LODWORD(v242) = v242 + 1;
        }
        else
        {
          if ( (_BYTE)v52 == 62 )
          {
            if ( !(_BYTE)v186 )
              goto LABEL_63;
            if ( (unsigned int)sub_BD2910((__int64)v50) == 1 )
            {
              if ( !(unsigned __int8)sub_24FF260((__int64)v208, v51, *(_QWORD *)(*((_QWORD *)v51 - 8) + 8LL), 0) )
                goto LABEL_63;
              goto LABEL_113;
            }
            v52 = *v51;
            if ( (unsigned __int8)v52 <= 0x1Cu )
              goto LABEL_63;
          }
          v53 = (unsigned int)(v52 - 34);
          if ( (unsigned __int8)v53 > 0x33u || !_bittest64(&v49, v53) )
            goto LABEL_63;
          v54 = *((_QWORD *)v51 - 4);
          v55 = *v50;
          if ( v54 )
          {
            if ( *(_BYTE *)v54 )
            {
              v54 = 0;
            }
            else if ( *(_QWORD *)(v54 + 24) != *((_QWORD *)v51 + 10) )
            {
              v54 = 0;
            }
          }
          if ( v54 != sub_B43CB0((__int64)v51) )
            goto LABEL_63;
          if ( v199 != v55 )
            goto LABEL_63;
          v56 = *(_DWORD *)(v55 + 32);
          if ( v56 != (unsigned int)sub_BD2910((__int64)v50) )
            goto LABEL_63;
          v58 = v198;
          if ( v198 )
          {
            if ( v198 < (unsigned int)v235 >> 1 )
              goto LABEL_63;
          }
          if ( !v213 )
            goto LABEL_188;
          v59 = v210;
          v57 = HIDWORD(v211);
          v58 = (__int64)&v210[HIDWORD(v211)];
          if ( v210 != (__int64 *)v58 )
          {
            while ( v51 != (unsigned __int8 *)*v59 )
            {
              if ( (__int64 *)v58 == ++v59 )
                goto LABEL_198;
            }
            goto LABEL_113;
          }
LABEL_198:
          if ( HIDWORD(v211) < (unsigned int)v211 )
          {
            ++HIDWORD(v211);
            *(_QWORD *)v58 = v51;
            ++v209;
          }
          else
          {
LABEL_188:
            sub_C8CC70((__int64)&v209, (__int64)v51, v58, v57, v25, v26);
          }
        }
LABEL_113:
        v48 = v239;
      }
      while ( (_DWORD)v239 );
    }
    if ( !v200 )
    {
      m128i_i64 = v197;
      if ( (unsigned __int64)(1LL << v197) <= 1 )
        goto LABEL_130;
    }
    v61 = (unsigned __int8 *)v199;
    v176 = v200;
    v62 = v197;
    v63 = *(_QWORD *)(v199 + 24);
    v180 = v199;
    v64 = sub_B2BEC0(v63);
    v250 = v176;
    v65 = v64;
    LODWORD(v251) = 64;
    v66 = sub_D30550(v61, v62, &v250, v64, 0, 0, 0, 0);
    v25 = v174;
    v26 = v175;
    if ( v66 )
    {
      if ( (unsigned int)v251 > 0x40 )
        goto LABEL_127;
      goto LABEL_130;
    }
    v67 = *(_QWORD *)(v63 + 16);
    if ( !v67 )
    {
LABEL_125:
      v72 = 0;
      goto LABEL_126;
    }
    v68 = v67;
    while ( 1 )
    {
      v69 = *(_QWORD *)(v68 + 24);
      if ( v213 )
        break;
      if ( !sub_C8CA60((__int64)&v209, v69) )
        goto LABEL_202;
LABEL_124:
      v68 = *(_QWORD *)(v68 + 8);
      if ( !v68 )
        goto LABEL_125;
    }
    v70 = v210;
    v71 = &v210[HIDWORD(v211)];
    if ( v210 != v71 )
    {
      while ( v69 != *v70 )
      {
        if ( v71 == ++v70 )
          goto LABEL_202;
      }
      goto LABEL_124;
    }
LABEL_202:
    if ( sub_D30550(
           *(unsigned __int8 **)(v69
                               + 32
                               * (*(unsigned int *)(v180 + 32) - (unsigned __int64)(*(_DWORD *)(v69 + 4) & 0x7FFFFFF))),
           v62,
           &v250,
           v65,
           0,
           0,
           0,
           0) )
    {
      goto LABEL_124;
    }
    v72 = v68;
LABEL_126:
    v66 = v72 == 0;
    if ( (unsigned int)v251 > 0x40 )
    {
LABEL_127:
      if ( v250 )
      {
        v181 = v66;
        j_j___libc_free_0_0(v250);
        v66 = v181;
      }
    }
    if ( !v66 )
    {
LABEL_63:
      v36 = 0;
      goto LABEL_64;
    }
LABEL_130:
    if ( (unsigned int)v235 >> 1 )
    {
      if ( (v235 & 1) != 0 )
      {
        v73 = (const __m128i *)&v238;
        v74 = (const __m128i *)&v236;
        goto LABEL_133;
      }
      v74 = v236;
      v73 = &v236[2 * v237];
      if ( v73 == v236 )
      {
LABEL_135:
        v75 = v232;
        goto LABEL_136;
      }
LABEL_133:
      while ( v74->m128i_i64[0] > 0x7FFFFFFFFFFFFFFDLL )
      {
        v74 += 2;
        if ( v74 == v73 )
          goto LABEL_135;
      }
      v75 = v232;
      if ( v74 == v73 )
      {
LABEL_136:
        if ( v75 <= HIDWORD(v232) )
        {
          LODWORD(v76) = 0;
          goto LABEL_143;
        }
        LODWORD(v76) = 0;
        sub_C8D5F0((__int64)&v231, v233, v75, 0x20u, v25, v26);
        v77 = v232;
        v78 = v231;
        m128i_i64 = (__int64)v231[2 * (unsigned int)v232].m128i_i64;
        if ( v73 != v74 )
          goto LABEL_138;
      }
      else
      {
        v132 = v74;
        v76 = 0;
        while ( 1 )
        {
          v133 = v132[2].m128i_i64;
          if ( &v132[2] == v73 )
            break;
          while ( 1 )
          {
            v132 = (const __m128i *)v133;
            if ( *v133 <= 0x7FFFFFFFFFFFFFFDLL )
              break;
            v133 += 4;
            if ( v73 == (const __m128i *)v133 )
              goto LABEL_227;
          }
          ++v76;
          if ( v133 == (__int64 *)v73 )
            goto LABEL_228;
        }
LABEL_227:
        ++v76;
LABEL_228:
        v134 = v76 + (unsigned int)v232;
        if ( HIDWORD(v232) < v134 )
          sub_C8D5F0((__int64)&v231, v233, v134, 0x20u, v25, v26);
        m128i_i64 = (__int64)v231[2 * (unsigned int)v232].m128i_i64;
        do
        {
LABEL_138:
          if ( m128i_i64 )
          {
            *(__m128i *)m128i_i64 = _mm_loadu_si128(v74);
            *(__m128i *)(m128i_i64 + 16) = _mm_loadu_si128(v74 + 1);
          }
          v79 = v74[2].m128i_i64;
          if ( &v74[2] == v73 )
            break;
          while ( 1 )
          {
            v74 = (const __m128i *)v79;
            if ( *v79 <= 0x7FFFFFFFFFFFFFFDLL )
              break;
            v79 += 4;
            if ( v73 == (const __m128i *)v79 )
              goto LABEL_143;
          }
          m128i_i64 += 32;
        }
        while ( v79 != (__int64 *)v73 );
LABEL_143:
        v77 = v232;
        v78 = v231;
      }
      LODWORD(v232) = v76 + v77;
      v80 = 32LL * (unsigned int)(v76 + v77);
      v81 = (const __m128i *)((char *)v78 + v80);
      v82 = v80;
      if ( v78 != (const __m128i *)&v78->m128i_i8[v80] )
      {
        v83 = (__int64 *)((char *)v78->m128i_i64 + v80);
        _BitScanReverse64((unsigned __int64 *)&v80, v80 >> 5);
        sub_24FD870((__int64)v78, v83, 2LL * (int)(63 - (v80 ^ 0x3F)), m128i_i64, v25);
        if ( v82 <= 0x200 )
        {
          sub_24FDF00(v78, v81);
        }
        else
        {
          sub_24FDF00(v78, v78 + 32);
          while ( v81 != (const __m128i *)v84 )
            sub_24FD750(v84);
        }
        v78 = v231;
      }
      v85 = &v78[2 * (unsigned int)v232];
      if ( v78 != v85 )
      {
        v86 = v78 + 2;
        v87 = v86[-2].m128i_i64[0];
        while ( 1 )
        {
          v88 = sub_9208B0(v183, v86[-2].m128i_i64[1]);
          v251 = (__int64 *)v89;
          v250 = (unsigned __int64)(v88 + 7) >> 3;
          v90 = v87 + sub_CA1930(&v250);
          if ( v85 == v86 )
            break;
          v87 = v86->m128i_i64[0];
          v86 += 2;
          if ( v87 < v90 )
            goto LABEL_63;
        }
      }
      if ( !(_BYTE)v186 )
      {
        v120 = v199;
        v121 = *(_QWORD *)(*(_QWORD *)(v199 + 24) + 16LL);
        if ( v121 )
        {
          while ( 1 )
          {
            v122 = *(unsigned __int8 **)(v121 + 24);
            sub_D669C0(&v215, (__int64)v122, *(_DWORD *)(v120 + 32), 0);
            v123 = sub_B43CB0((__int64)v122);
            v124 = sub_BC1CD0(a2, &unk_4F86540, v123);
            v125 = _mm_loadu_si128(&v215);
            v126 = _mm_loadu_si128(&v216);
            v224 = 1;
            v127 = (_QWORD *)(v124 + 8);
            v128 = &v253;
            v129 = _mm_loadu_si128(v217);
            v251 = 0;
            v250 = (unsigned __int64)v127;
            v252 = 1;
            v221 = v125;
            v222 = v126;
            v223 = v129;
            do
            {
              *v128 = -4;
              v128 += 5;
              *(v128 - 4) = -3;
              *(v128 - 3) = -4;
              *(v128 - 2) = -3;
            }
            while ( v128 != v262 );
            v262[1] = 0;
            v263 = 0;
            v262[0] = v268;
            v267 = 256;
            v264 = v266;
            v265 = 0x400000000LL;
            v268[1] = 0;
            v269 = 1;
            v268[0] = &unk_49DDBE8;
            v130 = &v270;
            do
            {
              *v130 = -4096;
              v130 += 2;
            }
            while ( v130 != (__int64 *)&v272 );
            v131 = sub_CF63E0(v127, v122, &v221, (__int64)&v250);
            v268[0] = &unk_49DDBE8;
            if ( (v269 & 1) == 0 )
              sub_C7D6A0(v270, 16LL * v271, 8);
            nullsub_184();
            if ( v264 != v266 )
              _libc_free((unsigned __int64)v264);
            if ( (v252 & 1) == 0 )
              sub_C7D6A0(v253, 40LL * v254, 8);
            if ( (v131 & 2) != 0 )
              break;
            v121 = *(_QWORD *)(v121 + 8);
            if ( !v121 )
              goto LABEL_221;
          }
          v139 = v241;
          v185 = &v241[(unsigned int)v242];
LABEL_252:
          if ( v185 == v139 )
            goto LABEL_221;
          v140 = *v139;
          v141 = *(_QWORD *)(*v139 + 40);
          sub_D665A0(v207, *v139);
          v142 = *(_QWORD *)(v141 + 56);
          if ( v142 )
            v142 -= 24;
          if ( (unsigned __int8)sub_CF66C0(v179, v142, v140, v207, 2u) )
            goto LABEL_63;
          v145 = *(_QWORD *)(v141 + 16);
          if ( !v145 )
            goto LABEL_307;
          while ( (unsigned __int8)(**(_BYTE **)(v145 + 24) - 30) > 0xAu )
          {
            v145 = *(_QWORD *)(v145 + 8);
            if ( !v145 )
              goto LABEL_307;
          }
LABEL_258:
          v201 = *(_QWORD *)(*(_QWORD *)(v145 + 24) + 40LL);
          sub_24FFF20(&v250, &v201, v143, v144, v25, v26);
          v146 = &v215;
          v147 = v217;
          sub_C8CD80((__int64)&v215, (__int64)v217, (__int64)&v250, v148, v149, v150);
          v218 = 0;
          v219 = 0;
          v220 = 0;
          v154 = (char *)v256 - (char *)v255;
          if ( v256 == v255 )
          {
            v156 = 0;
          }
          else
          {
            if ( v154 > 0x7FFFFFFFFFFFFFF8LL )
              goto LABEL_326;
            v177 = (char *)v256 - (char *)v255;
            v155 = sub_22077B0((char *)v256 - (char *)v255);
            v154 = v177;
            v156 = v155;
          }
          v157 = v256;
          v158 = (unsigned __int64)v255;
          v159 = v156 + v154;
          v218 = v156;
          v219 = v156;
          v220 = v159;
          if ( v256 != v255 )
          {
            v160 = (__m128i *)v156;
            v161 = v255;
            do
            {
              if ( v160 )
              {
                *v160 = _mm_loadu_si128(v161);
                v152 = v161[1].m128i_i64[0];
                v160[1].m128i_i64[0] = v152;
              }
              v161 = (const __m128i *)((char *)v161 + 24);
              v160 = (__m128i *)((char *)v160 + 24);
            }
            while ( v157 != v161 );
            v159 = 0x1FFFFFFFFFFFFFFFLL;
            v156 += 24
                  * (((0xAAAAAAAAAAAAAABLL * (((unsigned __int64)&v157[-2].m128i_u64[1] - v158) >> 3))
                    & 0x1FFFFFFFFFFFFFFFLL)
                   + 1);
          }
          v219 = v156;
          v146 = &v221;
          v147 = &v223;
          sub_C8CD80((__int64)&v221, (__int64)&v223, (__int64)v257, v159, v152, v153);
          v225 = 0;
          v226 = 0;
          v227 = 0;
          v162 = (char *)v261 - (char *)v260;
          if ( v261 == v260 )
          {
            v164 = 0;
          }
          else
          {
            if ( v162 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_326:
              sub_4261EA(v146, v147, v151);
            v178 = (char *)v261 - (char *)v260;
            v163 = sub_22077B0((char *)v261 - (char *)v260);
            v162 = v178;
            v164 = v163;
          }
          v165 = v261;
          v166 = (unsigned __int64)v260;
          v225 = v164;
          v226 = v164;
          v227 = v164 + v162;
          if ( v261 != v260 )
          {
            v167 = (__m128i *)v164;
            v168 = v260;
            do
            {
              if ( v167 )
              {
                *v167 = _mm_loadu_si128(v168);
                v25 = v168[1].m128i_i64[0];
                v167[1].m128i_i64[0] = v25;
              }
              v168 = (const __m128i *)((char *)v168 + 24);
              v167 = (__m128i *)((char *)v167 + 24);
            }
            while ( v165 != v168 );
            v164 += 24
                  * (((0xAAAAAAAAAAAAAABLL * (((unsigned __int64)&v165[-2].m128i_u64[1] - v166) >> 3))
                    & 0x1FFFFFFFFFFFFFFFLL)
                   + 1);
          }
          v226 = v164;
          while ( 1 )
          {
            v144 = v219;
            v172 = v218;
            v143 = v226 - v225;
            if ( v219 - v218 == v226 - v225 )
            {
              if ( v218 == v219 )
              {
LABEL_287:
                if ( v225 )
                  j_j___libc_free_0(v225);
                if ( !v222.m128i_i8[12] )
                  _libc_free(v221.m128i_u64[1]);
                if ( v218 )
                  j_j___libc_free_0(v218);
                if ( !v216.m128i_i8[12] )
                  _libc_free(v215.m128i_u64[1]);
                if ( v260 )
                  j_j___libc_free_0((unsigned __int64)v260);
                if ( !v259 )
                  _libc_free(v258);
                if ( v255 )
                  j_j___libc_free_0((unsigned __int64)v255);
                if ( !BYTE4(v253) )
                  _libc_free((unsigned __int64)v251);
                while ( 1 )
                {
                  v145 = *(_QWORD *)(v145 + 8);
                  if ( !v145 )
                    break;
                  if ( (unsigned __int8)(**(_BYTE **)(v145 + 24) - 30) <= 0xAu )
                    goto LABEL_258;
                }
LABEL_307:
                ++v139;
                goto LABEL_252;
              }
              v143 = v225;
              while ( *(_QWORD *)v172 == *(_QWORD *)v143 )
              {
                v173 = *(_BYTE *)(v172 + 16);
                if ( v173 != *(_BYTE *)(v143 + 16) || v173 && *(_QWORD *)(v172 + 8) != *(_QWORD *)(v143 + 8) )
                  break;
                v172 += 24LL;
                v143 += 24;
                if ( v219 == v172 )
                  goto LABEL_287;
              }
            }
            v169 = *(_QWORD *)(v219 - 24);
            if ( (unsigned __int8)sub_CF6770(v179, v169, v207) )
            {
              if ( v225 )
                j_j___libc_free_0(v225);
              if ( !v222.m128i_i8[12] )
                _libc_free(v221.m128i_u64[1]);
              if ( v218 )
                j_j___libc_free_0(v218);
              if ( !v216.m128i_i8[12] )
                _libc_free(v215.m128i_u64[1]);
              if ( v260 )
                j_j___libc_free_0((unsigned __int64)v260);
              if ( !v259 )
                _libc_free(v258);
              if ( v255 )
                j_j___libc_free_0((unsigned __int64)v255);
              if ( !BYTE4(v253) )
                _libc_free((unsigned __int64)v251);
              goto LABEL_63;
            }
            sub_24FFDC0((__int64)&v215, v169, v170, v171, v25, v26);
          }
        }
      }
    }
LABEL_221:
    v36 = 1;
LABEL_64:
    if ( !v213 )
      _libc_free((unsigned __int64)v210);
    if ( v241 != (__int64 *)v243 )
      _libc_free((unsigned __int64)v241);
    if ( !v248 )
      _libc_free(v245);
    if ( v238 != v240 )
      _libc_free((unsigned __int64)v238);
LABEL_72:
    if ( (v235 & 1) == 0 )
      sub_C7D6A0((__int64)v236, 32LL * v237, 8);
    if ( v36 )
    {
      v244 = &v246;
      v245 = 0x400000000LL;
      v106 = &v231[2 * (unsigned int)v232];
      if ( v106 == v231 )
      {
        v92 = 0;
        v91 = &v246;
      }
      else
      {
        v107 = v231->m128i_i64[1];
        v108 = &v246;
        v109 = v231 + 2;
        v92 = 0;
        while ( 1 )
        {
          v108[v92] = v107;
          v92 = (unsigned int)(v245 + 1);
          LODWORD(v245) = v245 + 1;
          if ( v109 == v106 )
            break;
          v107 = v109->m128i_i64[1];
          if ( v92 + 1 > HIDWORD(v245) )
          {
            sub_C8D5F0((__int64)&v244, &v246, v92 + 1, 8u, v25, v26);
            v92 = (unsigned int)v245;
          }
          v108 = v244;
          v109 += 2;
        }
        v91 = v244;
      }
LABEL_155:
      v251 = (__int64 *)v92;
      v250 = (unsigned __int64)v91;
      v93 = *(_QWORD *)(v182 + 16);
      if ( v93 )
      {
        v94 = 0x8000000000041LL;
        do
        {
          v95 = *(unsigned __int8 **)(v93 + 24);
          v96 = *v95;
          if ( (unsigned __int8)v96 <= 0x1Cu )
            goto LABEL_158;
          v97 = (unsigned int)(v96 - 34);
          if ( (unsigned __int8)v97 > 0x33u || !_bittest64(&v94, v97) )
            goto LABEL_158;
          v98 = sub_B491C0(*(_QWORD *)(v93 + 24));
          v99 = *((_QWORD *)v95 - 4);
          if ( v99 )
          {
            if ( *(_BYTE *)v99 )
            {
              v99 = 0;
            }
            else if ( *(_QWORD *)(v99 + 24) != *((_QWORD *)v95 + 10) )
            {
              v99 = 0;
            }
          }
          if ( !sub_DFE030(v187, v98, v99) )
            goto LABEL_158;
          v93 = *(_QWORD *)(v93 + 8);
        }
        while ( v93 );
      }
      v251 = &v253;
      v191 = v191 + v232 - 1;
      v250 = v188;
      v252 = 0x400000000LL;
      if ( (_DWORD)v232 )
        sub_24FE0F0((__int64)&v251, (__int64)&v231, (__int64)v91, 0x400000000LL, v25, v26);
      v100 = v206;
      if ( v206 )
      {
        v101 = v250;
        v102 = v206 - 1;
        v103 = v102 & (((unsigned int)v250 >> 9) ^ ((unsigned int)v250 >> 4));
        v104 = &v204[19 * v103];
        v105 = *v104;
        if ( v250 == *v104 )
        {
LABEL_172:
          if ( v251 != &v253 )
            _libc_free((unsigned __int64)v251);
LABEL_158:
          if ( v244 != &v246 )
            _libc_free((unsigned __int64)v244);
          goto LABEL_75;
        }
        v135 = 1;
        v136 = 0;
        while ( v105 != -4096 )
        {
          if ( v105 == -8192 && !v136 )
            v136 = v104;
          v103 = v102 & (v135 + v103);
          v104 = &v204[19 * v103];
          v105 = *v104;
          if ( v250 == *v104 )
            goto LABEL_172;
          ++v135;
        }
        if ( v136 )
          v104 = v136;
        ++v203;
        v137 = v205 + 1;
        v241 = (__int64 *)v104;
        if ( 4 * ((int)v205 + 1) < 3 * v206 )
        {
          v138 = v206 >> 3;
          if ( v206 - HIDWORD(v205) - v137 > (unsigned int)v138 )
          {
LABEL_241:
            LODWORD(v205) = v137;
            if ( *v104 != -4096 )
              --HIDWORD(v205);
            *v104 = v101;
            v104[1] = (unsigned __int64)(v104 + 3);
            v104[2] = 0x400000000LL;
            if ( (_DWORD)v252 )
              sub_24FE0F0((__int64)(v104 + 1), (__int64)&v251, (unsigned int)v252, 0x400000000LL, v138, v102);
            goto LABEL_172;
          }
LABEL_247:
          sub_24FE370((__int64)&v203, v100);
          sub_24FD4D0((__int64)&v203, (__int64 *)&v250, &v241);
          v101 = v250;
          v137 = v205 + 1;
          v104 = (unsigned __int64 *)v241;
          goto LABEL_241;
        }
      }
      else
      {
        ++v203;
        v241 = 0;
      }
      v100 = 2 * v206;
      goto LABEL_247;
    }
LABEL_75:
    if ( v231 != (const __m128i *)v233 )
      _libc_free((unsigned __int64)v231);
    ++v189;
  }
  while ( v184 != v189 );
  if ( (_DWORD)v205 && (unsigned int)sub_DFE4C0((__int64)v187) >= v191 )
  {
    v40 = sub_2500970(v182, a2, (__int64)&v203);
    v41 = v204;
    v42 = v206;
    v43 = 19LL * v206;
  }
  else
  {
    v41 = v204;
    v42 = v206;
    v40 = 0;
    v43 = 19LL * v206;
  }
  if ( v42 )
  {
    v44 = &v41[v43];
    v45 = v40;
    do
    {
      if ( *v41 != -4096 && *v41 != -8192 )
      {
        v46 = v41[1];
        if ( (_QWORD *)v46 != v41 + 3 )
          _libc_free(v46);
      }
      v41 += 19;
    }
    while ( v44 != v41 );
    v41 = v204;
    v40 = v45;
    v43 = 19LL * v206;
  }
LABEL_89:
  v195 = v40;
  sub_C7D6A0((__int64)v41, v43 * 8, 8);
  result = v195;
LABEL_26:
  if ( v228 != (__int64 *)v230 )
  {
    v194 = result;
    _libc_free((unsigned __int64)v228);
    return v194;
  }
  return result;
}
