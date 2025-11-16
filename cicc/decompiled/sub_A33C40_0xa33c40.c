// Function: sub_A33C40
// Address: 0xa33c40
//
__int64 __fastcall sub_A33C40(__int64 a1)
{
  __int64 *v1; // r13
  __int64 v3; // rdi
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // rdi
  volatile signed __int32 *v7; // rax
  __int64 i; // r12
  __int64 v9; // r14
  __int64 v10; // rax
  volatile signed __int32 **v11; // rbx
  unsigned int v12; // esi
  __int64 v13; // rdi
  __int64 v14; // rax
  volatile signed __int32 *v15; // rax
  unsigned int v16; // r12d
  __int64 *v17; // rbx
  __int64 *v18; // rax
  unsigned __int64 v19; // rdx
  unsigned int v20; // r15d
  __int64 *v21; // r14
  __int64 v22; // r12
  __int64 v23; // rdi
  __int64 v24; // rax
  volatile signed __int32 *v25; // rax
  __int64 v26; // rdi
  __int64 v27; // rax
  volatile signed __int32 *v28; // rax
  __int64 v29; // rdi
  __int64 v30; // rax
  volatile signed __int32 *v31; // rax
  __int64 v32; // rdi
  __int64 v33; // rax
  volatile signed __int32 *v34; // rax
  __int64 v35; // rdi
  __int64 v36; // rax
  volatile signed __int32 *v37; // rax
  __int64 v38; // rdi
  __int64 v39; // rax
  volatile signed __int32 *v40; // rax
  __int64 v41; // rax
  __int64 v42; // r14
  _QWORD *v43; // rax
  __int64 v44; // r8
  _QWORD *v45; // r13
  _QWORD *v46; // r15
  __int64 v47; // rdi
  int v48; // eax
  __int64 v49; // rbx
  __int64 v50; // r12
  __int64 *v51; // rdx
  __int64 v52; // rax
  __int64 *v53; // r13
  __int64 *v54; // rbx
  __int64 v55; // rax
  __int64 v56; // r15
  __int64 v57; // rdi
  int v58; // esi
  unsigned __int64 v59; // rbx
  int v60; // edi
  __int64 v61; // rcx
  unsigned int v62; // edx
  __int64 v63; // rax
  __int64 v64; // r11
  __int64 v65; // rsi
  __int64 v66; // rax
  bool v67; // zf
  __m128i *v68; // rax
  int v69; // edi
  unsigned int v70; // esi
  int v71; // edx
  __m128i v72; // xmm0
  __int64 v73; // rax
  unsigned __int16 v74; // cx
  __int16 v75; // dx
  __int64 v76; // rdx
  int v77; // esi
  int v78; // r8d
  __int64 v79; // rdi
  unsigned int v80; // ecx
  __int64 v81; // rax
  __int64 v82; // r9
  __int64 v83; // rsi
  __int64 v84; // rsi
  __int64 v85; // rbx
  _QWORD *v86; // rcx
  unsigned __int64 v87; // rdx
  _QWORD *v88; // rax
  __int64 v89; // rsi
  __m128i *v90; // rbx
  __m128i *v91; // r12
  __m128i *v92; // rdi
  __int64 v93; // rsi
  __int64 v94; // rsi
  __int64 result; // rax
  _QWORD *v96; // rcx
  _QWORD *v97; // r8
  __int64 v98; // rdi
  _QWORD *v99; // rdi
  __int64 v100; // rcx
  __int64 v101; // r14
  __int64 v102; // rbx
  __int64 v103; // r12
  __int64 v104; // rax
  __int64 v105; // rbx
  __int64 v106; // rax
  unsigned __int64 v107; // rdx
  __int64 v108; // rbx
  __int64 v109; // rax
  unsigned __int64 v110; // rdx
  __int64 v111; // rax
  __int64 v112; // rbx
  unsigned __int64 v113; // rdx
  __int64 v114; // rax
  __int64 v115; // rbx
  __int64 v116; // rax
  __int64 v117; // rbx
  __int64 m; // r12
  __int64 v119; // r14
  __int64 v120; // rbx
  __int64 v121; // rax
  __int64 v122; // rdx
  __int64 v123; // rbx
  __int64 v124; // rax
  __int64 v125; // r15
  unsigned int v126; // r12d
  __int64 v127; // rax
  unsigned __int64 v128; // rdx
  __int64 v129; // r8
  __int64 v130; // rax
  __int64 v131; // r12
  unsigned __int64 v132; // rdx
  __int64 v133; // rax
  __int64 v134; // r12
  __int64 v135; // rax
  __int64 v136; // r12
  __int64 v137; // rax
  __int64 v138; // r12
  __int64 v139; // r15
  unsigned int v140; // ebx
  unsigned int n; // r14d
  int v142; // ecx
  int v143; // r12d
  unsigned int v144; // ecx
  int v145; // r12d
  _QWORD *v146; // r13
  __int64 v147; // rdx
  unsigned int v148; // edx
  int v149; // eax
  int v150; // r13d
  unsigned __int64 v151; // r14
  unsigned int v152; // ebx
  int v153; // r12d
  unsigned int v154; // ecx
  int v155; // eax
  int v156; // eax
  _QWORD *v157; // r13
  __int64 v158; // rdx
  unsigned int v159; // eax
  unsigned int v160; // ecx
  int v161; // r12d
  _QWORD *v162; // rax
  __int64 v163; // rdx
  unsigned int v164; // edx
  int v165; // eax
  unsigned int v166; // ebx
  int v167; // ecx
  int v168; // r14d
  unsigned int v169; // ecx
  int v170; // r14d
  _QWORD *v171; // r13
  __int64 v172; // rdx
  unsigned int v173; // edx
  int v174; // eax
  int v175; // edx
  __int64 v176; // rcx
  int v177; // ecx
  __int64 v178; // r13
  __int64 v179; // rbx
  __int64 j; // r14
  _DWORD **v181; // rdi
  __int64 v182; // rdi
  __int64 *v183; // r14
  __int64 *v184; // r12
  __int64 *k; // rbx
  __int64 v186; // rdx
  __int64 v187; // [rsp+20h] [rbp-650h]
  __int64 v188; // [rsp+30h] [rbp-640h]
  __int64 v190; // [rsp+48h] [rbp-628h]
  unsigned __int64 *v191; // [rsp+50h] [rbp-620h]
  __int64 v192; // [rsp+50h] [rbp-620h]
  unsigned int v193; // [rsp+58h] [rbp-618h]
  __int64 v194; // [rsp+58h] [rbp-618h]
  __int64 *v195; // [rsp+58h] [rbp-618h]
  unsigned int v196; // [rsp+60h] [rbp-610h]
  unsigned int v197; // [rsp+60h] [rbp-610h]
  unsigned __int64 *v198; // [rsp+60h] [rbp-610h]
  unsigned int v199; // [rsp+60h] [rbp-610h]
  __int64 v200; // [rsp+60h] [rbp-610h]
  __int64 v201; // [rsp+68h] [rbp-608h]
  __int64 v202; // [rsp+68h] [rbp-608h]
  __int64 v203; // [rsp+68h] [rbp-608h]
  __int64 *v204; // [rsp+68h] [rbp-608h]
  int v205; // [rsp+68h] [rbp-608h]
  _QWORD *v206; // [rsp+68h] [rbp-608h]
  int v207; // [rsp+68h] [rbp-608h]
  __int64 *v208; // [rsp+68h] [rbp-608h]
  int v209; // [rsp+78h] [rbp-5F8h] BYREF
  int v210; // [rsp+7Ch] [rbp-5F4h] BYREF
  int v211; // [rsp+80h] [rbp-5F0h] BYREF
  int v212; // [rsp+84h] [rbp-5ECh] BYREF
  __int64 v213; // [rsp+88h] [rbp-5E8h] BYREF
  __int64 v214; // [rsp+90h] [rbp-5E0h] BYREF
  __int64 v215; // [rsp+98h] [rbp-5D8h] BYREF
  __int64 v216; // [rsp+A0h] [rbp-5D0h] BYREF
  volatile signed __int32 *v217; // [rsp+A8h] [rbp-5C8h]
  __int64 *v218; // [rsp+B0h] [rbp-5C0h] BYREF
  __int64 *v219; // [rsp+B8h] [rbp-5B8h]
  __int64 v220[2]; // [rsp+C0h] [rbp-5B0h] BYREF
  __int64 v221; // [rsp+D0h] [rbp-5A0h] BYREF
  __int64 v222; // [rsp+D8h] [rbp-598h]
  __int64 v223; // [rsp+E0h] [rbp-590h]
  unsigned int v224; // [rsp+E8h] [rbp-588h]
  __int64 v225; // [rsp+F0h] [rbp-580h] BYREF
  __int64 v226; // [rsp+F8h] [rbp-578h]
  __int64 *v227; // [rsp+100h] [rbp-570h]
  unsigned int v228; // [rsp+108h] [rbp-568h]
  __int64 v229; // [rsp+110h] [rbp-560h] BYREF
  __int64 v230; // [rsp+118h] [rbp-558h]
  __int64 v231; // [rsp+120h] [rbp-550h]
  __int64 v232; // [rsp+128h] [rbp-548h]
  __int64 v233[4]; // [rsp+130h] [rbp-540h] BYREF
  __int64 v234; // [rsp+150h] [rbp-520h] BYREF
  int v235; // [rsp+158h] [rbp-518h] BYREF
  __int64 v236; // [rsp+160h] [rbp-510h]
  int *v237; // [rsp+168h] [rbp-508h]
  int *v238; // [rsp+170h] [rbp-500h]
  __int64 v239; // [rsp+178h] [rbp-4F8h]
  __int64 v240; // [rsp+180h] [rbp-4F0h] BYREF
  __int64 v241; // [rsp+188h] [rbp-4E8h]
  __int64 v242; // [rsp+190h] [rbp-4E0h]
  unsigned int v243; // [rsp+198h] [rbp-4D8h]
  __m128i *v244; // [rsp+1A0h] [rbp-4D0h]
  __int64 v245; // [rsp+1A8h] [rbp-4C8h]
  __m128i v246; // [rsp+1B0h] [rbp-4C0h] BYREF
  __int64 *v247; // [rsp+1C0h] [rbp-4B0h] BYREF
  __int64 (__fastcall *v248)(__int64 *, int *); // [rsp+1C8h] [rbp-4A8h]
  __int64 *v249; // [rsp+1D0h] [rbp-4A0h]
  __int64 *v250; // [rsp+1D8h] [rbp-498h]
  int *v251; // [rsp+1E0h] [rbp-490h]
  __int64 **v252; // [rsp+1E8h] [rbp-488h]
  __int64 *v253; // [rsp+1F0h] [rbp-480h]
  int *v254; // [rsp+1F8h] [rbp-478h]
  int *v255; // [rsp+200h] [rbp-470h]
  __int64 *v256; // [rsp+208h] [rbp-468h]
  __int64 *v257; // [rsp+210h] [rbp-460h]
  int *v258; // [rsp+218h] [rbp-458h]
  __int64 v259; // [rsp+220h] [rbp-450h] BYREF
  __int64 v260; // [rsp+228h] [rbp-448h]
  _BYTE v261[512]; // [rsp+230h] [rbp-440h] BYREF
  __int64 *v262; // [rsp+430h] [rbp-240h] BYREF
  __int64 v263; // [rsp+438h] [rbp-238h] BYREF
  _BYTE v264[560]; // [rsp+440h] [rbp-230h] BYREF

  v1 = (__int64 *)&v262;
  sub_A19830(*(_QWORD *)a1, 0x14u, 4u);
  v3 = *(_QWORD *)a1;
  v259 = 12;
  v262 = &v259;
  v263 = 1;
  sub_A1B680(v3, 0xAu, (__int64 *)&v262, 0);
  v4 = *(_QWORD *)a1;
  v259 = sub_BAEE00(*(_QWORD *)(a1 + 16));
  v262 = &v259;
  v263 = 1;
  sub_A1B680(v4, 0x14u, (__int64 *)&v262, 0);
  sub_A23770(&v216);
  sub_A186C0(v216, 16, 1);
  sub_A186C0(v216, 6, 4);
  sub_A186C0(v216, 32, 2);
  sub_A186C0(v216, 32, 2);
  v5 = v216;
  v6 = *(_QWORD *)a1;
  v216 = 0;
  v262 = (__int64 *)v5;
  v7 = v217;
  v217 = 0;
  v263 = (__int64)v7;
  v196 = sub_A1AB30(v6, (__int64 *)&v262);
  if ( v263 )
    sub_A191D0((volatile signed __int32 *)v263);
  for ( i = *(_QWORD *)(a1 + 64); i != a1 + 48; i = sub_220EEE0(i) )
  {
    v9 = *(_QWORD *)a1;
    LODWORD(v262) = *(_DWORD *)(i + 40);
    v10 = *(_QWORD *)(i + 32);
    HIDWORD(v262) = HIDWORD(v10);
    LODWORD(v263) = v10;
    if ( v196 )
    {
      sub_A20C50(v9, v196, (__int64)&v262, 3, 0, 0, 0x10u, 1);
    }
    else
    {
      v11 = (volatile signed __int32 **)&v262;
      sub_A17B10(v9, 3u, *(_DWORD *)(v9 + 56));
      sub_A17CC0(v9, 0x10u, 6);
      sub_A17CC0(v9, 3u, 6);
      do
      {
        v12 = *(_DWORD *)v11;
        v11 = (volatile signed __int32 **)((char *)v11 + 4);
        sub_A17CC0(v9, v12, 6);
      }
      while ( (volatile signed __int32 **)((char *)&v263 + 4) != v11 );
    }
  }
  if ( *(_QWORD *)(a1 + 88) != *(_QWORD *)(a1 + 96) )
  {
    sub_A23770(&v259);
    sub_A186C0(v259, 30, 1);
    sub_A186C0(v259, 0, 6);
    sub_A186C0(v259, 32, 2);
    v13 = *(_QWORD *)a1;
    v14 = v259;
    v259 = 0;
    v262 = (__int64 *)v14;
    v15 = (volatile signed __int32 *)v260;
    v260 = 0;
    v263 = (__int64)v15;
    v16 = sub_A1AB30(v13, (__int64 *)&v262);
    if ( v263 )
      sub_A191D0((volatile signed __int32 *)v263);
    v262 = (__int64 *)v264;
    v17 = *(__int64 **)(a1 + 88);
    v263 = 0xC00000000LL;
    v18 = *(__int64 **)(a1 + 96);
    v19 = 2 * (v18 - v17);
    if ( v19 > 0xC )
    {
      sub_C8D5F0(&v262, v264, v19, 4);
      v17 = *(__int64 **)(a1 + 88);
      v18 = *(__int64 **)(a1 + 96);
      if ( v17 != v18 )
      {
LABEL_16:
        v20 = v16;
        v21 = v18;
        do
        {
          v22 = *v17++;
          sub_9C8C60((__int64)&v262, SHIDWORD(v22));
          sub_9C8C60((__int64)&v262, v22);
        }
        while ( v21 != v17 );
        v16 = v20;
      }
    }
    else if ( v17 != v18 )
    {
      goto LABEL_16;
    }
    sub_A23520(*(_QWORD *)a1, 0x1Eu, (__int64)&v262, v16);
    if ( v262 != (__int64 *)v264 )
      _libc_free(v262, 30);
    if ( v260 )
      sub_A191D0((volatile signed __int32 *)v260);
  }
  sub_A23770(&v262);
  sub_A19260(&v216, (__int64 *)&v262);
  if ( v263 )
    sub_A191D0((volatile signed __int32 *)v263);
  sub_A186C0(v216, 5, 1);
  sub_A186C0(v216, 8, 4);
  sub_A186C0(v216, 8, 4);
  sub_A186C0(v216, 6, 4);
  sub_A186C0(v216, 8, 4);
  sub_A186C0(v216, 4, 4);
  sub_A186C0(v216, 8, 4);
  sub_A186C0(v216, 4, 4);
  sub_A186C0(v216, 4, 4);
  sub_A186C0(v216, 4, 4);
  sub_A186C0(v216, 0, 6);
  sub_A186C0(v216, 8, 4);
  v23 = *(_QWORD *)a1;
  v24 = v216;
  v216 = 0;
  v262 = (__int64 *)v24;
  v25 = v217;
  v217 = 0;
  v263 = (__int64)v25;
  v209 = sub_A1AB30(v23, (__int64 *)&v262);
  if ( v263 )
    sub_A191D0((volatile signed __int32 *)v263);
  sub_A23770(&v262);
  sub_A19260(&v216, (__int64 *)&v262);
  if ( v263 )
    sub_A191D0((volatile signed __int32 *)v263);
  sub_A186C0(v216, 6, 1);
  sub_A186C0(v216, 8, 4);
  sub_A186C0(v216, 8, 4);
  sub_A186C0(v216, 6, 4);
  sub_A186C0(v216, 0, 6);
  sub_A186C0(v216, 8, 4);
  v26 = *(_QWORD *)a1;
  v27 = v216;
  v216 = 0;
  v262 = (__int64 *)v27;
  v28 = v217;
  v217 = 0;
  v263 = (__int64)v28;
  v210 = sub_A1AB30(v26, (__int64 *)&v262);
  if ( v263 )
    sub_A191D0((volatile signed __int32 *)v263);
  sub_A23770(&v262);
  sub_A19260(&v216, (__int64 *)&v262);
  if ( v263 )
    sub_A191D0((volatile signed __int32 *)v263);
  sub_A186C0(v216, 8, 1);
  sub_A186C0(v216, 8, 4);
  sub_A186C0(v216, 8, 4);
  sub_A186C0(v216, 6, 4);
  sub_A186C0(v216, 8, 4);
  v29 = *(_QWORD *)a1;
  v30 = v216;
  v216 = 0;
  v262 = (__int64 *)v30;
  v31 = v217;
  v217 = 0;
  v263 = (__int64)v31;
  v193 = sub_A1AB30(v29, (__int64 *)&v262);
  if ( v263 )
    sub_A191D0((volatile signed __int32 *)v263);
  sub_A23770(&v262);
  sub_A19260(&v216, (__int64 *)&v262);
  if ( v263 )
    sub_A191D0((volatile signed __int32 *)v263);
  sub_A186C0(v216, 28, 1);
  sub_A186C0(v216, 8, 4);
  sub_A186C0(v216, 4, 4);
  sub_A186C0(v216, 4, 4);
  sub_A186C0(v216, 0, 6);
  sub_A186C0(v216, 8, 4);
  v32 = *(_QWORD *)a1;
  v33 = v216;
  v216 = 0;
  v262 = (__int64 *)v33;
  v34 = v217;
  v217 = 0;
  v263 = (__int64)v34;
  v211 = sub_A1AB30(v32, (__int64 *)&v262);
  if ( v263 )
    sub_A191D0((volatile signed __int32 *)v263);
  sub_A23770(&v262);
  sub_A19260(&v216, (__int64 *)&v262);
  if ( v263 )
    sub_A191D0((volatile signed __int32 *)v263);
  sub_A186C0(v216, 29, 1);
  sub_A186C0(v216, 4, 4);
  sub_A186C0(v216, 4, 4);
  sub_A186C0(v216, 0, 6);
  sub_A186C0(v216, 8, 4);
  v35 = *(_QWORD *)a1;
  v36 = v216;
  v216 = 0;
  v262 = (__int64 *)v36;
  v37 = v217;
  v217 = 0;
  v263 = (__int64)v37;
  v212 = sub_A1AB30(v35, (__int64 *)&v262);
  if ( v263 )
    sub_A191D0((volatile signed __int32 *)v263);
  sub_A23770(&v262);
  sub_A19260(&v216, (__int64 *)&v262);
  if ( v263 )
    sub_A191D0((volatile signed __int32 *)v263);
  sub_A186C0(v216, 32, 1);
  sub_A186C0(v216, 0, 6);
  sub_A186C0(v216, 8, 4);
  v38 = *(_QWORD *)a1;
  v39 = v216;
  v216 = 0;
  v262 = (__int64 *)v39;
  v40 = v217;
  v217 = 0;
  v263 = (__int64)v40;
  v197 = sub_A1AB30(v38, (__int64 *)&v262);
  if ( v263 )
    sub_A191D0((volatile signed __int32 *)v263);
  v221 = 0;
  v259 = (__int64)v261;
  v260 = 0x4000000000LL;
  v263 = 0x4000000000LL;
  v237 = &v235;
  v238 = &v235;
  v244 = &v246;
  v41 = *(_QWORD *)(a1 + 32);
  v213 = a1;
  v222 = 0;
  v223 = 0;
  v224 = 0;
  v262 = (__int64 *)v264;
  v235 = 0;
  v236 = 0;
  v239 = 0;
  v218 = (__int64 *)a1;
  v219 = (__int64 *)&v262;
  v240 = 0;
  v241 = 0;
  v242 = 0;
  v243 = 0;
  v245 = 0;
  if ( v41 )
  {
    v201 = v41 + 8;
    if ( *(_QWORD *)(v41 + 24) != v41 + 8 )
    {
      v42 = *(_QWORD *)(v41 + 24);
      do
      {
        if ( *(_DWORD *)(v42 + 80) )
        {
          v43 = *(_QWORD **)(v42 + 72);
          v44 = 2LL * *(unsigned int *)(v42 + 88);
          v45 = &v43[v44];
          if ( v43 != &v43[v44] )
          {
            while ( 1 )
            {
              v46 = v43;
              if ( *v43 <= 0xFFFFFFFFFFFFFFFDLL )
                break;
              v43 += 2;
              if ( v45 == v43 )
                goto LABEL_50;
            }
            while ( v45 != v46 )
            {
              v47 = v46[1];
              if ( *(_DWORD *)(v47 + 8) == 1 )
              {
                v246.m128i_i64[0] = a1;
                v248 = sub_A31EF0;
                v247 = (__int64 *)sub_A15A80;
                sub_A2BA60((_DWORD **)v47, (__int64)&v246, (__int64)&v240);
                if ( v247 )
                  ((void (__fastcall *)(__m128i *, __m128i *, __int64))v247)(&v246, &v246, 3);
              }
              v46 += 2;
              if ( v46 == v45 )
                break;
              while ( *v46 > 0xFFFFFFFFFFFFFFFDLL )
              {
                v46 += 2;
                if ( v45 == v46 )
                  goto LABEL_50;
              }
            }
          }
        }
LABEL_50:
        v42 = sub_220EF30(v42);
      }
      while ( v201 != v42 );
      v1 = (__int64 *)&v262;
      v48 = v245;
      goto LABEL_65;
    }
    v225 = 0;
    v226 = 0;
    v227 = 0;
    v49 = *(_QWORD *)(a1 + 16);
    v228 = 0;
  }
  else
  {
    v49 = *(_QWORD *)(a1 + 16);
    v192 = v49 + 8;
    if ( *(_QWORD *)(v49 + 24) != v49 + 8 )
    {
      v178 = *(_QWORD *)(v49 + 24);
      do
      {
        v179 = *(_QWORD *)(v178 + 64);
        for ( j = *(_QWORD *)(v178 + 56); v179 != j; j += 8 )
        {
          v181 = *(_DWORD ***)j;
          if ( *(_DWORD *)(*(_QWORD *)j + 8LL) == 1 )
          {
            v246.m128i_i64[0] = a1;
            v248 = sub_A31EF0;
            v247 = (__int64 *)sub_A15A80;
            sub_A2BA60(v181, (__int64)&v246, (__int64)&v240);
            if ( v247 )
              ((void (__fastcall *)(__m128i *, __m128i *, __int64))v247)(&v246, &v246, 3);
          }
        }
        v178 = sub_220EF30(v178);
      }
      while ( v192 != v178 );
      v1 = (__int64 *)&v262;
      v48 = v245;
LABEL_65:
      v225 = 0;
      v226 = 0;
      v227 = 0;
      v228 = 0;
      if ( v48 )
      {
        sub_A21370((__int64)&v246, (__int64)&v240, *(_QWORD *)a1, v197);
        sub_C7D6A0(v226, 16LL * v228, 8);
        ++v225;
        v226 = v246.m128i_i64[1];
        v246 = (__m128i)(unsigned __int64)(v246.m128i_i64[0] + 1);
        v227 = v247;
        v228 = (unsigned int)v248;
        v247 = 0;
        LODWORD(v248) = 0;
        sub_C7D6A0(0, 0, 8);
      }
      v41 = *(_QWORD *)(a1 + 32);
      v49 = *(_QWORD *)(a1 + 16);
      goto LABEL_68;
    }
    v225 = 0;
    v226 = 0;
    v227 = 0;
    v228 = 0;
  }
LABEL_68:
  v214 = 0;
  v250 = &v213;
  v251 = &v210;
  v252 = &v218;
  v254 = &v211;
  v246.m128i_i64[0] = (__int64)&v229;
  v255 = &v212;
  v256 = &v225;
  v246.m128i_i64[1] = a1;
  v257 = &v214;
  v229 = 0;
  v230 = 0;
  v231 = 0;
  v232 = 0;
  v247 = &v221;
  v248 = (__int64 (__fastcall *)(__int64 *, int *))&v259;
  v249 = (__int64 *)&v262;
  v253 = &v234;
  v258 = &v209;
  if ( v41 )
  {
    v50 = *(_QWORD *)(v41 + 24);
    v202 = v41 + 8;
    if ( v50 != v41 + 8 )
    {
      do
      {
        if ( *(_DWORD *)(v50 + 80) )
        {
          v51 = *(__int64 **)(v50 + 72);
          v52 = 2LL * *(unsigned int *)(v50 + 88);
          v53 = &v51[v52];
          if ( v51 != &v51[v52] )
          {
            while ( 1 )
            {
              v54 = v51;
              if ( (unsigned __int64)*v51 <= 0xFFFFFFFFFFFFFFFDLL )
                break;
              v51 += 2;
              if ( v53 == v51 )
                goto LABEL_71;
            }
            while ( v53 != v54 )
            {
              sub_A323F0(v246.m128i_i64, *v54, v54[1], 0);
              v55 = v54[1];
              if ( !*(_DWORD *)(v55 + 8) )
                sub_A323F0(
                  v246.m128i_i64,
                  *(_QWORD *)(*(_QWORD *)(v55 + 56) & 0xFFFFFFFFFFFFFFF8LL),
                  *(_QWORD *)(v55 + 64),
                  1);
              v54 += 2;
              if ( v54 == v53 )
                break;
              while ( (unsigned __int64)*v54 > 0xFFFFFFFFFFFFFFFDLL )
              {
                v54 += 2;
                if ( v53 == v54 )
                  goto LABEL_71;
              }
            }
          }
        }
LABEL_71:
        v50 = sub_220EF30(v50);
      }
      while ( v202 != v50 );
      v1 = (__int64 *)&v262;
    }
  }
  else
  {
    v208 = (__int64 *)(v49 + 8);
    if ( *(_QWORD *)(v49 + 24) != v49 + 8 )
    {
      v183 = *(__int64 **)(v49 + 24);
      do
      {
        v184 = (__int64 *)v183[8];
        for ( k = (__int64 *)v183[7]; v184 != k; ++k )
        {
          v186 = *k;
          sub_A323F0(v246.m128i_i64, v183[4], v186, 0);
        }
        v183 = (__int64 *)sub_220EF30(v183);
      }
      while ( v208 != v183 );
    }
  }
  v191 = (unsigned __int64 *)(v259 + 8LL * (unsigned int)v260);
  if ( v191 != (unsigned __int64 *)v259 )
  {
    v198 = (unsigned __int64 *)v259;
    v56 = a1 + 152;
    while ( 1 )
    {
      v58 = v224;
      v59 = *v198;
      v233[0] = *v198;
      if ( !v224 )
        break;
      v60 = 1;
      v61 = 0;
      v62 = (v224 - 1) & (((unsigned int)v59 >> 9) ^ ((unsigned int)v59 >> 4));
      v63 = v222 + 16LL * v62;
      v64 = *(_QWORD *)v63;
      if ( v59 == *(_QWORD *)v63 )
      {
LABEL_92:
        v65 = *(unsigned int *)(v63 + 8);
        goto LABEL_93;
      }
      while ( v64 != -4096 )
      {
        if ( !v61 && v64 == -8192 )
          v61 = v63;
        v62 = (v224 - 1) & (v60 + v62);
        v63 = v222 + 16LL * v62;
        v64 = *(_QWORD *)v63;
        if ( v59 == *(_QWORD *)v63 )
          goto LABEL_92;
        ++v60;
      }
      if ( v61 )
        v63 = v61;
      ++v221;
      v175 = v223 + 1;
      v246.m128i_i64[0] = v63;
      if ( 4 * ((int)v223 + 1) >= 3 * v224 )
        goto LABEL_249;
      v176 = v59;
      if ( v224 - HIDWORD(v223) - v175 <= v224 >> 3 )
        goto LABEL_250;
LABEL_232:
      LODWORD(v223) = v175;
      if ( *(_QWORD *)v63 != -4096 )
        --HIDWORD(v223);
      *(_QWORD *)v63 = v176;
      v65 = 0;
      *(_DWORD *)(v63 + 8) = 0;
LABEL_93:
      sub_A188E0((__int64)&v262, v65);
      v66 = *(_QWORD *)(v59 + 32);
      v246.m128i_i64[0] = *(_QWORD *)(v59 + 24);
      v246.m128i_i64[1] = v66;
      v67 = (unsigned __int8)sub_A19D80(v56, &v246, v220) == 0;
      v68 = (__m128i *)v220[0];
      if ( !v67 )
        goto LABEL_99;
      v69 = *(_DWORD *)(a1 + 168);
      v70 = *(_DWORD *)(a1 + 176);
      v233[0] = v220[0];
      ++*(_QWORD *)(a1 + 152);
      v71 = v69 + 1;
      if ( 4 * (v69 + 1) >= 3 * v70 )
      {
        v70 *= 2;
LABEL_221:
        sub_A2B260(v56, v70);
        sub_A19D80(v56, &v246, v233);
        v71 = *(_DWORD *)(a1 + 168) + 1;
        v68 = (__m128i *)v233[0];
        goto LABEL_96;
      }
      if ( v70 - *(_DWORD *)(a1 + 172) - v71 <= v70 >> 3 )
        goto LABEL_221;
LABEL_96:
      *(_DWORD *)(a1 + 168) = v71;
      if ( v68->m128i_i64[0] != -1 )
        --*(_DWORD *)(a1 + 172);
      v72 = _mm_loadu_si128(&v246);
      v68[1].m128i_i64[0] = 0;
      *v68 = v72;
LABEL_99:
      sub_A188E0((__int64)&v262, v68[1].m128i_i64[0]);
      v73 = *(_QWORD *)(v213 + 24);
      if ( v73 )
        LOBYTE(v74) = sub_A15B10(*(_QWORD *)v73, *(_QWORD *)(v73 + 8), v59) != 0;
      else
        v74 = 0;
      v75 = *(unsigned __int8 *)(v59 + 12);
      LOBYTE(v75) = (unsigned __int8)v75 >> 4;
      sub_A188E0(
        (__int64)&v262,
        (16
       * ((4 * (*(_BYTE *)(v59 + 13) & 1))
        | (2 * (*(_BYTE *)(v59 + 12) >> 7))
        | ((*(_BYTE *)(v59 + 12) & 0x40) != 0)
        | (unsigned __int64)((8 * (*(_BYTE *)(v59 + 13) >> 1)) & 8)))
      | *(_BYTE *)(v59 + 12) & 0xF
      | (unsigned __int64)((v75 << 8) & 0x300)
      | ((v74 | (*(_BYTE *)(v59 + 13) >> 2)) << 10) & 0x400);
      v76 = *(_QWORD *)(v59 + 64);
      v77 = v224;
      v233[0] = v76;
      if ( !v224 )
      {
        ++v221;
        v246.m128i_i64[0] = 0;
        goto LABEL_252;
      }
      v78 = 1;
      v79 = 0;
      v80 = (v224 - 1) & (((unsigned int)v76 >> 9) ^ ((unsigned int)v76 >> 4));
      v81 = v222 + 16LL * v80;
      v82 = *(_QWORD *)v81;
      if ( v76 != *(_QWORD *)v81 )
      {
        while ( v82 != -4096 )
        {
          if ( !v79 && v82 == -8192 )
            v79 = v81;
          v80 = (v224 - 1) & (v78 + v80);
          v81 = v222 + 16LL * v80;
          v82 = *(_QWORD *)v81;
          if ( v76 == *(_QWORD *)v81 )
            goto LABEL_103;
          ++v78;
        }
        if ( v79 )
          v81 = v79;
        ++v221;
        v177 = v223 + 1;
        v246.m128i_i64[0] = v81;
        if ( 4 * ((int)v223 + 1) >= 3 * v224 )
        {
LABEL_252:
          v77 = 2 * v224;
        }
        else if ( v224 - HIDWORD(v223) - v177 > v224 >> 3 )
        {
LABEL_245:
          LODWORD(v223) = v177;
          if ( *(_QWORD *)v81 != -4096 )
            --HIDWORD(v223);
          *(_QWORD *)v81 = v76;
          v83 = 0;
          *(_DWORD *)(v81 + 8) = 0;
          goto LABEL_104;
        }
        sub_A32030((__int64)&v221, v77);
        sub_A1A190((__int64)&v221, v233, &v246);
        v76 = v233[0];
        v177 = v223 + 1;
        v81 = v246.m128i_i64[0];
        goto LABEL_245;
      }
LABEL_103:
      v83 = *(unsigned int *)(v81 + 8);
LABEL_104:
      sub_A188E0((__int64)&v262, v83);
      sub_A1FB70(*(_QWORD *)a1, 8u, (__int64)&v262, v193);
      LODWORD(v263) = 0;
      if ( !v218[4] && (*(_BYTE *)(v59 + 12) & 0xFu) - 7 <= 1 )
      {
        sub_A188E0((__int64)v219, *(_QWORD *)(v59 + 16));
        sub_A1FB70(*v218, 9u, (__int64)v219, 0);
        *((_DWORD *)v219 + 2) = 0;
      }
      v57 = *(_QWORD *)(v59 + 64);
      if ( *(_DWORD *)(v57 + 8) == 1 )
        sub_A19FD0(v57, &v234);
      if ( v191 == ++v198 )
        goto LABEL_108;
    }
    ++v221;
    v246.m128i_i64[0] = 0;
LABEL_249:
    v58 = 2 * v224;
LABEL_250:
    sub_A32030((__int64)&v221, v58);
    sub_A1A190((__int64)&v221, v233, &v246);
    v176 = v233[0];
    v175 = v223 + 1;
    v63 = v246.m128i_i64[0];
    goto LABEL_232;
  }
LABEL_108:
  v246.m128i_i64[0] = (__int64)&v247;
  v246.m128i_i64[1] = 0x400000000LL;
  v233[2] = (__int64)&v262;
  v233[0] = (__int64)&v229;
  v233[3] = a1;
  v233[1] = (__int64)&v246;
  sub_A20890((__int64)v233, *(_QWORD *)(a1 + 16) + 352LL, 0x11u);
  v84 = *(_QWORD *)(a1 + 16) + 384LL;
  sub_A20890((__int64)v233, v84, 0x12u);
  v187 = (__int64)v237;
  if ( v237 == &v235 )
  {
    v85 = *(_QWORD *)(a1 + 16);
    goto LABEL_117;
  }
  v85 = *(_QWORD *)(a1 + 16);
  do
  {
    v86 = *(_QWORD **)(v85 + 224);
    v84 = v85 + 216;
    if ( !v86 )
      goto LABEL_116;
    v87 = *(_QWORD *)(v187 + 32);
    while ( 1 )
    {
      while ( v86[4] < v87 )
      {
        v86 = (_QWORD *)v86[3];
        if ( !v86 )
          goto LABEL_116;
      }
      v88 = (_QWORD *)v86[2];
      if ( v86[4] <= v87 )
        break;
      v84 = (__int64)v86;
      v86 = (_QWORD *)v86[2];
      if ( !v88 )
        goto LABEL_116;
    }
    v190 = (__int64)v86;
    v96 = (_QWORD *)v86[3];
    v188 = v84;
    v97 = (_QWORD *)v84;
    if ( v96 )
    {
      do
      {
        while ( 1 )
        {
          v98 = v96[2];
          v84 = v96[3];
          if ( v87 < v96[4] )
            break;
          v96 = (_QWORD *)v96[3];
          if ( !v84 )
            goto LABEL_140;
        }
        v97 = v96;
        v96 = (_QWORD *)v96[2];
      }
      while ( v98 );
LABEL_140:
      v188 = (__int64)v97;
    }
    v99 = (_QWORD *)v190;
    if ( v88 )
    {
      do
      {
        while ( 1 )
        {
          v84 = v88[2];
          v100 = v88[3];
          if ( v87 <= v88[4] )
            break;
          v88 = (_QWORD *)v88[3];
          if ( !v100 )
            goto LABEL_146;
        }
        v99 = v88;
        v88 = (_QWORD *)v88[2];
      }
      while ( v84 );
LABEL_146:
      v190 = (__int64)v99;
    }
    if ( v188 == v190 )
      goto LABEL_116;
    do
    {
      v101 = *(_QWORD *)(a1 + 8);
      v102 = *(_QWORD *)(v190 + 40);
      v103 = *(_QWORD *)(v190 + 48);
      v194 = v101;
      v104 = sub_C94890(v102, v103);
      v105 = sub_C0CA60(v101, v102, (v104 << 32) | (unsigned int)v103);
      v106 = (unsigned int)v263;
      v107 = (unsigned int)v263 + 1LL;
      if ( v107 > HIDWORD(v263) )
      {
        sub_C8D5F0(v1, v264, v107, 8);
        v106 = (unsigned int)v263;
      }
      v262[v106] = v105;
      LODWORD(v263) = v263 + 1;
      sub_A188E0((__int64)v1, v103);
      v108 = *(unsigned int *)(v190 + 56);
      v109 = (unsigned int)v263;
      v110 = (unsigned int)v263 + 1LL;
      if ( v110 > HIDWORD(v263) )
      {
        sub_C8D5F0(v1, v264, v110, 8);
        v109 = (unsigned int)v263;
      }
      v262[v109] = v108;
      LODWORD(v263) = v263 + 1;
      sub_A188E0((__int64)v1, *(unsigned int *)(v190 + 60));
      v111 = (unsigned int)v263;
      v112 = *(_QWORD *)(v190 + 64);
      v113 = (unsigned int)v263 + 1LL;
      if ( v113 > HIDWORD(v263) )
      {
        sub_C8D5F0(v1, v264, v113, 8);
        v111 = (unsigned int)v263;
      }
      v262[v111] = v112;
      LODWORD(v263) = v263 + 1;
      v114 = (unsigned int)v263;
      v115 = *(_QWORD *)(v190 + 72);
      if ( (unsigned __int64)(unsigned int)v263 + 1 > HIDWORD(v263) )
      {
        sub_C8D5F0(v1, v264, (unsigned int)v263 + 1LL, 8);
        v114 = (unsigned int)v263;
      }
      v262[v114] = v115;
      LODWORD(v263) = v263 + 1;
      v116 = (unsigned int)v263;
      v117 = *(unsigned __int8 *)(v190 + 80);
      if ( (unsigned __int64)(unsigned int)v263 + 1 > HIDWORD(v263) )
      {
        sub_C8D5F0(v1, v264, (unsigned int)v263 + 1LL, 8);
        v116 = (unsigned int)v263;
      }
      v262[v116] = v117;
      LODWORD(v263) = v263 + 1;
      sub_A188E0((__int64)v1, *(_QWORD *)(v190 + 88));
      for ( m = *(_QWORD *)(v190 + 120); v190 + 104 != m; m = sub_220EF30(m) )
      {
        sub_A188E0((__int64)v1, *(_QWORD *)(m + 32));
        sub_A188E0((__int64)v1, *(unsigned int *)(m + 40));
        v119 = *(_QWORD *)(m + 48);
        v120 = *(_QWORD *)(m + 56);
        v121 = sub_C94890(v119, v120);
        v122 = (unsigned int)v120;
        v123 = m + 88;
        v124 = sub_C0CA60(v194, v119, (v121 << 32) | v122);
        sub_A188E0((__int64)v1, v124);
        sub_A188E0((__int64)v1, *(_QWORD *)(m + 56));
        sub_A188E0((__int64)v1, *(_QWORD *)(m + 120));
        v125 = *(_QWORD *)(m + 104);
        if ( v125 != m + 88 )
        {
          v203 = m;
          v126 = v263;
          do
          {
            v127 = v126;
            v128 = v126 + 1LL;
            v129 = (__int64)(*(_QWORD *)(v125 + 40) - *(_QWORD *)(v125 + 32)) >> 3;
            if ( v128 > HIDWORD(v263) )
            {
              v200 = (__int64)(*(_QWORD *)(v125 + 40) - *(_QWORD *)(v125 + 32)) >> 3;
              sub_C8D5F0(v1, v264, v128, 8);
              v127 = (unsigned int)v263;
              v129 = v200;
            }
            v262[v127] = v129;
            LODWORD(v263) = v263 + 1;
            sub_A16520((__int64)v1, (char *)&v262[(unsigned int)v263], *(char **)(v125 + 32), *(char **)(v125 + 40));
            v130 = (unsigned int)v263;
            v131 = *(unsigned int *)(v125 + 56);
            v132 = (unsigned int)v263 + 1LL;
            if ( v132 > HIDWORD(v263) )
            {
              sub_C8D5F0(v1, v264, v132, 8);
              v130 = (unsigned int)v263;
            }
            v262[v130] = v131;
            LODWORD(v263) = v263 + 1;
            v133 = (unsigned int)v263;
            v134 = *(_QWORD *)(v125 + 64);
            if ( (unsigned __int64)(unsigned int)v263 + 1 > HIDWORD(v263) )
            {
              sub_C8D5F0(v1, v264, (unsigned int)v263 + 1LL, 8);
              v133 = (unsigned int)v263;
            }
            v262[v133] = v134;
            LODWORD(v263) = v263 + 1;
            v135 = (unsigned int)v263;
            v136 = *(unsigned int *)(v125 + 72);
            if ( (unsigned __int64)(unsigned int)v263 + 1 > HIDWORD(v263) )
            {
              sub_C8D5F0(v1, v264, (unsigned int)v263 + 1LL, 8);
              v135 = (unsigned int)v263;
            }
            v262[v135] = v136;
            LODWORD(v263) = v263 + 1;
            v137 = (unsigned int)v263;
            v138 = *(unsigned int *)(v125 + 76);
            if ( (unsigned __int64)(unsigned int)v263 + 1 > HIDWORD(v263) )
            {
              sub_C8D5F0(v1, v264, (unsigned int)v263 + 1LL, 8);
              v137 = (unsigned int)v263;
            }
            v262[v137] = v138;
            v126 = v263 + 1;
            LODWORD(v263) = v263 + 1;
            v125 = sub_220EF30(v125);
          }
          while ( v123 != v125 );
          m = v203;
        }
      }
      v139 = *(_QWORD *)a1;
      v140 = v263;
      v199 = v263;
      sub_A17B10(*(_QWORD *)a1, 3u, *(_DWORD *)(*(_QWORD *)a1 + 56LL));
      sub_A17B10(v139, 0x15u, 6);
      if ( v140 <= 0x1F )
      {
        v84 = v199;
        sub_A17B10(v139, v199, 6);
        if ( !v199 )
          goto LABEL_200;
      }
      else
      {
        v204 = v1;
        for ( n = v140; n > 0x1F; n >>= 5 )
        {
          while ( 1 )
          {
            v142 = *(_DWORD *)(v139 + 48);
            v143 = (n & 0x1F | 0x20) << v142;
            v144 = v142 + 6;
            v145 = *(_DWORD *)(v139 + 52) | v143;
            *(_DWORD *)(v139 + 52) = v145;
            if ( v144 > 0x1F )
              break;
            n >>= 5;
            *(_DWORD *)(v139 + 48) = v144;
            if ( n <= 0x1F )
              goto LABEL_183;
          }
          v146 = *(_QWORD **)(v139 + 24);
          v147 = v146[1];
          if ( (unsigned __int64)(v147 + 4) > v146[2] )
          {
            sub_C8D290(*(_QWORD *)(v139 + 24), v146 + 3, v147 + 4, 1);
            v147 = v146[1];
          }
          *(_DWORD *)(*v146 + v147) = v145;
          v148 = 0;
          v146[1] += 4LL;
          v149 = *(_DWORD *)(v139 + 48);
          if ( v149 )
            v148 = (n & 0x1F | 0x20) >> (32 - v149);
          *(_DWORD *)(v139 + 52) = v148;
          *(_DWORD *)(v139 + 48) = ((_BYTE)v149 + 6) & 0x1F;
        }
LABEL_183:
        v84 = n;
        v1 = v204;
        sub_A17B10(v139, n, 6);
      }
      v195 = v1;
      v150 = 0;
      do
      {
        v151 = v262[v150];
        v152 = v151;
        if ( v151 == (unsigned int)v151 )
        {
          if ( (unsigned int)v151 > 0x1F )
          {
            v207 = v150;
            do
            {
              while ( 1 )
              {
                v167 = *(_DWORD *)(v139 + 48);
                v168 = (v152 & 0x1F | 0x20) << v167;
                v169 = v167 + 6;
                v170 = *(_DWORD *)(v139 + 52) | v168;
                *(_DWORD *)(v139 + 52) = v170;
                if ( v169 > 0x1F )
                  break;
                v152 >>= 5;
                *(_DWORD *)(v139 + 48) = v169;
                if ( v152 <= 0x1F )
                  goto LABEL_216;
              }
              v171 = *(_QWORD **)(v139 + 24);
              v172 = v171[1];
              if ( (unsigned __int64)(v172 + 4) > v171[2] )
              {
                sub_C8D290(*(_QWORD *)(v139 + 24), v171 + 3, v172 + 4, 1);
                v172 = v171[1];
              }
              *(_DWORD *)(*v171 + v172) = v170;
              v173 = 0;
              v171[1] += 4LL;
              v174 = *(_DWORD *)(v139 + 48);
              if ( v174 )
                v173 = (v152 & 0x1F | 0x20) >> (32 - v174);
              v152 >>= 5;
              *(_DWORD *)(v139 + 52) = v173;
              *(_DWORD *)(v139 + 48) = ((_BYTE)v174 + 6) & 0x1F;
            }
            while ( v152 > 0x1F );
LABEL_216:
            v150 = v207;
          }
          v84 = v152;
          sub_A17B10(v139, v152, 6);
        }
        else
        {
          v153 = *(_DWORD *)(v139 + 52);
          v154 = *(_DWORD *)(v139 + 48);
          if ( v151 > 0x1F )
          {
            v205 = v150;
            do
            {
              v156 = (v151 & 0x1F | 0x20) << v154;
              v154 += 6;
              v153 |= v156;
              *(_DWORD *)(v139 + 52) = v153;
              if ( v154 > 0x1F )
              {
                v157 = *(_QWORD **)(v139 + 24);
                v158 = v157[1];
                if ( (unsigned __int64)(v158 + 4) > v157[2] )
                {
                  v84 = (__int64)(v157 + 3);
                  sub_C8D290(*(_QWORD *)(v139 + 24), v157 + 3, v158 + 4, 1);
                  v158 = v157[1];
                }
                *(_DWORD *)(*v157 + v158) = v153;
                v153 = 0;
                v157[1] += 4LL;
                v155 = *(_DWORD *)(v139 + 48);
                if ( v155 )
                  v153 = (v151 & 0x1F | 0x20) >> (32 - (unsigned __int8)v155);
                v154 = ((_BYTE)v155 + 6) & 0x1F;
                *(_DWORD *)(v139 + 52) = v153;
              }
              v151 >>= 5;
              *(_DWORD *)(v139 + 48) = v154;
            }
            while ( v151 > 0x1F );
            v150 = v205;
            v152 = v151;
          }
          v159 = v152 << v154;
          v160 = v154 + 6;
          v161 = v159 | v153;
          *(_DWORD *)(v139 + 52) = v161;
          if ( v160 > 0x1F )
          {
            v162 = *(_QWORD **)(v139 + 24);
            v163 = v162[1];
            if ( (unsigned __int64)(v163 + 4) > v162[2] )
            {
              v84 = (__int64)(v162 + 3);
              v206 = *(_QWORD **)(v139 + 24);
              sub_C8D290(v206, v162 + 3, v163 + 4, 1);
              v162 = v206;
              v163 = v206[1];
            }
            *(_DWORD *)(*v162 + v163) = v161;
            v164 = 0;
            v162[1] += 4LL;
            v165 = *(_DWORD *)(v139 + 48);
            v166 = v152 >> (32 - v165);
            if ( v165 )
              v164 = v166;
            *(_DWORD *)(v139 + 52) = v164;
            *(_DWORD *)(v139 + 48) = ((_BYTE)v165 + 6) & 0x1F;
          }
          else
          {
            *(_DWORD *)(v139 + 48) = v160;
          }
        }
        ++v150;
      }
      while ( v199 != v150 );
      v1 = v195;
LABEL_200:
      LODWORD(v263) = 0;
      v190 = sub_220EF30(v190);
    }
    while ( v190 != v188 );
    v85 = *(_QWORD *)(a1 + 16);
LABEL_116:
    v187 = sub_220EF30(v187);
  }
  while ( (int *)v187 != &v235 );
LABEL_117:
  if ( *(_QWORD *)(v85 + 520) )
  {
    v215 = *(_QWORD *)(v85 + 520);
    v84 = 24;
    v220[1] = 1;
    v182 = *(_QWORD *)a1;
    v220[0] = (__int64)&v215;
    sub_A1B680(v182, 0x18u, v220, 0);
  }
  sub_A192A0(*(_QWORD *)a1);
  if ( (__int64 **)v246.m128i_i64[0] != &v247 )
    _libc_free(v246.m128i_i64[0], v84);
  sub_C7D6A0(v230, 8LL * (unsigned int)v232, 8);
  v89 = 16LL * v228;
  sub_C7D6A0(v226, v89, 8);
  v90 = v244;
  v91 = (__m128i *)((char *)v244 + 72 * (unsigned int)v245);
  if ( v244 != v91 )
  {
    do
    {
      v91 = (__m128i *)((char *)v91 - 72);
      v92 = (__m128i *)v91->m128i_i64[1];
      if ( v92 != (__m128i *)&v91[1].m128i_u64[1] )
        _libc_free(v92, v89);
    }
    while ( v90 != v91 );
    v91 = v244;
  }
  if ( v91 != &v246 )
    _libc_free(v91, v89);
  v93 = 16LL * v243;
  sub_C7D6A0(v241, v93, 8);
  sub_A16180(v236);
  if ( v262 != (__int64 *)v264 )
    _libc_free(v262, v93);
  v94 = 16LL * v224;
  result = sub_C7D6A0(v222, v94, 8);
  if ( (_BYTE *)v259 != v261 )
    result = _libc_free(v259, v94);
  if ( v217 )
    return sub_A191D0(v217);
  return result;
}
