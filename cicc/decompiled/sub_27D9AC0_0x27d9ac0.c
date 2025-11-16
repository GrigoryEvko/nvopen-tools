// Function: sub_27D9AC0
// Address: 0x27d9ac0
//
__int64 __fastcall sub_27D9AC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // r9
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // edi
  unsigned int i; // eax
  __int64 v12; // r8
  int v13; // eax
  __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // r14
  __int64 *v17; // r15
  __int64 v18; // r14
  __int64 *v19; // r12
  __int64 v20; // rbx
  __int64 v21; // r10
  __int64 v22; // rbx
  __int64 v23; // r13
  __int64 v24; // rax
  _DWORD *v25; // rax
  __int64 v26; // r10
  unsigned int v27; // r11d
  __int64 *v28; // rax
  unsigned __int64 *v29; // rbx
  unsigned __int64 v30; // r13
  unsigned __int64 v31; // rdi
  __int64 *v32; // r13
  _QWORD *v33; // rax
  __int64 v34; // rdi
  const char *v35; // rax
  unsigned __int64 v36; // rdx
  __int8 v37; // cl
  unsigned __int64 *v38; // rsi
  __int64 v39; // rax
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 v42; // rax
  unsigned __int64 v43; // rdx
  unsigned __int64 *v44; // rax
  unsigned __int64 v45; // rax
  int v46; // edx
  _QWORD *v47; // rdi
  _QWORD *v48; // rax
  __int64 v49; // rbx
  __int64 v50; // r12
  __int64 v51; // rax
  _QWORD *v52; // rax
  __int64 v53; // r12
  const void **v54; // r15
  const void **v55; // rbx
  __int64 v56; // rdx
  unsigned int v57; // esi
  __int64 v58; // r12
  __int64 v59; // r8
  __int64 v60; // r9
  __int64 v61; // r12
  const void **v62; // rbx
  __int64 v63; // rdx
  unsigned int v64; // esi
  __int64 v65; // rax
  unsigned __int64 v66; // rdx
  unsigned __int64 *v67; // rax
  const char *v68; // rsi
  __int64 v69; // r9
  const char *v70; // r12
  unsigned __int64 v71; // r8
  __int64 v72; // rax
  int v73; // ecx
  unsigned __int64 *v74; // rdx
  __int64 v75; // rcx
  __int64 v76; // r12
  __int64 *v77; // r13
  __int64 v78; // r15
  __int64 v79; // r14
  __int64 v80; // rax
  __int64 v81; // r8
  __int64 v82; // r9
  __int64 v83; // rbx
  __int64 v84; // rax
  unsigned __int64 v85; // rdx
  unsigned __int64 *v86; // rax
  __int64 v87; // rax
  unsigned __int64 *v88; // rax
  _QWORD *v89; // r14
  __int64 v90; // rax
  bool v91; // zf
  __int64 v92; // rdx
  __int64 v93; // rdx
  __int64 v94; // rax
  _QWORD *v95; // rdi
  int v96; // eax
  int v97; // eax
  unsigned int v98; // edx
  __int64 v99; // rax
  __int64 v100; // rdx
  __int64 v101; // rdx
  __int64 v102; // r13
  __int64 v103; // rax
  __int64 v104; // rdx
  __int64 v105; // rcx
  __int64 v106; // r8
  __int64 v107; // r9
  __m128i v108; // xmm0
  __m128i v109; // xmm1
  __m128i v110; // xmm2
  int v111; // ebx
  unsigned __int64 *v112; // r13
  unsigned __int64 *v113; // rbx
  unsigned __int64 *v114; // r13
  unsigned __int64 v115; // rdi
  __int64 v116; // rsi
  __int64 v117; // rdx
  __int64 v118; // rcx
  __int64 v119; // r8
  __int64 v120; // r9
  _QWORD *v121; // rbx
  _QWORD *v122; // r14
  void (__fastcall *v123)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v124; // rax
  __int64 v126; // rsi
  __int64 v127; // rax
  unsigned int j; // eax
  int v129; // eax
  __int64 v130; // rcx
  __int64 v131; // rax
  __int64 v132; // rax
  __int64 v133; // rax
  __m128i *v134; // r15
  __int64 v135; // rsi
  unsigned __int64 *v136; // rbx
  unsigned __int64 *v137; // r15
  unsigned __int64 v138; // rdi
  unsigned __int64 v139; // rbx
  int v140; // eax
  bool v141; // al
  __int64 v142; // rax
  unsigned __int64 v143; // rdx
  unsigned __int64 v144; // rax
  __int64 v145; // rcx
  unsigned __int64 v146; // rdi
  __int64 v147; // rdx
  unsigned __int64 v148; // r12
  _BYTE *v149; // rax
  __int64 v150; // r14
  __int64 v151; // r8
  __int64 v152; // r9
  __int64 v153; // rax
  unsigned __int64 v154; // rdx
  __int64 v155; // [rsp+10h] [rbp-A40h]
  __int64 *v156; // [rsp+18h] [rbp-A38h]
  __int64 v158; // [rsp+38h] [rbp-A18h]
  __int64 v159; // [rsp+40h] [rbp-A10h]
  __int64 v160; // [rsp+48h] [rbp-A08h]
  __int64 v161; // [rsp+50h] [rbp-A00h]
  __int64 v162; // [rsp+58h] [rbp-9F8h]
  __int64 v163; // [rsp+A0h] [rbp-9B0h]
  char v164; // [rsp+AFh] [rbp-9A1h]
  __int64 *v165; // [rsp+D8h] [rbp-978h]
  __int64 v166; // [rsp+E0h] [rbp-970h]
  unsigned __int64 v167; // [rsp+E8h] [rbp-968h]
  __int64 *v168; // [rsp+E8h] [rbp-968h]
  __int64 v169; // [rsp+F0h] [rbp-960h]
  __int64 v170; // [rsp+F0h] [rbp-960h]
  __int64 *v171; // [rsp+F8h] [rbp-958h]
  __int64 v172; // [rsp+100h] [rbp-950h]
  __int64 v173; // [rsp+108h] [rbp-948h]
  const char *v174; // [rsp+110h] [rbp-940h]
  __int64 v175; // [rsp+110h] [rbp-940h]
  __int64 v176; // [rsp+118h] [rbp-938h]
  unsigned __int16 v177; // [rsp+118h] [rbp-938h]
  unsigned int v178; // [rsp+118h] [rbp-938h]
  unsigned int v179; // [rsp+118h] [rbp-938h]
  char v180; // [rsp+118h] [rbp-938h]
  unsigned __int64 v181; // [rsp+118h] [rbp-938h]
  __int64 v182; // [rsp+120h] [rbp-930h]
  char v183; // [rsp+128h] [rbp-928h]
  __int64 v184; // [rsp+128h] [rbp-928h]
  __int64 v185; // [rsp+128h] [rbp-928h]
  _BYTE *v186; // [rsp+130h] [rbp-920h]
  __int64 v187; // [rsp+130h] [rbp-920h]
  __int64 v188; // [rsp+130h] [rbp-920h]
  int v189; // [rsp+130h] [rbp-920h]
  __int64 v190; // [rsp+138h] [rbp-918h]
  __int64 v191; // [rsp+148h] [rbp-908h] BYREF
  unsigned __int64 v192; // [rsp+150h] [rbp-900h]
  __int64 *v193; // [rsp+158h] [rbp-8F8h] BYREF
  unsigned int v194; // [rsp+160h] [rbp-8F0h]
  char v195; // [rsp+168h] [rbp-8E8h] BYREF
  char v196; // [rsp+1B8h] [rbp-898h]
  unsigned __int64 *v197; // [rsp+1C0h] [rbp-890h] BYREF
  __int64 v198; // [rsp+1C8h] [rbp-888h]
  _BYTE v199[128]; // [rsp+1D0h] [rbp-880h] BYREF
  const void **v200; // [rsp+250h] [rbp-800h] BYREF
  __int64 v201; // [rsp+258h] [rbp-7F8h]
  _BYTE v202[32]; // [rsp+260h] [rbp-7F0h] BYREF
  __int64 v203; // [rsp+280h] [rbp-7D0h]
  __int64 v204; // [rsp+288h] [rbp-7C8h]
  __int64 v205; // [rsp+290h] [rbp-7C0h]
  __int64 v206; // [rsp+298h] [rbp-7B8h]
  void **v207; // [rsp+2A0h] [rbp-7B0h]
  void **v208; // [rsp+2A8h] [rbp-7A8h]
  __int64 v209; // [rsp+2B0h] [rbp-7A0h]
  int v210; // [rsp+2B8h] [rbp-798h]
  __int16 v211; // [rsp+2BCh] [rbp-794h]
  char v212; // [rsp+2BEh] [rbp-792h]
  __int64 v213; // [rsp+2C0h] [rbp-790h]
  __int64 v214; // [rsp+2C8h] [rbp-788h]
  void *v215; // [rsp+2D0h] [rbp-780h] BYREF
  void *v216; // [rsp+2D8h] [rbp-778h] BYREF
  const void **v217; // [rsp+2E0h] [rbp-770h] BYREF
  __int64 v218; // [rsp+2E8h] [rbp-768h]
  _BYTE v219[32]; // [rsp+2F0h] [rbp-760h] BYREF
  __int64 v220; // [rsp+310h] [rbp-740h]
  __int64 v221; // [rsp+318h] [rbp-738h]
  __int64 v222; // [rsp+320h] [rbp-730h]
  __int64 v223; // [rsp+328h] [rbp-728h]
  void **v224; // [rsp+330h] [rbp-720h]
  void **v225; // [rsp+338h] [rbp-718h]
  __int64 v226; // [rsp+340h] [rbp-710h]
  int v227; // [rsp+348h] [rbp-708h]
  __int16 v228; // [rsp+34Ch] [rbp-704h]
  char v229; // [rsp+34Eh] [rbp-702h]
  __int64 v230; // [rsp+350h] [rbp-700h]
  __int64 v231; // [rsp+358h] [rbp-6F8h]
  void *v232; // [rsp+360h] [rbp-6F0h] BYREF
  void *v233; // [rsp+368h] [rbp-6E8h] BYREF
  const void **v234; // [rsp+370h] [rbp-6E0h] BYREF
  __int64 v235; // [rsp+378h] [rbp-6D8h]
  _BYTE v236[32]; // [rsp+380h] [rbp-6D0h] BYREF
  __int64 v237; // [rsp+3A0h] [rbp-6B0h]
  __int64 *v238; // [rsp+3A8h] [rbp-6A8h]
  __int16 v239; // [rsp+3B0h] [rbp-6A0h]
  __int64 v240; // [rsp+3B8h] [rbp-698h]
  void **v241; // [rsp+3C0h] [rbp-690h]
  void **v242; // [rsp+3C8h] [rbp-688h]
  __int64 v243; // [rsp+3D0h] [rbp-680h]
  int v244; // [rsp+3D8h] [rbp-678h]
  __int16 v245; // [rsp+3DCh] [rbp-674h]
  char v246; // [rsp+3DEh] [rbp-672h]
  __int64 v247; // [rsp+3E0h] [rbp-670h]
  __int64 v248; // [rsp+3E8h] [rbp-668h]
  void *v249; // [rsp+3F0h] [rbp-660h] BYREF
  void *v250; // [rsp+3F8h] [rbp-658h] BYREF
  unsigned __int64 v251; // [rsp+400h] [rbp-650h] BYREF
  __m128i *v252; // [rsp+408h] [rbp-648h] BYREF
  __int64 v253; // [rsp+410h] [rbp-640h]
  __m128i v254; // [rsp+418h] [rbp-638h] BYREF
  __int64 v255; // [rsp+428h] [rbp-628h]
  __m128i v256; // [rsp+430h] [rbp-620h]
  __m128i v257; // [rsp+440h] [rbp-610h]
  __m128i *v258; // [rsp+450h] [rbp-600h] BYREF
  __int64 v259; // [rsp+458h] [rbp-5F8h]
  _BYTE v260[320]; // [rsp+460h] [rbp-5F0h] BYREF
  char v261; // [rsp+5A0h] [rbp-4B0h]
  int v262; // [rsp+5A4h] [rbp-4ACh]
  __int64 v263; // [rsp+5A8h] [rbp-4A8h]
  const char *v264; // [rsp+5B0h] [rbp-4A0h] BYREF
  unsigned __int64 v265; // [rsp+5B8h] [rbp-498h]
  __int64 v266; // [rsp+5C0h] [rbp-490h] BYREF
  __m128i v267; // [rsp+5C8h] [rbp-488h] BYREF
  __int64 v268; // [rsp+5D8h] [rbp-478h]
  __m128i v269; // [rsp+5E0h] [rbp-470h] BYREF
  __m128i v270; // [rsp+5F0h] [rbp-460h] BYREF
  unsigned __int64 *v271; // [rsp+600h] [rbp-450h] BYREF
  __int64 v272; // [rsp+608h] [rbp-448h]
  _BYTE v273[320]; // [rsp+610h] [rbp-440h] BYREF
  char v274; // [rsp+750h] [rbp-300h]
  int v275; // [rsp+754h] [rbp-2FCh]
  __int64 v276; // [rsp+758h] [rbp-2F8h]
  unsigned __int64 v277[2]; // [rsp+760h] [rbp-2F0h] BYREF
  _BYTE v278[512]; // [rsp+770h] [rbp-2E0h] BYREF
  __int64 v279; // [rsp+970h] [rbp-E0h]
  __int64 v280; // [rsp+978h] [rbp-D8h]
  __int64 v281; // [rsp+980h] [rbp-D0h]
  __int64 v282; // [rsp+988h] [rbp-C8h]
  char v283; // [rsp+990h] [rbp-C0h]
  __int64 v284; // [rsp+998h] [rbp-B8h]
  char *v285; // [rsp+9A0h] [rbp-B0h]
  __int64 v286; // [rsp+9A8h] [rbp-A8h]
  int v287; // [rsp+9B0h] [rbp-A0h]
  char v288; // [rsp+9B4h] [rbp-9Ch]
  char v289; // [rsp+9B8h] [rbp-98h] BYREF
  __int16 v290; // [rsp+9F8h] [rbp-58h]
  _QWORD *v291; // [rsp+A00h] [rbp-50h]
  _QWORD *v292; // [rsp+A08h] [rbp-48h]
  __int64 v293; // [rsp+A10h] [rbp-40h]

  v6 = sub_BC1CD0(a4, &unk_4F8FAE8, a3);
  v8 = *(unsigned int *)(a4 + 88);
  v9 = *(_QWORD *)(a4 + 72);
  v160 = v6;
  v156 = (__int64 *)(v6 + 8);
  if ( (_DWORD)v8 )
  {
    v7 = 1;
    v10 = v8 - 1;
    for ( i = (v8 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
                | ((unsigned __int64)(((unsigned int)&unk_4F81450 >> 9) ^ ((unsigned int)&unk_4F81450 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = v10 & v13 )
    {
      v12 = v9 + 24LL * i;
      if ( *(_UNKNOWN **)v12 == &unk_4F81450 && a3 == *(_QWORD *)(v12 + 8) )
        break;
      if ( *(_QWORD *)v12 == -4096 && *(_QWORD *)(v12 + 8) == -4096 )
        goto LABEL_7;
      v13 = v7 + i;
      v7 = (unsigned int)(v7 + 1);
    }
    v8 = v9 + 24 * v8;
    if ( v8 != v12 )
    {
      v126 = *(_QWORD *)(*(_QWORD *)(v12 + 16) + 24LL);
      v127 = v126 + 8;
      if ( !v126 )
        v127 = 0;
      v159 = v127;
      goto LABEL_170;
    }
  }
  else
  {
LABEL_7:
    v12 = v9 + 24LL * (unsigned int)v8;
    if ( !(_DWORD)v8 )
    {
      v159 = 0;
LABEL_9:
      v158 = 0;
      goto LABEL_10;
    }
    v10 = v8 - 1;
  }
  v159 = 0;
  v8 = v12;
LABEL_170:
  v7 = 1;
  for ( j = v10
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F8FBC8 >> 9) ^ ((unsigned int)&unk_4F8FBC8 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; j = v10 & v129 )
  {
    v12 = v9 + 24LL * j;
    if ( *(_UNKNOWN **)v12 == &unk_4F8FBC8 && a3 == *(_QWORD *)(v12 + 8) )
      break;
    if ( *(_QWORD *)v12 == -4096 && *(_QWORD *)(v12 + 8) == -4096 )
      goto LABEL_9;
    v129 = v7 + j;
    v7 = (unsigned int)(v7 + 1);
  }
  if ( v12 == v8 )
    goto LABEL_9;
  v130 = *(_QWORD *)(*(_QWORD *)(v12 + 16) + 24LL);
  v131 = v130 + 8;
  if ( !v130 )
    v131 = 0;
  v158 = v131;
LABEL_10:
  v14 = *(_QWORD *)(a3 + 80);
  v277[0] = (unsigned __int64)v278;
  v277[1] = 0x1000000000LL;
  v281 = v159;
  v279 = 0;
  v282 = v158;
  v285 = &v289;
  v280 = 0;
  v283 = 1;
  v284 = 0;
  v286 = 8;
  v287 = 0;
  v288 = 1;
  v290 = 0;
  v291 = 0;
  v292 = 0;
  v293 = 0;
  v172 = a3 + 72;
  v182 = v14;
  v183 = 0;
  if ( v14 == a3 + 72 )
  {
LABEL_164:
    v116 = a1;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    goto LABEL_149;
  }
  do
  {
    v15 = v182 - 24;
    v182 = *(_QWORD *)(v182 + 8);
    v16 = v15;
    do
    {
      v17 = *(__int64 **)(v16 + 56);
      if ( (__int64 *)(v16 + 48) == v17 )
        break;
      v18 = v16 + 48;
      while ( 1 )
      {
        v19 = v17;
        v17 = (__int64 *)v17[1];
        if ( *((_BYTE *)v19 - 24) != 85 )
          goto LABEL_14;
        v20 = *(v19 - 7);
        if ( v20 )
        {
          if ( !*(_BYTE *)v20 )
            goto LABEL_14;
        }
        if ( (*((_WORD *)v19 - 11) & 3) == 2 )
          goto LABEL_14;
        if ( *(_BYTE *)v20 != 61 )
          goto LABEL_14;
        if ( sub_B46500((unsigned __int8 *)*(v19 - 7)) )
          goto LABEL_14;
        if ( (*(_BYTE *)(v20 + 2) & 1) != 0 )
          goto LABEL_14;
        v21 = *(_QWORD *)(v20 - 32);
        if ( *(_BYTE *)v21 != 63 )
          goto LABEL_14;
        v22 = *(_QWORD *)(v20 + 8);
        if ( *(_BYTE *)(v22 + 8) != 14 )
          v22 = 0;
        v23 = *(_QWORD *)(v21 - 32LL * (*(_DWORD *)(v21 + 4) & 0x7FFFFFF));
        if ( *(_BYTE *)v23 != 3 )
          goto LABEL_14;
        if ( (*(_BYTE *)(v23 + 80) & 1) == 0 )
          goto LABEL_14;
        v190 = v21;
        if ( sub_B2FC80(*(_QWORD *)(v21 - 32LL * (*(_DWORD *)(v21 + 4) & 0x7FFFFFF)))
          || (unsigned __int8)sub_B2F6B0(v23)
          || (*(_BYTE *)(v23 + 80) & 2) != 0 )
        {
          goto LABEL_14;
        }
        v186 = (_BYTE *)sub_B2BEC0(*(_QWORD *)(*(_QWORD *)(v190 + 40) + 72LL));
        v24 = *(_QWORD *)(*(_QWORD *)(v190 - 32LL * (*(_DWORD *)(v190 + 4) & 0x7FFFFFF)) + 8LL);
        if ( (unsigned int)*(unsigned __int8 *)(v24 + 8) - 17 <= 1 )
          v24 = **(_QWORD **)(v24 + 16);
        v25 = sub_AE2980((__int64)v186, *(_DWORD *)(v24 + 8) >> 8);
        v26 = v190;
        v27 = v25[3];
        v28 = &v266;
        v264 = 0;
        v265 = 1;
        do
        {
          *v28 = -4096;
          v28 += 2;
        }
        while ( v28 != (__int64 *)&v271 );
        LODWORD(v198) = v27;
        v271 = (unsigned __int64 *)v273;
        v272 = 0x400000000LL;
        if ( v27 > 0x40 )
        {
          v178 = v27;
          sub_C43690((__int64)&v197, 0, 0);
          v27 = v178;
          v26 = v190;
        }
        else
        {
          v197 = 0;
        }
        if ( !(unsigned __int8)sub_B4DE70(v26, v186, v27, &v264, &v197) )
        {
          v196 = 0;
          v8 = (unsigned int)v198;
          goto LABEL_38;
        }
        v8 = (unsigned int)v198;
        if ( (_DWORD)v272 != 1
          || ((unsigned int)v198 <= 0x40
            ? (v141 = v197 == 0)
            : (v179 = v198, v140 = sub_C444A0((__int64)&v197), v8 = v179, v141 = v179 == v140),
              !v141) )
        {
          v196 = 0;
          goto LABEL_38;
        }
        LODWORD(v201) = *((_DWORD *)v271 + 4);
        if ( (unsigned int)v201 > 0x40 )
          sub_C43780((__int64)&v200, (const void **)v271 + 1);
        else
          v200 = (const void **)v271[1];
        v175 = *(_QWORD *)(v23 + 24);
        v180 = sub_AE5020((__int64)v186, v175);
        v142 = sub_9208B0((__int64)v186, v175);
        v252 = (__m128i *)v143;
        v251 = (((unsigned __int64)(v142 + 7) >> 3) + (1LL << v180) - 1) >> v180 << v180;
        v144 = sub_CA1930(&v251);
        v145 = (unsigned int)v201;
        v146 = (unsigned __int64)v200;
        if ( (unsigned int)v201 > 0x40 )
          v146 = (unsigned __int64)*v200;
        v147 = v144 % v146;
        v181 = v144 / v146;
        if ( !(v144 % v146) && v181 <= (unsigned int)qword_4FFD8A8 )
        {
          v252 = &v254;
          v253 = 0xA00000000LL;
          v251 = *v271;
          if ( v181 > 0xA )
          {
            sub_C8D5F0((__int64)&v252, &v254, v181, 8u, v12, v7);
            LODWORD(v145) = v201;
LABEL_223:
            v168 = v19;
            v148 = 0;
            v170 = v18;
            LODWORD(v235) = v145;
            if ( (unsigned int)v145 <= 0x40 )
            {
LABEL_224:
              v234 = v200;
              goto LABEL_225;
            }
            while ( 1 )
            {
              sub_C43780((__int64)&v234, (const void **)&v200);
LABEL_225:
              sub_C47170((__int64)&v234, v148);
              LODWORD(v218) = v235;
              v217 = v234;
              v149 = (_BYTE *)sub_9714E0(*(_QWORD *)(v23 - 32), v22, (__int64)&v217, v186);
              v150 = (__int64)v149;
              if ( !v149
                || *v149
                || sub_B2FC80((__int64)v149)
                || (unsigned int)sub_B2BED0(v150) > (unsigned int)qword_4FFD7C8 )
              {
                break;
              }
              v153 = (unsigned int)v253;
              v154 = (unsigned int)v253 + 1LL;
              if ( v154 > HIDWORD(v253) )
              {
                sub_C8D5F0((__int64)&v252, &v254, v154, 8u, v151, v152);
                v153 = (unsigned int)v253;
              }
              ++v148;
              v252->m128i_i64[v153] = v150;
              LODWORD(v253) = v253 + 1;
              sub_969240((__int64 *)&v217);
              if ( v181 <= v148 )
              {
                v18 = v170;
                v19 = v168;
                goto LABEL_242;
              }
              LODWORD(v235) = v201;
              if ( (unsigned int)v201 <= 0x40 )
                goto LABEL_224;
            }
            v196 = 0;
            v18 = v170;
            v19 = v168;
            sub_969240((__int64 *)&v217);
          }
          else
          {
            if ( v144 >= v146 )
              goto LABEL_223;
LABEL_242:
            v192 = v251;
            sub_27D99E0((__int64)&v193, (__int64)&v252, v147, v145, v12, v7);
            v196 = 1;
          }
          if ( v252 != &v254 )
            _libc_free((unsigned __int64)v252);
          LODWORD(v145) = v201;
          goto LABEL_246;
        }
        v196 = 0;
LABEL_246:
        if ( (unsigned int)v145 > 0x40 && v200 )
          j_j___libc_free_0_0((unsigned __int64)v200);
        v8 = (unsigned int)v198;
LABEL_38:
        if ( (unsigned int)v8 > 0x40 && v197 )
          j_j___libc_free_0_0((unsigned __int64)v197);
        v29 = v271;
        v30 = (unsigned __int64)&v271[3 * (unsigned int)v272];
        if ( v271 != (unsigned __int64 *)v30 )
        {
          do
          {
            v30 -= 24LL;
            if ( *(_DWORD *)(v30 + 16) > 0x40u )
            {
              v31 = *(_QWORD *)(v30 + 8);
              if ( v31 )
                j_j___libc_free_0_0(v31);
            }
          }
          while ( v29 != (unsigned __int64 *)v30 );
          v30 = (unsigned __int64)v271;
        }
        if ( (_BYTE *)v30 != v273 )
          _libc_free(v30);
        if ( (v265 & 1) == 0 )
          sub_C7D6A0(v266, 16LL * v267.m128i_u32[0], 8);
        if ( v196 )
          break;
LABEL_14:
        if ( (__int64 *)v18 == v17 )
          goto LABEL_139;
      }
      v164 = v196;
      v32 = v19;
      v184 = *(v19 - 2);
      v171 = v19 - 3;
      v33 = (_QWORD *)sub_BD5C60((__int64)(v19 - 3));
      v176 = sub_BCB120(v33);
      v197 = (unsigned __int64 *)v199;
      v198 = 0x800000000LL;
      v34 = v19[2];
      v173 = v34;
      v251 = (unsigned __int64)".tail";
      v254.m128i_i16[4] = 259;
      v35 = sub_BD5D20(v34);
      v37 = v254.m128i_i8[8];
      if ( v254.m128i_i8[8] )
      {
        if ( v254.m128i_i8[8] == 1 )
        {
          v264 = v35;
          v265 = v36;
          v267.m128i_i16[4] = 261;
        }
        else
        {
          if ( v254.m128i_i8[9] == 1 )
          {
            v161 = (__int64)v252;
            v38 = (unsigned __int64 *)v251;
          }
          else
          {
            v38 = &v251;
            v37 = 2;
          }
          v264 = v35;
          v265 = v36;
          v266 = (__int64)v38;
          v267.m128i_i64[0] = v161;
          v267.m128i_i8[8] = 5;
          v267.m128i_i8[9] = v37;
        }
      }
      else
      {
        v267.m128i_i16[4] = 256;
      }
      v39 = v163;
      LOWORD(v39) = 0;
      v163 = v39;
      v16 = sub_F36990(v34, v19, v39, (__int64)v277, 0, 0, (void **)&v264, 0);
      v167 = v16 & 0xFFFFFFFFFFFFFFFBLL;
      v42 = (unsigned int)v198;
      v43 = (unsigned int)v198 + 1LL;
      if ( v43 > HIDWORD(v198) )
      {
        sub_C8D5F0((__int64)&v197, v199, v43, 0x10u, v40, v41);
        v42 = (unsigned int)v198;
      }
      v44 = &v197[2 * v42];
      v44[1] = v16 | 4;
      *v44 = v34;
      LODWORD(v198) = v198 + 1;
      v45 = *(_QWORD *)(v34 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v34 + 48 == v45 )
      {
        v47 = 0;
      }
      else
      {
        if ( !v45 )
          BUG();
        v46 = *(unsigned __int8 *)(v45 - 24);
        v47 = 0;
        v48 = (_QWORD *)(v45 - 24);
        if ( (unsigned int)(v46 - 30) < 0xB )
          v47 = v48;
      }
      sub_B43D60(v47);
      v49 = *(_QWORD *)(v173 + 72);
      v264 = "default.switch.case.unreachable";
      v166 = v49;
      v267.m128i_i16[4] = 259;
      v50 = sub_B2BE50(v49);
      v51 = sub_22077B0(0x50u);
      v187 = v51;
      if ( v51 )
        sub_AA4D50(v51, v50, (__int64)&v264, v49, v16);
      v206 = sub_AA48A0(v187);
      v207 = &v215;
      v208 = &v216;
      v201 = 0x200000000LL;
      LOWORD(v205) = 0;
      v215 = &unk_49DA100;
      v200 = (const void **)v202;
      v216 = &unk_49DA0B0;
      v204 = v187 + 48;
      v267.m128i_i16[4] = 257;
      v209 = 0;
      v211 = 512;
      v210 = 0;
      v212 = 7;
      v213 = 0;
      v214 = 0;
      v203 = v187;
      v52 = sub_BD2C40(72, unk_3F148B8);
      v53 = (__int64)v52;
      if ( v52 )
        sub_B4C8A0((__int64)v52, v206, 0, 0);
      (*((void (__fastcall **)(void **, __int64, const char **, __int64, __int64))*v208 + 2))(
        v208,
        v53,
        &v264,
        v204,
        v205);
      v54 = v200;
      v55 = &v200[2 * (unsigned int)v201];
      if ( v200 != v55 )
      {
        do
        {
          v56 = (__int64)v54[1];
          v57 = *(_DWORD *)v54;
          v54 += 2;
          sub_B99FD0(v53, v57, v56);
        }
        while ( v55 != v54 );
      }
      v223 = sub_AA48A0(v173);
      v58 = v192;
      v224 = &v232;
      v225 = &v233;
      v217 = (const void **)v219;
      v232 = &unk_49DA100;
      v218 = 0x200000000LL;
      LOWORD(v222) = 0;
      v233 = &unk_49DA0B0;
      v226 = 0;
      v227 = 0;
      v228 = 512;
      v229 = 7;
      v230 = 0;
      v231 = 0;
      v220 = v173;
      v221 = v173 + 48;
      v267.m128i_i16[4] = 257;
      v169 = sub_BD2DA0(80);
      if ( v169 )
        sub_B53A60(v169, v58, v187, 10, 0, 0);
      (*((void (__fastcall **)(void **, __int64, const char **, __int64, __int64))*v225 + 2))(
        v225,
        v169,
        &v264,
        v221,
        v222);
      v61 = (__int64)v217;
      v62 = &v217[2 * (unsigned int)v218];
      if ( v217 != v62 )
      {
        do
        {
          v63 = *(_QWORD *)(v61 + 8);
          v64 = *(_DWORD *)v61;
          v61 += 16;
          sub_B99FD0(v169, v64, v63);
        }
        while ( v62 != (const void **)v61 );
      }
      v65 = (unsigned int)v198;
      v66 = (unsigned int)v198 + 1LL;
      if ( v66 > HIDWORD(v198) )
      {
        sub_C8D5F0((__int64)&v197, v199, v66, 0x10u, v59, v60);
        v65 = (unsigned int)v198;
      }
      v67 = &v197[2 * v65];
      v67[1] = v187 & 0xFFFFFFFFFFFFFFFBLL;
      *v67 = v173;
      LODWORD(v198) = v198 + 1;
      v240 = sub_BD5C60((__int64)v171);
      v241 = &v249;
      v242 = &v250;
      v235 = 0x200000000LL;
      v245 = 512;
      v249 = &unk_49DA100;
      v237 = 0;
      v238 = 0;
      v234 = (const void **)v236;
      v243 = 0;
      v244 = 0;
      v246 = 7;
      v247 = 0;
      v248 = 0;
      v239 = 0;
      v250 = &unk_49DA0B0;
      v237 = v32[2];
      v238 = v32;
      v68 = *(const char **)sub_B46C60((__int64)v171);
      v264 = v68;
      if ( !v68 || (sub_B96E90((__int64)&v264, (__int64)v68, 1), (v70 = v264) == 0) )
      {
        sub_93FB40((__int64)&v234, 0);
        v70 = v264;
        goto LABEL_184;
      }
      v71 = (unsigned int)v235;
      v72 = (__int64)v234;
      v73 = v235;
      v74 = (unsigned __int64 *)&v234[2 * (unsigned int)v235];
      if ( v234 != (const void **)v74 )
      {
        while ( *(_DWORD *)v72 )
        {
          v72 += 16;
          if ( v74 == (unsigned __int64 *)v72 )
            goto LABEL_180;
        }
        *(_QWORD *)(v72 + 8) = v264;
        goto LABEL_82;
      }
LABEL_180:
      if ( (unsigned int)v235 >= (unsigned __int64)HIDWORD(v235) )
      {
        v71 = (unsigned int)v235 + 1LL;
        v139 = v155 & 0xFFFFFFFF00000000LL;
        v155 &= 0xFFFFFFFF00000000LL;
        if ( HIDWORD(v235) < v71 )
        {
          sub_C8D5F0((__int64)&v234, v236, v71, 0x10u, v71, v69);
          v74 = (unsigned __int64 *)&v234[2 * (unsigned int)v235];
        }
        *v74 = v139;
        v74[1] = (unsigned __int64)v70;
        v70 = v264;
        LODWORD(v235) = v235 + 1;
      }
      else
      {
        if ( v74 )
        {
          *(_DWORD *)v74 = 0;
          v74[1] = (unsigned __int64)v70;
          v73 = v235;
          v70 = v264;
        }
        LODWORD(v235) = v73 + 1;
      }
LABEL_184:
      if ( v70 )
LABEL_82:
        sub_B91220((__int64)&v264, (__int64)v70);
      v75 = v176;
      v76 = 0;
      if ( v184 != v176 )
      {
        v267.m128i_i16[4] = 257;
        v76 = sub_D5C860((__int64 *)&v234, *(v32 - 2), v194, (__int64)&v264);
      }
      v77 = v193;
      v185 = 0;
      v165 = &v193[v194];
      if ( v193 != v165 )
      {
        v188 = v16;
        v78 = v162;
        do
        {
          v191 = v185;
          v264 = "call.";
          v266 = (__int64)&v191;
          v267.m128i_i16[4] = 2819;
          v79 = sub_B2BE50(*v77);
          v80 = sub_22077B0(0x50u);
          v83 = v80;
          if ( v80 )
            sub_AA4D50(v80, v79, (__int64)&v264, v166, v188);
          v84 = (unsigned int)v198;
          v85 = (unsigned int)v198 + 1LL;
          if ( v85 > HIDWORD(v198) )
          {
            sub_C8D5F0((__int64)&v197, v199, v85, 0x10u, v81, v82);
            v84 = (unsigned int)v198;
          }
          v86 = &v197[2 * v84];
          v86[1] = v83 & 0xFFFFFFFFFFFFFFFBLL;
          *v86 = v173;
          LODWORD(v198) = v198 + 1;
          v87 = (unsigned int)v198;
          if ( (unsigned __int64)(unsigned int)v198 + 1 > HIDWORD(v198) )
          {
            sub_C8D5F0((__int64)&v197, v199, (unsigned int)v198 + 1LL, 0x10u, v81, v82);
            v87 = (unsigned int)v198;
          }
          v88 = &v197[2 * v87];
          *v88 = v83;
          v88[1] = v167;
          LODWORD(v198) = v198 + 1;
          v89 = (_QWORD *)sub_B47F80(v171);
          v90 = *v77;
          v91 = *(v89 - 4) == 0;
          v89[10] = *(_QWORD *)(*v77 + 24);
          if ( !v91 )
          {
            v92 = *(v89 - 3);
            *(_QWORD *)*(v89 - 2) = v92;
            if ( v92 )
              *(_QWORD *)(v92 + 16) = *(v89 - 2);
          }
          *(v89 - 4) = v90;
          v93 = *(_QWORD *)(v90 + 16);
          *(v89 - 3) = v93;
          if ( v93 )
            *(_QWORD *)(v93 + 16) = v89 - 3;
          *(v89 - 2) = v90 + 16;
          LOWORD(v78) = 0;
          *(_QWORD *)(v90 + 16) = v89 - 4;
          sub_B44240(v89, v83, (unsigned __int64 *)(v83 + 48), v78);
          v94 = sub_AD64C0(*(_QWORD *)(v192 + 8), v191, 0);
          sub_B53E30(v169, v94, v83);
          sub_B43C20((__int64)&v264, v83);
          v174 = v264;
          v177 = v265;
          v95 = sub_BD2C40(72, 1u);
          if ( v95 )
            sub_B4C8F0((__int64)v95, v188, 1u, (__int64)v174, v177);
          if ( v76 )
          {
            v96 = *(_DWORD *)(v76 + 4) & 0x7FFFFFF;
            if ( v96 == *(_DWORD *)(v76 + 72) )
            {
              sub_B48D90(v76);
              v96 = *(_DWORD *)(v76 + 4) & 0x7FFFFFF;
            }
            v97 = (v96 + 1) & 0x7FFFFFF;
            v98 = v97 | *(_DWORD *)(v76 + 4) & 0xF8000000;
            v99 = *(_QWORD *)(v76 - 8) + 32LL * (unsigned int)(v97 - 1);
            *(_DWORD *)(v76 + 4) = v98;
            if ( *(_QWORD *)v99 )
            {
              v100 = *(_QWORD *)(v99 + 8);
              **(_QWORD **)(v99 + 16) = v100;
              if ( v100 )
                *(_QWORD *)(v100 + 16) = *(_QWORD *)(v99 + 16);
            }
            *(_QWORD *)v99 = v89;
            v101 = v89[2];
            v75 = (__int64)(v89 + 2);
            *(_QWORD *)(v99 + 8) = v101;
            if ( v101 )
              *(_QWORD *)(v101 + 16) = v99 + 8;
            *(_QWORD *)(v99 + 16) = v75;
            v89[2] = v99;
            *(_QWORD *)(*(_QWORD *)(v76 - 8)
                      + 32LL * *(unsigned int *)(v76 + 72)
                      + 8LL * ((*(_DWORD *)(v76 + 4) & 0x7FFFFFFu) - 1)) = v83;
          }
          ++v185;
          ++v77;
        }
        while ( v165 != v77 );
        v162 = v78;
        v16 = v188;
      }
      sub_FFB3D0((__int64)v277, v197, (unsigned int)v198, v75, v71, v69);
      v102 = *(_QWORD *)(v160 + 8);
      v103 = sub_B2BE50(v102);
      if ( sub_B6EA50(v103)
        || (v132 = sub_B2BE50(v102),
            v133 = sub_B6F970(v132),
            (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v133 + 48LL))(v133)) )
      {
        sub_B174A0(
          (__int64)&v264,
          (__int64)"jump-table-to-switch",
          (__int64)"ReplacedJumpTableWithSwitch",
          27,
          (__int64)v171);
        sub_B18290((__int64)&v264, "expanded indirect call into switch", 0x22u);
        v108 = _mm_loadu_si128(&v267);
        v109 = _mm_load_si128(&v269);
        v110 = _mm_load_si128(&v270);
        LODWORD(v252) = v265;
        v111 = v272;
        v112 = v271;
        v254 = v108;
        BYTE4(v252) = BYTE4(v265);
        v256 = v109;
        v253 = v266;
        v257 = v110;
        v251 = (unsigned __int64)&unk_49D9D40;
        v255 = v268;
        v258 = (__m128i *)v260;
        v259 = 0x400000000LL;
        if ( !(_DWORD)v272 )
          goto LABEL_113;
        v134 = (__m128i *)v260;
        v135 = (unsigned int)v272;
        if ( (unsigned int)v272 > 4 )
        {
          sub_11F02D0((__int64)&v258, (unsigned int)v272, v104, v105, v106, v107);
          v134 = v258;
          v135 = (unsigned int)v272;
        }
        v112 = &v271[10 * v135];
        if ( v271 == v112 )
        {
          LODWORD(v259) = v111;
LABEL_113:
          v261 = v274;
          v262 = v275;
          v263 = v276;
          v251 = (unsigned __int64)&unk_49D9D78;
        }
        else
        {
          v189 = v111;
          v136 = v271;
          do
          {
            if ( v134 )
            {
              v134->m128i_i64[0] = (__int64)v134[1].m128i_i64;
              sub_27D97C0(v134->m128i_i64, (_BYTE *)*v136, *v136 + v136[1]);
              v134[2].m128i_i64[0] = (__int64)v134[3].m128i_i64;
              sub_27D97C0(v134[2].m128i_i64, (_BYTE *)v136[4], v136[4] + v136[5]);
              v134[4] = _mm_loadu_si128((const __m128i *)v136 + 4);
            }
            v136 += 10;
            v134 += 5;
          }
          while ( v112 != v136 );
          v112 = v271;
          LODWORD(v259) = v189;
          v261 = v274;
          v137 = &v271[10 * (unsigned int)v272];
          v262 = v275;
          v263 = v276;
          v251 = (unsigned __int64)&unk_49D9D78;
          v264 = (const char *)&unk_49D9D40;
          if ( v137 != v271 )
          {
            do
            {
              v137 -= 10;
              v138 = v137[4];
              if ( (unsigned __int64 *)v138 != v137 + 6 )
                j_j___libc_free_0(v138);
              if ( (unsigned __int64 *)*v137 != v137 + 2 )
                j_j___libc_free_0(*v137);
            }
            while ( v112 != v137 );
            v112 = v271;
          }
        }
        if ( v112 != (unsigned __int64 *)v273 )
          _libc_free((unsigned __int64)v112);
        sub_1049740(v156, (__int64)&v251);
        v113 = (unsigned __int64 *)v258;
        v251 = (unsigned __int64)&unk_49D9D40;
        v114 = (unsigned __int64 *)&v258[5 * (unsigned int)v259];
        if ( v258 != (__m128i *)v114 )
        {
          do
          {
            v114 -= 10;
            v115 = v114[4];
            if ( (unsigned __int64 *)v115 != v114 + 6 )
              j_j___libc_free_0(v115);
            if ( (unsigned __int64 *)*v114 != v114 + 2 )
              j_j___libc_free_0(*v114);
          }
          while ( v113 != v114 );
          v114 = (unsigned __int64 *)v258;
        }
        if ( v114 != (unsigned __int64 *)v260 )
          _libc_free((unsigned __int64)v114);
      }
      if ( v76 )
        sub_BD84D0((__int64)v171, v76);
      sub_B43D60(v171);
      nullsub_61();
      v249 = &unk_49DA100;
      nullsub_63();
      if ( v234 != (const void **)v236 )
        _libc_free((unsigned __int64)v234);
      nullsub_61();
      v232 = &unk_49DA100;
      nullsub_63();
      if ( v217 != (const void **)v219 )
        _libc_free((unsigned __int64)v217);
      nullsub_61();
      v215 = &unk_49DA100;
      nullsub_63();
      if ( v200 != (const void **)v202 )
        _libc_free((unsigned __int64)v200);
      if ( v197 != (unsigned __int64 *)v199 )
        _libc_free((unsigned __int64)v197);
      if ( v196 && v193 != (__int64 *)&v195 )
        _libc_free((unsigned __int64)v193);
      v183 = v164;
    }
    while ( v16 );
LABEL_139:
    v14 = v182;
  }
  while ( v172 != v182 );
  if ( !v183 )
    goto LABEL_164;
  v264 = 0;
  v265 = (unsigned __int64)&v267.m128i_u64[1];
  v266 = 2;
  v267.m128i_i32[0] = 0;
  v267.m128i_i8[4] = 1;
  v269.m128i_i64[0] = 0;
  v269.m128i_i64[1] = (__int64)&v271;
  v270.m128i_i64[0] = 2;
  v270.m128i_i32[2] = 0;
  v270.m128i_i8[12] = 1;
  if ( v159 )
    sub_27D9870((__int64)&v264, (__int64)&unk_4F81450, v8, v182, v12, v7);
  if ( v158 )
    sub_27D9870((__int64)&v264, (__int64)&unk_4F8FBC8, v8, v14, v12, v7);
  sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)&v267.m128i_i64[1], (__int64)&v264);
  v116 = a1 + 80;
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)&v271, (__int64)&v269);
  if ( !v270.m128i_i8[12] )
    _libc_free(v269.m128i_u64[1]);
  if ( !v267.m128i_i8[4] )
    _libc_free(v265);
LABEL_149:
  sub_FFCE90((__int64)v277, v116, v8, v14, v12, v7);
  sub_FFD870((__int64)v277, v116, v117, v118, v119, v120);
  sub_FFBC40((__int64)v277, v116);
  v121 = v292;
  v122 = v291;
  if ( v292 != v291 )
  {
    do
    {
      v123 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v122[7];
      *v122 = &unk_49E5048;
      if ( v123 )
        v123(v122 + 5, v122 + 5, 3);
      *v122 = &unk_49DB368;
      v124 = v122[3];
      if ( v124 != -4096 && v124 != 0 && v124 != -8192 )
        sub_BD60C0(v122 + 1);
      v122 += 9;
    }
    while ( v121 != v122 );
    v122 = v291;
  }
  if ( v122 )
    j_j___libc_free_0((unsigned __int64)v122);
  if ( !v288 )
    _libc_free((unsigned __int64)v285);
  if ( (_BYTE *)v277[0] != v278 )
    _libc_free(v277[0]);
  return a1;
}
