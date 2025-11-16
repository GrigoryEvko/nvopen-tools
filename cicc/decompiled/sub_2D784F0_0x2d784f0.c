// Function: sub_2D784F0
// Address: 0x2d784f0
//
_BOOL8 __fastcall sub_2D784F0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, int a5)
{
  __int64 v7; // rdx
  __int64 v8; // rcx
  _QWORD *v9; // rdx
  unsigned int v10; // eax
  __int64 v11; // rbx
  __int64 v12; // r8
  __int64 v13; // r9
  char v14; // dl
  char v15; // r12
  unsigned __int8 v16; // al
  char v17; // r13
  __int64 v18; // r9
  __int64 v19; // r8
  __int64 v20; // r11
  __int64 v21; // rax
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rcx
  __int64 v25; // rax
  __int64 v26; // rdx
  bool v27; // al
  unsigned __int64 v28; // r14
  unsigned __int64 v29; // r12
  __int64 v30; // rax
  unsigned int v31; // ebx
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // rax
  __m128i *v35; // rax
  char v36; // dl
  __int64 v37; // rsi
  __int64 v38; // rax
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 *v41; // r13
  __m128i *v42; // r14
  __int64 v43; // rbx
  __int64 v44; // rcx
  int v45; // r12d
  unsigned int v46; // edi
  __int64 *v47; // rdx
  __int64 *v48; // rax
  __int64 v49; // r11
  __int64 *v50; // rax
  __m128i *v51; // r13
  __int64 *v52; // rax
  _QWORD *v53; // rbx
  __int64 v54; // r12
  _BYTE *v55; // rbx
  __int64 v56; // rdi
  unsigned __int64 v57; // r13
  _QWORD *v58; // rax
  __int64 v59; // rdx
  _QWORD *v60; // rdi
  __int64 v61; // rcx
  _QWORD *v62; // rsi
  __int64 v63; // rdx
  __int64 v64; // rcx
  __int64 v65; // rcx
  __int64 v66; // rcx
  _BYTE *v67; // r12
  _QWORD *v69; // rax
  __int64 v70; // rbx
  __int64 v71; // rsi
  __int64 v72; // rax
  __int64 v73; // rdx
  __int64 v74; // rdi
  int v75; // esi
  int v76; // eax
  unsigned __int64 *v77; // rax
  __int64 v78; // rcx
  __int64 v79; // rax
  __int64 v80; // rax
  __int64 v81; // rcx
  __int64 v82; // r14
  __int64 v83; // rax
  unsigned __int64 v84; // rdx
  unsigned __int64 v85; // r8
  int v86; // esi
  __int64 v87; // rdx
  _QWORD *v88; // r8
  unsigned __int64 *v89; // rax
  __int32 v90; // eax
  unsigned int v91; // esi
  int v92; // eax
  _BYTE *v93; // r12
  int v94; // eax
  _QWORD *v95; // rax
  _QWORD *v96; // r8
  __m128i *v97; // rax
  char *v98; // rax
  __int64 v99; // rax
  __int64 v100; // r13
  unsigned __int64 v101; // rdx
  __int64 v102; // rax
  __int64 v103; // rbx
  __int64 v104; // rax
  int v105; // edx
  unsigned int v106; // eax
  _BYTE *v107; // r10
  __int64 v108; // rsi
  __int64 v109; // rax
  __int64 v110; // rsi
  _BYTE *v111; // rax
  __int64 v112; // rax
  __int64 v113; // rax
  _BYTE *v114; // rbx
  __int64 v115; // rdi
  __int64 v116; // r9
  unsigned __int64 v117; // rax
  __int64 **v118; // rcx
  unsigned __int64 v119; // rdx
  _QWORD *v120; // rax
  _QWORD *v121; // rbx
  _QWORD *v122; // rcx
  __int64 v123; // rax
  __int64 v124; // rbx
  unsigned __int64 v125; // rdx
  __int64 v126; // rax
  __int64 (__fastcall *v127)(const __m128i **, const __m128i *, int); // rbx
  __int64 v128; // rax
  __int64 v129; // rax
  __m128i v130; // xmm7
  __m128i v131; // xmm6
  __m128i v132; // xmm7
  __m128i *v133; // rax
  __m128i v134; // xmm7
  __int64 v135; // r8
  __m128i *v136; // rbx
  __m128i v137; // xmm1
  __m128i v138; // xmm2
  __m128i *v139; // rax
  __int64 *v140; // r14
  __int64 v141; // rdx
  __m128i *v142; // r8
  int v143; // r11d
  unsigned __int32 v144; // ecx
  __m128i *v145; // rax
  __int64 v146; // rdi
  __int64 v147; // rax
  __int64 v148; // rbx
  unsigned __int32 v149; // ecx
  __int64 v150; // r11
  __int32 v151; // ecx
  __int32 v152; // edx
  __int64 v153; // rax
  _QWORD *v154; // rdx
  __int64 v155; // rdx
  __int64 v156; // rax
  unsigned __int64 v157; // rbx
  __int64 v158; // rdi
  __int64 (__fastcall *v159)(__int64); // rax
  __int64 v161; // rdx
  __int64 v162; // rbx
  __int64 v163; // rdi
  __int64 v164; // rax
  __int64 **v165; // rax
  __int64 v166; // rbx
  __int64 v167; // rax
  unsigned __int64 v168; // r10
  _QWORD *v169; // r11
  __int64 v170; // rax
  __int64 v171; // rax
  _BYTE *v172; // rax
  __int64 v173; // rax
  _BYTE *v174; // r13
  __int64 **v175; // rcx
  __int64 v176; // rax
  __int64 *v177; // rdi
  bool v178; // zf
  __int64 *v179; // rsi
  __int64 *v180; // rdx
  __int64 v181; // rcx
  __int64 v182; // rax
  __int64 v183; // rcx
  __int64 **v184; // rcx
  __int8 *v185; // rbx
  __int64 v186; // rax
  __int64 **v187; // rcx
  __int64 v188; // rax
  unsigned __int64 v189; // rdx
  unsigned __int64 v190; // rax
  int v191; // edi
  _BYTE *v192; // rax
  __int64 v193; // [rsp+28h] [rbp-918h]
  __int8 v194; // [rsp+30h] [rbp-910h]
  char v195; // [rsp+38h] [rbp-908h]
  unsigned int v196; // [rsp+38h] [rbp-908h]
  _QWORD *v197; // [rsp+38h] [rbp-908h]
  unsigned __int64 v198; // [rsp+38h] [rbp-908h]
  __int64 v200; // [rsp+40h] [rbp-900h]
  __int64 v203; // [rsp+58h] [rbp-8E8h]
  __int64 v204; // [rsp+58h] [rbp-8E8h]
  _QWORD *v205; // [rsp+58h] [rbp-8E8h]
  __int64 v206; // [rsp+58h] [rbp-8E8h]
  __int64 v207; // [rsp+58h] [rbp-8E8h]
  unsigned int v208; // [rsp+58h] [rbp-8E8h]
  __int64 v209; // [rsp+60h] [rbp-8E0h]
  bool v210; // [rsp+60h] [rbp-8E0h]
  __int64 v211; // [rsp+60h] [rbp-8E0h]
  __int64 v212; // [rsp+68h] [rbp-8D8h]
  __int64 v213; // [rsp+68h] [rbp-8D8h]
  unsigned __int64 v214; // [rsp+68h] [rbp-8D8h]
  __int64 v215; // [rsp+68h] [rbp-8D8h]
  __int64 v216; // [rsp+68h] [rbp-8D8h]
  unsigned __int64 *v217; // [rsp+68h] [rbp-8D8h]
  __int64 v218; // [rsp+68h] [rbp-8D8h]
  __int32 v219; // [rsp+68h] [rbp-8D8h]
  _BYTE *v220; // [rsp+68h] [rbp-8D8h]
  bool v221; // [rsp+70h] [rbp-8D0h]
  __int64 v222; // [rsp+70h] [rbp-8D0h]
  _QWORD *v223; // [rsp+70h] [rbp-8D0h]
  _BYTE *v224; // [rsp+70h] [rbp-8D0h]
  _QWORD *v225; // [rsp+70h] [rbp-8D0h]
  __int64 v226; // [rsp+78h] [rbp-8C8h]
  __int64 v227; // [rsp+78h] [rbp-8C8h]
  _BYTE *v228; // [rsp+78h] [rbp-8C8h]
  _QWORD *v229; // [rsp+78h] [rbp-8C8h]
  _BYTE *v230; // [rsp+78h] [rbp-8C8h]
  __int64 v231; // [rsp+78h] [rbp-8C8h]
  unsigned __int64 v232; // [rsp+78h] [rbp-8C8h]
  unsigned __int64 v234; // [rsp+88h] [rbp-8B8h]
  __int64 v235; // [rsp+88h] [rbp-8B8h]
  _QWORD v236[2]; // [rsp+90h] [rbp-8B0h] BYREF
  __int64 v237; // [rsp+A0h] [rbp-8A0h]
  __int64 v238; // [rsp+A8h] [rbp-898h]
  __m128i v239; // [rsp+B0h] [rbp-890h] BYREF
  __int64 (__fastcall *v240)(const __m128i **, const __m128i *, int); // [rsp+C0h] [rbp-880h]
  __int64 (__fastcall *v241)(_QWORD *); // [rsp+C8h] [rbp-878h]
  _BYTE *v242[2]; // [rsp+D0h] [rbp-870h] BYREF
  __int64 (__fastcall *v243)(const __m128i **, const __m128i *, int); // [rsp+E0h] [rbp-860h]
  __int64 (__fastcall *v244)(_QWORD *); // [rsp+E8h] [rbp-858h]
  _QWORD v245[8]; // [rsp+F0h] [rbp-850h] BYREF
  __int16 v246; // [rsp+130h] [rbp-810h]
  __m128i v247; // [rsp+140h] [rbp-800h] BYREF
  __m128i v248; // [rsp+150h] [rbp-7F0h] BYREF
  __m128i v249; // [rsp+160h] [rbp-7E0h] BYREF
  __m128i v250; // [rsp+170h] [rbp-7D0h] BYREF
  __int64 v251; // [rsp+180h] [rbp-7C0h]
  _QWORD *v252; // [rsp+190h] [rbp-7B0h] BYREF
  __int64 v253; // [rsp+198h] [rbp-7A8h]
  _QWORD v254[8]; // [rsp+1A0h] [rbp-7A0h] BYREF
  _BYTE *v255; // [rsp+1E0h] [rbp-760h] BYREF
  __int64 v256; // [rsp+1E8h] [rbp-758h]
  _BYTE v257[128]; // [rsp+1F0h] [rbp-750h] BYREF
  _BYTE *v258; // [rsp+270h] [rbp-6D0h] BYREF
  __int64 v259; // [rsp+278h] [rbp-6C8h]
  _BYTE v260[128]; // [rsp+280h] [rbp-6C0h] BYREF
  __int64 v261; // [rsp+300h] [rbp-640h]
  __int64 v262; // [rsp+310h] [rbp-630h] BYREF
  char *v263; // [rsp+318h] [rbp-628h]
  __int64 v264; // [rsp+320h] [rbp-620h]
  int v265; // [rsp+328h] [rbp-618h]
  char v266; // [rsp+32Ch] [rbp-614h]
  char v267; // [rsp+330h] [rbp-610h] BYREF
  __m128i v268; // [rsp+3B0h] [rbp-590h] BYREF
  __m128i v269; // [rsp+3C0h] [rbp-580h] BYREF
  __m128i v270; // [rsp+3D0h] [rbp-570h] BYREF
  __m128i v271; // [rsp+3E0h] [rbp-560h] BYREF
  __int64 v272; // [rsp+3F0h] [rbp-550h]
  __int64 *v273; // [rsp+3F8h] [rbp-548h]
  int v274; // [rsp+400h] [rbp-540h]
  __int64 v275; // [rsp+408h] [rbp-538h]
  __m128i *v276; // [rsp+410h] [rbp-530h]
  __int64 v277; // [rsp+418h] [rbp-528h]
  __int64 v278; // [rsp+420h] [rbp-520h]
  _BYTE **v279; // [rsp+428h] [rbp-518h]
  _QWORD *v280; // [rsp+430h] [rbp-510h]
  char v281; // [rsp+438h] [rbp-508h]
  char v282; // [rsp+439h] [rbp-507h]
  __int64 v283; // [rsp+440h] [rbp-500h]
  __int64 v284; // [rsp+448h] [rbp-4F8h]
  unsigned __int64 v285; // [rsp+450h] [rbp-4F0h] BYREF
  __int64 v286; // [rsp+458h] [rbp-4E8h]
  _BYTE v287[1152]; // [rsp+460h] [rbp-4E0h] BYREF
  unsigned int v288; // [rsp+8E0h] [rbp-60h]
  char v289; // [rsp+8E4h] [rbp-5Ch]
  __int64 v290; // [rsp+8E8h] [rbp-58h]
  __int64 *v291; // [rsp+8F0h] [rbp-50h]
  __int64 v292; // [rsp+8F8h] [rbp-48h]
  _QWORD *v293; // [rsp+900h] [rbp-40h]

  v263 = &v267;
  v254[0] = a3;
  v7 = *(_QWORD *)(a1 + 48);
  v8 = *(_QWORD *)(a1 + 816);
  v253 = 0x800000001LL;
  v255 = v257;
  v245[1] = v7;
  v252 = v254;
  v256 = 0x1000000000LL;
  v262 = 0;
  v264 = 16;
  v265 = 0;
  v266 = 1;
  v245[0] = v8;
  memset(&v245[2], 0, 48);
  v291 = v245;
  v9 = v254;
  v246 = 257;
  v292 = a3;
  v285 = (unsigned __int64)v287;
  v288 = 0;
  v289 = 1;
  v290 = 0;
  v293 = 0;
  v258 = v260;
  v195 = 0;
  v286 = 0x1000000000LL;
  v259 = 0x1000000000LL;
  v261 = a1 + 376;
  v10 = 1;
  while ( 1 )
  {
    v11 = v9[v10 - 1];
    LODWORD(v253) = v10 - 1;
    sub_AE6EC0((__int64)&v262, v11);
    v15 = v14;
    if ( !v14 )
      goto LABEL_63;
    v16 = *(_BYTE *)v11;
    if ( *(_BYTE *)v11 <= 0x1Cu )
      break;
    if ( v16 == 84 )
    {
      v98 = (char *)sub_986520(v11);
      sub_2D57390((__int64)&v252, (char *)&v252[(unsigned int)v253], v98, &v98[32 * (*(_DWORD *)(v11 + 4) & 0x7FFFFFF)]);
      v195 = v15;
      v10 = v253;
      goto LABEL_64;
    }
    if ( v16 != 86 )
      break;
    v99 = (unsigned int)v253;
    v100 = *(_QWORD *)(v11 - 32);
    v101 = (unsigned int)v253 + 1LL;
    if ( v101 > HIDWORD(v253) )
    {
      sub_C8D5F0((__int64)&v252, v254, v101, 8u, v12, v13);
      v99 = (unsigned int)v253;
    }
    v252[v99] = v100;
    LODWORD(v253) = v253 + 1;
    v102 = (unsigned int)v253;
    v103 = *(_QWORD *)(v11 - 64);
    if ( (unsigned __int64)(unsigned int)v253 + 1 > HIDWORD(v253) )
    {
      sub_C8D5F0((__int64)&v252, v254, (unsigned int)v253 + 1LL, 8u, v12, v13);
      v102 = (unsigned int)v253;
    }
    v195 = v15;
    v252[v102] = v103;
    v10 = v253 + 1;
    LODWORD(v253) = v253 + 1;
LABEL_64:
    if ( !v10 )
      goto LABEL_69;
    v9 = v252;
  }
  v209 = *(_QWORD *)(a1 + 64);
  v203 = *(_QWORD *)(a1 + 56);
  v212 = *(_QWORD *)(a1 + 80);
  v17 = *(_BYTE *)(a1 + 808);
  v241 = sub_2D58060;
  v240 = sub_2D56B10;
  v239.m128i_i64[0] = a2;
  v251 = 1;
  v247 = 0;
  v248 = 0;
  v249 = 0;
  v250 = 0;
  LODWORD(v256) = 0;
  v236[0] = 0;
  v236[1] = 0;
  v237 = 0;
  v238 = 0;
  v239.m128i_i64[1] = a1;
  v243 = 0;
  sub_2D56B10((const __m128i **)v242, &v239, 2);
  v268.m128i_i64[1] = v18;
  v269.m128i_i64[0] = v19;
  v244 = v241;
  v243 = v240;
  v268.m128i_i64[0] = (__int64)&v255;
  v21 = sub_B43CC0(v20);
  v271.m128i_i64[1] = 0;
  v269.m128i_i64[1] = v21;
  v270.m128i_i64[0] = v203;
  if ( v243 )
  {
    v243((const __m128i **)&v270.m128i_i64[1], (const __m128i *)v242, 2);
    v272 = (__int64)v244;
    v271.m128i_i64[1] = (__int64)v243;
  }
  v282 = v17;
  v278 = a1 + 344;
  v273 = a4;
  v281 = 0;
  v274 = a5;
  v275 = a2;
  v276 = &v247;
  v277 = a1 + 184;
  v279 = &v258;
  v280 = v236;
  v283 = v212;
  v284 = v209;
  sub_2D65BF0((__int64)&v268, (unsigned __int8 *)v11, 0);
  if ( v271.m128i_i64[1] )
    ((void (__fastcall *)(unsigned __int64 *, unsigned __int64 *, __int64))v271.m128i_i64[1])(
      &v270.m128i_u64[1],
      &v270.m128i_u64[1],
      3);
  if ( v243 )
    v243((const __m128i **)v242, (const __m128i *)v242, 3);
  if ( v240 )
    v240((const __m128i **)&v239, &v239, 3);
  v24 = v237;
  if ( v237 )
  {
    v269.m128i_i64[0] = v237;
    v268 = 0u;
    v210 = v237 != -8192 && v237 != -4096;
    if ( v210 )
    {
      v213 = v237;
      sub_BD73F0((__int64)&v268);
      v24 = v213;
    }
    if ( *(_QWORD *)(a1 + 720) )
    {
      v95 = *(_QWORD **)(a1 + 696);
      if ( v95 )
      {
        v96 = (_QWORD *)(a1 + 688);
        do
        {
          if ( v95[6] < v269.m128i_i64[0] )
          {
            v95 = (_QWORD *)v95[3];
          }
          else
          {
            v96 = v95;
            v95 = (_QWORD *)v95[2];
          }
        }
        while ( v95 );
        if ( (_QWORD *)(a1 + 688) != v96 && v269.m128i_i64[0] >= v96[6] )
          goto LABEL_23;
      }
    }
    else
    {
      v25 = *(_QWORD *)(a1 + 616);
      v26 = v25 + 24LL * *(unsigned int *)(a1 + 624);
      if ( v25 != v26 )
      {
        while ( *(_QWORD *)(v25 + 16) != v269.m128i_i64[0] )
        {
          v25 += 24;
          if ( v26 == v25 )
            goto LABEL_164;
        }
        if ( v26 != v25 )
        {
LABEL_23:
          sub_D68D70(&v268);
          goto LABEL_24;
        }
      }
    }
LABEL_164:
    v215 = v24;
    sub_D68D70(&v268);
    v78 = v215;
    v79 = *(_QWORD *)(v215 - 32LL * (*(_DWORD *)(v215 + 4) & 0x7FFFFFF));
    if ( v79 )
    {
      v268 = 0u;
      v269.m128i_i64[0] = v79;
      if ( v79 != -4096 && v79 != -8192 )
      {
        sub_BD73F0((__int64)&v268);
        v78 = v215;
      }
    }
    else
    {
      v268 = 0u;
      v269.m128i_i64[0] = 0;
    }
    v216 = v78;
    v80 = sub_2D749D0(a1 + 568, &v268);
    v81 = v216;
    v82 = v80;
    v83 = *(unsigned int *)(v80 + 8);
    v84 = *(unsigned int *)(v82 + 12);
    v85 = v83 + 1;
    v86 = v83;
    if ( v83 + 1 > v84 )
    {
      v116 = *(_QWORD *)v82;
      if ( *(_QWORD *)v82 > (unsigned __int64)v236 || (v206 = *(_QWORD *)v82, (unsigned __int64)v236 >= v116 + 32 * v83) )
      {
        sub_2D68450(v82, v85, v84, v216, v85, v116);
        v83 = *(unsigned int *)(v82 + 8);
        v87 = *(_QWORD *)v82;
        v88 = v236;
        v81 = v216;
        v86 = *(_DWORD *)(v82 + 8);
      }
      else
      {
        sub_2D68450(v82, v85, v84, v216, v85, v116);
        v87 = *(_QWORD *)v82;
        v83 = *(unsigned int *)(v82 + 8);
        v81 = v216;
        v86 = *(_DWORD *)(v82 + 8);
        v88 = (_QWORD *)((char *)v236 + *(_QWORD *)v82 - v206);
      }
    }
    else
    {
      v87 = *(_QWORD *)v82;
      v88 = v236;
    }
    v89 = (unsigned __int64 *)(v87 + 32 * v83);
    if ( v89 )
    {
      v193 = v81;
      v205 = v88;
      v217 = v89;
      sub_D68CD0(v89, 0, v88);
      v81 = v193;
      v217[3] = v205[3];
      v86 = *(_DWORD *)(v82 + 8);
    }
    v218 = v81;
    *(_DWORD *)(v82 + 8) = v86 + 1;
    sub_D68D70(&v268);
    v268 = 0u;
    v90 = *(_DWORD *)(a1 + 744);
    v269.m128i_i64[0] = v218;
    if ( v210 )
    {
      v219 = v90;
      sub_BD73F0((__int64)&v268);
      v90 = v219;
    }
    v269.m128i_i32[2] = v90;
    if ( (unsigned __int8)sub_2D67BB0(a1 + 728, (__int64)&v268, &v239) )
      goto LABEL_23;
    v91 = *(_DWORD *)(a1 + 752);
    v92 = *(_DWORD *)(a1 + 744);
    v93 = (_BYTE *)v239.m128i_i64[0];
    ++*(_QWORD *)(a1 + 728);
    v94 = v92 + 1;
    v242[0] = v93;
    if ( 4 * v94 >= 3 * v91 )
    {
      v91 *= 2;
    }
    else if ( v91 - *(_DWORD *)(a1 + 748) - v94 > v91 >> 3 )
    {
LABEL_177:
      *(_DWORD *)(a1 + 744) = v94;
      if ( *((_QWORD *)v93 + 2) != -4096 )
        --*(_DWORD *)(a1 + 748);
      sub_2D57220(v93, v269.m128i_i64[0]);
      *((_DWORD *)v93 + 6) = v269.m128i_i32[2];
      goto LABEL_23;
    }
    sub_2D6E640(a1 + 728, v91);
    sub_2D67BB0(a1 + 728, (__int64)&v268, v242);
    v93 = v242[0];
    v94 = *(_DWORD *)(a1 + 744) + 1;
    goto LABEL_177;
  }
LABEL_24:
  v27 = v289;
  v250.m128i_i64[1] = v11;
  if ( v289 )
  {
    if ( !(v248.m128i_i64[1] | v247.m128i_i64[1]) )
    {
      if ( v247.m128i_i64[0] )
        v27 = v249.m128i_i64[1] == 0;
      goto LABEL_28;
    }
    v28 = (unsigned int)v286;
    v289 = 0;
    if ( !(_DWORD)v286 )
    {
LABEL_187:
      if ( HIDWORD(v286) )
      {
        v97 = (__m128i *)v285;
        if ( v285 )
        {
          *(__m128i *)v285 = _mm_loadu_si128(&v247);
          LODWORD(v28) = v286;
          v97[1] = _mm_loadu_si128(&v248);
          v97[2] = _mm_loadu_si128(&v249);
          v97[3] = _mm_loadu_si128(&v250);
          v97[4].m128i_i64[0] = v251;
        }
        LODWORD(v286) = v28 + 1;
      }
      else
      {
        v130 = _mm_loadu_si128(&v248);
        v268 = _mm_loadu_si128(&v247);
        v131 = _mm_loadu_si128(&v249);
        v269 = v130;
        v132 = _mm_loadu_si128(&v250);
        v272 = v251;
        v270 = v131;
        v271 = v132;
        sub_C8D5F0((__int64)&v285, v287, 1u, 0x48u, v22, v23);
        v133 = (__m128i *)(v285 + 72LL * (unsigned int)v286);
        *v133 = _mm_loadu_si128(&v268);
        v133[1] = _mm_loadu_si128(&v269);
        v133[2] = _mm_loadu_si128(&v270);
        v133[3] = _mm_loadu_si128(&v271);
        v133[4].m128i_i64[0] = v272;
        LODWORD(v286) = v286 + 1;
      }
LABEL_60:
      if ( v237 != -4096 && v237 != 0 && v237 != -8192 )
        sub_BD60C0(v236);
LABEL_63:
      v10 = v253;
      goto LABEL_64;
    }
  }
  else
  {
LABEL_28:
    v28 = (unsigned int)v286;
    v289 = v27;
    if ( !(_DWORD)v286 )
      goto LABEL_187;
  }
  v29 = v285;
  v30 = *(_QWORD *)(v285 + 40);
  if ( !v30 || !v249.m128i_i64[1] || (v31 = 255, *(_QWORD *)(v30 + 8) == *(_QWORD *)(v249.m128i_i64[1] + 8)) )
  {
    v32 = *(_QWORD *)v285;
    if ( !*(_QWORD *)v285
      || !v247.m128i_i64[0]
      || (v31 = 255, *(_QWORD *)(v32 + 8) == *(_QWORD *)(v247.m128i_i64[0] + 8)) )
    {
      v33 = *(_QWORD *)(v285 + 48);
      if ( !v33 || !v250.m128i_i64[0] || (v31 = 255, *(_QWORD *)(v33 + 8) == *(_QWORD *)(v250.m128i_i64[0] + 8)) )
      {
        v31 = 255;
        if ( *(_BYTE *)(v285 + 64) == (_BYTE)v251 )
        {
          v31 = v249.m128i_i64[1] != v30;
          if ( v32 != v247.m128i_i64[0] )
            v31 |= 2u;
          if ( *(_QWORD *)(v285 + 8) != v247.m128i_i64[1] )
            v31 |= 4u;
          if ( v33 != v250.m128i_i64[0] )
            v31 |= 8u;
          v34 = *(_QWORD *)(v285 + 24);
          if ( v34 && v248.m128i_i64[1] && v34 != v248.m128i_i64[1] )
            v31 |= 0x10u;
          if ( (int)sub_39FAC40(v31) > 1 )
            v31 = 255;
        }
      }
    }
  }
  if ( !v288 )
  {
    v288 = v31;
    goto LABEL_53;
  }
  if ( v288 == v31 )
  {
LABEL_53:
    if ( v31 == 255 || v31 == 16 )
      goto LABEL_68;
    if ( v31 == 4 )
    {
      if ( v250.m128i_i64[0] )
        goto LABEL_68;
    }
    else if ( v31 == 2 && v248.m128i_i8[0] )
    {
      goto LABEL_68;
    }
    v35 = (__m128i *)(v29 + 72 * v28);
    if ( v28 >= HIDWORD(v286) )
    {
      v134 = _mm_loadu_si128(&v248);
      v135 = v28 + 1;
      v136 = &v268;
      v137 = _mm_loadu_si128(&v249);
      v138 = _mm_loadu_si128(&v250);
      v268 = _mm_loadu_si128(&v247);
      v269 = v134;
      v272 = v251;
      v270 = v137;
      v271 = v138;
      if ( HIDWORD(v286) < v28 + 1 )
      {
        if ( v29 > (unsigned __int64)&v268 || v35 <= &v268 )
        {
          v136 = &v268;
          sub_C8D5F0((__int64)&v285, v287, v28 + 1, 0x48u, v135, v23);
          v29 = v285;
          v28 = (unsigned int)v286;
        }
        else
        {
          v185 = &v268.m128i_i8[-v29];
          sub_C8D5F0((__int64)&v285, v287, v28 + 1, 0x48u, v135, v23);
          v29 = v285;
          v28 = (unsigned int)v286;
          v136 = (__m128i *)&v185[v285];
        }
      }
      v139 = (__m128i *)(v29 + 72 * v28);
      *v139 = _mm_loadu_si128(v136);
      v139[1] = _mm_loadu_si128(v136 + 1);
      v139[2] = _mm_loadu_si128(v136 + 2);
      v139[3] = _mm_loadu_si128(v136 + 3);
      v139[4].m128i_i64[0] = v136[4].m128i_i64[0];
      LODWORD(v286) = v286 + 1;
    }
    else
    {
      *v35 = _mm_loadu_si128(&v247);
      v35[1] = _mm_loadu_si128(&v248);
      v35[2] = _mm_loadu_si128(&v249);
      v35[3] = _mm_loadu_si128(&v250);
      v35[4].m128i_i64[0] = v251;
      LODWORD(v286) = v286 + 1;
    }
    goto LABEL_60;
  }
  v288 = 255;
LABEL_68:
  LODWORD(v286) = 0;
  sub_D68D70(v236);
LABEL_69:
  if ( !(_DWORD)v286 )
    goto LABEL_228;
  if ( (unsigned int)v286 == 1 || !v288 )
    goto LABEL_105;
  if ( v289 || byte_5017568 )
    goto LABEL_228;
  if ( v288 == 4 )
  {
    v36 = byte_5017108;
  }
  else if ( v288 > 4 )
  {
    v36 = byte_5017028;
    if ( v288 != 8 )
      goto LABEL_228;
  }
  else if ( v288 == 1 )
  {
    v36 = byte_50172C8;
  }
  else
  {
    if ( v288 != 2 )
      goto LABEL_228;
    v36 = byte_50171E8;
  }
  if ( !v36 )
    goto LABEL_228;
  v247 = 0u;
  v268.m128i_i64[0] = (__int64)&v269;
  v268.m128i_i64[1] = 0x200000000LL;
  v248.m128i_i64[0] = 0;
  v248.m128i_i32[2] = 0;
  v37 = *(_QWORD *)(*(_QWORD *)(v285 + 56) + 8LL);
  v38 = sub_AE4450(*v291, v37);
  v41 = (__int64 *)v285;
  v226 = v38;
  v42 = (__m128i *)(v285 + 72LL * (unsigned int)v286);
  if ( (__m128i *)v285 == v42 )
  {
LABEL_267:
    v140 = (__int64 *)v268.m128i_i64[0];
    v51 = (__m128i *)(v268.m128i_i64[0] + 8LL * v268.m128i_u32[2]);
    if ( (__m128i *)v268.m128i_i64[0] == v51 )
    {
LABEL_97:
      if ( v51 != &v269 )
        _libc_free((unsigned __int64)v51);
      v293 = (_QWORD *)sub_2D75700((__int64 *)&v285, (__int64)&v247);
      if ( !v293 )
        goto LABEL_314;
      v52 = (__int64 *)v285;
      if ( v288 == 4 )
      {
        *(_QWORD *)(v285 + 48) = v293;
        v52[3] = 1;
        v52[1] = 0;
        goto LABEL_104;
      }
      if ( v288 > 4 )
      {
        if ( v288 == 8 )
        {
          v178 = *(_QWORD *)(v285 + 24) == 0;
          *(_QWORD *)(v285 + 48) = v293;
          if ( v178 )
          {
            v179 = &v52[9 * (unsigned int)v286];
            if ( v52 != v179 )
            {
              v180 = v52;
              while ( 1 )
              {
                v180 += 9;
                if ( v179 == v180 )
                  break;
                v181 = v180[3];
                if ( v181 )
                {
                  v52[3] = v181;
                  goto LABEL_104;
                }
              }
            }
          }
          goto LABEL_104;
        }
      }
      else
      {
        if ( v288 == 1 )
        {
          *(_QWORD *)(v285 + 40) = v293;
LABEL_104:
          v53 = v293;
          sub_C7D6A0(v247.m128i_i64[1], 16LL * v248.m128i_u32[2], 8);
          if ( !v53 )
            goto LABEL_228;
LABEL_105:
          v54 = (__int64)v258;
          v55 = &v258[8 * (unsigned int)v259];
          v221 = (_DWORD)v259 != 0;
          while ( (_BYTE *)v54 != v55 )
          {
            while ( 1 )
            {
              v56 = *((_QWORD *)v55 - 1);
              v55 -= 8;
              if ( !v56 )
                break;
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v56 + 8LL))(v56);
              if ( (_BYTE *)v54 == v55 )
                goto LABEL_109;
            }
          }
LABEL_109:
          LODWORD(v259) = 0;
          v57 = *(_QWORD *)(v285 + 40);
          v234 = *(_QWORD *)v285;
          v211 = *(_QWORD *)(v285 + 8);
          v227 = *(_QWORD *)(v285 + 24);
          v214 = *(_QWORD *)(v285 + 48);
          v194 = *(_BYTE *)(v285 + 64);
          if ( !v195 )
          {
            v58 = v255;
            v59 = 8LL * (unsigned int)v256;
            v60 = &v255[v59];
            v61 = v59 >> 3;
            if ( v59 >> 5 )
            {
              v62 = &v255[32 * (v59 >> 5)];
              v63 = *(_QWORD *)(a2 + 40);
              while ( *(_BYTE *)*v58 <= 0x1Cu || v63 == *(_QWORD *)(*v58 + 40LL) )
              {
                v64 = v58[1];
                if ( *(_BYTE *)v64 > 0x1Cu && v63 != *(_QWORD *)(v64 + 40) )
                {
                  ++v58;
                  goto LABEL_142;
                }
                v65 = v58[2];
                if ( *(_BYTE *)v65 > 0x1Cu && v63 != *(_QWORD *)(v65 + 40) )
                {
                  v58 += 2;
                  goto LABEL_142;
                }
                v66 = v58[3];
                if ( *(_BYTE *)v66 > 0x1Cu && v63 != *(_QWORD *)(v66 + 40) )
                {
                  v58 += 3;
                  goto LABEL_142;
                }
                v58 += 4;
                if ( v62 == v58 )
                {
                  v61 = v60 - v58;
                  goto LABEL_122;
                }
              }
              goto LABEL_142;
            }
LABEL_122:
            if ( v61 == 2 )
            {
              v155 = *(_QWORD *)(a2 + 40);
              goto LABEL_398;
            }
            if ( v61 != 3 )
            {
              if ( v61 != 1 )
                goto LABEL_125;
              v155 = *(_QWORD *)(a2 + 40);
              goto LABEL_303;
            }
            v155 = *(_QWORD *)(a2 + 40);
            if ( *(_BYTE *)*v58 <= 0x1Cu || v155 == *(_QWORD *)(*v58 + 40LL) )
            {
              ++v58;
LABEL_398:
              if ( *(_BYTE *)*v58 <= 0x1Cu || v155 == *(_QWORD *)(*v58 + 40LL) )
              {
                ++v58;
LABEL_303:
                v67 = v258;
                if ( *(_BYTE *)*v58 <= 0x1Cu || v155 == *(_QWORD *)(*v58 + 40LL) )
                  goto LABEL_126;
              }
            }
LABEL_142:
            if ( v60 == v58 )
              goto LABEL_125;
          }
          sub_23D0AB0((__int64)&v268, a2, 0, 0, 0);
          v200 = a1 + 104;
          v69 = sub_2D737B0(a1 + 104, a3);
          sub_D68CD0((unsigned __int64 *)&v239, 3u, v69);
          if ( v240 == 0
            || (__int64 (__fastcall *)(const __m128i **, const __m128i *, int))((char *)v240 + 4096) == 0
            || v240 == (__int64 (__fastcall *)(const __m128i **, const __m128i *, int))-8192LL )
          {
            v247 = (__m128i)6uLL;
            v248.m128i_i64[0] = 0;
            sub_D68D70(&v247);
            v71 = *(_QWORD *)(a3 + 8);
            v204 = sub_AE4450(*(_QWORD *)(a1 + 816), v71);
          }
          else
          {
            sub_D68CD0((unsigned __int64 *)&v247, 3u, &v239);
            v70 = v248.m128i_i64[0];
            sub_D68D70(&v247);
            v71 = *(_QWORD *)(a3 + 8);
            v204 = sub_AE4450(*(_QWORD *)(a1 + 816), v71);
            if ( v70 )
              goto LABEL_146;
          }
          if ( !byte_50180C8 )
          {
            v197 = sub_C52410();
            v119 = sub_C959E0();
            v120 = (_QWORD *)v197[2];
            v121 = v197 + 1;
            if ( v120 )
            {
              v122 = v197 + 1;
              do
              {
                v71 = v120[3];
                if ( v119 > v120[4] )
                {
                  v120 = (_QWORD *)v120[3];
                }
                else
                {
                  v122 = v120;
                  v120 = (_QWORD *)v120[2];
                }
              }
              while ( v120 );
              if ( v122 != v121 && v119 >= v122[4] )
                v121 = v122;
            }
            if ( v121 != (_QWORD *)((char *)sub_C52410() + 8) )
            {
              v153 = v121[7];
              v71 = (__int64)(v121 + 6);
              if ( v153 )
              {
                v154 = v121 + 6;
                do
                {
                  if ( *(_DWORD *)(v153 + 32) < dword_5018048 )
                  {
                    v153 = *(_QWORD *)(v153 + 24);
                  }
                  else
                  {
                    v154 = (_QWORD *)v153;
                    v153 = *(_QWORD *)(v153 + 16);
                  }
                }
                while ( v153 );
                if ( (_QWORD *)v71 != v154 && dword_5018048 >= *((_DWORD *)v154 + 8) )
                {
                  v71 = *((unsigned int *)v154 + 9);
                  if ( (_DWORD)v71 )
                    goto LABEL_321;
                }
              }
            }
            v158 = *(_QWORD *)(a1 + 8);
            v159 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v158 + 416LL);
            if ( !(v159 == sub_2D56570
                 ? (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v158 + 408LL))(v158, v71)
                 : ((unsigned __int8 (__fastcall *)(__int64, __int64))v159)(v158, v71)) )
            {
LABEL_321:
              if ( v57 )
              {
                v161 = *(_QWORD *)(v57 + 8);
                if ( v227 )
                {
                  v162 = *(_QWORD *)(v214 + 8);
                  if ( v161 && *(_BYTE *)(v161 + 8) != 14 )
                    v161 = 0;
LABEL_326:
                  if ( v162 && *(_BYTE *)(v162 + 8) != 14 )
                    v162 = 0;
                  v163 = *(_QWORD *)(a1 + 816);
                  v164 = *(_QWORD *)(a3 + 8);
                  if ( *(_BYTE *)(v164 + 8) == 14 )
                  {
LABEL_330:
                    v207 = v161;
                    v71 = *(_DWORD *)(v164 + 8) >> 8;
                    if ( !*((_BYTE *)sub_AE2980(v163, v71) + 16) )
                    {
                      v163 = *(_QWORD *)(a1 + 816);
                      v161 = v207;
                      goto LABEL_332;
                    }
LABEL_161:
                    sub_D68D70(&v239);
                    sub_F94A20(&v268, v71);
                    goto LABEL_229;
                  }
LABEL_332:
                  if ( v161 )
                  {
LABEL_333:
                    v71 = *(_DWORD *)(v161 + 8) >> 8;
                    if ( *((_BYTE *)sub_AE2980(v163, v71) + 16) )
                      goto LABEL_161;
                    v163 = *(_QWORD *)(a1 + 816);
                  }
                  if ( v162 )
                  {
                    v71 = *(_DWORD *)(v162 + 8) >> 8;
                    if ( *((_BYTE *)sub_AE2980(v163, v71) + 16) )
                      goto LABEL_161;
                    v163 = *(_QWORD *)(a1 + 816);
                  }
LABEL_338:
                  if ( v234 )
                  {
                    v71 = *(_DWORD *)(*(_QWORD *)(v234 + 8) + 8LL) >> 8;
                    if ( *((_BYTE *)sub_AE2980(v163, v71) + 16) )
                      goto LABEL_161;
                    v163 = *(_QWORD *)(a1 + 816);
                  }
                  v71 = *(_QWORD *)(a3 + 8);
                  v165 = (__int64 **)sub_AE4450(v163, v71);
                  v166 = (__int64)v165;
                  if ( v57 )
                  {
                    v167 = *(_QWORD *)(v57 + 8);
                    v168 = v57;
                    if ( *(_BYTE *)(v167 + 8) == 14 )
                    {
                      v247.m128i_i64[0] = (__int64)"sunkaddr";
                      v71 = 47;
                      v249.m128i_i16[0] = 259;
                      v168 = sub_2D5B7B0(
                               v268.m128i_i64,
                               0x2Fu,
                               v57,
                               (__int64 **)v166,
                               (__int64)&v247,
                               0,
                               (int)v242[0],
                               0);
                      v167 = *(_QWORD *)(v168 + 8);
                    }
                    if ( v166 != v167 )
                    {
                      v247.m128i_i64[0] = (__int64)"sunkaddr";
                      v249.m128i_i16[0] = 259;
                      v198 = v168;
                      v208 = sub_BCB060(*(_QWORD *)(v168 + 8));
                      v71 = (unsigned int)sub_BCB060(v166) < v208 ? 38 : 40;
                      v168 = sub_2D5B7B0(
                               v268.m128i_i64,
                               v71,
                               v198,
                               (__int64 **)v166,
                               (__int64)&v247,
                               0,
                               (int)v242[0],
                               0);
                    }
                    v169 = (_QWORD *)v168;
                    if ( v227 )
                    {
LABEL_347:
                      v170 = *(_QWORD *)(v214 + 8);
                      if ( v166 == v170 )
                      {
                        v168 = v214;
                      }
                      else if ( *(_BYTE *)(v170 + 8) == 14 )
                      {
                        v247.m128i_i64[0] = (__int64)"sunkaddr";
                        v225 = v169;
                        v249.m128i_i16[0] = 259;
                        v190 = sub_2D5B7B0(
                                 v268.m128i_i64,
                                 0x2Fu,
                                 v214,
                                 (__int64 **)v166,
                                 (__int64)&v247,
                                 0,
                                 (int)v242[0],
                                 0);
                        v169 = v225;
                        v168 = v190;
                      }
                      else
                      {
                        if ( *(_DWORD *)(v170 + 8) >> 8 <= *(_DWORD *)(v166 + 8) >> 8 )
                        {
                          if ( v169 && *(_BYTE *)v169 > 0x1Cu && v169 != (_QWORD *)v57 )
                            sub_B43D60(v169);
                          goto LABEL_161;
                        }
                        v223 = v169;
                        v247.m128i_i64[0] = (__int64)"sunkaddr";
                        v249.m128i_i16[0] = 259;
                        v171 = sub_A82DA0((unsigned int **)&v268, v214, v166, (__int64)&v247, 0, 0);
                        v169 = v223;
                        v168 = v171;
                      }
                      v71 = v227;
                      if ( v227 != 1 )
                      {
                        v229 = v169;
                        v224 = (_BYTE *)v168;
                        v247.m128i_i64[0] = (__int64)"sunkaddr";
                        v249.m128i_i16[0] = 259;
                        v172 = (_BYTE *)sub_AD64C0(v166, v71, 0);
                        v71 = (__int64)v224;
                        v173 = sub_A81850((unsigned int **)&v268, v224, v172, (__int64)&v247, 0, 0);
                        v169 = v229;
                        v168 = v173;
                      }
                      if ( v169 )
                      {
                        v71 = (__int64)v169;
                        v247.m128i_i64[0] = (__int64)"sunkaddr";
                        v249.m128i_i16[0] = 259;
                        v168 = sub_929C50((unsigned int **)&v268, v169, (_BYTE *)v168, (__int64)&v247, 0, 0);
                      }
                    }
                    v174 = (_BYTE *)v168;
                    if ( v234 )
                    {
                      if ( (*(_BYTE *)(v234 + 33) & 0x1C) == 0 )
                      {
LABEL_357:
                        v71 = 47;
                        v230 = (_BYTE *)v168;
                        v247.m128i_i64[0] = (__int64)"sunkaddr";
                        v249.m128i_i16[0] = 259;
                        v174 = (_BYTE *)sub_2D5B7B0(
                                          v268.m128i_i64,
                                          0x2Fu,
                                          v234,
                                          (__int64 **)v166,
                                          (__int64)&v247,
                                          0,
                                          (int)v242[0],
                                          0);
                        if ( v230 )
                        {
                          v71 = (__int64)v230;
                          v247.m128i_i64[0] = (__int64)"sunkaddr";
                          v249.m128i_i16[0] = 259;
                          v174 = (_BYTE *)sub_929C50((unsigned int **)&v268, v230, v174, (__int64)&v247, 0, 0);
                        }
                        goto LABEL_359;
                      }
LABEL_473:
                      v232 = v168;
                      v188 = sub_B34A60((__int64)&v268, (unsigned __int8 *)v234);
                      v168 = v232;
                      v234 = v188;
                      goto LABEL_357;
                    }
LABEL_359:
                    if ( v211 )
                    {
                      v71 = v211;
                      v192 = (_BYTE *)sub_AD64C0(v166, v211, 0);
                      if ( v174 )
                      {
                        v71 = (__int64)v174;
                        v247.m128i_i64[0] = (__int64)"sunkaddr";
                        v249.m128i_i16[0] = 259;
                        v174 = (_BYTE *)sub_929C50((unsigned int **)&v268, v174, v192, (__int64)&v247, 0, 0);
                      }
                      else
                      {
                        v174 = v192;
                      }
                    }
LABEL_360:
                    if ( v174 )
                    {
                      v175 = *(__int64 ***)(a3 + 8);
                      v247.m128i_i64[0] = (__int64)"sunkaddr";
                      v249.m128i_i16[0] = 259;
                      v70 = sub_2D5B7B0(
                              v268.m128i_i64,
                              0x30u,
                              (unsigned __int64)v174,
                              v175,
                              (__int64)&v247,
                              0,
                              (int)v242[0],
                              0);
                      goto LABEL_156;
                    }
                  }
                  else
                  {
                    if ( v227 )
                    {
                      v169 = 0;
                      goto LABEL_347;
                    }
                    if ( v234 )
                    {
                      if ( (*(_BYTE *)(v234 + 33) & 0x1C) == 0 )
                      {
                        v247.m128i_i64[0] = (__int64)"sunkaddr";
                        v71 = 47;
                        v249.m128i_i16[0] = 259;
                        v174 = (_BYTE *)sub_2D5B7B0(
                                          v268.m128i_i64,
                                          0x2Fu,
                                          v234,
                                          v165,
                                          (__int64)&v247,
                                          0,
                                          (int)v242[0],
                                          0);
                        goto LABEL_359;
                      }
                      v168 = 0;
                      goto LABEL_473;
                    }
                    if ( v211 )
                    {
                      v71 = v211;
                      v174 = (_BYTE *)sub_AD64C0((__int64)v165, v211, 0);
                      goto LABEL_360;
                    }
                  }
LABEL_378:
                  v70 = sub_AD6530(*(_QWORD *)(a3 + 8), v71);
LABEL_156:
                  sub_BD2ED0(a2, a3, v70);
                  v248.m128i_i64[0] = v70;
                  v247 = (__m128i)6uLL;
                  if ( v70 )
                  {
LABEL_157:
                    if ( v70 != -8192 && v70 != -4096 )
                      sub_BD73F0((__int64)&v247);
                  }
                  v77 = sub_2D737B0(v200, a3);
                  v71 = (__int64)&v247;
                  sub_2D57190(v77, &v247);
                  sub_D68D70(&v247);
                  v221 = 1;
                  if ( !*(_QWORD *)(a3 + 16) )
                  {
                    v126 = *(_QWORD *)(a1 + 88);
                    if ( !v126 )
                      BUG();
                    v127 = (__int64 (__fastcall *)(const __m128i **, const __m128i *, int))(v126 - 24);
                    v128 = *(_QWORD *)(v126 + 16);
                    v242[0] = (_BYTE *)6;
                    v242[1] = 0;
                    v235 = v128;
                    v243 = v127;
                    if ( v127 != (__int64 (__fastcall *)(const __m128i **, const __m128i *, int))-8192LL
                      && v127 != (__int64 (__fastcall *)(const __m128i **, const __m128i *, int))-4096LL )
                    {
                      sub_BD73F0((__int64)v242);
                    }
                    v71 = *(_QWORD *)(a1 + 48);
                    v247.m128i_i64[0] = a1;
                    v248.m128i_i64[1] = (__int64)sub_2D69700;
                    v248.m128i_i64[0] = (__int64)sub_2D56B40;
                    sub_F5CAB0((char *)a3, (__int64 *)v71, 0, (__int64)&v247);
                    sub_A17130((__int64)&v247);
                    if ( v127 != v243 )
                    {
                      v129 = *(_QWORD *)(v235 + 56);
                      *(_WORD *)(a1 + 96) = 1;
                      *(_QWORD *)(a1 + 88) = v129;
                      sub_2D69E90(v200);
                    }
                    sub_D68D70(v242);
                    v221 = 1;
                  }
                  goto LABEL_161;
                }
                if ( v161 && *(_BYTE *)(v161 + 8) == 14 )
                {
                  v163 = *(_QWORD *)(a1 + 816);
                  v164 = *(_QWORD *)(a3 + 8);
                  if ( *(_BYTE *)(v164 + 8) == 14 )
                  {
                    v162 = 0;
                    goto LABEL_330;
                  }
                  v162 = 0;
                  goto LABEL_333;
                }
              }
              else if ( v227 )
              {
                v161 = 0;
                v162 = *(_QWORD *)(v214 + 8);
                goto LABEL_326;
              }
              v163 = *(_QWORD *)(a1 + 816);
              v164 = *(_QWORD *)(a3 + 8);
              if ( *(_BYTE *)(v164 + 8) == 14 )
              {
                v161 = 0;
                v162 = 0;
                goto LABEL_330;
              }
              goto LABEL_338;
            }
          }
          if ( v57 )
          {
            if ( *(_BYTE *)(*(_QWORD *)(v57 + 8) + 8LL) != 14 )
            {
              if ( !v227 )
              {
                if ( v234 )
                  goto LABEL_205;
                v183 = *(_QWORD *)(a3 + 8);
                if ( *(_BYTE *)(v183 + 8) == 14 )
                {
                  v157 = v57;
                  v71 = *(_DWORD *)(v183 + 8) >> 8;
                  if ( *((_BYTE *)sub_AE2980(*(_QWORD *)(a1 + 816), v71) + 16) )
                    goto LABEL_161;
LABEL_431:
                  v71 = 48;
                  v184 = *(__int64 ***)(a3 + 8);
                  v247.m128i_i64[0] = (__int64)"sunkaddr";
                  v249.m128i_i16[0] = 259;
                  v234 = sub_2D5B7B0(v268.m128i_i64, 0x30u, v157, v184, (__int64)&v247, 0, (int)v242[0], 0);
                  if ( v234 )
                  {
LABEL_432:
                    v57 = 0;
                    v104 = *(_QWORD *)(a3 + 8);
                    if ( (unsigned int)*(unsigned __int8 *)(v104 + 8) - 17 > 1 )
                    {
                      v70 = v234;
                      v222 = sub_BCE3C0(v273, *(_DWORD *)(v104 + 8) >> 8);
LABEL_434:
                      v107 = 0;
                      if ( !v227 )
                      {
LABEL_435:
                        if ( v211 )
                        {
                          v57 = sub_AD64C0(v204, v211, 0);
                          goto LABEL_223;
                        }
                        goto LABEL_146;
                      }
LABEL_216:
                      v57 = v214;
                      if ( *(_QWORD *)(v214 + 8) != v204 )
                      {
                        v108 = v214;
                        v220 = v107;
                        v247.m128i_i64[0] = (__int64)"sunkaddr";
                        v249.m128i_i16[0] = 259;
                        v109 = sub_A82DA0((unsigned int **)&v268, v108, v204, (__int64)&v247, 0, 0);
                        v107 = v220;
                        v57 = v109;
                      }
                      v110 = v227;
                      if ( v227 != 1 )
                      {
                        v228 = v107;
                        v247.m128i_i64[0] = (__int64)"sunkaddr";
                        v249.m128i_i16[0] = 259;
                        v111 = (_BYTE *)sub_AD64C0(v204, v110, 0);
                        v112 = sub_A81850((unsigned int **)&v268, (_BYTE *)v57, v111, (__int64)&v247, 0, 0);
                        v107 = v228;
                        v57 = v112;
                      }
                      if ( v107 )
                      {
                        v247.m128i_i64[0] = (__int64)"sunkaddr";
                        v249.m128i_i16[0] = 259;
                        v57 = sub_929C50((unsigned int **)&v268, v107, (_BYTE *)v57, (__int64)&v247, 0, 0);
                      }
                      goto LABEL_222;
                    }
LABEL_210:
                    v104 = **(_QWORD **)(v104 + 16);
LABEL_211:
                    v70 = v234;
                    v222 = sub_BCE3C0(v273, *(_DWORD *)(v104 + 8) >> 8);
                    if ( !v57 )
                      goto LABEL_434;
                    if ( *(_QWORD *)(v57 + 8) != v204 )
                    {
                      v247.m128i_i64[0] = (__int64)"sunkaddr";
                      v249.m128i_i16[0] = 259;
                      v196 = sub_BCB060(*(_QWORD *)(v57 + 8));
                      v106 = sub_BCB060(v204);
                      v57 = sub_2D5B7B0(
                              v268.m128i_i64,
                              v106 < v196 ? 38 : 40,
                              v57,
                              (__int64 **)v204,
                              (__int64)&v247,
                              0,
                              (int)v242[0],
                              0);
                    }
                    if ( v227 )
                    {
                      v107 = (_BYTE *)v57;
                      goto LABEL_216;
                    }
LABEL_222:
                    if ( v211 )
                    {
                      v231 = sub_AD64C0(v204, v211, 0);
                      if ( v57 )
                      {
                        if ( *(_QWORD *)(v234 + 8) != v222 )
                        {
                          v249.m128i_i16[0] = 257;
                          v70 = (__int64)sub_94BCF0((unsigned int **)&v268, v234, v222, (__int64)&v247);
                        }
                        v242[0] = (_BYTE *)v57;
                        v247.m128i_i64[0] = (__int64)"sunkaddr";
                        v249.m128i_i16[0] = 259;
                        v182 = sub_BCB2B0(v273);
                        v57 = v231;
                        v70 = sub_921130((unsigned int **)&v268, v182, v70, v242, 1, (__int64)&v247, v194 != 0 ? 3 : 0);
                      }
                      else
                      {
                        v57 = v231;
                      }
                    }
LABEL_223:
                    if ( v57 )
                    {
                      if ( v222 != *(_QWORD *)(v70 + 8) )
                      {
                        v249.m128i_i16[0] = 257;
                        v70 = (__int64)sub_94BCF0((unsigned int **)&v268, v70, v222, (__int64)&v247);
                      }
                      v242[0] = (_BYTE *)v57;
                      v247.m128i_i64[0] = (__int64)"sunkaddr";
                      v249.m128i_i16[0] = 259;
                      v113 = sub_BCB2B0(v273);
                      v70 = sub_921130((unsigned int **)&v268, v113, v70, v242, 1, (__int64)&v247, v194 != 0 ? 3 : 0);
                    }
LABEL_146:
                    v72 = *(_QWORD *)(v70 + 8);
                    v73 = *(_QWORD *)(a3 + 8);
                    if ( v73 != v72 )
                    {
                      if ( (unsigned int)*(unsigned __int8 *)(v72 + 8) - 17 <= 1 )
                        v72 = **(_QWORD **)(v72 + 16);
                      v74 = *(_QWORD *)(a3 + 8);
                      v75 = *(unsigned __int8 *)(v73 + 8);
                      v76 = *(_DWORD *)(v72 + 8) >> 8;
                      if ( (unsigned int)(v75 - 17) <= 1 )
                        v74 = **(_QWORD **)(v73 + 16);
                      if ( *(_DWORD *)(v74 + 8) >> 8 != v76 )
                      {
                        if ( (_BYTE)v75 != 14
                          || !*((_BYTE *)sub_AE2980(*(_QWORD *)(a1 + 816), *(_DWORD *)(v73 + 8) >> 8) + 16) )
                        {
                          v247.m128i_i64[0] = (__int64)"sunkaddr";
                          v249.m128i_i16[0] = 259;
                          v117 = sub_2D5B7B0(
                                   v268.m128i_i64,
                                   0x2Fu,
                                   v70,
                                   (__int64 **)v204,
                                   (__int64)&v247,
                                   0,
                                   (int)v242[0],
                                   0);
                          v247.m128i_i64[0] = (__int64)"sunkaddr";
                          v118 = *(__int64 ***)(a3 + 8);
                          v249.m128i_i16[0] = 259;
                          v70 = sub_2D5B7B0(v268.m128i_i64, 0x30u, v117, v118, (__int64)&v247, 0, (int)v242[0], 0);
                          goto LABEL_156;
                        }
                        v73 = *(_QWORD *)(a3 + 8);
                      }
                      v249.m128i_i16[0] = 257;
                      v70 = (__int64)sub_94BCF0((unsigned int **)&v268, v70, v73, (__int64)&v247);
                      goto LABEL_156;
                    }
                    sub_BD2ED0(a2, a3, v70);
                    v248.m128i_i64[0] = v70;
                    v247 = (__m128i)6uLL;
                    goto LABEL_157;
                  }
LABEL_426:
                  if ( v227 )
                    goto LABEL_161;
                }
                else
                {
                  v247.m128i_i64[0] = (__int64)"sunkaddr";
                  v71 = 48;
                  v249.m128i_i16[0] = 259;
                  v234 = sub_2D5B7B0(v268.m128i_i64, 0x30u, v57, (__int64 **)v183, (__int64)&v247, 0, (int)v242[0], 0);
                  if ( v234 )
                    goto LABEL_432;
                }
                goto LABEL_377;
              }
              v156 = *(_QWORD *)(v214 + 8);
              if ( *(_BYTE *)(v156 + 8) != 14 )
              {
                v157 = v57;
                v57 = 0;
LABEL_308:
                if ( *(_DWORD *)(v156 + 8) >> 8 < *(_DWORD *)(v204 + 8) >> 8 )
                  goto LABEL_161;
LABEL_309:
                if ( v234 )
                {
                  if ( v57 )
                    goto LABEL_161;
                  v57 = v157;
                  goto LABEL_205;
                }
LABEL_450:
                v186 = *(_QWORD *)(a3 + 8);
                if ( *(_BYTE *)(v186 + 8) != 14
                  || (v71 = *(_DWORD *)(v186 + 8) >> 8, !*((_BYTE *)sub_AE2980(*(_QWORD *)(a1 + 816), v71) + 16)) )
                {
                  if ( !v57 )
                  {
                    if ( v157 )
                      goto LABEL_431;
                    if ( v227 != 1 )
                      goto LABEL_426;
                    v247.m128i_i64[0] = (__int64)"sunkaddr";
                    v71 = 48;
                    v187 = *(__int64 ***)(a3 + 8);
                    v249.m128i_i16[0] = 259;
                    v234 = sub_2D5B7B0(v268.m128i_i64, 0x30u, v214, v187, (__int64)&v247, 0, (int)v242[0], 0);
                    if ( v234 )
                    {
                      v227 = 0;
                      v104 = *(_QWORD *)(a3 + 8);
                      if ( (unsigned int)*(unsigned __int8 *)(v104 + 8) - 17 > 1 )
                      {
                        v70 = v234;
                        v222 = sub_BCE3C0(v273, *(_DWORD *)(v104 + 8) >> 8);
                        goto LABEL_435;
                      }
                      goto LABEL_210;
                    }
LABEL_377:
                    if ( v211 )
                      goto LABEL_161;
                    goto LABEL_378;
                  }
LABEL_428:
                  v234 = v57;
                  v57 = v157;
                  v104 = *(_QWORD *)(a3 + 8);
LABEL_208:
                  v105 = *(unsigned __int8 *)(v104 + 8);
LABEL_209:
                  if ( (unsigned int)(v105 - 17) <= 1 )
                    goto LABEL_210;
                  goto LABEL_211;
                }
LABEL_419:
                if ( !v57 )
                {
                  if ( v157 )
                    goto LABEL_161;
                  goto LABEL_426;
                }
                v234 = v57;
                v57 = v157;
                v104 = *(_QWORD *)(a3 + 8);
                v105 = *(unsigned __int8 *)(v104 + 8);
                goto LABEL_209;
              }
LABEL_459:
              if ( v227 != 1 )
                goto LABEL_161;
              v227 = 0;
              v157 = v57;
              v57 = v214;
              goto LABEL_309;
            }
            if ( !v227 )
            {
              if ( v234 )
                goto LABEL_161;
              v104 = *(_QWORD *)(a3 + 8);
              if ( *(_BYTE *)(v104 + 8) != 14 )
              {
                v189 = v57;
                v57 = 0;
                v234 = v189;
                goto LABEL_208;
              }
              v157 = 0;
              v71 = *(_DWORD *)(v104 + 8) >> 8;
              if ( !*((_BYTE *)sub_AE2980(*(_QWORD *)(a1 + 816), v71) + 16) )
                goto LABEL_428;
              goto LABEL_419;
            }
            v156 = *(_QWORD *)(v214 + 8);
            if ( *(_BYTE *)(v156 + 8) == 14 )
              goto LABEL_161;
          }
          else
          {
            if ( !v227 )
            {
              if ( !v234 )
              {
                v176 = *(_QWORD *)(a3 + 8);
                if ( *(_BYTE *)(v176 + 8) == 14 )
                {
                  v71 = *(_DWORD *)(v176 + 8) >> 8;
                  sub_AE2980(*(_QWORD *)(a1 + 816), v71);
                }
                goto LABEL_377;
              }
LABEL_205:
              if ( (*(_BYTE *)(v234 + 33) & 0x1C) != 0 )
              {
                v71 = v234;
                v157 = v57;
                v57 = sub_B34A60((__int64)&v268, (unsigned __int8 *)v234);
                goto LABEL_450;
              }
              v104 = *(_QWORD *)(a3 + 8);
              v105 = *(unsigned __int8 *)(v104 + 8);
              if ( (_BYTE)v105 == 14 )
              {
                sub_AE2980(*(_QWORD *)(a1 + 816), *(_DWORD *)(v104 + 8) >> 8);
                v104 = *(_QWORD *)(a3 + 8);
                goto LABEL_208;
              }
              goto LABEL_209;
            }
            v156 = *(_QWORD *)(v214 + 8);
            if ( *(_BYTE *)(v156 + 8) == 14 )
              goto LABEL_459;
          }
          v157 = 0;
          goto LABEL_308;
        }
        if ( v288 == 2 )
        {
          *(_QWORD *)(v285 + 40) = v293;
          *v52 = 0;
          goto LABEL_104;
        }
      }
      BUG();
    }
    while ( 2 )
    {
      v239.m128i_i64[0] = *v140;
      v147 = sub_AD6530(v290, v37);
      v37 = v248.m128i_u32[2];
      v148 = v147;
      if ( !v248.m128i_i32[2] )
      {
        ++v247.m128i_i64[0];
        v242[0] = 0;
        goto LABEL_273;
      }
      v141 = v239.m128i_i64[0];
      v142 = 0;
      v143 = 1;
      v144 = (v248.m128i_i32[2] - 1)
           & (((unsigned __int32)v239.m128i_i32[0] >> 9)
            ^ ((unsigned __int32)v239.m128i_i32[0] >> 4));
      v145 = (__m128i *)(v247.m128i_i64[1] + 16LL * v144);
      v146 = v145->m128i_i64[0];
      if ( v239.m128i_i64[0] != v145->m128i_i64[0] )
      {
        while ( v146 != -4096 )
        {
          if ( !v142 && v146 == -8192 )
            v142 = v145;
          v144 = (v248.m128i_i32[2] - 1) & (v143 + v144);
          v145 = (__m128i *)(v247.m128i_i64[1] + 16LL * v144);
          v146 = v145->m128i_i64[0];
          if ( v239.m128i_i64[0] == v145->m128i_i64[0] )
            goto LABEL_270;
          ++v143;
        }
        if ( v142 )
          v145 = v142;
        ++v247.m128i_i64[0];
        v151 = v248.m128i_i32[0] + 1;
        v242[0] = v145;
        if ( 4 * (v248.m128i_i32[0] + 1) >= (unsigned int)(3 * v248.m128i_i32[2]) )
        {
LABEL_273:
          v37 = (unsigned int)(2 * v248.m128i_i32[2]);
          sub_FAA400((__int64)&v247, v37);
          if ( v248.m128i_i32[2] )
          {
            v141 = v239.m128i_i64[0];
            v149 = (v248.m128i_i32[2] - 1)
                 & (((unsigned __int32)v239.m128i_i32[0] >> 9)
                  ^ ((unsigned __int32)v239.m128i_i32[0] >> 4));
            v145 = (__m128i *)(v247.m128i_i64[1] + 16LL * v149);
            v150 = v145->m128i_i64[0];
            if ( v239.m128i_i64[0] == v145->m128i_i64[0] )
            {
LABEL_275:
              v37 = v248.m128i_u32[0];
              v242[0] = v145;
              v151 = v248.m128i_i32[0] + 1;
            }
            else
            {
              v191 = 1;
              v37 = 0;
              while ( v150 != -4096 )
              {
                if ( !v37 && v150 == -8192 )
                  v37 = (__int64)v145;
                v149 = (v248.m128i_i32[2] - 1) & (v191 + v149);
                v145 = (__m128i *)(v247.m128i_i64[1] + 16LL * v149);
                v150 = v145->m128i_i64[0];
                if ( v239.m128i_i64[0] == v145->m128i_i64[0] )
                  goto LABEL_275;
                ++v191;
              }
              if ( !v37 )
                v37 = (__int64)v145;
              v242[0] = (_BYTE *)v37;
              v151 = v248.m128i_i32[0] + 1;
              v145 = (__m128i *)v37;
            }
          }
          else
          {
            v141 = v239.m128i_i64[0];
            v242[0] = 0;
            v151 = v248.m128i_i32[0] + 1;
            v145 = 0;
          }
        }
        else if ( v248.m128i_i32[2] - v248.m128i_i32[1] - v151 <= (unsigned __int32)v248.m128i_i32[2] >> 3 )
        {
          sub_FAA400((__int64)&v247, v248.m128i_i32[2]);
          v37 = (__int64)&v239;
          sub_F9D990((__int64)&v247, v239.m128i_i64, v242);
          v141 = v239.m128i_i64[0];
          v151 = v248.m128i_i32[0] + 1;
          v145 = (__m128i *)v242[0];
        }
        v248.m128i_i32[0] = v151;
        if ( v145->m128i_i64[0] != -4096 )
          --v248.m128i_i32[1];
        v145->m128i_i64[0] = v141;
        v145->m128i_i64[1] = 0;
      }
LABEL_270:
      ++v140;
      v145->m128i_i64[1] = v148;
      if ( v51 == (__m128i *)v140 )
      {
        v51 = (__m128i *)v268.m128i_i64[0];
        goto LABEL_97;
      }
      continue;
    }
  }
  while ( 2 )
  {
    if ( v288 == 4 )
    {
      v37 = v41[1];
      v43 = sub_AD64C0(v226, v37, 0);
    }
    else
    {
      if ( v288 <= 4 )
      {
        if ( v288 == 1 )
        {
          v43 = v41[5];
          break;
        }
        if ( v288 == 2 )
        {
          v43 = *v41;
          break;
        }
LABEL_248:
        v123 = v268.m128i_u32[2];
        v124 = v41[7];
        v125 = v268.m128i_u32[2] + 1LL;
        if ( v125 > v268.m128i_u32[3] )
        {
          v37 = (__int64)&v269;
          sub_C8D5F0((__int64)&v268, &v269, v125, 8u, v39, v40);
          v123 = v268.m128i_u32[2];
        }
        *(_QWORD *)(v268.m128i_i64[0] + 8 * v123) = v124;
        ++v268.m128i_i32[2];
LABEL_91:
        v41 += 9;
        if ( v42 == (__m128i *)v41 )
          goto LABEL_267;
        continue;
      }
      if ( v288 != 8 )
        goto LABEL_248;
      v43 = v41[6];
    }
    break;
  }
  if ( !v43 )
    goto LABEL_248;
  if ( !v290 || *(_QWORD *)(v43 + 8) == v290 )
  {
    v37 = v248.m128i_u32[2];
    v290 = *(_QWORD *)(v43 + 8);
    if ( v248.m128i_i32[2] )
    {
      v44 = v41[7];
      v40 = (unsigned int)(v248.m128i_i32[2] - 1);
      v39 = v247.m128i_i64[1];
      v45 = 1;
      v46 = v40 & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
      v47 = (__int64 *)(v247.m128i_i64[1] + 16LL * v46);
      v48 = 0;
      v49 = *v47;
      if ( v44 == *v47 )
      {
LABEL_89:
        v50 = v47 + 1;
LABEL_90:
        *v50 = v43;
        goto LABEL_91;
      }
      while ( v49 != -4096 )
      {
        if ( v49 == -8192 && !v48 )
          v48 = v47;
        v46 = v40 & (v45 + v46);
        v47 = (__int64 *)(v247.m128i_i64[1] + 16LL * v46);
        v49 = *v47;
        if ( v44 == *v47 )
          goto LABEL_89;
        ++v45;
      }
      if ( !v48 )
        v48 = v47;
      ++v247.m128i_i64[0];
      v152 = v248.m128i_i32[0] + 1;
      if ( 4 * (v248.m128i_i32[0] + 1) < (unsigned int)(3 * v248.m128i_i32[2]) )
      {
        v39 = (unsigned __int32)v248.m128i_i32[2] >> 3;
        if ( v248.m128i_i32[2] - v248.m128i_i32[1] - v152 <= (unsigned int)v39 )
        {
          sub_FAA400((__int64)&v247, v248.m128i_i32[2]);
          if ( !v248.m128i_i32[2] )
          {
LABEL_504:
            ++v248.m128i_i32[0];
            BUG();
          }
          v39 = v41[7];
          v40 = 1;
          v152 = v248.m128i_i32[0] + 1;
          v177 = 0;
          v37 = (v248.m128i_i32[2] - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
          v48 = (__int64 *)(v247.m128i_i64[1] + 16 * v37);
          v44 = *v48;
          if ( v39 != *v48 )
          {
            while ( v44 != -4096 )
            {
              if ( v44 == -8192 && !v177 )
                v177 = v48;
              v37 = (v248.m128i_i32[2] - 1) & (unsigned int)(v40 + v37);
              v48 = (__int64 *)(v247.m128i_i64[1] + 16LL * (unsigned int)v37);
              v44 = *v48;
              if ( v39 == *v48 )
                goto LABEL_289;
              v40 = (unsigned int)(v40 + 1);
            }
            goto LABEL_384;
          }
        }
        goto LABEL_289;
      }
    }
    else
    {
      ++v247.m128i_i64[0];
    }
    sub_FAA400((__int64)&v247, 2 * v248.m128i_i32[2]);
    if ( !v248.m128i_i32[2] )
      goto LABEL_504;
    v39 = v41[7];
    v152 = v248.m128i_i32[0] + 1;
    v37 = (v248.m128i_i32[2] - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
    v48 = (__int64 *)(v247.m128i_i64[1] + 16 * v37);
    v44 = *v48;
    if ( v39 != *v48 )
    {
      v40 = 1;
      v177 = 0;
      while ( v44 != -4096 )
      {
        if ( v44 == -8192 && !v177 )
          v177 = v48;
        v37 = (v248.m128i_i32[2] - 1) & (unsigned int)(v40 + v37);
        v48 = (__int64 *)(v247.m128i_i64[1] + 16LL * (unsigned int)v37);
        v44 = *v48;
        if ( v39 == *v48 )
          goto LABEL_289;
        v40 = (unsigned int)(v40 + 1);
      }
LABEL_384:
      v44 = v39;
      if ( v177 )
        v48 = v177;
    }
LABEL_289:
    v248.m128i_i32[0] = v152;
    if ( *v48 != -4096 )
      --v248.m128i_i32[1];
    *v48 = v44;
    v50 = v48 + 1;
    *v50 = 0;
    goto LABEL_90;
  }
  if ( (__m128i *)v268.m128i_i64[0] != &v269 )
    _libc_free(v268.m128i_u64[0]);
LABEL_314:
  sub_C7D6A0(v247.m128i_i64[1], 16LL * v248.m128i_u32[2], 8);
LABEL_228:
  sub_2D57BD0((__int64 *)&v258, 0);
  v221 = 0;
LABEL_229:
  v67 = v258;
  v114 = &v258[8 * (unsigned int)v259];
  if ( v258 != v114 )
  {
    do
    {
      v115 = *((_QWORD *)v114 - 1);
      v114 -= 8;
      if ( v115 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v115 + 8LL))(v115);
    }
    while ( v67 != v114 );
LABEL_125:
    v67 = v258;
  }
LABEL_126:
  if ( v67 != v260 )
    _libc_free((unsigned __int64)v67);
  if ( v293 && !(unsigned int)sub_BD3960((__int64)v293) && *(_BYTE *)v293 > 0x1Cu )
    sub_B43D60(v293);
  if ( (_BYTE *)v285 != v287 )
    _libc_free(v285);
  if ( v255 != v257 )
    _libc_free((unsigned __int64)v255);
  if ( !v266 )
    _libc_free((unsigned __int64)v263);
  if ( v252 != v254 )
    _libc_free((unsigned __int64)v252);
  return v221;
}
