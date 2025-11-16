// Function: sub_2C4B640
// Address: 0x2c4b640
//
__int64 __fastcall sub_2C4B640(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  _QWORD *v10; // rax
  __int64 v11; // rax
  __int64 v12; // rsi
  const __m128i *v13; // rsi
  char *v14; // rdi
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rdx
  __int64 v19; // r9
  __int64 v20; // rcx
  __int64 v21; // r8
  unsigned __int64 v22; // rbx
  __int64 v23; // rax
  unsigned __int64 v24; // rdi
  __m128i *v25; // rdx
  const __m128i *v26; // rax
  __int64 v27; // r9
  __int64 v28; // rcx
  __int64 v29; // r8
  unsigned __int64 v30; // rbx
  __int64 v31; // rax
  unsigned __int64 v32; // rdi
  __m128i *v33; // rdx
  const __m128i *v34; // rax
  __int64 v35; // r9
  __int64 v36; // rcx
  __int64 v37; // r8
  unsigned __int64 v38; // rbx
  __int64 v39; // rax
  unsigned __int64 v40; // rdi
  __m128i *v41; // rdx
  const __m128i *v42; // rax
  __int64 v43; // r9
  unsigned __int64 v44; // rcx
  __int64 v45; // r8
  unsigned __int64 v46; // rbx
  __int64 v47; // rax
  unsigned __int64 v48; // rdi
  __m128i *v49; // rdx
  const __m128i *v50; // rax
  __int64 v51; // r9
  __int64 v52; // rcx
  __int64 v53; // r8
  unsigned __int64 v54; // rbx
  __int64 v55; // rax
  unsigned __int64 v56; // rdi
  __m128i *v57; // rdx
  const __m128i *v58; // rax
  __int64 v59; // r9
  __int64 v60; // rcx
  __int64 v61; // r8
  unsigned __int64 v62; // rbx
  __int64 v63; // rax
  unsigned __int64 v64; // rdi
  __m128i *v65; // rdx
  const __m128i *v66; // rax
  __int64 v67; // r9
  unsigned __int64 v68; // rcx
  __int64 v69; // r8
  unsigned __int64 v70; // rbx
  __int64 v71; // rax
  unsigned __int64 v72; // rdi
  __m128i *v73; // rdx
  const __m128i *v74; // rax
  __int64 v75; // r9
  unsigned __int64 v76; // rcx
  __int64 v77; // r8
  unsigned __int64 v78; // rbx
  __int64 v79; // rax
  unsigned __int64 v80; // rdi
  __m128i *v81; // rdx
  const __m128i *v82; // rax
  __int64 v83; // r9
  unsigned __int64 v84; // rcx
  __int64 v85; // r8
  unsigned __int64 v86; // rbx
  __int64 v87; // rax
  unsigned __int64 v88; // rdi
  __m128i *v89; // rdx
  const __m128i *v90; // rax
  __int64 v91; // r9
  __int64 v92; // r8
  unsigned __int64 v93; // r12
  __int64 v94; // rax
  __int64 v95; // rcx
  __m128i *v96; // rdx
  const __m128i *v97; // rax
  unsigned __int64 v98; // rsi
  unsigned __int64 v99; // rdx
  const __m128i *v100; // rcx
  const __m128i *v101; // r8
  unsigned __int64 v102; // r13
  __int64 v103; // rax
  unsigned __int64 v104; // rsi
  __m128i *v105; // rdx
  const __m128i *v106; // rax
  __int64 v107; // rsi
  __int64 v108; // rdx
  __int64 v109; // rcx
  __int64 v110; // r8
  __int64 v111; // r9
  char v112; // r13
  char v113; // al
  unsigned __int64 v114; // rdx
  char v115; // bl
  unsigned int v116; // r13d
  _BYTE *v117; // rbx
  _BYTE *v118; // r12
  unsigned __int64 v119; // r14
  unsigned __int64 v120; // rdi
  _QWORD *v122; // rbx
  __int64 v123; // rdx
  __int64 v124; // rax
  __int64 v125; // rax
  unsigned __int64 v126; // rdx
  _QWORD *v127; // rdi
  __m128i *v128; // rax
  __m128i v129; // xmm0
  unsigned __int64 v130; // rax
  char v131; // si
  _QWORD *v132; // rax
  __m128i *v133; // rdx
  __m128i v134; // xmm0
  _QWORD *v135; // rax
  __m128i *v136; // rdx
  __m128i si128; // xmm0
  _QWORD *v138; // rdi
  __m128i *v139; // rax
  __m128i v140; // xmm0
  _QWORD *v141; // rdi
  __m128i *v142; // rax
  __m128i v143; // xmm0
  _QWORD *v144; // rdi
  __m128i *v145; // rax
  __m128i v146; // xmm0
  __m128i v147; // xmm0
  __int64 v149; // [rsp+20h] [rbp-7E0h] BYREF
  __int64 v150; // [rsp+28h] [rbp-7D8h]
  __int64 v151; // [rsp+30h] [rbp-7D0h]
  __int64 v152; // [rsp+38h] [rbp-7C8h]
  _QWORD *v153; // [rsp+40h] [rbp-7C0h]
  __int64 v154; // [rsp+48h] [rbp-7B8h]
  _QWORD v155[3]; // [rsp+50h] [rbp-7B0h] BYREF
  char *v156; // [rsp+68h] [rbp-798h]
  __int64 v157; // [rsp+70h] [rbp-790h]
  int v158; // [rsp+78h] [rbp-788h]
  char v159; // [rsp+7Ch] [rbp-784h]
  char v160; // [rsp+80h] [rbp-780h] BYREF
  char v161[8]; // [rsp+C0h] [rbp-740h] BYREF
  unsigned __int64 v162; // [rsp+C8h] [rbp-738h]
  char v163; // [rsp+DCh] [rbp-724h]
  _BYTE v164[64]; // [rsp+E0h] [rbp-720h] BYREF
  unsigned __int64 v165; // [rsp+120h] [rbp-6E0h]
  unsigned __int64 v166; // [rsp+128h] [rbp-6D8h]
  unsigned __int64 v167; // [rsp+130h] [rbp-6D0h]
  char v168[8]; // [rsp+140h] [rbp-6C0h] BYREF
  unsigned __int64 v169; // [rsp+148h] [rbp-6B8h]
  char v170; // [rsp+15Ch] [rbp-6A4h]
  _BYTE v171[64]; // [rsp+160h] [rbp-6A0h] BYREF
  unsigned __int64 v172; // [rsp+1A0h] [rbp-660h]
  unsigned __int64 v173; // [rsp+1A8h] [rbp-658h]
  unsigned __int64 v174; // [rsp+1B0h] [rbp-650h]
  char v175[8]; // [rsp+1C0h] [rbp-640h] BYREF
  unsigned __int64 v176; // [rsp+1C8h] [rbp-638h]
  char v177; // [rsp+1DCh] [rbp-624h]
  _BYTE v178[64]; // [rsp+1E0h] [rbp-620h] BYREF
  unsigned __int64 v179; // [rsp+220h] [rbp-5E0h]
  unsigned __int64 v180; // [rsp+228h] [rbp-5D8h]
  unsigned __int64 v181; // [rsp+230h] [rbp-5D0h]
  char v182[8]; // [rsp+240h] [rbp-5C0h] BYREF
  unsigned __int64 v183; // [rsp+248h] [rbp-5B8h]
  char v184; // [rsp+25Ch] [rbp-5A4h]
  _BYTE v185[64]; // [rsp+260h] [rbp-5A0h] BYREF
  unsigned __int64 v186; // [rsp+2A0h] [rbp-560h]
  unsigned __int64 v187; // [rsp+2A8h] [rbp-558h]
  unsigned __int64 v188; // [rsp+2B0h] [rbp-550h]
  char v189[8]; // [rsp+2C0h] [rbp-540h] BYREF
  unsigned __int64 v190; // [rsp+2C8h] [rbp-538h]
  char v191; // [rsp+2DCh] [rbp-524h]
  _BYTE v192[64]; // [rsp+2E0h] [rbp-520h] BYREF
  unsigned __int64 v193; // [rsp+320h] [rbp-4E0h]
  unsigned __int64 v194; // [rsp+328h] [rbp-4D8h]
  unsigned __int64 v195; // [rsp+330h] [rbp-4D0h]
  char v196[8]; // [rsp+340h] [rbp-4C0h] BYREF
  unsigned __int64 v197; // [rsp+348h] [rbp-4B8h]
  char v198; // [rsp+35Ch] [rbp-4A4h]
  _BYTE v199[64]; // [rsp+360h] [rbp-4A0h] BYREF
  unsigned __int64 v200; // [rsp+3A0h] [rbp-460h]
  unsigned __int64 v201; // [rsp+3A8h] [rbp-458h]
  unsigned __int64 v202; // [rsp+3B0h] [rbp-450h]
  char v203[8]; // [rsp+3C0h] [rbp-440h] BYREF
  unsigned __int64 v204; // [rsp+3C8h] [rbp-438h]
  char v205; // [rsp+3DCh] [rbp-424h]
  _BYTE v206[64]; // [rsp+3E0h] [rbp-420h] BYREF
  const __m128i *v207; // [rsp+420h] [rbp-3E0h]
  const __m128i *v208; // [rsp+428h] [rbp-3D8h]
  unsigned __int64 v209; // [rsp+430h] [rbp-3D0h]
  char v210[8]; // [rsp+440h] [rbp-3C0h] BYREF
  unsigned __int64 v211; // [rsp+448h] [rbp-3B8h]
  char v212; // [rsp+45Ch] [rbp-3A4h]
  _BYTE v213[64]; // [rsp+460h] [rbp-3A0h] BYREF
  unsigned __int64 v214; // [rsp+4A0h] [rbp-360h]
  unsigned __int64 v215; // [rsp+4A8h] [rbp-358h]
  unsigned __int64 v216; // [rsp+4B0h] [rbp-350h]
  char v217[8]; // [rsp+4C0h] [rbp-340h] BYREF
  unsigned __int64 v218; // [rsp+4C8h] [rbp-338h]
  char v219; // [rsp+4DCh] [rbp-324h]
  _BYTE v220[64]; // [rsp+4E0h] [rbp-320h] BYREF
  unsigned __int64 v221; // [rsp+520h] [rbp-2E0h]
  unsigned __int64 i; // [rsp+528h] [rbp-2D8h]
  unsigned __int64 v223; // [rsp+530h] [rbp-2D0h]
  char v224[8]; // [rsp+540h] [rbp-2C0h] BYREF
  unsigned __int64 v225; // [rsp+548h] [rbp-2B8h]
  char v226; // [rsp+55Ch] [rbp-2A4h]
  _BYTE v227[64]; // [rsp+560h] [rbp-2A0h] BYREF
  unsigned __int64 v228; // [rsp+5A0h] [rbp-260h]
  unsigned __int64 v229; // [rsp+5A8h] [rbp-258h]
  unsigned __int64 v230; // [rsp+5B0h] [rbp-250h]
  char v231[8]; // [rsp+5C0h] [rbp-240h] BYREF
  unsigned __int64 v232; // [rsp+5C8h] [rbp-238h]
  char v233; // [rsp+5DCh] [rbp-224h]
  _BYTE v234[64]; // [rsp+5E0h] [rbp-220h] BYREF
  unsigned __int64 v235; // [rsp+620h] [rbp-1E0h]
  unsigned __int64 v236; // [rsp+628h] [rbp-1D8h]
  unsigned __int64 v237; // [rsp+630h] [rbp-1D0h]
  unsigned __int64 v238[2]; // [rsp+640h] [rbp-1C0h] BYREF
  char v239; // [rsp+650h] [rbp-1B0h] BYREF
  _BYTE *v240; // [rsp+658h] [rbp-1A8h]
  __int64 v241; // [rsp+660h] [rbp-1A0h]
  _BYTE v242[48]; // [rsp+668h] [rbp-198h] BYREF
  __int64 v243; // [rsp+698h] [rbp-168h]
  __int64 v244; // [rsp+6A0h] [rbp-160h]
  __int64 v245; // [rsp+6A8h] [rbp-158h]
  unsigned int v246; // [rsp+6B0h] [rbp-150h]
  __int64 v247; // [rsp+6B8h] [rbp-148h]
  __int64 *v248; // [rsp+6C0h] [rbp-140h]
  char v249; // [rsp+6C8h] [rbp-138h]
  __int64 v250; // [rsp+6CCh] [rbp-134h]
  _QWORD v251[3]; // [rsp+6E0h] [rbp-120h] BYREF
  char v252; // [rsp+6FCh] [rbp-104h]
  unsigned __int64 v253; // [rsp+740h] [rbp-C0h]
  __int64 v254; // [rsp+748h] [rbp-B8h]
  char v255[8]; // [rsp+758h] [rbp-A8h] BYREF
  unsigned __int64 v256; // [rsp+760h] [rbp-A0h]
  char v257; // [rsp+774h] [rbp-8Ch]
  unsigned __int64 v258; // [rsp+7B8h] [rbp-48h]
  __int64 v259; // [rsp+7C0h] [rbp-40h]

  v238[0] = (unsigned __int64)&v239;
  v238[1] = 0x100000000LL;
  v240 = v242;
  v248 = a1;
  v241 = 0x600000000LL;
  v243 = 0;
  v244 = 0;
  v245 = 0;
  v246 = 0;
  v247 = 0;
  v249 = 0;
  v250 = 0;
  sub_2C06B20((__int64)v238, a2, a3, a4, a5, a6);
  v6 = sub_2BF3F10(a1);
  v7 = sub_2BF04D0(v6);
  if ( v7 + 112 == (*(_QWORD *)(v7 + 112) & 0xFFFFFFFFFFFFFFF8LL) )
  {
    if ( *(_DWORD *)(v7 + 88) != 1 )
      BUG();
    v7 = **(_QWORD **)(v7 + 80);
  }
  v8 = *(_QWORD *)(v7 + 120);
  if ( !v8 )
    BUG();
  if ( !*(_DWORD *)(v8 + 32) )
    BUG();
  v9 = **(_QWORD **)(v8 + 24);
  v149 = 0;
  v10 = *(_QWORD **)(*(_QWORD *)(v9 + 40) + 8LL);
  v150 = 0;
  v151 = 0;
  v152 = 0;
  v153 = v10;
  v11 = *v10;
  v155[0] = v238;
  v154 = v11;
  v155[1] = &v149;
  v156 = &v160;
  v155[2] = 0;
  v12 = *a1;
  v159 = 1;
  v157 = 8;
  v158 = 0;
  sub_2C48240(v251, v12);
  v13 = (const __m128i *)v171;
  v14 = v168;
  sub_C8CD80((__int64)v168, (__int64)v171, (__int64)v255, v15, v16, v17);
  v20 = v259;
  v21 = v258;
  v172 = 0;
  v173 = 0;
  v174 = 0;
  v22 = v259 - v258;
  if ( v259 == v258 )
  {
    v22 = 0;
    v24 = 0;
  }
  else
  {
    if ( v22 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_236;
    v23 = sub_22077B0(v259 - v258);
    v20 = v259;
    v21 = v258;
    v24 = v23;
  }
  v172 = v24;
  v173 = v24;
  v174 = v24 + v22;
  if ( v20 != v21 )
  {
    v25 = (__m128i *)v24;
    v26 = (const __m128i *)v21;
    do
    {
      if ( v25 )
      {
        *v25 = _mm_loadu_si128(v26);
        v25[1].m128i_i64[0] = v26[1].m128i_i64[0];
      }
      v26 = (const __m128i *)((char *)v26 + 24);
      v25 = (__m128i *)((char *)v25 + 24);
    }
    while ( v26 != (const __m128i *)v20 );
    v24 += 8 * (((unsigned __int64)&v26[-2].m128i_u64[1] - v21) >> 3) + 24;
  }
  v173 = v24;
  v13 = (const __m128i *)v164;
  v14 = v161;
  sub_C8CD80((__int64)v161, (__int64)v164, (__int64)v251, v20, v21, v19);
  v28 = v254;
  v29 = v253;
  v165 = 0;
  v166 = 0;
  v167 = 0;
  v30 = v254 - v253;
  if ( v254 == v253 )
  {
    v30 = 0;
    v32 = 0;
  }
  else
  {
    if ( v30 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_236;
    v31 = sub_22077B0(v254 - v253);
    v28 = v254;
    v29 = v253;
    v32 = v31;
  }
  v165 = v32;
  v166 = v32;
  v167 = v32 + v30;
  if ( v29 != v28 )
  {
    v33 = (__m128i *)v32;
    v34 = (const __m128i *)v29;
    do
    {
      if ( v33 )
      {
        *v33 = _mm_loadu_si128(v34);
        v33[1].m128i_i64[0] = v34[1].m128i_i64[0];
      }
      v34 = (const __m128i *)((char *)v34 + 24);
      v33 = (__m128i *)((char *)v33 + 24);
    }
    while ( v34 != (const __m128i *)v28 );
    v32 += 8 * (((unsigned __int64)&v34[-2].m128i_u64[1] - v29) >> 3) + 24;
  }
  v166 = v32;
  v13 = (const __m128i *)v185;
  v14 = v182;
  sub_C8CD80((__int64)v182, (__int64)v185, (__int64)v168, v28, v29, v27);
  v36 = v173;
  v37 = v172;
  v186 = 0;
  v187 = 0;
  v188 = 0;
  v38 = v173 - v172;
  if ( v173 == v172 )
  {
    v40 = 0;
  }
  else
  {
    if ( v38 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_236;
    v39 = sub_22077B0(v173 - v172);
    v36 = v173;
    v37 = v172;
    v40 = v39;
  }
  v186 = v40;
  v187 = v40;
  v188 = v40 + v38;
  if ( v36 != v37 )
  {
    v41 = (__m128i *)v40;
    v42 = (const __m128i *)v37;
    do
    {
      if ( v41 )
      {
        *v41 = _mm_loadu_si128(v42);
        v41[1].m128i_i64[0] = v42[1].m128i_i64[0];
      }
      v42 = (const __m128i *)((char *)v42 + 24);
      v41 = (__m128i *)((char *)v41 + 24);
    }
    while ( v42 != (const __m128i *)v36 );
    v40 += 8 * (((unsigned __int64)&v42[-2].m128i_u64[1] - v37) >> 3) + 24;
  }
  v187 = v40;
  v13 = (const __m128i *)v178;
  v14 = v175;
  sub_C8CD80((__int64)v175, (__int64)v178, (__int64)v161, v36, v37, v35);
  v44 = v166;
  v45 = v165;
  v179 = 0;
  v180 = 0;
  v181 = 0;
  v46 = v166 - v165;
  if ( v166 == v165 )
  {
    v48 = 0;
  }
  else
  {
    if ( v46 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_236;
    v47 = sub_22077B0(v166 - v165);
    v44 = v166;
    v45 = v165;
    v48 = v47;
  }
  v179 = v48;
  v180 = v48;
  v181 = v48 + v46;
  if ( v44 != v45 )
  {
    v49 = (__m128i *)v48;
    v50 = (const __m128i *)v45;
    do
    {
      if ( v49 )
      {
        *v49 = _mm_loadu_si128(v50);
        v49[1].m128i_i64[0] = v50[1].m128i_i64[0];
      }
      v50 = (const __m128i *)((char *)v50 + 24);
      v49 = (__m128i *)((char *)v49 + 24);
    }
    while ( (const __m128i *)v44 != v50 );
    v44 = (v44 - 24 - v45) >> 3;
    v48 += 8 * v44 + 24;
  }
  v180 = v48;
  v13 = (const __m128i *)v199;
  v14 = v196;
  sub_C8CD80((__int64)v196, (__int64)v199, (__int64)v182, v44, v45, v43);
  v52 = v187;
  v53 = v186;
  v200 = 0;
  v201 = 0;
  v202 = 0;
  v54 = v187 - v186;
  if ( v187 == v186 )
  {
    v54 = 0;
    v56 = 0;
  }
  else
  {
    if ( v54 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_236;
    v55 = sub_22077B0(v187 - v186);
    v52 = v187;
    v53 = v186;
    v56 = v55;
  }
  v200 = v56;
  v201 = v56;
  v202 = v56 + v54;
  if ( v53 != v52 )
  {
    v57 = (__m128i *)v56;
    v58 = (const __m128i *)v53;
    do
    {
      if ( v57 )
      {
        *v57 = _mm_loadu_si128(v58);
        v57[1].m128i_i64[0] = v58[1].m128i_i64[0];
      }
      v58 = (const __m128i *)((char *)v58 + 24);
      v57 = (__m128i *)((char *)v57 + 24);
    }
    while ( v58 != (const __m128i *)v52 );
    v56 += 8 * (((unsigned __int64)&v58[-2].m128i_u64[1] - v53) >> 3) + 24;
  }
  v201 = v56;
  v13 = (const __m128i *)v192;
  v14 = v189;
  sub_C8CD80((__int64)v189, (__int64)v192, (__int64)v175, v52, v53, v51);
  v60 = v180;
  v61 = v179;
  v193 = 0;
  v194 = 0;
  v195 = 0;
  v62 = v180 - v179;
  if ( v180 == v179 )
  {
    v62 = 0;
    v64 = 0;
  }
  else
  {
    if ( v62 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_236;
    v63 = sub_22077B0(v180 - v179);
    v60 = v180;
    v61 = v179;
    v64 = v63;
  }
  v193 = v64;
  v194 = v64;
  v195 = v64 + v62;
  if ( v61 != v60 )
  {
    v65 = (__m128i *)v64;
    v66 = (const __m128i *)v61;
    do
    {
      if ( v65 )
      {
        *v65 = _mm_loadu_si128(v66);
        v65[1].m128i_i64[0] = v66[1].m128i_i64[0];
      }
      v66 = (const __m128i *)((char *)v66 + 24);
      v65 = (__m128i *)((char *)v65 + 24);
    }
    while ( v66 != (const __m128i *)v60 );
    v64 += 8 * (((unsigned __int64)&v66[-2].m128i_u64[1] - v61) >> 3) + 24;
  }
  v194 = v64;
  v13 = (const __m128i *)v213;
  v14 = v210;
  sub_C8CD80((__int64)v210, (__int64)v213, (__int64)v196, v60, v61, v59);
  v68 = v201;
  v69 = v200;
  v214 = 0;
  v215 = 0;
  v216 = 0;
  v70 = v201 - v200;
  if ( v201 == v200 )
  {
    v70 = 0;
    v72 = 0;
  }
  else
  {
    if ( v70 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_236;
    v71 = sub_22077B0(v201 - v200);
    v68 = v201;
    v69 = v200;
    v72 = v71;
  }
  v214 = v72;
  v215 = v72;
  v216 = v72 + v70;
  if ( v69 != v68 )
  {
    v73 = (__m128i *)v72;
    v74 = (const __m128i *)v69;
    do
    {
      if ( v73 )
      {
        *v73 = _mm_loadu_si128(v74);
        v73[1].m128i_i64[0] = v74[1].m128i_i64[0];
      }
      v74 = (const __m128i *)((char *)v74 + 24);
      v73 = (__m128i *)((char *)v73 + 24);
    }
    while ( (const __m128i *)v68 != v74 );
    v68 = (v68 - 24 - v69) >> 3;
    v72 += 8 * v68 + 24;
  }
  v215 = v72;
  v13 = (const __m128i *)v206;
  v14 = v203;
  sub_C8CD80((__int64)v203, (__int64)v206, (__int64)v189, v68, v69, v67);
  v76 = v194;
  v77 = v193;
  v207 = 0;
  v208 = 0;
  v209 = 0;
  v78 = v194 - v193;
  if ( v194 == v193 )
  {
    v78 = 0;
    v80 = 0;
  }
  else
  {
    if ( v78 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_236;
    v79 = sub_22077B0(v194 - v193);
    v76 = v194;
    v77 = v193;
    v80 = v79;
  }
  v207 = (const __m128i *)v80;
  v208 = (const __m128i *)v80;
  v209 = v80 + v78;
  if ( v77 != v76 )
  {
    v81 = (__m128i *)v80;
    v82 = (const __m128i *)v77;
    do
    {
      if ( v81 )
      {
        *v81 = _mm_loadu_si128(v82);
        v81[1].m128i_i64[0] = v82[1].m128i_i64[0];
      }
      v82 = (const __m128i *)((char *)v82 + 24);
      v81 = (__m128i *)((char *)v81 + 24);
    }
    while ( (const __m128i *)v76 != v82 );
    v76 = (v76 - 24 - v77) >> 3;
    v80 += 8 * v76 + 24;
  }
  v208 = (const __m128i *)v80;
  v13 = (const __m128i *)v227;
  v14 = v224;
  sub_C8CD80((__int64)v224, (__int64)v227, (__int64)v210, v76, v77, v75);
  v84 = v215;
  v85 = v214;
  v228 = 0;
  v229 = 0;
  v230 = 0;
  v86 = v215 - v214;
  if ( v215 == v214 )
  {
    v86 = 0;
    v88 = 0;
  }
  else
  {
    if ( v86 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_236;
    v87 = sub_22077B0(v215 - v214);
    v84 = v215;
    v85 = v214;
    v88 = v87;
  }
  v228 = v88;
  v229 = v88;
  v230 = v88 + v86;
  if ( v85 != v84 )
  {
    v89 = (__m128i *)v88;
    v90 = (const __m128i *)v85;
    do
    {
      if ( v89 )
      {
        *v89 = _mm_loadu_si128(v90);
        v89[1].m128i_i64[0] = v90[1].m128i_i64[0];
      }
      v90 = (const __m128i *)((char *)v90 + 24);
      v89 = (__m128i *)((char *)v89 + 24);
    }
    while ( (const __m128i *)v84 != v90 );
    v84 = (v84 - 24 - v85) >> 3;
    v88 += 8 * v84 + 24;
  }
  v229 = v88;
  v14 = v217;
  sub_C8CD80((__int64)v217, (__int64)v220, (__int64)v203, v84, v85, v83);
  v13 = v208;
  v92 = (__int64)v207;
  v221 = 0;
  i = 0;
  v223 = 0;
  v93 = (char *)v208 - (char *)v207;
  if ( v208 != v207 )
  {
    if ( v93 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v94 = sub_22077B0((char *)v208 - (char *)v207);
      v13 = v208;
      v92 = (__int64)v207;
      v95 = v94;
      goto LABEL_88;
    }
LABEL_236:
    sub_4261EA(v14, v13, v18);
  }
  v95 = 0;
LABEL_88:
  v221 = v95;
  i = v95;
  v223 = v95 + v93;
  if ( (const __m128i *)v92 == v13 )
  {
    v98 = v95;
  }
  else
  {
    v96 = (__m128i *)v95;
    v97 = (const __m128i *)v92;
    do
    {
      if ( v96 )
      {
        *v96 = _mm_loadu_si128(v97);
        v96[1].m128i_i64[0] = v97[1].m128i_i64[0];
      }
      v97 = (const __m128i *)((char *)v97 + 24);
      v96 = (__m128i *)((char *)v96 + 24);
    }
    while ( v13 != v97 );
    v98 = v95 + 8 * (((unsigned __int64)&v13[-2].m128i_u64[1] - v92) >> 3) + 24;
  }
  for ( i = v98; ; v98 = i )
  {
    v99 = v228;
    if ( v98 - v95 == v229 - v228 )
      break;
LABEL_96:
    v13 = (const __m128i *)v234;
    v14 = v231;
    sub_C8CD80((__int64)v231, (__int64)v234, (__int64)v217, v95, v92, v91);
    v100 = (const __m128i *)i;
    v101 = (const __m128i *)v221;
    v235 = 0;
    v236 = 0;
    v237 = 0;
    v102 = i - v221;
    if ( i == v221 )
    {
      v104 = 0;
    }
    else
    {
      if ( v102 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_236;
      v103 = sub_22077B0(i - v221);
      v100 = (const __m128i *)i;
      v101 = (const __m128i *)v221;
      v104 = v103;
    }
    v235 = v104;
    v236 = v104;
    v237 = v104 + v102;
    if ( v100 != v101 )
    {
      v105 = (__m128i *)v104;
      v106 = v101;
      do
      {
        if ( v105 )
        {
          *v105 = _mm_loadu_si128(v106);
          v105[1].m128i_i64[0] = v106[1].m128i_i64[0];
        }
        v106 = (const __m128i *)((char *)v106 + 24);
        v105 = (__m128i *)((char *)v105 + 24);
      }
      while ( v100 != v106 );
      v104 += 8
            * (3
             * ((0xAAAAAAAAAAAAAABLL * ((unsigned __int64)((char *)&v100[-2].m128i_u64[1] - (char *)v101) >> 3))
              & 0x1FFFFFFFFFFFFFFFLL)
             + 3);
    }
    v236 = v104;
    v107 = *(_QWORD *)(v104 - 24);
    v112 = sub_2C48D10((__int64)v155, v107);
    if ( v235 )
    {
      v107 = v237 - v235;
      j_j___libc_free_0(v235);
    }
    if ( v233 )
    {
      if ( !v112 )
        goto LABEL_118;
    }
    else
    {
      _libc_free(v232);
      if ( !v112 )
        goto LABEL_118;
    }
    sub_2C48140((__int64)v217, v107, v108, v109, v110, v111);
    v95 = v221;
  }
  while ( v95 != v98 )
  {
    if ( *(_QWORD *)v95 != *(_QWORD *)v99 )
      goto LABEL_96;
    v113 = *(_BYTE *)(v95 + 16);
    if ( v113 != *(_BYTE *)(v99 + 16) || v113 && *(_QWORD *)(v95 + 8) != *(_QWORD *)(v99 + 8) )
      goto LABEL_96;
    v95 += 24;
    v99 += 24LL;
  }
LABEL_118:
  sub_C8CF70((__int64)v231, v234, 8, (__int64)v220, (__int64)v217);
  v235 = v221;
  v236 = i;
  v237 = v223;
  if ( !v219 )
    _libc_free(v218);
  if ( v228 )
    j_j___libc_free_0(v228);
  if ( !v226 )
    _libc_free(v225);
  if ( v207 )
    j_j___libc_free_0((unsigned __int64)v207);
  if ( !v205 )
    _libc_free(v204);
  if ( v214 )
    j_j___libc_free_0(v214);
  if ( !v212 )
    _libc_free(v211);
  v114 = v186;
  v115 = 0;
  if ( v187 - v186 == v236 - v235 )
  {
    if ( v186 == v187 )
    {
LABEL_218:
      v115 = 1;
    }
    else
    {
      v130 = v235;
      while ( *(_QWORD *)v114 == *(_QWORD *)v130 )
      {
        v131 = *(_BYTE *)(v114 + 16);
        if ( v131 != *(_BYTE *)(v130 + 16) || v131 && *(_QWORD *)(v114 + 8) != *(_QWORD *)(v130 + 8) )
          break;
        v114 += 24LL;
        v130 += 24LL;
        if ( v187 == v114 )
          goto LABEL_218;
      }
      v115 = 0;
    }
  }
  if ( v235 )
    j_j___libc_free_0(v235);
  if ( !v233 )
    _libc_free(v232);
  if ( v193 )
    j_j___libc_free_0(v193);
  if ( !v191 )
    _libc_free(v190);
  if ( v200 )
    j_j___libc_free_0(v200);
  if ( !v198 )
    _libc_free(v197);
  if ( v179 )
    j_j___libc_free_0(v179);
  if ( !v177 )
    _libc_free(v176);
  if ( v186 )
    j_j___libc_free_0(v186);
  if ( !v184 )
    _libc_free(v183);
  if ( v165 )
    j_j___libc_free_0(v165);
  if ( !v163 )
    _libc_free(v162);
  if ( v172 )
    j_j___libc_free_0(v172);
  if ( !v170 )
    _libc_free(v169);
  if ( v258 )
    j_j___libc_free_0(v258);
  if ( !v257 )
    _libc_free(v256);
  if ( v253 )
    j_j___libc_free_0(v253);
  if ( !v252 )
    _libc_free(v251[1]);
  if ( !v115 )
    goto LABEL_170;
  v122 = (_QWORD *)sub_2BF5D50(a1);
  v116 = sub_2C4A350((__int64)v155, (__int64)v122);
  if ( !(_BYTE)v116 )
    goto LABEL_170;
  if ( v122[6] )
  {
    v135 = sub_CB72A0();
    v136 = (__m128i *)v135[4];
    if ( v135[3] - (_QWORD)v136 <= 0x27u )
    {
      sub_CB6200((__int64)v135, "VPlan Top Region should have no parent.\n", 0x28u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_43A1690);
      v136[2].m128i_i64[0] = 0xA2E746E65726170LL;
      *v136 = si128;
      v136[1] = _mm_load_si128((const __m128i *)&xmmword_43A16A0);
      v135[4] += 40LL;
    }
    goto LABEL_170;
  }
  v123 = v122[14];
  if ( (unsigned int)*(unsigned __int8 *)(v123 + 8) - 1 > 1 )
  {
    v132 = sub_CB72A0();
    v133 = (__m128i *)v132[4];
    if ( v132[3] - (_QWORD)v133 <= 0x27u )
    {
      sub_CB6200((__int64)v132, "VPlan entry block is not a VPBasicBlock\n", 0x28u);
    }
    else
    {
      v134 = _mm_load_si128((const __m128i *)&xmmword_43A16B0);
      v133[2].m128i_i64[0] = 0xA6B636F6C426369LL;
      *v133 = v134;
      v133[1] = _mm_load_si128((const __m128i *)&xmmword_43A16C0);
      v132[4] += 40LL;
    }
    goto LABEL_170;
  }
  v124 = *(_QWORD *)(v123 + 120);
  if ( !v124 )
    goto LABEL_237;
  if ( *(_BYTE *)(v124 - 16) != 29 )
  {
    v141 = sub_CB72A0();
    v142 = (__m128i *)v141[4];
    if ( v141[3] - (_QWORD)v142 <= 0x45u )
    {
      sub_CB6200((__int64)v141, "VPlan vector loop header does not start with a VPCanonicalIVPHIRecipe\n", 0x46u);
    }
    else
    {
      v143 = _mm_load_si128((const __m128i *)&xmmword_43A16D0);
      v142[4].m128i_i32[0] = 1885954917;
      v142[4].m128i_i16[2] = 2661;
      *v142 = v143;
      v142[1] = _mm_load_si128((const __m128i *)&xmmword_43A16E0);
      v142[2] = _mm_load_si128((const __m128i *)&xmmword_43A16F0);
      v142[3] = _mm_load_si128((const __m128i *)&xmmword_43A1700);
      v141[4] += 70LL;
    }
    goto LABEL_170;
  }
  v125 = v122[15];
  if ( (unsigned int)*(unsigned __int8 *)(v125 + 8) - 1 > 1 )
  {
    v138 = sub_CB72A0();
    v139 = (__m128i *)v138[4];
    if ( v138[3] - (_QWORD)v139 <= 0x29u )
    {
      sub_CB6200((__int64)v138, "VPlan exiting block is not a VPBasicBlock\n", 0x2Au);
    }
    else
    {
      v140 = _mm_load_si128((const __m128i *)&xmmword_43A1710);
      qmemcpy(&v139[2], "asicBlock\n", 10);
      *v139 = v140;
      v139[1] = _mm_load_si128((const __m128i *)&xmmword_43A1720);
      v138[4] += 42LL;
    }
    goto LABEL_170;
  }
  v126 = *(_QWORD *)(v125 + 112) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v126 == v125 + 112 )
  {
    v144 = sub_CB72A0();
    v145 = (__m128i *)v144[4];
    if ( v144[3] - (_QWORD)v145 <= 0x66u )
    {
      sub_CB6200(
        (__int64)v144,
        "VPlan vector loop exiting block must end with BranchOnCount or BranchOnCond VPInstruction but is empty\n",
        0x67u);
    }
    else
    {
      v146 = _mm_load_si128((const __m128i *)&xmmword_43A16D0);
      v145[6].m128i_i32[0] = 1886217504;
      v145[6].m128i_i16[2] = 31092;
      *v145 = v146;
      v147 = _mm_load_si128((const __m128i *)&xmmword_43A1730);
      v145[6].m128i_i8[6] = 10;
      v145[1] = v147;
      v145[2] = _mm_load_si128((const __m128i *)&xmmword_43A1740);
      v145[3] = _mm_load_si128((const __m128i *)&xmmword_43A1750);
      v145[4] = _mm_load_si128((const __m128i *)&xmmword_43A1760);
      v145[5] = _mm_load_si128((const __m128i *)&xmmword_43A1770);
      v144[4] += 103LL;
    }
    goto LABEL_170;
  }
  if ( !v126 )
LABEL_237:
    BUG();
  if ( *(_BYTE *)(v126 - 16) != 4 || (unsigned int)*(unsigned __int8 *)(v126 + 136) - 78 > 1 )
  {
    v127 = sub_CB72A0();
    v128 = (__m128i *)v127[4];
    if ( v127[3] - (_QWORD)v128 <= 0x50u )
    {
      sub_CB6200(
        (__int64)v127,
        "VPlan vector loop exit must end with BranchOnCount or BranchOnCond VPInstruction\n",
        0x51u);
    }
    else
    {
      v129 = _mm_load_si128((const __m128i *)&xmmword_43A16D0);
      v128[5].m128i_i8[0] = 10;
      *v128 = v129;
      v128[1] = _mm_load_si128((const __m128i *)&xmmword_43A1780);
      v128[2] = _mm_load_si128((const __m128i *)&xmmword_43A1790);
      v128[3] = _mm_load_si128((const __m128i *)&xmmword_43A17A0);
      v128[4] = _mm_load_si128((const __m128i *)&xmmword_43A17B0);
      v127[4] += 81LL;
    }
LABEL_170:
    v116 = 0;
  }
  if ( !v159 )
    _libc_free((unsigned __int64)v156);
  sub_C7D6A0(v150, 16LL * (unsigned int)v152, 8);
  sub_C7D6A0(v244, 16LL * v246, 8);
  v117 = v240;
  v118 = &v240[8 * (unsigned int)v241];
  if ( v240 != v118 )
  {
    do
    {
      v119 = *((_QWORD *)v118 - 1);
      v118 -= 8;
      if ( v119 )
      {
        v120 = *(_QWORD *)(v119 + 24);
        if ( v120 != v119 + 40 )
          _libc_free(v120);
        j_j___libc_free_0(v119);
      }
    }
    while ( v117 != v118 );
    v118 = v240;
  }
  if ( v118 != v242 )
    _libc_free((unsigned __int64)v118);
  if ( (char *)v238[0] != &v239 )
    _libc_free(v238[0]);
  return v116;
}
