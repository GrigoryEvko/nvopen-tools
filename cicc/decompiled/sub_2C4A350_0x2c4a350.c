// Function: sub_2C4A350
// Address: 0x2c4a350
//
__int64 __fastcall sub_2C4A350(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  const __m128i *v5; // rsi
  char *v6; // rdi
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __m128i *v10; // rdx
  __int64 v11; // r9
  unsigned __int64 v12; // rcx
  __int64 v13; // r8
  unsigned __int64 v14; // r15
  __int64 v15; // rax
  unsigned __int64 v16; // rdi
  __m128i *v17; // rdx
  const __m128i *v18; // rax
  const __m128i *v19; // rcx
  unsigned __int64 v20; // r8
  unsigned __int64 v21; // r15
  __int64 v22; // rax
  unsigned __int64 v23; // rdi
  __m128i *v24; // rdx
  const __m128i *v25; // rax
  unsigned __int64 v26; // rax
  __int64 v27; // rsi
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  unsigned __int64 v32; // rcx
  unsigned __int64 v33; // rax
  char v34; // si
  __int64 v35; // rcx
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // r9
  __int64 v39; // rcx
  __int64 v40; // r8
  unsigned __int64 v41; // r12
  __int64 v42; // rax
  unsigned __int64 v43; // rdi
  __m128i *v44; // rdx
  const __m128i *v45; // rax
  __int64 v46; // r9
  unsigned __int64 v47; // rcx
  __int64 v48; // r8
  unsigned __int64 v49; // r14
  __int64 v50; // rax
  unsigned __int64 v51; // rdi
  __m128i *v52; // rdx
  const __m128i *v53; // rax
  __int64 v54; // r9
  unsigned __int64 v55; // rcx
  __int64 v56; // r8
  unsigned __int64 v57; // r15
  __int64 v58; // rax
  unsigned __int64 v59; // rdi
  __m128i *v60; // rdx
  const __m128i *v61; // rax
  __int64 v62; // r9
  __int64 v63; // rcx
  __int64 v64; // r8
  unsigned __int64 v65; // r12
  __int64 v66; // rax
  unsigned __int64 v67; // rdi
  __m128i *v68; // rdx
  const __m128i *v69; // rax
  __int64 v70; // r9
  unsigned __int64 v71; // rcx
  __int64 v72; // r8
  unsigned __int64 v73; // r14
  __int64 v74; // rax
  unsigned __int64 v75; // rdi
  __m128i *v76; // rdx
  const __m128i *v77; // rax
  __int64 v78; // r9
  unsigned __int64 v79; // rcx
  __int64 v80; // r8
  unsigned __int64 v81; // r15
  __int64 v82; // rax
  unsigned __int64 v83; // rdi
  __m128i *v84; // rdx
  const __m128i *v85; // rax
  __int64 v86; // r9
  unsigned __int64 v87; // rcx
  __int64 v88; // r8
  unsigned __int64 v89; // r12
  __int64 v90; // rax
  unsigned __int64 v91; // rdi
  __m128i *v92; // rdx
  const __m128i *v93; // rax
  __int64 v94; // r9
  __int64 v95; // r8
  unsigned __int64 v96; // r14
  __int64 v97; // rax
  __int64 v98; // rcx
  __m128i *v99; // rdx
  const __m128i *v100; // rax
  unsigned __int64 v101; // rsi
  unsigned __int64 v102; // rdx
  __int64 v103; // r9
  __int64 v104; // rcx
  __int64 v105; // r8
  unsigned __int64 v106; // r15
  __int64 v107; // rax
  unsigned __int64 v108; // rdi
  const __m128i *v109; // rax
  unsigned __int64 v110; // rax
  unsigned __int64 v111; // rax
  __int64 v112; // rsi
  char v113; // r13
  char v114; // al
  unsigned int v115; // r12d
  _QWORD *v117; // rax
  __m128i *v118; // rdx
  __m128i v119; // xmm0
  _QWORD *v120; // rax
  __m128i *v121; // rdx
  __m128i si128; // xmm0
  char v123; // al
  _QWORD *v124; // rax
  __m128i *v125; // rdx
  __m128i v126; // xmm0
  unsigned __int64 v127; // rdx
  unsigned __int64 v128; // rax
  char v129; // si
  char v130[8]; // [rsp+10h] [rbp-790h] BYREF
  unsigned __int64 v131; // [rsp+18h] [rbp-788h]
  char v132; // [rsp+2Ch] [rbp-774h]
  _BYTE v133[64]; // [rsp+30h] [rbp-770h] BYREF
  unsigned __int64 v134; // [rsp+70h] [rbp-730h]
  unsigned __int64 v135; // [rsp+78h] [rbp-728h]
  unsigned __int64 v136; // [rsp+80h] [rbp-720h]
  char v137[8]; // [rsp+90h] [rbp-710h] BYREF
  unsigned __int64 v138; // [rsp+98h] [rbp-708h]
  char v139; // [rsp+ACh] [rbp-6F4h]
  _BYTE v140[64]; // [rsp+B0h] [rbp-6F0h] BYREF
  unsigned __int64 v141; // [rsp+F0h] [rbp-6B0h]
  unsigned __int64 i; // [rsp+F8h] [rbp-6A8h]
  unsigned __int64 v143; // [rsp+100h] [rbp-6A0h]
  char v144[8]; // [rsp+110h] [rbp-690h] BYREF
  unsigned __int64 v145; // [rsp+118h] [rbp-688h]
  char v146; // [rsp+12Ch] [rbp-674h]
  _BYTE v147[64]; // [rsp+130h] [rbp-670h] BYREF
  unsigned __int64 v148; // [rsp+170h] [rbp-630h]
  unsigned __int64 v149; // [rsp+178h] [rbp-628h]
  unsigned __int64 v150; // [rsp+180h] [rbp-620h]
  char v151[8]; // [rsp+190h] [rbp-610h] BYREF
  unsigned __int64 v152; // [rsp+198h] [rbp-608h]
  char v153; // [rsp+1ACh] [rbp-5F4h]
  _BYTE v154[64]; // [rsp+1B0h] [rbp-5F0h] BYREF
  unsigned __int64 v155; // [rsp+1F0h] [rbp-5B0h]
  unsigned __int64 v156; // [rsp+1F8h] [rbp-5A8h]
  unsigned __int64 v157; // [rsp+200h] [rbp-5A0h]
  char v158[8]; // [rsp+210h] [rbp-590h] BYREF
  unsigned __int64 v159; // [rsp+218h] [rbp-588h]
  char v160; // [rsp+22Ch] [rbp-574h]
  _BYTE v161[64]; // [rsp+230h] [rbp-570h] BYREF
  unsigned __int64 v162; // [rsp+270h] [rbp-530h]
  unsigned __int64 v163; // [rsp+278h] [rbp-528h]
  unsigned __int64 v164; // [rsp+280h] [rbp-520h]
  char v165[8]; // [rsp+290h] [rbp-510h] BYREF
  unsigned __int64 v166; // [rsp+298h] [rbp-508h]
  char v167; // [rsp+2ACh] [rbp-4F4h]
  _BYTE v168[64]; // [rsp+2B0h] [rbp-4F0h] BYREF
  unsigned __int64 v169; // [rsp+2F0h] [rbp-4B0h]
  unsigned __int64 v170; // [rsp+2F8h] [rbp-4A8h]
  unsigned __int64 v171; // [rsp+300h] [rbp-4A0h]
  char v172[8]; // [rsp+310h] [rbp-490h] BYREF
  unsigned __int64 v173; // [rsp+318h] [rbp-488h]
  char v174; // [rsp+32Ch] [rbp-474h]
  _BYTE v175[64]; // [rsp+330h] [rbp-470h] BYREF
  const __m128i *v176; // [rsp+370h] [rbp-430h]
  const __m128i *v177; // [rsp+378h] [rbp-428h]
  unsigned __int64 v178; // [rsp+380h] [rbp-420h]
  char v179[8]; // [rsp+390h] [rbp-410h] BYREF
  unsigned __int64 v180; // [rsp+398h] [rbp-408h]
  char v181; // [rsp+3ACh] [rbp-3F4h]
  _BYTE v182[64]; // [rsp+3B0h] [rbp-3F0h] BYREF
  unsigned __int64 v183; // [rsp+3F0h] [rbp-3B0h]
  unsigned __int64 v184; // [rsp+3F8h] [rbp-3A8h]
  unsigned __int64 v185; // [rsp+400h] [rbp-3A0h]
  char v186[8]; // [rsp+410h] [rbp-390h] BYREF
  unsigned __int64 v187; // [rsp+418h] [rbp-388h]
  char v188; // [rsp+42Ch] [rbp-374h]
  _BYTE v189[64]; // [rsp+430h] [rbp-370h] BYREF
  unsigned __int64 v190; // [rsp+470h] [rbp-330h]
  unsigned __int64 j; // [rsp+478h] [rbp-328h]
  unsigned __int64 v192; // [rsp+480h] [rbp-320h]
  char v193[8]; // [rsp+490h] [rbp-310h] BYREF
  unsigned __int64 v194; // [rsp+498h] [rbp-308h]
  char v195; // [rsp+4ACh] [rbp-2F4h]
  _BYTE v196[64]; // [rsp+4B0h] [rbp-2F0h] BYREF
  unsigned __int64 v197; // [rsp+4F0h] [rbp-2B0h]
  unsigned __int64 v198; // [rsp+4F8h] [rbp-2A8h]
  unsigned __int64 v199; // [rsp+500h] [rbp-2A0h]
  char v200[8]; // [rsp+510h] [rbp-290h] BYREF
  unsigned __int64 v201; // [rsp+518h] [rbp-288h]
  char v202; // [rsp+52Ch] [rbp-274h]
  _BYTE v203[64]; // [rsp+530h] [rbp-270h] BYREF
  unsigned __int64 v204; // [rsp+570h] [rbp-230h]
  unsigned __int64 v205; // [rsp+578h] [rbp-228h]
  unsigned __int64 v206; // [rsp+580h] [rbp-220h]
  _QWORD v207[3]; // [rsp+590h] [rbp-210h] BYREF
  char v208; // [rsp+5ACh] [rbp-1F4h]
  __int64 v209; // [rsp+5F0h] [rbp-1B0h]
  unsigned __int64 v210; // [rsp+5F8h] [rbp-1A8h]
  char v211[8]; // [rsp+608h] [rbp-198h] BYREF
  unsigned __int64 v212; // [rsp+610h] [rbp-190h]
  char v213; // [rsp+624h] [rbp-17Ch]
  unsigned __int64 v214; // [rsp+668h] [rbp-138h]
  __int64 v215; // [rsp+670h] [rbp-130h]
  __int64 v216; // [rsp+680h] [rbp-120h] BYREF
  unsigned __int64 v217; // [rsp+688h] [rbp-118h]
  char v218; // [rsp+69Ch] [rbp-104h]
  char v219[64]; // [rsp+6A0h] [rbp-100h] BYREF
  unsigned __int64 v220; // [rsp+6E0h] [rbp-C0h]
  unsigned __int64 v221; // [rsp+6E8h] [rbp-B8h]
  unsigned __int64 v222; // [rsp+6F0h] [rbp-B0h]
  char v223[8]; // [rsp+6F8h] [rbp-A8h] BYREF
  unsigned __int64 v224; // [rsp+700h] [rbp-A0h]
  char v225; // [rsp+714h] [rbp-8Ch]
  const __m128i *v226; // [rsp+758h] [rbp-48h]
  const __m128i *v227; // [rsp+760h] [rbp-40h]

  v3 = *(_QWORD *)(a2 + 112);
  if ( *(_DWORD *)(v3 + 64) )
  {
    v120 = sub_CB72A0();
    v121 = (__m128i *)v120[4];
    if ( v120[3] - (_QWORD)v121 <= 0x23u )
    {
      sub_CB6200((__int64)v120, "region entry block has predecessors\n", 0x24u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_43A1640);
      v121[2].m128i_i32[0] = 175338095;
      *v121 = si128;
      v121[1] = _mm_load_si128((const __m128i *)&xmmword_43A1650);
      v120[4] += 36LL;
    }
    return 0;
  }
  if ( *(_DWORD *)(*(_QWORD *)(a2 + 120) + 88LL) )
  {
    v117 = sub_CB72A0();
    v118 = (__m128i *)v117[4];
    if ( v117[3] - (_QWORD)v118 <= 0x23u )
    {
      sub_CB6200((__int64)v117, "region exiting block has successors\n", 0x24u);
    }
    else
    {
      v119 = _mm_load_si128((const __m128i *)&xmmword_43A1660);
      v118[2].m128i_i32[0] = 175338095;
      *v118 = v119;
      v118[1] = _mm_load_si128((const __m128i *)&xmmword_43A1670);
      v117[4] += 36LL;
    }
    return 0;
  }
  sub_2C48240(&v216, v3);
  v5 = (const __m128i *)v133;
  v6 = v130;
  sub_C8CD80((__int64)v130, (__int64)v133, (__int64)&v216, v7, v8, v9);
  v12 = v221;
  v13 = v220;
  v134 = 0;
  v135 = 0;
  v136 = 0;
  v14 = v221 - v220;
  if ( v221 == v220 )
  {
    v16 = 0;
  }
  else
  {
    if ( v14 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_241;
    v15 = sub_22077B0(v221 - v220);
    v12 = v221;
    v13 = v220;
    v16 = v15;
  }
  v134 = v16;
  v135 = v16;
  v136 = v16 + v14;
  if ( v12 != v13 )
  {
    v17 = (__m128i *)v16;
    v18 = (const __m128i *)v13;
    do
    {
      if ( v17 )
      {
        *v17 = _mm_loadu_si128(v18);
        v17[1].m128i_i64[0] = v18[1].m128i_i64[0];
      }
      v18 = (const __m128i *)((char *)v18 + 24);
      v17 = (__m128i *)((char *)v17 + 24);
    }
    while ( (const __m128i *)v12 != v18 );
    v12 = (v12 - 24 - v13) >> 3;
    v16 += 8 * v12 + 24;
  }
  v135 = v16;
  v6 = v137;
  v5 = (const __m128i *)v140;
  sub_C8CD80((__int64)v137, (__int64)v140, (__int64)v223, v12, v13, v11);
  v19 = v227;
  v20 = (unsigned __int64)v226;
  v141 = 0;
  i = 0;
  v143 = 0;
  v21 = (char *)v227 - (char *)v226;
  if ( v227 == v226 )
  {
    v23 = 0;
  }
  else
  {
    if ( v21 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_241;
    v22 = sub_22077B0((char *)v227 - (char *)v226);
    v19 = v227;
    v20 = (unsigned __int64)v226;
    v23 = v22;
  }
  v141 = v23;
  i = v23;
  v143 = v23 + v21;
  if ( (const __m128i *)v20 == v19 )
  {
    v26 = v23;
  }
  else
  {
    v24 = (__m128i *)v23;
    v25 = (const __m128i *)v20;
    do
    {
      if ( v24 )
      {
        *v24 = _mm_loadu_si128(v25);
        v24[1].m128i_i64[0] = v25[1].m128i_i64[0];
      }
      v25 = (const __m128i *)((char *)v25 + 24);
      v24 = (__m128i *)((char *)v24 + 24);
    }
    while ( v25 != v19 );
    v26 = v23 + 8 * (((unsigned __int64)&v25[-2].m128i_u64[1] - v20) >> 3) + 24;
  }
  for ( i = v26; ; v26 = i )
  {
    v32 = v134;
    if ( v135 - v134 != v26 - v23 )
      goto LABEL_22;
    if ( v134 == v135 )
      break;
    v33 = v23;
    while ( *(_QWORD *)v32 == *(_QWORD *)v33 )
    {
      v34 = *(_BYTE *)(v32 + 16);
      if ( v34 != *(_BYTE *)(v33 + 16) || v34 && *(_QWORD *)(v32 + 8) != *(_QWORD *)(v33 + 8) )
        break;
      v32 += 24LL;
      v33 += 24LL;
      if ( v135 == v32 )
        goto LABEL_32;
    }
LABEL_22:
    v27 = *(_QWORD *)(v135 - 24);
    if ( a2 != *(_QWORD *)(v27 + 48) )
    {
      v124 = sub_CB72A0();
      v125 = (__m128i *)v124[4];
      if ( v124[3] - (_QWORD)v125 <= 0x1Cu )
      {
        sub_CB6200((__int64)v124, "VPBlockBase has wrong parent\n", 0x1Du);
      }
      else
      {
        v126 = _mm_load_si128((const __m128i *)&xmmword_43A1680);
        qmemcpy(&v125[1], "wrong parent\n", 13);
        *v125 = v126;
        v124[4] += 29LL;
      }
LABEL_141:
      if ( v141 )
        j_j___libc_free_0(v141);
      if ( !v139 )
        _libc_free(v138);
      if ( v134 )
        j_j___libc_free_0(v134);
      if ( !v132 )
        _libc_free(v131);
      if ( v226 )
        j_j___libc_free_0((unsigned __int64)v226);
      if ( !v225 )
        _libc_free(v224);
      if ( v220 )
        j_j___libc_free_0(v220);
      if ( !v218 )
        _libc_free(v217);
      return 0;
    }
    if ( !(unsigned __int8)sub_2C48D10(a1, v27) )
      goto LABEL_141;
    sub_2C48140((__int64)v130, v27, v28, v29, v30, v31);
    v23 = v141;
  }
LABEL_32:
  if ( v23 )
    j_j___libc_free_0(v23);
  if ( !v139 )
    _libc_free(v138);
  if ( v134 )
    j_j___libc_free_0(v134);
  if ( !v132 )
    _libc_free(v131);
  if ( v226 )
    j_j___libc_free_0((unsigned __int64)v226);
  if ( !v225 )
    _libc_free(v224);
  if ( v220 )
    j_j___libc_free_0(v220);
  if ( !v218 )
    _libc_free(v217);
  sub_2C48240(v207, *(_QWORD *)(a2 + 112));
  v5 = (const __m128i *)v154;
  v6 = v151;
  sub_C8CD80((__int64)v151, (__int64)v154, (__int64)v211, v35, v36, v37);
  v39 = v215;
  v40 = v214;
  v155 = 0;
  v156 = 0;
  v157 = 0;
  v41 = v215 - v214;
  if ( v215 != v214 )
  {
    if ( v41 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v42 = sub_22077B0(v215 - v214);
      v39 = v215;
      v40 = v214;
      v43 = v42;
      goto LABEL_51;
    }
LABEL_241:
    sub_4261EA(v6, v5, v10);
  }
  v43 = 0;
LABEL_51:
  v155 = v43;
  v156 = v43;
  v157 = v43 + v41;
  if ( v40 != v39 )
  {
    v44 = (__m128i *)v43;
    v45 = (const __m128i *)v40;
    do
    {
      if ( v44 )
      {
        *v44 = _mm_loadu_si128(v45);
        v44[1].m128i_i64[0] = v45[1].m128i_i64[0];
      }
      v45 = (const __m128i *)((char *)v45 + 24);
      v44 = (__m128i *)((char *)v44 + 24);
    }
    while ( v45 != (const __m128i *)v39 );
    v43 += 8 * (((unsigned __int64)&v45[-2].m128i_u64[1] - v40) >> 3) + 24;
  }
  v156 = v43;
  v5 = (const __m128i *)v147;
  v6 = v144;
  sub_C8CD80((__int64)v144, (__int64)v147, (__int64)v207, v39, v40, v38);
  v47 = v210;
  v48 = v209;
  v148 = 0;
  v149 = 0;
  v150 = 0;
  v49 = v210 - v209;
  if ( v210 == v209 )
  {
    v51 = 0;
  }
  else
  {
    if ( v49 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_241;
    v50 = sub_22077B0(v210 - v209);
    v47 = v210;
    v48 = v209;
    v51 = v50;
  }
  v148 = v51;
  v149 = v51;
  v150 = v51 + v49;
  if ( v48 != v47 )
  {
    v52 = (__m128i *)v51;
    v53 = (const __m128i *)v48;
    do
    {
      if ( v52 )
      {
        *v52 = _mm_loadu_si128(v53);
        v52[1].m128i_i64[0] = v53[1].m128i_i64[0];
      }
      v53 = (const __m128i *)((char *)v53 + 24);
      v52 = (__m128i *)((char *)v52 + 24);
    }
    while ( (const __m128i *)v47 != v53 );
    v47 = (v47 - 24 - v48) >> 3;
    v51 += 8 * v47 + 24;
  }
  v149 = v51;
  v5 = (const __m128i *)v168;
  v6 = v165;
  sub_C8CD80((__int64)v165, (__int64)v168, (__int64)v151, v47, v48, v46);
  v55 = v156;
  v56 = v155;
  v169 = 0;
  v170 = 0;
  v171 = 0;
  v57 = v156 - v155;
  if ( v156 == v155 )
  {
    v59 = 0;
  }
  else
  {
    if ( v57 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_241;
    v58 = sub_22077B0(v156 - v155);
    v55 = v156;
    v56 = v155;
    v59 = v58;
  }
  v169 = v59;
  v170 = v59;
  v171 = v59 + v57;
  if ( v56 != v55 )
  {
    v60 = (__m128i *)v59;
    v61 = (const __m128i *)v56;
    do
    {
      if ( v60 )
      {
        *v60 = _mm_loadu_si128(v61);
        v60[1].m128i_i64[0] = v61[1].m128i_i64[0];
      }
      v61 = (const __m128i *)((char *)v61 + 24);
      v60 = (__m128i *)((char *)v60 + 24);
    }
    while ( (const __m128i *)v55 != v61 );
    v55 = (v55 - 24 - v56) >> 3;
    v59 += 8 * v55 + 24;
  }
  v170 = v59;
  v5 = (const __m128i *)v161;
  v6 = v158;
  sub_C8CD80((__int64)v158, (__int64)v161, (__int64)v144, v55, v56, v54);
  v63 = v149;
  v64 = v148;
  v162 = 0;
  v163 = 0;
  v164 = 0;
  v65 = v149 - v148;
  if ( v149 == v148 )
  {
    v65 = 0;
    v67 = 0;
  }
  else
  {
    if ( v65 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_241;
    v66 = sub_22077B0(v149 - v148);
    v63 = v149;
    v64 = v148;
    v67 = v66;
  }
  v162 = v67;
  v163 = v67;
  v164 = v67 + v65;
  if ( v64 != v63 )
  {
    v68 = (__m128i *)v67;
    v69 = (const __m128i *)v64;
    do
    {
      if ( v68 )
      {
        *v68 = _mm_loadu_si128(v69);
        v68[1].m128i_i64[0] = v69[1].m128i_i64[0];
      }
      v69 = (const __m128i *)((char *)v69 + 24);
      v68 = (__m128i *)((char *)v68 + 24);
    }
    while ( v69 != (const __m128i *)v63 );
    v67 += 8 * (((unsigned __int64)&v69[-2].m128i_u64[1] - v64) >> 3) + 24;
  }
  v163 = v67;
  v5 = (const __m128i *)v182;
  v6 = v179;
  sub_C8CD80((__int64)v179, (__int64)v182, (__int64)v165, v63, v64, v62);
  v71 = v170;
  v72 = v169;
  v183 = 0;
  v184 = 0;
  v185 = 0;
  v73 = v170 - v169;
  if ( v170 == v169 )
  {
    v75 = 0;
  }
  else
  {
    if ( v73 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_241;
    v74 = sub_22077B0(v170 - v169);
    v71 = v170;
    v72 = v169;
    v75 = v74;
  }
  v183 = v75;
  v184 = v75;
  v185 = v75 + v73;
  if ( v71 != v72 )
  {
    v76 = (__m128i *)v75;
    v77 = (const __m128i *)v72;
    do
    {
      if ( v76 )
      {
        *v76 = _mm_loadu_si128(v77);
        v76[1].m128i_i64[0] = v77[1].m128i_i64[0];
      }
      v77 = (const __m128i *)((char *)v77 + 24);
      v76 = (__m128i *)((char *)v76 + 24);
    }
    while ( (const __m128i *)v71 != v77 );
    v71 = (v71 - 24 - v72) >> 3;
    v75 += 8 * v71 + 24;
  }
  v184 = v75;
  v5 = (const __m128i *)v175;
  v6 = v172;
  sub_C8CD80((__int64)v172, (__int64)v175, (__int64)v158, v71, v72, v70);
  v79 = v163;
  v80 = v162;
  v176 = 0;
  v177 = 0;
  v178 = 0;
  v81 = v163 - v162;
  if ( v163 == v162 )
  {
    v83 = 0;
  }
  else
  {
    if ( v81 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_241;
    v82 = sub_22077B0(v163 - v162);
    v79 = v163;
    v80 = v162;
    v83 = v82;
  }
  v176 = (const __m128i *)v83;
  v177 = (const __m128i *)v83;
  v178 = v83 + v81;
  if ( v79 != v80 )
  {
    v84 = (__m128i *)v83;
    v85 = (const __m128i *)v80;
    do
    {
      if ( v84 )
      {
        *v84 = _mm_loadu_si128(v85);
        v84[1].m128i_i64[0] = v85[1].m128i_i64[0];
      }
      v85 = (const __m128i *)((char *)v85 + 24);
      v84 = (__m128i *)((char *)v84 + 24);
    }
    while ( (const __m128i *)v79 != v85 );
    v79 = (v79 - 24 - v80) >> 3;
    v83 += 8 * v79 + 24;
  }
  v177 = (const __m128i *)v83;
  v6 = v193;
  v5 = (const __m128i *)v196;
  sub_C8CD80((__int64)v193, (__int64)v196, (__int64)v179, v79, v80, v78);
  v87 = v184;
  v88 = v183;
  v197 = 0;
  v198 = 0;
  v199 = 0;
  v89 = v184 - v183;
  if ( v184 == v183 )
  {
    v91 = 0;
  }
  else
  {
    if ( v89 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_241;
    v90 = sub_22077B0(v184 - v183);
    v87 = v184;
    v88 = v183;
    v91 = v90;
  }
  v197 = v91;
  v198 = v91;
  v199 = v91 + v89;
  if ( v88 != v87 )
  {
    v92 = (__m128i *)v91;
    v93 = (const __m128i *)v88;
    do
    {
      if ( v92 )
      {
        *v92 = _mm_loadu_si128(v93);
        v92[1].m128i_i64[0] = v93[1].m128i_i64[0];
      }
      v93 = (const __m128i *)((char *)v93 + 24);
      v92 = (__m128i *)((char *)v92 + 24);
    }
    while ( (const __m128i *)v87 != v93 );
    v87 = (v87 - 24 - v88) >> 3;
    v91 += 8 * v87 + 24;
  }
  v198 = v91;
  v6 = v186;
  sub_C8CD80((__int64)v186, (__int64)v189, (__int64)v172, v87, v88, v86);
  v5 = v177;
  v95 = (__int64)v176;
  v190 = 0;
  j = 0;
  v192 = 0;
  v96 = (char *)v177 - (char *)v176;
  if ( v177 == v176 )
  {
    v98 = 0;
  }
  else
  {
    if ( v96 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_241;
    v97 = sub_22077B0((char *)v177 - (char *)v176);
    v5 = v177;
    v95 = (__int64)v176;
    v98 = v97;
  }
  v190 = v98;
  j = v98;
  v192 = v98 + v96;
  if ( (const __m128i *)v95 == v5 )
  {
    v101 = v98;
  }
  else
  {
    v99 = (__m128i *)v98;
    v100 = (const __m128i *)v95;
    do
    {
      if ( v99 )
      {
        *v99 = _mm_loadu_si128(v100);
        v99[1].m128i_i64[0] = v100[1].m128i_i64[0];
      }
      v100 = (const __m128i *)((char *)v100 + 24);
      v99 = (__m128i *)((char *)v99 + 24);
    }
    while ( v5 != v100 );
    v101 = v98 + 8 * (((unsigned __int64)&v5[-2].m128i_u64[1] - v95) >> 3) + 24;
  }
  for ( j = v101; ; v101 = j )
  {
    v102 = v197;
    if ( v101 - v98 == v198 - v197 )
      break;
LABEL_122:
    v5 = (const __m128i *)v203;
    v6 = v200;
    sub_C8CD80((__int64)v200, (__int64)v203, (__int64)v186, v98, v95, v94);
    v104 = j;
    v105 = v190;
    v204 = 0;
    v205 = 0;
    v206 = 0;
    v106 = j - v190;
    if ( j == v190 )
    {
      v108 = 0;
    }
    else
    {
      if ( v106 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_241;
      v107 = sub_22077B0(j - v190);
      v104 = j;
      v105 = v190;
      v108 = v107;
    }
    v204 = v108;
    v205 = v108;
    v206 = v108 + v106;
    if ( v105 == v104 )
    {
      v111 = v108;
    }
    else
    {
      v10 = (__m128i *)v108;
      v109 = (const __m128i *)v105;
      do
      {
        if ( v10 )
        {
          *v10 = _mm_loadu_si128(v109);
          v10[1].m128i_i64[0] = v109[1].m128i_i64[0];
        }
        v109 = (const __m128i *)((char *)v109 + 24);
        v10 = (__m128i *)((char *)v10 + 24);
      }
      while ( (const __m128i *)v104 != v109 );
      v110 = 0xAAAAAAAAAAAAAABLL * ((unsigned __int64)(v104 - 24 - v105) >> 3);
      v104 = 0x1FFFFFFFFFFFFFFFLL;
      v111 = v108 + 8 * (3 * (v110 & 0x1FFFFFFFFFFFFFFFLL) + 3);
    }
    v112 = *(_QWORD *)(v111 - 24);
    v205 = v111;
    v113 = 1;
    if ( !*(_BYTE *)(v112 + 8) )
    {
      v114 = sub_2C4A350(a1, v112, v10);
      v108 = v204;
      v113 = v114;
      v106 = v206 - v204;
    }
    if ( v108 )
    {
      v112 = v106;
      j_j___libc_free_0(v108);
    }
    if ( !v202 )
      _libc_free(v201);
    if ( !v113 )
      goto LABEL_186;
    sub_2C48140((__int64)v186, v112, (__int64)v10, v104, v105, v103);
    v98 = v190;
  }
  while ( v101 != v98 )
  {
    if ( *(_QWORD *)v98 != *(_QWORD *)v102 )
      goto LABEL_122;
    v123 = *(_BYTE *)(v98 + 16);
    if ( v123 != *(_BYTE *)(v102 + 16) || v123 && *(_QWORD *)(v98 + 8) != *(_QWORD *)(v102 + 8) )
      goto LABEL_122;
    v98 += 24;
    v102 += 24LL;
  }
LABEL_186:
  sub_C8CF70((__int64)&v216, v219, 8, (__int64)v189, (__int64)v186);
  v220 = v190;
  v221 = j;
  v222 = v192;
  if ( !v188 )
    _libc_free(v187);
  if ( v197 )
    j_j___libc_free_0(v197);
  if ( !v195 )
    _libc_free(v194);
  if ( v176 )
    j_j___libc_free_0((unsigned __int64)v176);
  if ( !v174 )
    _libc_free(v173);
  if ( v183 )
    j_j___libc_free_0(v183);
  if ( !v181 )
    _libc_free(v180);
  v127 = v155;
  v115 = 0;
  if ( v156 - v155 == v221 - v220 )
  {
    if ( v155 == v156 )
    {
LABEL_239:
      v115 = 1;
    }
    else
    {
      v128 = v220;
      while ( *(_QWORD *)v127 == *(_QWORD *)v128 )
      {
        v129 = *(_BYTE *)(v127 + 16);
        if ( v129 != *(_BYTE *)(v128 + 16) || v129 && *(_QWORD *)(v127 + 8) != *(_QWORD *)(v128 + 8) )
          break;
        v127 += 24LL;
        v128 += 24LL;
        if ( v156 == v127 )
          goto LABEL_239;
      }
      v115 = 0;
    }
  }
  if ( v220 )
    j_j___libc_free_0(v220);
  if ( !v218 )
    _libc_free(v217);
  if ( v162 )
    j_j___libc_free_0(v162);
  if ( !v160 )
    _libc_free(v159);
  if ( v169 )
    j_j___libc_free_0(v169);
  if ( !v167 )
    _libc_free(v166);
  if ( v148 )
    j_j___libc_free_0(v148);
  if ( !v146 )
    _libc_free(v145);
  if ( v155 )
    j_j___libc_free_0(v155);
  if ( !v153 )
    _libc_free(v152);
  if ( v214 )
    j_j___libc_free_0(v214);
  if ( !v213 )
    _libc_free(v212);
  if ( v209 )
    j_j___libc_free_0(v209);
  if ( !v208 )
    _libc_free(v207[1]);
  return v115;
}
