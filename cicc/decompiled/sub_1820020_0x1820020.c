// Function: sub_1820020
// Address: 0x1820020
//
void __fastcall sub_1820020(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v6; // rdi
  __int64 v7; // rcx
  __int64 v8; // rdx
  _BYTE *v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rcx
  char v12; // si
  __int64 v13; // rcx
  unsigned __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v16; // rcx
  char v17; // si
  int v18; // r8d
  int v19; // r9d
  __int64 v20; // rcx
  unsigned __int64 v21; // r12
  __int64 v22; // rax
  __int64 v23; // rdi
  __int64 v24; // rax
  __int64 v25; // rdx
  char v26; // si
  unsigned __int64 v27; // r12
  __int64 v28; // rcx
  __int64 v29; // rax
  char v30; // si
  bool v31; // si
  __int64 v32; // rdx
  __int64 v33; // rbx
  __int64 v34; // rcx
  __int64 v35; // rax
  __int64 v36; // rsi
  __int64 v37; // rcx
  char v38; // si
  __int64 v39; // rcx
  __int64 v40; // rax
  __int64 v41; // rcx
  char v42; // si
  __int64 v43; // rcx
  __int64 v44; // rax
  __int64 v45; // rcx
  char v46; // si
  __int64 v47; // rcx
  unsigned __int64 v48; // r15
  __int64 v49; // rax
  __int64 v50; // rcx
  char v51; // si
  __int64 v52; // rcx
  unsigned __int64 v53; // r13
  __int64 v54; // rax
  __int64 v55; // rcx
  char v56; // si
  __int64 v57; // rcx
  unsigned __int64 v58; // r15
  __int64 v59; // rax
  __int64 v60; // rcx
  char v61; // si
  __int64 v62; // rcx
  __int64 v63; // rax
  __int64 v64; // rcx
  char v65; // si
  __int64 v66; // rcx
  unsigned __int64 v67; // r13
  __int64 v68; // rax
  __int64 v69; // rcx
  char v70; // si
  __int64 v71; // rcx
  unsigned __int64 v72; // r15
  __int64 v73; // rax
  __int64 v74; // rcx
  char v75; // si
  __int64 v76; // rcx
  unsigned __int64 v77; // r13
  __int64 v78; // rax
  __int64 v79; // rcx
  char v80; // si
  unsigned __int64 v81; // r15
  __int64 v82; // rax
  _BYTE *v83; // rsi
  char v84; // di
  __int64 v85; // rax
  __int64 v86; // rax
  unsigned __int64 v87; // rax
  __int64 v88; // rcx
  unsigned __int64 v89; // r13
  __int64 v90; // rax
  __int64 v91; // rcx
  char v92; // si
  __int64 v93; // rcx
  __int64 v94; // rax
  __int64 v95; // rcx
  char v96; // si
  __int64 v97; // rcx
  unsigned __int64 v98; // r15
  __int64 v99; // rax
  __int64 v100; // rcx
  char v101; // si
  __int64 v102; // rcx
  unsigned __int64 v103; // r13
  __int64 v104; // rax
  __int64 v105; // rdi
  __int64 v106; // rax
  __int64 v107; // rdx
  char v108; // si
  __int64 v109; // rcx
  __int64 v110; // rax
  char v111; // si
  char v112; // r8
  bool v113; // si
  __int64 v114; // [rsp+0h] [rbp-C40h]
  _BYTE *v115; // [rsp+0h] [rbp-C40h]
  _BYTE *v116; // [rsp+8h] [rbp-C38h]
  _BYTE *v117; // [rsp+8h] [rbp-C38h]
  _BYTE *v118; // [rsp+8h] [rbp-C38h]
  _BYTE *v119; // [rsp+8h] [rbp-C38h]
  _QWORD v120[2]; // [rsp+10h] [rbp-C30h] BYREF
  unsigned __int64 v121; // [rsp+20h] [rbp-C20h]
  char v122[64]; // [rsp+38h] [rbp-C08h] BYREF
  __int64 v123; // [rsp+78h] [rbp-BC8h]
  __int64 v124; // [rsp+80h] [rbp-BC0h]
  _BYTE *v125; // [rsp+88h] [rbp-BB8h]
  _QWORD v126[2]; // [rsp+90h] [rbp-BB0h] BYREF
  unsigned __int64 v127; // [rsp+A0h] [rbp-BA0h]
  __int64 v128; // [rsp+F8h] [rbp-B48h]
  __int64 v129; // [rsp+100h] [rbp-B40h]
  __int64 v130; // [rsp+108h] [rbp-B38h]
  _QWORD v131[2]; // [rsp+110h] [rbp-B30h] BYREF
  unsigned __int64 v132; // [rsp+120h] [rbp-B20h]
  char v133[64]; // [rsp+138h] [rbp-B08h] BYREF
  __int64 v134; // [rsp+178h] [rbp-AC8h]
  __int64 v135; // [rsp+180h] [rbp-AC0h]
  _BYTE *v136; // [rsp+188h] [rbp-AB8h]
  _QWORD v137[2]; // [rsp+190h] [rbp-AB0h] BYREF
  unsigned __int64 v138; // [rsp+1A0h] [rbp-AA0h]
  char v139[64]; // [rsp+1B8h] [rbp-A88h] BYREF
  __int64 v140; // [rsp+1F8h] [rbp-A48h]
  __int64 v141; // [rsp+200h] [rbp-A40h]
  __int64 v142; // [rsp+208h] [rbp-A38h]
  _QWORD v143[2]; // [rsp+210h] [rbp-A30h] BYREF
  unsigned __int64 v144; // [rsp+220h] [rbp-A20h]
  _BYTE v145[64]; // [rsp+238h] [rbp-A08h] BYREF
  __int64 v146; // [rsp+278h] [rbp-9C8h]
  __int64 v147; // [rsp+280h] [rbp-9C0h]
  unsigned __int64 v148; // [rsp+288h] [rbp-9B8h]
  _QWORD v149[2]; // [rsp+290h] [rbp-9B0h] BYREF
  unsigned __int64 v150; // [rsp+2A0h] [rbp-9A0h]
  _BYTE v151[64]; // [rsp+2B8h] [rbp-988h] BYREF
  __int64 v152; // [rsp+2F8h] [rbp-948h]
  __int64 v153; // [rsp+300h] [rbp-940h]
  unsigned __int64 v154; // [rsp+308h] [rbp-938h]
  _QWORD v155[2]; // [rsp+310h] [rbp-930h] BYREF
  unsigned __int64 v156; // [rsp+320h] [rbp-920h]
  _BYTE v157[64]; // [rsp+338h] [rbp-908h] BYREF
  __int64 v158; // [rsp+378h] [rbp-8C8h]
  __int64 v159; // [rsp+380h] [rbp-8C0h]
  unsigned __int64 v160; // [rsp+388h] [rbp-8B8h]
  _QWORD v161[2]; // [rsp+390h] [rbp-8B0h] BYREF
  unsigned __int64 v162; // [rsp+3A0h] [rbp-8A0h]
  char v163[64]; // [rsp+3B8h] [rbp-888h] BYREF
  __int64 v164; // [rsp+3F8h] [rbp-848h]
  __int64 v165; // [rsp+400h] [rbp-840h]
  _BYTE *v166; // [rsp+408h] [rbp-838h]
  _QWORD v167[2]; // [rsp+410h] [rbp-830h] BYREF
  unsigned __int64 v168; // [rsp+420h] [rbp-820h]
  _BYTE v169[64]; // [rsp+438h] [rbp-808h] BYREF
  __int64 v170; // [rsp+478h] [rbp-7C8h]
  __int64 v171; // [rsp+480h] [rbp-7C0h]
  unsigned __int64 v172; // [rsp+488h] [rbp-7B8h]
  _QWORD v173[2]; // [rsp+490h] [rbp-7B0h] BYREF
  unsigned __int64 v174; // [rsp+4A0h] [rbp-7A0h]
  __int64 v175; // [rsp+4F8h] [rbp-748h]
  __int64 v176; // [rsp+500h] [rbp-740h]
  __int64 v177; // [rsp+508h] [rbp-738h]
  _QWORD v178[2]; // [rsp+510h] [rbp-730h] BYREF
  unsigned __int64 v179; // [rsp+520h] [rbp-720h]
  _BYTE v180[64]; // [rsp+538h] [rbp-708h] BYREF
  __int64 v181; // [rsp+578h] [rbp-6C8h]
  __int64 v182; // [rsp+580h] [rbp-6C0h]
  unsigned __int64 v183; // [rsp+588h] [rbp-6B8h]
  _QWORD v184[2]; // [rsp+590h] [rbp-6B0h] BYREF
  unsigned __int64 v185; // [rsp+5A0h] [rbp-6A0h]
  char v186[64]; // [rsp+5B8h] [rbp-688h] BYREF
  __int64 v187; // [rsp+5F8h] [rbp-648h]
  __int64 v188; // [rsp+600h] [rbp-640h]
  _BYTE *v189; // [rsp+608h] [rbp-638h]
  _QWORD v190[2]; // [rsp+610h] [rbp-630h] BYREF
  unsigned __int64 v191; // [rsp+620h] [rbp-620h]
  char v192[64]; // [rsp+638h] [rbp-608h] BYREF
  __int64 v193; // [rsp+678h] [rbp-5C8h]
  __int64 v194; // [rsp+680h] [rbp-5C0h]
  unsigned __int64 v195; // [rsp+688h] [rbp-5B8h]
  _QWORD v196[2]; // [rsp+690h] [rbp-5B0h] BYREF
  unsigned __int64 v197; // [rsp+6A0h] [rbp-5A0h]
  _BYTE v198[64]; // [rsp+6B8h] [rbp-588h] BYREF
  __int64 v199; // [rsp+6F8h] [rbp-548h]
  __int64 v200; // [rsp+700h] [rbp-540h]
  unsigned __int64 v201; // [rsp+708h] [rbp-538h]
  _QWORD v202[2]; // [rsp+710h] [rbp-530h] BYREF
  unsigned __int64 v203; // [rsp+720h] [rbp-520h]
  char v204[64]; // [rsp+738h] [rbp-508h] BYREF
  __int64 v205; // [rsp+778h] [rbp-4C8h]
  _BYTE *v206; // [rsp+780h] [rbp-4C0h]
  unsigned __int64 v207; // [rsp+788h] [rbp-4B8h]
  _QWORD v208[2]; // [rsp+790h] [rbp-4B0h] BYREF
  unsigned __int64 v209; // [rsp+7A0h] [rbp-4A0h]
  _BYTE v210[64]; // [rsp+7B8h] [rbp-488h] BYREF
  __int64 v211; // [rsp+7F8h] [rbp-448h]
  __int64 v212; // [rsp+800h] [rbp-440h]
  unsigned __int64 v213; // [rsp+808h] [rbp-438h]
  _QWORD v214[2]; // [rsp+810h] [rbp-430h] BYREF
  unsigned __int64 v215; // [rsp+820h] [rbp-420h]
  char v216[64]; // [rsp+838h] [rbp-408h] BYREF
  __int64 v217; // [rsp+878h] [rbp-3C8h]
  __int64 v218; // [rsp+880h] [rbp-3C0h]
  unsigned __int64 v219; // [rsp+888h] [rbp-3B8h]
  _QWORD v220[2]; // [rsp+890h] [rbp-3B0h] BYREF
  unsigned __int64 v221; // [rsp+8A0h] [rbp-3A0h]
  _BYTE v222[64]; // [rsp+8B8h] [rbp-388h] BYREF
  __int64 v223; // [rsp+8F8h] [rbp-348h]
  __int64 v224; // [rsp+900h] [rbp-340h]
  unsigned __int64 v225; // [rsp+908h] [rbp-338h]
  _QWORD v226[2]; // [rsp+910h] [rbp-330h] BYREF
  unsigned __int64 v227; // [rsp+920h] [rbp-320h]
  char v228[64]; // [rsp+938h] [rbp-308h] BYREF
  __int64 v229; // [rsp+978h] [rbp-2C8h]
  __int64 v230; // [rsp+980h] [rbp-2C0h]
  unsigned __int64 v231; // [rsp+988h] [rbp-2B8h]
  _QWORD v232[2]; // [rsp+990h] [rbp-2B0h] BYREF
  unsigned __int64 v233; // [rsp+9A0h] [rbp-2A0h]
  char v234[64]; // [rsp+9B8h] [rbp-288h] BYREF
  __int64 v235; // [rsp+9F8h] [rbp-248h]
  __int64 v236; // [rsp+A00h] [rbp-240h]
  unsigned __int64 v237; // [rsp+A08h] [rbp-238h]
  _QWORD v238[2]; // [rsp+A10h] [rbp-230h] BYREF
  unsigned __int64 v239; // [rsp+A20h] [rbp-220h]
  _BYTE v240[64]; // [rsp+A38h] [rbp-208h] BYREF
  __int64 v241; // [rsp+A78h] [rbp-1C8h]
  __int64 v242; // [rsp+A80h] [rbp-1C0h]
  unsigned __int64 v243; // [rsp+A88h] [rbp-1B8h]
  _QWORD v244[2]; // [rsp+A90h] [rbp-1B0h] BYREF
  unsigned __int64 v245; // [rsp+AA0h] [rbp-1A0h]
  char v246[64]; // [rsp+AB8h] [rbp-188h] BYREF
  __int64 v247; // [rsp+AF8h] [rbp-148h]
  __int64 v248; // [rsp+B00h] [rbp-140h]
  _BYTE *v249; // [rsp+B08h] [rbp-138h]
  _QWORD v250[2]; // [rsp+B10h] [rbp-130h] BYREF
  unsigned __int64 v251; // [rsp+B20h] [rbp-120h]
  _BYTE v252[64]; // [rsp+B38h] [rbp-108h] BYREF
  __int64 v253; // [rsp+B78h] [rbp-C8h]
  __int64 v254; // [rsp+B80h] [rbp-C0h]
  unsigned __int64 v255; // [rsp+B88h] [rbp-B8h]
  _QWORD v256[2]; // [rsp+B90h] [rbp-B0h] BYREF
  unsigned __int64 v257; // [rsp+BA0h] [rbp-A0h]
  _BYTE v258[64]; // [rsp+BB8h] [rbp-88h] BYREF
  __int64 v259; // [rsp+BF8h] [rbp-48h]
  __int64 v260; // [rsp+C00h] [rbp-40h]
  unsigned __int64 v261; // [rsp+C08h] [rbp-38h]

  sub_1817790(v126, a3);
  v6 = v120;
  sub_16CCCB0(v120, (__int64)v122, a2);
  v7 = *(_QWORD *)(a2 + 112);
  v8 = *(_QWORD *)(a2 + 104);
  v123 = 0;
  v124 = 0;
  v125 = 0;
  v9 = (_BYTE *)(v7 - v8);
  if ( v7 == v8 )
  {
    v10 = 0;
  }
  else
  {
    if ( (unsigned __int64)v9 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_322;
    v116 = (_BYTE *)(v7 - v8);
    v10 = sub_22077B0(v7 - v8);
    v7 = *(_QWORD *)(a2 + 112);
    v8 = *(_QWORD *)(a2 + 104);
    v9 = v116;
  }
  v123 = v10;
  v124 = v10;
  v125 = &v9[v10];
  if ( v7 == v8 )
  {
    v11 = v10;
  }
  else
  {
    v11 = v10 + v7 - v8;
    do
    {
      if ( v10 )
      {
        *(_QWORD *)v10 = *(_QWORD *)v8;
        v12 = *(_BYTE *)(v8 + 24);
        *(_BYTE *)(v10 + 24) = v12;
        if ( v12 )
          *(__m128i *)(v10 + 8) = _mm_loadu_si128((const __m128i *)(v8 + 8));
      }
      v10 += 32;
      v8 += 32;
    }
    while ( v10 != v11 );
  }
  v6 = v149;
  v9 = v151;
  v124 = v11;
  sub_16CCCB0(v149, (__int64)v151, (__int64)v126);
  v13 = v129;
  v8 = v128;
  v152 = 0;
  v153 = 0;
  v154 = 0;
  v14 = v129 - v128;
  if ( v129 == v128 )
  {
    v15 = 0;
  }
  else
  {
    if ( v14 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_322;
    v15 = sub_22077B0(v129 - v128);
    v13 = v129;
    v8 = v128;
  }
  v152 = v15;
  v153 = v15;
  v154 = v15 + v14;
  if ( v13 == v8 )
  {
    v16 = v15;
  }
  else
  {
    v16 = v15 + v13 - v8;
    do
    {
      if ( v15 )
      {
        *(_QWORD *)v15 = *(_QWORD *)v8;
        v17 = *(_BYTE *)(v8 + 24);
        *(_BYTE *)(v15 + 24) = v17;
        if ( v17 )
          *(__m128i *)(v15 + 8) = _mm_loadu_si128((const __m128i *)(v8 + 8));
      }
      v15 += 32;
      v8 += 32;
    }
    while ( v15 != v16 );
  }
  v9 = v145;
  v153 = v16;
  v6 = v143;
  sub_16CCCB0(v143, (__int64)v145, (__int64)v120);
  v8 = v124;
  v20 = v123;
  v146 = 0;
  v147 = 0;
  v148 = 0;
  v21 = v124 - v123;
  if ( v124 == v123 )
  {
    v23 = 0;
  }
  else
  {
    if ( v21 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_322;
    v22 = sub_22077B0(v124 - v123);
    v8 = v124;
    v20 = v123;
    v23 = v22;
  }
  v146 = v23;
  v147 = v23;
  v148 = v23 + v21;
  if ( v20 == v8 )
  {
    v25 = v23;
  }
  else
  {
    v24 = v23;
    v25 = v23 + v8 - v20;
    do
    {
      if ( v24 )
      {
        *(_QWORD *)v24 = *(_QWORD *)v20;
        v26 = *(_BYTE *)(v20 + 24);
        *(_BYTE *)(v24 + 24) = v26;
        if ( v26 )
          *(__m128i *)(v24 + 8) = _mm_loadu_si128((const __m128i *)(v20 + 8));
      }
      v24 += 32;
      v20 += 32;
    }
    while ( v24 != v25 );
  }
  v147 = v25;
  v27 = 0;
  while ( 1 )
  {
    v28 = v152;
    if ( v25 - v23 != v153 - v152 )
      goto LABEL_29;
    if ( v23 == v25 )
      break;
    v29 = v23;
    while ( *(_QWORD *)v29 == *(_QWORD *)v28 )
    {
      v30 = *(_BYTE *)(v29 + 24);
      v18 = *(unsigned __int8 *)(v28 + 24);
      if ( v30 && (_BYTE)v18 )
        v31 = *(_DWORD *)(v29 + 16) == *(_DWORD *)(v28 + 16);
      else
        v31 = v30 == (char)v18;
      if ( !v31 )
        break;
      v29 += 32;
      v28 += 32;
      if ( v29 == v25 )
        goto LABEL_39;
    }
LABEL_29:
    ++v27;
    sub_17D3A30((__int64)v143);
    v23 = v146;
    v25 = v147;
  }
LABEL_39:
  if ( v23 )
    j_j___libc_free_0(v23, v148 - v23);
  if ( v144 != v143[1] )
    _libc_free(v144);
  if ( v152 )
    j_j___libc_free_0(v152, v154 - v152);
  if ( v150 != v149[1] )
    _libc_free(v150);
  if ( v123 )
    j_j___libc_free_0(v123, &v125[-v123]);
  if ( v121 != v120[1] )
    _libc_free(v121);
  if ( v128 )
    j_j___libc_free_0(v128, v130 - v128);
  if ( v127 != v126[1] )
    _libc_free(v127);
  v32 = *(unsigned int *)(a1 + 8);
  if ( (unsigned __int64)*(unsigned int *)(a1 + 12) - v32 < v27 )
  {
    sub_16CD150(a1, (const void *)(a1 + 16), v27 + v32, 8, v18, v19);
    v32 = *(unsigned int *)(a1 + 8);
  }
  v33 = *(_QWORD *)a1 + 8 * v32;
  v6 = v137;
  sub_16CCCB0(v137, (__int64)v139, a3);
  v34 = *(_QWORD *)(a3 + 112);
  v8 = *(_QWORD *)(a3 + 104);
  v140 = 0;
  v141 = 0;
  v142 = 0;
  v9 = (_BYTE *)(v34 - v8);
  if ( v34 != v8 )
  {
    if ( (unsigned __int64)v9 <= 0x7FFFFFFFFFFFFFE0LL )
    {
      v114 = v34 - v8;
      v35 = sub_22077B0(v34 - v8);
      v34 = *(_QWORD *)(a3 + 112);
      v8 = *(_QWORD *)(a3 + 104);
      v36 = v114;
      goto LABEL_60;
    }
LABEL_322:
    sub_4261EA(v6, v9, v8);
  }
  v36 = 0;
  v35 = 0;
LABEL_60:
  v140 = v35;
  v141 = v35;
  v142 = v35 + v36;
  if ( v8 == v34 )
  {
    v37 = v35;
  }
  else
  {
    v37 = v35 + v34 - v8;
    do
    {
      if ( v35 )
      {
        *(_QWORD *)v35 = *(_QWORD *)v8;
        v38 = *(_BYTE *)(v8 + 24);
        *(_BYTE *)(v35 + 24) = v38;
        if ( v38 )
          *(__m128i *)(v35 + 8) = _mm_loadu_si128((const __m128i *)(v8 + 8));
      }
      v35 += 32;
      v8 += 32;
    }
    while ( v37 != v35 );
  }
  v141 = v37;
  v6 = v131;
  sub_16CCCB0(v131, (__int64)v133, a2);
  v39 = *(_QWORD *)(a2 + 112);
  v8 = *(_QWORD *)(a2 + 104);
  v134 = 0;
  v135 = 0;
  v136 = 0;
  v9 = (_BYTE *)(v39 - v8);
  if ( v39 == v8 )
  {
    v40 = 0;
  }
  else
  {
    if ( (unsigned __int64)v9 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_322;
    v115 = (_BYTE *)(v39 - v8);
    v40 = sub_22077B0(v39 - v8);
    v39 = *(_QWORD *)(a2 + 112);
    v8 = *(_QWORD *)(a2 + 104);
    v9 = v115;
  }
  v134 = v40;
  v135 = v40;
  v136 = &v9[v40];
  if ( v8 == v39 )
  {
    v41 = v40;
  }
  else
  {
    v41 = v40 + v39 - v8;
    do
    {
      if ( v40 )
      {
        *(_QWORD *)v40 = *(_QWORD *)v8;
        v42 = *(_BYTE *)(v8 + 24);
        *(_BYTE *)(v40 + 24) = v42;
        if ( v42 )
          *(__m128i *)(v40 + 8) = _mm_loadu_si128((const __m128i *)(v8 + 8));
      }
      v40 += 32;
      v8 += 32;
    }
    while ( v40 != v41 );
  }
  v135 = v41;
  v6 = v161;
  sub_16CCCB0(v161, (__int64)v163, (__int64)v137);
  v43 = v141;
  v8 = v140;
  v164 = 0;
  v165 = 0;
  v166 = 0;
  v9 = (_BYTE *)(v141 - v140);
  if ( v141 == v140 )
  {
    v44 = 0;
  }
  else
  {
    if ( (unsigned __int64)v9 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_322;
    v117 = (_BYTE *)(v141 - v140);
    v44 = sub_22077B0(v141 - v140);
    v43 = v141;
    v8 = v140;
    v9 = v117;
  }
  v164 = v44;
  v165 = v44;
  v166 = &v9[v44];
  if ( v43 == v8 )
  {
    v45 = v44;
  }
  else
  {
    v45 = v44 + v43 - v8;
    do
    {
      if ( v44 )
      {
        *(_QWORD *)v44 = *(_QWORD *)v8;
        v46 = *(_BYTE *)(v8 + 24);
        *(_BYTE *)(v44 + 24) = v46;
        if ( v46 )
          *(__m128i *)(v44 + 8) = _mm_loadu_si128((const __m128i *)(v8 + 8));
      }
      v44 += 32;
      v8 += 32;
    }
    while ( v44 != v45 );
  }
  v9 = v157;
  v165 = v45;
  v6 = v155;
  sub_16CCCB0(v155, (__int64)v157, (__int64)v131);
  v47 = v135;
  v8 = v134;
  v158 = 0;
  v159 = 0;
  v160 = 0;
  v48 = v135 - v134;
  if ( v135 == v134 )
  {
    v49 = 0;
  }
  else
  {
    if ( v48 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_322;
    v49 = sub_22077B0(v135 - v134);
    v47 = v135;
    v8 = v134;
  }
  v158 = v49;
  v159 = v49;
  v160 = v49 + v48;
  if ( v47 == v8 )
  {
    v50 = v49;
  }
  else
  {
    v50 = v49 + v47 - v8;
    do
    {
      if ( v49 )
      {
        *(_QWORD *)v49 = *(_QWORD *)v8;
        v51 = *(_BYTE *)(v8 + 24);
        *(_BYTE *)(v49 + 24) = v51;
        if ( v51 )
          *(__m128i *)(v49 + 8) = _mm_loadu_si128((const __m128i *)(v8 + 8));
      }
      v49 += 32;
      v8 += 32;
    }
    while ( v50 != v49 );
  }
  v9 = v169;
  v159 = v50;
  v6 = v167;
  sub_16CCCB0(v167, (__int64)v169, (__int64)v161);
  v52 = v165;
  v8 = v164;
  v170 = 0;
  v171 = 0;
  v172 = 0;
  v53 = v165 - v164;
  if ( v165 == v164 )
  {
    v54 = 0;
  }
  else
  {
    if ( v53 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_322;
    v54 = sub_22077B0(v165 - v164);
    v52 = v165;
    v8 = v164;
  }
  v170 = v54;
  v171 = v54;
  v172 = v54 + v53;
  if ( v52 == v8 )
  {
    v55 = v54;
  }
  else
  {
    v55 = v54 + v52 - v8;
    do
    {
      if ( v54 )
      {
        *(_QWORD *)v54 = *(_QWORD *)v8;
        v56 = *(_BYTE *)(v8 + 24);
        *(_BYTE *)(v54 + 24) = v56;
        if ( v56 )
          *(__m128i *)(v54 + 8) = _mm_loadu_si128((const __m128i *)(v8 + 8));
      }
      v54 += 32;
      v8 += 32;
    }
    while ( v55 != v54 );
  }
  v171 = v55;
  sub_1817790(v173, (__int64)v155);
  v9 = v180;
  v6 = v178;
  sub_16CCCB0(v178, (__int64)v180, (__int64)v167);
  v57 = v171;
  v8 = v170;
  v181 = 0;
  v182 = 0;
  v183 = 0;
  v58 = v171 - v170;
  if ( v171 == v170 )
  {
    v59 = 0;
  }
  else
  {
    if ( v58 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_322;
    v59 = sub_22077B0(v171 - v170);
    v57 = v171;
    v8 = v170;
  }
  v181 = v59;
  v182 = v59;
  v183 = v59 + v58;
  if ( v57 == v8 )
  {
    v60 = v59;
  }
  else
  {
    v60 = v59 + v57 - v8;
    do
    {
      if ( v59 )
      {
        *(_QWORD *)v59 = *(_QWORD *)v8;
        v61 = *(_BYTE *)(v8 + 24);
        *(_BYTE *)(v59 + 24) = v61;
        if ( v61 )
          *(__m128i *)(v59 + 8) = _mm_loadu_si128((const __m128i *)(v8 + 8));
      }
      v59 += 32;
      v8 += 32;
    }
    while ( v60 != v59 );
  }
  v182 = v60;
  v6 = v184;
  sub_16CCCB0(v184, (__int64)v186, (__int64)v173);
  v62 = v176;
  v8 = v175;
  v187 = 0;
  v188 = 0;
  v189 = 0;
  v9 = (_BYTE *)(v176 - v175);
  if ( v176 == v175 )
  {
    v63 = 0;
  }
  else
  {
    if ( (unsigned __int64)v9 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_322;
    v118 = (_BYTE *)(v176 - v175);
    v63 = sub_22077B0(v176 - v175);
    v62 = v176;
    v8 = v175;
    v9 = v118;
  }
  v187 = v63;
  v188 = v63;
  v189 = &v9[v63];
  if ( v62 == v8 )
  {
    v64 = v63;
  }
  else
  {
    v64 = v63 + v62 - v8;
    do
    {
      if ( v63 )
      {
        *(_QWORD *)v63 = *(_QWORD *)v8;
        v65 = *(_BYTE *)(v8 + 24);
        *(_BYTE *)(v63 + 24) = v65;
        if ( v65 )
          *(__m128i *)(v63 + 8) = _mm_loadu_si128((const __m128i *)(v8 + 8));
      }
      v63 += 32;
      v8 += 32;
    }
    while ( v64 != v63 );
  }
  v9 = v198;
  v188 = v64;
  v6 = v196;
  sub_16CCCB0(v196, (__int64)v198, (__int64)v178);
  v66 = v182;
  v8 = v181;
  v199 = 0;
  v200 = 0;
  v201 = 0;
  v67 = v182 - v181;
  if ( v182 == v181 )
  {
    v67 = 0;
    v68 = 0;
  }
  else
  {
    if ( v67 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_322;
    v68 = sub_22077B0(v182 - v181);
    v66 = v182;
    v8 = v181;
  }
  v199 = v68;
  v200 = v68;
  v201 = v68 + v67;
  if ( v8 == v66 )
  {
    v69 = v68;
  }
  else
  {
    v69 = v68 + v66 - v8;
    do
    {
      if ( v68 )
      {
        *(_QWORD *)v68 = *(_QWORD *)v8;
        v70 = *(_BYTE *)(v8 + 24);
        *(_BYTE *)(v68 + 24) = v70;
        if ( v70 )
          *(__m128i *)(v68 + 8) = _mm_loadu_si128((const __m128i *)(v8 + 8));
      }
      v68 += 32;
      v8 += 32;
    }
    while ( v69 != v68 );
  }
  v200 = v69;
  sub_16CCEE0(v190, (__int64)v192, 8, (__int64)v196);
  v6 = v208;
  v9 = v210;
  v193 = v199;
  v199 = 0;
  v194 = v200;
  v200 = 0;
  v195 = v201;
  v201 = 0;
  sub_16CCCB0(v208, (__int64)v210, (__int64)v184);
  v71 = v188;
  v8 = v187;
  v211 = 0;
  v212 = 0;
  v213 = 0;
  v72 = v188 - v187;
  if ( v188 == v187 )
  {
    v73 = 0;
  }
  else
  {
    if ( v72 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_322;
    v73 = sub_22077B0(v188 - v187);
    v71 = v188;
    v8 = v187;
  }
  v211 = v73;
  v212 = v73;
  v213 = v73 + v72;
  if ( v71 == v8 )
  {
    v74 = v73;
  }
  else
  {
    v74 = v73 + v71 - v8;
    do
    {
      if ( v73 )
      {
        *(_QWORD *)v73 = *(_QWORD *)v8;
        v75 = *(_BYTE *)(v8 + 24);
        *(_BYTE *)(v73 + 24) = v75;
        if ( v75 )
          *(__m128i *)(v73 + 8) = _mm_loadu_si128((const __m128i *)(v8 + 8));
      }
      v73 += 32;
      v8 += 32;
    }
    while ( v73 != v74 );
  }
  v212 = v74;
  sub_16CCEE0(v202, (__int64)v204, 8, (__int64)v208);
  v6 = v220;
  v9 = v222;
  v205 = v211;
  v211 = 0;
  v206 = (_BYTE *)v212;
  v212 = 0;
  v207 = v213;
  v213 = 0;
  sub_16CCCB0(v220, (__int64)v222, (__int64)v190);
  v76 = v194;
  v8 = v193;
  v223 = 0;
  v224 = 0;
  v225 = 0;
  v77 = v194 - v193;
  if ( v194 == v193 )
  {
    v78 = 0;
  }
  else
  {
    if ( v77 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_322;
    v78 = sub_22077B0(v194 - v193);
    v76 = v194;
    v8 = v193;
  }
  v223 = v78;
  v224 = v78;
  v225 = v78 + v77;
  if ( v76 == v8 )
  {
    v79 = v78;
  }
  else
  {
    v79 = v78 + v76 - v8;
    do
    {
      if ( v78 )
      {
        *(_QWORD *)v78 = *(_QWORD *)v8;
        v80 = *(_BYTE *)(v8 + 24);
        *(_BYTE *)(v78 + 24) = v80;
        if ( v80 )
          *(__m128i *)(v78 + 8) = _mm_loadu_si128((const __m128i *)(v8 + 8));
      }
      v78 += 32;
      v8 += 32;
    }
    while ( v78 != v79 );
  }
  v224 = v79;
  sub_16CCEE0(v214, (__int64)v216, 8, (__int64)v220);
  v6 = v232;
  v217 = v223;
  v223 = 0;
  v218 = v224;
  v224 = 0;
  v219 = v225;
  v225 = 0;
  sub_16CCCB0(v232, (__int64)v234, (__int64)v202);
  v9 = v206;
  v8 = v205;
  v235 = 0;
  v236 = 0;
  v237 = 0;
  v81 = (unsigned __int64)&v206[-v205];
  if ( v206 == (_BYTE *)v205 )
  {
    v82 = 0;
  }
  else
  {
    if ( v81 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_322;
    v82 = sub_22077B0(&v206[-v205]);
    v9 = v206;
    v8 = v205;
  }
  v235 = v82;
  v236 = v82;
  v237 = v82 + v81;
  if ( v9 == (_BYTE *)v8 )
  {
    v83 = (_BYTE *)v82;
  }
  else
  {
    v83 = &v9[v82 - v8];
    do
    {
      if ( v82 )
      {
        *(_QWORD *)v82 = *(_QWORD *)v8;
        v84 = *(_BYTE *)(v8 + 24);
        *(_BYTE *)(v82 + 24) = v84;
        if ( v84 )
          *(__m128i *)(v82 + 8) = _mm_loadu_si128((const __m128i *)(v8 + 8));
      }
      v82 += 32;
      v8 += 32;
    }
    while ( v83 != (_BYTE *)v82 );
  }
  v236 = (__int64)v83;
  sub_16CCEE0(v226, (__int64)v228, 8, (__int64)v232);
  v85 = v235;
  v6 = v238;
  v9 = v240;
  v235 = 0;
  v229 = v85;
  v86 = v236;
  v236 = 0;
  v230 = v86;
  v87 = v237;
  v237 = 0;
  v231 = v87;
  sub_16CCCB0(v238, (__int64)v240, (__int64)v214);
  v88 = v218;
  v8 = v217;
  v241 = 0;
  v242 = 0;
  v243 = 0;
  v89 = v218 - v217;
  if ( v218 == v217 )
  {
    v90 = 0;
  }
  else
  {
    if ( v89 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_322;
    v90 = sub_22077B0(v218 - v217);
    v88 = v218;
    v8 = v217;
  }
  v241 = v90;
  v242 = v90;
  v243 = v90 + v89;
  if ( v88 == v8 )
  {
    v91 = v90;
  }
  else
  {
    v91 = v90 + v88 - v8;
    do
    {
      if ( v90 )
      {
        *(_QWORD *)v90 = *(_QWORD *)v8;
        v92 = *(_BYTE *)(v8 + 24);
        *(_BYTE *)(v90 + 24) = v92;
        if ( v92 )
          *(__m128i *)(v90 + 8) = _mm_loadu_si128((const __m128i *)(v8 + 8));
      }
      v90 += 32;
      v8 += 32;
    }
    while ( v91 != v90 );
  }
  v242 = v91;
  v6 = v244;
  sub_16CCCB0(v244, (__int64)v246, (__int64)v226);
  v93 = v230;
  v8 = v229;
  v247 = 0;
  v248 = 0;
  v249 = 0;
  v9 = (_BYTE *)(v230 - v229);
  if ( v230 == v229 )
  {
    v94 = 0;
  }
  else
  {
    if ( (unsigned __int64)v9 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_322;
    v119 = (_BYTE *)(v230 - v229);
    v94 = sub_22077B0(v230 - v229);
    v93 = v230;
    v8 = v229;
    v9 = v119;
  }
  v247 = v94;
  v248 = v94;
  v249 = &v9[v94];
  if ( v8 == v93 )
  {
    v95 = v94;
  }
  else
  {
    v95 = v94 + v93 - v8;
    do
    {
      if ( v94 )
      {
        *(_QWORD *)v94 = *(_QWORD *)v8;
        v96 = *(_BYTE *)(v8 + 24);
        *(_BYTE *)(v94 + 24) = v96;
        if ( v96 )
          *(__m128i *)(v94 + 8) = _mm_loadu_si128((const __m128i *)(v8 + 8));
      }
      v94 += 32;
      v8 += 32;
    }
    while ( v95 != v94 );
  }
  v6 = v250;
  v9 = v252;
  v248 = v95;
  sub_16CCCB0(v250, (__int64)v252, (__int64)v238);
  v97 = v242;
  v8 = v241;
  v253 = 0;
  v254 = 0;
  v255 = 0;
  v98 = v242 - v241;
  if ( v242 == v241 )
  {
    v99 = 0;
  }
  else
  {
    if ( v98 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_322;
    v99 = sub_22077B0(v242 - v241);
    v97 = v242;
    v8 = v241;
  }
  v253 = v99;
  v254 = v99;
  v255 = v99 + v98;
  if ( v8 == v97 )
  {
    v100 = v99;
  }
  else
  {
    v100 = v99 + v97 - v8;
    do
    {
      if ( v99 )
      {
        *(_QWORD *)v99 = *(_QWORD *)v8;
        v101 = *(_BYTE *)(v8 + 24);
        *(_BYTE *)(v99 + 24) = v101;
        if ( v101 )
          *(__m128i *)(v99 + 8) = _mm_loadu_si128((const __m128i *)(v8 + 8));
      }
      v99 += 32;
      v8 += 32;
    }
    while ( v100 != v99 );
  }
  v9 = v258;
  v254 = v100;
  v6 = v256;
  sub_16CCCB0(v256, (__int64)v258, (__int64)v244);
  v8 = v248;
  v102 = v247;
  v259 = 0;
  v260 = 0;
  v261 = 0;
  v103 = v248 - v247;
  if ( v248 == v247 )
  {
    v103 = 0;
    v105 = 0;
  }
  else
  {
    if ( v103 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_322;
    v104 = sub_22077B0(v248 - v247);
    v8 = v248;
    v102 = v247;
    v105 = v104;
  }
  v259 = v105;
  v260 = v105;
  v261 = v105 + v103;
  if ( v102 == v8 )
  {
    v107 = v105;
  }
  else
  {
    v106 = v105;
    v107 = v105 + v8 - v102;
    do
    {
      if ( v106 )
      {
        *(_QWORD *)v106 = *(_QWORD *)v102;
        v108 = *(_BYTE *)(v102 + 24);
        *(_BYTE *)(v106 + 24) = v108;
        if ( v108 )
          *(__m128i *)(v106 + 8) = _mm_loadu_si128((const __m128i *)(v102 + 8));
      }
      v106 += 32;
      v102 += 32;
    }
    while ( v107 != v106 );
  }
  v260 = v107;
  while ( 2 )
  {
    v109 = v253;
    if ( v107 - v105 != v254 - v253 )
    {
LABEL_193:
      v33 += 8;
      *(_QWORD *)(v33 - 8) = *(_QWORD *)(v107 - 32);
      sub_17D3A30((__int64)v256);
      v105 = v259;
      v107 = v260;
      continue;
    }
    break;
  }
  if ( v105 != v107 )
  {
    v110 = v105;
    while ( *(_QWORD *)v110 == *(_QWORD *)v109 )
    {
      v111 = *(_BYTE *)(v110 + 24);
      v112 = *(_BYTE *)(v109 + 24);
      if ( v111 && v112 )
        v113 = *(_DWORD *)(v110 + 16) == *(_DWORD *)(v109 + 16);
      else
        v113 = v112 == v111;
      if ( !v113 )
        break;
      v110 += 32;
      v109 += 32;
      if ( v107 == v110 )
        goto LABEL_203;
    }
    goto LABEL_193;
  }
LABEL_203:
  if ( v105 )
    j_j___libc_free_0(v105, v261 - v105);
  if ( v257 != v256[1] )
    _libc_free(v257);
  if ( v253 )
    j_j___libc_free_0(v253, v255 - v253);
  if ( v251 != v250[1] )
    _libc_free(v251);
  if ( v247 )
    j_j___libc_free_0(v247, &v249[-v247]);
  if ( v245 != v244[1] )
    _libc_free(v245);
  if ( v241 )
    j_j___libc_free_0(v241, v243 - v241);
  if ( v239 != v238[1] )
    _libc_free(v239);
  if ( v229 )
    j_j___libc_free_0(v229, v231 - v229);
  if ( v227 != v226[1] )
    _libc_free(v227);
  if ( v235 )
    j_j___libc_free_0(v235, v237 - v235);
  if ( v233 != v232[1] )
    _libc_free(v233);
  if ( v217 )
    j_j___libc_free_0(v217, v219 - v217);
  if ( v215 != v214[1] )
    _libc_free(v215);
  if ( v223 )
    j_j___libc_free_0(v223, v225 - v223);
  if ( v221 != v220[1] )
    _libc_free(v221);
  if ( v205 )
    j_j___libc_free_0(v205, v207 - v205);
  if ( v203 != v202[1] )
    _libc_free(v203);
  if ( v211 )
    j_j___libc_free_0(v211, v213 - v211);
  if ( v209 != v208[1] )
    _libc_free(v209);
  if ( v193 )
    j_j___libc_free_0(v193, v195 - v193);
  if ( v191 != v190[1] )
    _libc_free(v191);
  if ( v199 )
    j_j___libc_free_0(v199, v201 - v199);
  if ( v197 != v196[1] )
    _libc_free(v197);
  if ( v187 )
    j_j___libc_free_0(v187, &v189[-v187]);
  if ( v185 != v184[1] )
    _libc_free(v185);
  if ( v181 )
    j_j___libc_free_0(v181, v183 - v181);
  if ( v179 != v178[1] )
    _libc_free(v179);
  if ( v175 )
    j_j___libc_free_0(v175, v177 - v175);
  if ( v174 != v173[1] )
    _libc_free(v174);
  if ( v170 )
    j_j___libc_free_0(v170, v172 - v170);
  if ( v168 != v167[1] )
    _libc_free(v168);
  if ( v158 )
    j_j___libc_free_0(v158, v160 - v158);
  if ( v156 != v155[1] )
    _libc_free(v156);
  if ( v164 )
    j_j___libc_free_0(v164, &v166[-v164]);
  if ( v162 != v161[1] )
    _libc_free(v162);
  if ( v134 )
    j_j___libc_free_0(v134, &v136[-v134]);
  if ( v132 != v131[1] )
    _libc_free(v132);
  if ( v140 )
    j_j___libc_free_0(v140, v142 - v140);
  if ( v138 != v137[1] )
    _libc_free(v138);
  *(_DWORD *)(a1 + 8) += v27;
}
