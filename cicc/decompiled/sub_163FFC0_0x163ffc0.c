// Function: sub_163FFC0
// Address: 0x163ffc0
//
void __fastcall sub_163FFC0(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rax
  _BYTE *v4; // rsi
  __m128i *v5; // rdi
  __int64 v6; // rdx
  const __m128i *v7; // rcx
  const __m128i *v8; // r8
  unsigned __int64 v9; // r14
  __int64 v10; // rax
  __m128i *v11; // rdi
  __m128i *v12; // rdx
  const __m128i *v13; // rax
  __m128i *v14; // rax
  __m128i *v15; // rax
  __int8 *v16; // rax
  const __m128i *v17; // rcx
  const __m128i *v18; // r8
  unsigned __int64 v19; // r13
  __int64 v20; // rax
  __m128i *v21; // rdi
  __m128i *v22; // rdx
  const __m128i *v23; // rax
  __m128i *v24; // rax
  __m128i *v25; // rax
  __int8 *v26; // rax
  const __m128i *v27; // rcx
  const __m128i *v28; // r8
  unsigned __int64 v29; // r14
  __int64 v30; // rax
  __m128i *v31; // rdi
  __m128i *v32; // rdx
  const __m128i *v33; // rax
  const __m128i *v34; // rcx
  const __m128i *v35; // r9
  unsigned __int64 v36; // r13
  __int64 v37; // rax
  __m128i *v38; // rdi
  __m128i *v39; // rdx
  const __m128i *v40; // rax
  __m128i *v41; // rax
  __m128i *v42; // rax
  __int8 *v43; // rax
  const __m128i *v44; // rcx
  const __m128i *v45; // r8
  unsigned __int64 v46; // r14
  __int64 v47; // rax
  __m128i *v48; // rdi
  __m128i *v49; // rdx
  const __m128i *v50; // rax
  const __m128i *v51; // rcx
  const __m128i *v52; // r8
  unsigned __int64 v53; // r15
  __int64 v54; // rax
  __m128i *v55; // rdi
  __m128i *v56; // rdx
  const __m128i *v57; // rax
  const __m128i *v58; // rcx
  const __m128i *v59; // r8
  unsigned __int64 v60; // r12
  __int64 v61; // rax
  __m128i *v62; // rdi
  __m128i *v63; // rdx
  const __m128i *v64; // rax
  const __m128i *v65; // rcx
  const __m128i *v66; // r8
  unsigned __int64 v67; // r13
  __int64 v68; // rax
  __m128i *v69; // rdi
  __m128i *v70; // rdx
  const __m128i *v71; // rax
  __m128i *v72; // rcx
  _BYTE *v73; // rsi
  __m128i *v74; // rax
  __int64 v75; // [rsp+10h] [rbp-730h] BYREF
  _QWORD *v76; // [rsp+18h] [rbp-728h]
  _QWORD *v77; // [rsp+20h] [rbp-720h]
  __int64 v78; // [rsp+28h] [rbp-718h]
  int v79; // [rsp+30h] [rbp-710h]
  _QWORD v80[8]; // [rsp+38h] [rbp-708h] BYREF
  const __m128i *v81; // [rsp+78h] [rbp-6C8h] BYREF
  const __m128i *v82; // [rsp+80h] [rbp-6C0h]
  __int64 v83; // [rsp+88h] [rbp-6B8h]
  _QWORD v84[16]; // [rsp+90h] [rbp-6B0h] BYREF
  char v85[8]; // [rsp+110h] [rbp-630h] BYREF
  __int64 v86; // [rsp+118h] [rbp-628h]
  unsigned __int64 v87; // [rsp+120h] [rbp-620h]
  _BYTE v88[64]; // [rsp+138h] [rbp-608h] BYREF
  __m128i *v89; // [rsp+178h] [rbp-5C8h]
  __m128i *v90; // [rsp+180h] [rbp-5C0h]
  __int8 *v91; // [rsp+188h] [rbp-5B8h]
  char v92[8]; // [rsp+190h] [rbp-5B0h] BYREF
  __int64 v93; // [rsp+198h] [rbp-5A8h]
  unsigned __int64 v94; // [rsp+1A0h] [rbp-5A0h]
  char v95[64]; // [rsp+1B8h] [rbp-588h] BYREF
  const __m128i *v96; // [rsp+1F8h] [rbp-548h]
  const __m128i *v97; // [rsp+200h] [rbp-540h]
  __int8 *v98; // [rsp+208h] [rbp-538h]
  char v99[8]; // [rsp+210h] [rbp-530h] BYREF
  __int64 v100; // [rsp+218h] [rbp-528h]
  unsigned __int64 v101; // [rsp+220h] [rbp-520h]
  _BYTE v102[64]; // [rsp+238h] [rbp-508h] BYREF
  __m128i *v103; // [rsp+278h] [rbp-4C8h]
  __m128i *v104; // [rsp+280h] [rbp-4C0h]
  __int8 *v105; // [rsp+288h] [rbp-4B8h]
  char v106[8]; // [rsp+290h] [rbp-4B0h] BYREF
  __int64 v107; // [rsp+298h] [rbp-4A8h]
  unsigned __int64 v108; // [rsp+2A0h] [rbp-4A0h]
  char v109[64]; // [rsp+2B8h] [rbp-488h] BYREF
  const __m128i *v110; // [rsp+2F8h] [rbp-448h]
  const __m128i *v111; // [rsp+300h] [rbp-440h]
  __int8 *v112; // [rsp+308h] [rbp-438h]
  char v113[8]; // [rsp+310h] [rbp-430h] BYREF
  __int64 v114; // [rsp+318h] [rbp-428h]
  unsigned __int64 v115; // [rsp+320h] [rbp-420h]
  _BYTE v116[64]; // [rsp+338h] [rbp-408h] BYREF
  __m128i *v117; // [rsp+378h] [rbp-3C8h]
  __m128i *v118; // [rsp+380h] [rbp-3C0h]
  __int8 *v119; // [rsp+388h] [rbp-3B8h]
  char v120[8]; // [rsp+390h] [rbp-3B0h] BYREF
  __int64 v121; // [rsp+398h] [rbp-3A8h]
  unsigned __int64 v122; // [rsp+3A0h] [rbp-3A0h]
  char v123[64]; // [rsp+3B8h] [rbp-388h] BYREF
  const __m128i *v124; // [rsp+3F8h] [rbp-348h]
  const __m128i *v125; // [rsp+400h] [rbp-340h]
  __int8 *v126; // [rsp+408h] [rbp-338h]
  char v127[8]; // [rsp+410h] [rbp-330h] BYREF
  __int64 v128; // [rsp+418h] [rbp-328h]
  unsigned __int64 v129; // [rsp+420h] [rbp-320h]
  _BYTE v130[64]; // [rsp+438h] [rbp-308h] BYREF
  __m128i *v131; // [rsp+478h] [rbp-2C8h]
  __m128i *v132; // [rsp+480h] [rbp-2C0h]
  __int8 *v133; // [rsp+488h] [rbp-2B8h]
  char v134[8]; // [rsp+490h] [rbp-2B0h] BYREF
  __int64 v135; // [rsp+498h] [rbp-2A8h]
  unsigned __int64 v136; // [rsp+4A0h] [rbp-2A0h]
  char v137[64]; // [rsp+4B8h] [rbp-288h] BYREF
  const __m128i *v138; // [rsp+4F8h] [rbp-248h]
  const __m128i *v139; // [rsp+500h] [rbp-240h]
  __int8 *v140; // [rsp+508h] [rbp-238h]
  char v141[8]; // [rsp+510h] [rbp-230h] BYREF
  __int64 v142; // [rsp+518h] [rbp-228h]
  unsigned __int64 v143; // [rsp+520h] [rbp-220h]
  _BYTE v144[64]; // [rsp+538h] [rbp-208h] BYREF
  __m128i *v145; // [rsp+578h] [rbp-1C8h]
  __m128i *v146; // [rsp+580h] [rbp-1C0h]
  __int8 *v147; // [rsp+588h] [rbp-1B8h]
  char v148[8]; // [rsp+590h] [rbp-1B0h] BYREF
  __int64 v149; // [rsp+598h] [rbp-1A8h]
  unsigned __int64 v150; // [rsp+5A0h] [rbp-1A0h]
  _BYTE v151[64]; // [rsp+5B8h] [rbp-188h] BYREF
  __m128i *v152; // [rsp+5F8h] [rbp-148h]
  __m128i *v153; // [rsp+600h] [rbp-140h]
  __int8 *v154; // [rsp+608h] [rbp-138h]
  char v155[8]; // [rsp+610h] [rbp-130h] BYREF
  __int64 v156; // [rsp+618h] [rbp-128h]
  unsigned __int64 v157; // [rsp+620h] [rbp-120h]
  _BYTE v158[64]; // [rsp+638h] [rbp-108h] BYREF
  __m128i *v159; // [rsp+678h] [rbp-C8h]
  __m128i *v160; // [rsp+680h] [rbp-C0h]
  __int8 *v161; // [rsp+688h] [rbp-B8h]
  __m128i v162; // [rsp+690h] [rbp-B0h] BYREF
  unsigned __int64 v163; // [rsp+6A0h] [rbp-A0h]
  _BYTE v164[64]; // [rsp+6B8h] [rbp-88h] BYREF
  __m128i *v165; // [rsp+6F8h] [rbp-48h]
  __m128i *v166; // [rsp+700h] [rbp-40h]
  __int8 *v167; // [rsp+708h] [rbp-38h]

  v80[0] = a2;
  memset(v84, 0, sizeof(v84));
  LODWORD(v84[3]) = 8;
  v84[1] = &v84[5];
  v84[2] = &v84[5];
  v76 = v80;
  v77 = v80;
  v81 = 0;
  v82 = 0;
  v83 = 0;
  v78 = 0x100000008LL;
  v79 = 0;
  v75 = 1;
  v3 = sub_157EBA0(a2);
  v162.m128i_i64[0] = a2;
  v162.m128i_i64[1] = v3;
  LODWORD(v163) = 0;
  sub_136D560(&v81, 0, &v162);
  sub_136D710((__int64)&v75);
  v4 = v102;
  v5 = (__m128i *)v99;
  sub_16CCCB0(v99, v102, v84);
  v7 = (const __m128i *)v84[14];
  v8 = (const __m128i *)v84[13];
  v103 = 0;
  v104 = 0;
  v105 = 0;
  v9 = v84[14] - v84[13];
  if ( v84[14] == v84[13] )
  {
    v11 = 0;
  }
  else
  {
    if ( v9 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_152;
    v10 = sub_22077B0(v84[14] - v84[13]);
    v7 = (const __m128i *)v84[14];
    v8 = (const __m128i *)v84[13];
    v11 = (__m128i *)v10;
  }
  v103 = v11;
  v104 = v11;
  v105 = &v11->m128i_i8[v9];
  if ( v7 != v8 )
  {
    v12 = v11;
    v13 = v8;
    do
    {
      if ( v12 )
      {
        *v12 = _mm_loadu_si128(v13);
        v12[1].m128i_i64[0] = v13[1].m128i_i64[0];
      }
      v13 = (const __m128i *)((char *)v13 + 24);
      v12 = (__m128i *)((char *)v12 + 24);
    }
    while ( v13 != v7 );
    v11 = (__m128i *)((char *)v11 + 8 * ((unsigned __int64)((char *)&v13[-2].m128i_u64[1] - (char *)v8) >> 3) + 24);
  }
  v104 = v11;
  sub_16CCEE0(v106, v109, 8, v99);
  v14 = v103;
  v5 = (__m128i *)v85;
  v4 = v88;
  v103 = 0;
  v110 = v14;
  v15 = v104;
  v104 = 0;
  v111 = v15;
  v16 = v105;
  v105 = 0;
  v112 = v16;
  sub_16CCCB0(v85, v88, &v75);
  v17 = v82;
  v18 = v81;
  v89 = 0;
  v90 = 0;
  v91 = 0;
  v19 = (char *)v82 - (char *)v81;
  if ( v82 == v81 )
  {
    v21 = 0;
  }
  else
  {
    if ( v19 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_152;
    v20 = sub_22077B0((char *)v82 - (char *)v81);
    v17 = v82;
    v18 = v81;
    v21 = (__m128i *)v20;
  }
  v89 = v21;
  v90 = v21;
  v91 = &v21->m128i_i8[v19];
  if ( v17 != v18 )
  {
    v22 = v21;
    v23 = v18;
    do
    {
      if ( v22 )
      {
        *v22 = _mm_loadu_si128(v23);
        v22[1].m128i_i64[0] = v23[1].m128i_i64[0];
      }
      v23 = (const __m128i *)((char *)v23 + 24);
      v22 = (__m128i *)((char *)v22 + 24);
    }
    while ( v17 != v23 );
    v21 = (__m128i *)((char *)v21 + 8 * ((unsigned __int64)((char *)&v17[-2].m128i_u64[1] - (char *)v18) >> 3) + 24);
  }
  v90 = v21;
  sub_16CCEE0(v92, v95, 8, v85);
  v24 = v89;
  v5 = (__m128i *)v127;
  v4 = v130;
  v89 = 0;
  v96 = v24;
  v25 = v90;
  v90 = 0;
  v97 = v25;
  v26 = v91;
  v91 = 0;
  v98 = v26;
  sub_16CCCB0(v127, v130, v106);
  v27 = v111;
  v28 = v110;
  v131 = 0;
  v132 = 0;
  v133 = 0;
  v29 = (char *)v111 - (char *)v110;
  if ( v111 == v110 )
  {
    v31 = 0;
  }
  else
  {
    if ( v29 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_152;
    v30 = sub_22077B0((char *)v111 - (char *)v110);
    v27 = v111;
    v28 = v110;
    v31 = (__m128i *)v30;
  }
  v131 = v31;
  v132 = v31;
  v133 = &v31->m128i_i8[v29];
  if ( v28 != v27 )
  {
    v32 = v31;
    v33 = v28;
    do
    {
      if ( v32 )
      {
        *v32 = _mm_loadu_si128(v33);
        v32[1].m128i_i64[0] = v33[1].m128i_i64[0];
      }
      v33 = (const __m128i *)((char *)v33 + 24);
      v32 = (__m128i *)((char *)v32 + 24);
    }
    while ( v33 != v27 );
    v31 = (__m128i *)((char *)v31 + 8 * ((unsigned __int64)((char *)&v33[-2].m128i_u64[1] - (char *)v28) >> 3) + 24);
  }
  v132 = v31;
  sub_16CCEE0(v134, v137, 8, v127);
  v5 = (__m128i *)v113;
  v4 = v116;
  v138 = v131;
  v131 = 0;
  v139 = v132;
  v132 = 0;
  v140 = v133;
  v133 = 0;
  sub_16CCCB0(v113, v116, v92);
  v34 = v97;
  v35 = v96;
  v117 = 0;
  v118 = 0;
  v119 = 0;
  v36 = (char *)v97 - (char *)v96;
  if ( v97 == v96 )
  {
    v38 = 0;
  }
  else
  {
    if ( v36 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_152;
    v37 = sub_22077B0((char *)v97 - (char *)v96);
    v34 = v97;
    v35 = v96;
    v38 = (__m128i *)v37;
  }
  v117 = v38;
  v118 = v38;
  v119 = &v38->m128i_i8[v36];
  if ( v34 != v35 )
  {
    v39 = v38;
    v40 = v35;
    do
    {
      if ( v39 )
      {
        *v39 = _mm_loadu_si128(v40);
        v39[1].m128i_i64[0] = v40[1].m128i_i64[0];
      }
      v40 = (const __m128i *)((char *)v40 + 24);
      v39 = (__m128i *)((char *)v39 + 24);
    }
    while ( v34 != v40 );
    v38 = (__m128i *)((char *)v38 + 8 * ((unsigned __int64)((char *)&v34[-2].m128i_u64[1] - (char *)v35) >> 3) + 24);
  }
  v118 = v38;
  sub_16CCEE0(v120, v123, 8, v113);
  v41 = v117;
  v5 = (__m128i *)v148;
  v4 = v151;
  v117 = 0;
  v124 = v41;
  v42 = v118;
  v118 = 0;
  v125 = v42;
  v43 = v119;
  v119 = 0;
  v126 = v43;
  sub_16CCCB0(v148, v151, v134);
  v44 = v139;
  v45 = v138;
  v152 = 0;
  v153 = 0;
  v154 = 0;
  v46 = (char *)v139 - (char *)v138;
  if ( v139 == v138 )
  {
    v48 = 0;
  }
  else
  {
    if ( v46 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_152;
    v47 = sub_22077B0((char *)v139 - (char *)v138);
    v44 = v139;
    v45 = v138;
    v48 = (__m128i *)v47;
  }
  v152 = v48;
  v153 = v48;
  v154 = &v48->m128i_i8[v46];
  if ( v45 != v44 )
  {
    v49 = v48;
    v50 = v45;
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
    while ( v50 != v44 );
    v48 = (__m128i *)((char *)v48 + 8 * ((unsigned __int64)((char *)&v50[-2].m128i_u64[1] - (char *)v45) >> 3) + 24);
  }
  v153 = v48;
  v4 = v144;
  v5 = (__m128i *)v141;
  sub_16CCCB0(v141, v144, v120);
  v51 = v125;
  v52 = v124;
  v145 = 0;
  v146 = 0;
  v147 = 0;
  v53 = (char *)v125 - (char *)v124;
  if ( v125 == v124 )
  {
    v55 = 0;
  }
  else
  {
    if ( v53 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_152;
    v54 = sub_22077B0((char *)v125 - (char *)v124);
    v51 = v125;
    v52 = v124;
    v55 = (__m128i *)v54;
  }
  v145 = v55;
  v146 = v55;
  v147 = &v55->m128i_i8[v53];
  if ( v51 != v52 )
  {
    v56 = v55;
    v57 = v52;
    do
    {
      if ( v56 )
      {
        *v56 = _mm_loadu_si128(v57);
        v56[1].m128i_i64[0] = v57[1].m128i_i64[0];
      }
      v57 = (const __m128i *)((char *)v57 + 24);
      v56 = (__m128i *)((char *)v56 + 24);
    }
    while ( v51 != v57 );
    v55 = (__m128i *)((char *)v55 + 8 * ((unsigned __int64)((char *)&v51[-2].m128i_u64[1] - (char *)v52) >> 3) + 24);
  }
  v146 = v55;
  v4 = v164;
  v5 = &v162;
  sub_16CCCB0(&v162, v164, v148);
  v58 = v153;
  v59 = v152;
  v165 = 0;
  v166 = 0;
  v167 = 0;
  v60 = (char *)v153 - (char *)v152;
  if ( v153 == v152 )
  {
    v62 = 0;
  }
  else
  {
    if ( v60 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_152;
    v61 = sub_22077B0((char *)v153 - (char *)v152);
    v58 = v153;
    v59 = v152;
    v62 = (__m128i *)v61;
  }
  v165 = v62;
  v166 = v62;
  v167 = &v62->m128i_i8[v60];
  if ( v58 != v59 )
  {
    v63 = v62;
    v64 = v59;
    do
    {
      if ( v63 )
      {
        *v63 = _mm_loadu_si128(v64);
        v63[1].m128i_i64[0] = v64[1].m128i_i64[0];
      }
      v64 = (const __m128i *)((char *)v64 + 24);
      v63 = (__m128i *)((char *)v63 + 24);
    }
    while ( v58 != v64 );
    v62 = (__m128i *)((char *)v62 + 8 * ((unsigned __int64)((char *)&v58[-2].m128i_u64[1] - (char *)v59) >> 3) + 24);
  }
  v166 = v62;
  v4 = v158;
  v5 = (__m128i *)v155;
  sub_16CCCB0(v155, v158, v141);
  v65 = v146;
  v66 = v145;
  v159 = 0;
  v160 = 0;
  v161 = 0;
  v67 = (char *)v146 - (char *)v145;
  if ( v146 == v145 )
  {
    v69 = 0;
    goto LABEL_67;
  }
  if ( v67 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_152:
    sub_4261EA(v5, v4, v6);
  v68 = sub_22077B0((char *)v146 - (char *)v145);
  v65 = v146;
  v66 = v145;
  v69 = (__m128i *)v68;
LABEL_67:
  v159 = v69;
  v70 = v69;
  v160 = v69;
  v161 = &v69->m128i_i8[v67];
  if ( v65 != v66 )
  {
    v71 = v66;
    do
    {
      if ( v70 )
      {
        *v70 = _mm_loadu_si128(v71);
        v70[1].m128i_i64[0] = v71[1].m128i_i64[0];
      }
      v71 = (const __m128i *)((char *)v71 + 24);
      v70 = (__m128i *)((char *)v70 + 24);
    }
    while ( v65 != v71 );
    v70 = (__m128i *)((char *)v69 + 8 * ((unsigned __int64)((char *)&v65[-2].m128i_u64[1] - (char *)v66) >> 3) + 24);
  }
  v160 = v70;
  while ( 1 )
  {
    v72 = v165;
    if ( (char *)v70 - (char *)v69 != (char *)v166 - (char *)v165 )
      goto LABEL_75;
    if ( v69 == v70 )
      break;
    v74 = v69;
    while ( v74->m128i_i64[0] == v72->m128i_i64[0] && v74[1].m128i_i32[0] == v72[1].m128i_i32[0] )
    {
      v74 = (__m128i *)((char *)v74 + 24);
      v72 = (__m128i *)((char *)v72 + 24);
      if ( v74 == v70 )
        goto LABEL_86;
    }
LABEL_75:
    v73 = *(_BYTE **)(a1 + 8);
    if ( v73 == *(_BYTE **)(a1 + 16) )
    {
      sub_136D8A0(a1, v73, &v70[-2].m128i_i64[1]);
      v70 = v160;
    }
    else
    {
      if ( v73 )
      {
        *(_QWORD *)v73 = v70[-2].m128i_i64[1];
        v73 = *(_BYTE **)(a1 + 8);
        v70 = v160;
      }
      *(_QWORD *)(a1 + 8) = v73 + 8;
    }
    v69 = v159;
    v70 = (__m128i *)((char *)v70 - 24);
    v160 = v70;
    if ( v70 != v159 )
    {
      sub_136D710((__int64)v155);
      v69 = v159;
      v70 = v160;
    }
  }
LABEL_86:
  if ( v69 )
    j_j___libc_free_0(v69, v161 - (__int8 *)v69);
  if ( v157 != v156 )
    _libc_free(v157);
  if ( v165 )
    j_j___libc_free_0(v165, v167 - (__int8 *)v165);
  if ( v163 != v162.m128i_i64[1] )
    _libc_free(v163);
  if ( v145 )
    j_j___libc_free_0(v145, v147 - (__int8 *)v145);
  if ( v143 != v142 )
    _libc_free(v143);
  if ( v152 )
    j_j___libc_free_0(v152, v154 - (__int8 *)v152);
  if ( v150 != v149 )
    _libc_free(v150);
  if ( v124 )
    j_j___libc_free_0(v124, v126 - (__int8 *)v124);
  if ( v122 != v121 )
    _libc_free(v122);
  if ( v117 )
    j_j___libc_free_0(v117, v119 - (__int8 *)v117);
  if ( v115 != v114 )
    _libc_free(v115);
  if ( v138 )
    j_j___libc_free_0(v138, v140 - (__int8 *)v138);
  if ( v136 != v135 )
    _libc_free(v136);
  if ( v131 )
    j_j___libc_free_0(v131, v133 - (__int8 *)v131);
  if ( v129 != v128 )
    _libc_free(v129);
  if ( v96 )
    j_j___libc_free_0(v96, v98 - (__int8 *)v96);
  if ( v94 != v93 )
    _libc_free(v94);
  if ( v89 )
    j_j___libc_free_0(v89, v91 - (__int8 *)v89);
  if ( v87 != v86 )
    _libc_free(v87);
  if ( v110 )
    j_j___libc_free_0(v110, v112 - (__int8 *)v110);
  if ( v108 != v107 )
    _libc_free(v108);
  if ( v103 )
    j_j___libc_free_0(v103, v105 - (__int8 *)v103);
  if ( v101 != v100 )
    _libc_free(v101);
  if ( v81 )
    j_j___libc_free_0(v81, v83 - (_QWORD)v81);
  if ( v77 != v76 )
    _libc_free((unsigned __int64)v77);
  if ( v84[13] )
    j_j___libc_free_0(v84[13], v84[15] - v84[13]);
  if ( v84[2] != v84[1] )
    _libc_free(v84[2]);
}
