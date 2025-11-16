// Function: sub_191E690
// Address: 0x191e690
//
void __fastcall sub_191E690(__int64 a1, __int64 a2)
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
  _QWORD v85[2]; // [rsp+110h] [rbp-630h] BYREF
  unsigned __int64 v86; // [rsp+120h] [rbp-620h]
  _BYTE v87[64]; // [rsp+138h] [rbp-608h] BYREF
  __m128i *v88; // [rsp+178h] [rbp-5C8h]
  __m128i *v89; // [rsp+180h] [rbp-5C0h]
  __int8 *v90; // [rsp+188h] [rbp-5B8h]
  _QWORD v91[2]; // [rsp+190h] [rbp-5B0h] BYREF
  unsigned __int64 v92; // [rsp+1A0h] [rbp-5A0h]
  char v93[64]; // [rsp+1B8h] [rbp-588h] BYREF
  const __m128i *v94; // [rsp+1F8h] [rbp-548h]
  const __m128i *v95; // [rsp+200h] [rbp-540h]
  __int8 *v96; // [rsp+208h] [rbp-538h]
  _QWORD v97[2]; // [rsp+210h] [rbp-530h] BYREF
  unsigned __int64 v98; // [rsp+220h] [rbp-520h]
  _BYTE v99[64]; // [rsp+238h] [rbp-508h] BYREF
  __m128i *v100; // [rsp+278h] [rbp-4C8h]
  __m128i *v101; // [rsp+280h] [rbp-4C0h]
  __int8 *v102; // [rsp+288h] [rbp-4B8h]
  _QWORD v103[2]; // [rsp+290h] [rbp-4B0h] BYREF
  unsigned __int64 v104; // [rsp+2A0h] [rbp-4A0h]
  char v105[64]; // [rsp+2B8h] [rbp-488h] BYREF
  const __m128i *v106; // [rsp+2F8h] [rbp-448h]
  const __m128i *v107; // [rsp+300h] [rbp-440h]
  __int8 *v108; // [rsp+308h] [rbp-438h]
  _QWORD v109[2]; // [rsp+310h] [rbp-430h] BYREF
  unsigned __int64 v110; // [rsp+320h] [rbp-420h]
  _BYTE v111[64]; // [rsp+338h] [rbp-408h] BYREF
  __m128i *v112; // [rsp+378h] [rbp-3C8h]
  __m128i *v113; // [rsp+380h] [rbp-3C0h]
  __int8 *v114; // [rsp+388h] [rbp-3B8h]
  _QWORD v115[2]; // [rsp+390h] [rbp-3B0h] BYREF
  unsigned __int64 v116; // [rsp+3A0h] [rbp-3A0h]
  char v117[64]; // [rsp+3B8h] [rbp-388h] BYREF
  const __m128i *v118; // [rsp+3F8h] [rbp-348h]
  const __m128i *v119; // [rsp+400h] [rbp-340h]
  __int8 *v120; // [rsp+408h] [rbp-338h]
  _QWORD v121[2]; // [rsp+410h] [rbp-330h] BYREF
  unsigned __int64 v122; // [rsp+420h] [rbp-320h]
  _BYTE v123[64]; // [rsp+438h] [rbp-308h] BYREF
  __m128i *v124; // [rsp+478h] [rbp-2C8h]
  __m128i *v125; // [rsp+480h] [rbp-2C0h]
  __int8 *v126; // [rsp+488h] [rbp-2B8h]
  _QWORD v127[2]; // [rsp+490h] [rbp-2B0h] BYREF
  unsigned __int64 v128; // [rsp+4A0h] [rbp-2A0h]
  char v129[64]; // [rsp+4B8h] [rbp-288h] BYREF
  const __m128i *v130; // [rsp+4F8h] [rbp-248h]
  const __m128i *v131; // [rsp+500h] [rbp-240h]
  __int8 *v132; // [rsp+508h] [rbp-238h]
  _QWORD v133[2]; // [rsp+510h] [rbp-230h] BYREF
  unsigned __int64 v134; // [rsp+520h] [rbp-220h]
  _BYTE v135[64]; // [rsp+538h] [rbp-208h] BYREF
  __m128i *v136; // [rsp+578h] [rbp-1C8h]
  __m128i *v137; // [rsp+580h] [rbp-1C0h]
  __int8 *v138; // [rsp+588h] [rbp-1B8h]
  _QWORD v139[2]; // [rsp+590h] [rbp-1B0h] BYREF
  unsigned __int64 v140; // [rsp+5A0h] [rbp-1A0h]
  _BYTE v141[64]; // [rsp+5B8h] [rbp-188h] BYREF
  __m128i *v142; // [rsp+5F8h] [rbp-148h]
  __m128i *v143; // [rsp+600h] [rbp-140h]
  __int8 *v144; // [rsp+608h] [rbp-138h]
  _QWORD v145[2]; // [rsp+610h] [rbp-130h] BYREF
  unsigned __int64 v146; // [rsp+620h] [rbp-120h]
  _BYTE v147[64]; // [rsp+638h] [rbp-108h] BYREF
  __m128i *v148; // [rsp+678h] [rbp-C8h]
  __m128i *v149; // [rsp+680h] [rbp-C0h]
  __int8 *v150; // [rsp+688h] [rbp-B8h]
  __m128i v151; // [rsp+690h] [rbp-B0h] BYREF
  unsigned __int64 v152; // [rsp+6A0h] [rbp-A0h]
  _BYTE v153[64]; // [rsp+6B8h] [rbp-88h] BYREF
  __m128i *v154; // [rsp+6F8h] [rbp-48h]
  __m128i *v155; // [rsp+700h] [rbp-40h]
  __int8 *v156; // [rsp+708h] [rbp-38h]

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
  v151.m128i_i64[0] = a2;
  v151.m128i_i64[1] = v3;
  LODWORD(v152) = 0;
  sub_13FDF40(&v81, 0, &v151);
  sub_13FE0F0((__int64)&v75);
  v4 = v99;
  v5 = (__m128i *)v97;
  sub_16CCCB0(v97, (__int64)v99, (__int64)v84);
  v7 = (const __m128i *)v84[14];
  v8 = (const __m128i *)v84[13];
  v100 = 0;
  v101 = 0;
  v102 = 0;
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
  v100 = v11;
  v101 = v11;
  v102 = &v11->m128i_i8[v9];
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
  v101 = v11;
  sub_16CCEE0(v103, (__int64)v105, 8, (__int64)v97);
  v14 = v100;
  v5 = (__m128i *)v85;
  v4 = v87;
  v100 = 0;
  v106 = v14;
  v15 = v101;
  v101 = 0;
  v107 = v15;
  v16 = v102;
  v102 = 0;
  v108 = v16;
  sub_16CCCB0(v85, (__int64)v87, (__int64)&v75);
  v17 = v82;
  v18 = v81;
  v88 = 0;
  v89 = 0;
  v90 = 0;
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
  v88 = v21;
  v89 = v21;
  v90 = &v21->m128i_i8[v19];
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
  v89 = v21;
  sub_16CCEE0(v91, (__int64)v93, 8, (__int64)v85);
  v24 = v88;
  v5 = (__m128i *)v121;
  v4 = v123;
  v88 = 0;
  v94 = v24;
  v25 = v89;
  v89 = 0;
  v95 = v25;
  v26 = v90;
  v90 = 0;
  v96 = v26;
  sub_16CCCB0(v121, (__int64)v123, (__int64)v103);
  v27 = v107;
  v28 = v106;
  v124 = 0;
  v125 = 0;
  v126 = 0;
  v29 = (char *)v107 - (char *)v106;
  if ( v107 == v106 )
  {
    v31 = 0;
  }
  else
  {
    if ( v29 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_152;
    v30 = sub_22077B0((char *)v107 - (char *)v106);
    v27 = v107;
    v28 = v106;
    v31 = (__m128i *)v30;
  }
  v124 = v31;
  v125 = v31;
  v126 = &v31->m128i_i8[v29];
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
  v125 = v31;
  sub_16CCEE0(v127, (__int64)v129, 8, (__int64)v121);
  v5 = (__m128i *)v109;
  v4 = v111;
  v130 = v124;
  v124 = 0;
  v131 = v125;
  v125 = 0;
  v132 = v126;
  v126 = 0;
  sub_16CCCB0(v109, (__int64)v111, (__int64)v91);
  v34 = v95;
  v35 = v94;
  v112 = 0;
  v113 = 0;
  v114 = 0;
  v36 = (char *)v95 - (char *)v94;
  if ( v95 == v94 )
  {
    v38 = 0;
  }
  else
  {
    if ( v36 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_152;
    v37 = sub_22077B0((char *)v95 - (char *)v94);
    v34 = v95;
    v35 = v94;
    v38 = (__m128i *)v37;
  }
  v112 = v38;
  v113 = v38;
  v114 = &v38->m128i_i8[v36];
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
  v113 = v38;
  sub_16CCEE0(v115, (__int64)v117, 8, (__int64)v109);
  v41 = v112;
  v5 = (__m128i *)v139;
  v4 = v141;
  v112 = 0;
  v118 = v41;
  v42 = v113;
  v113 = 0;
  v119 = v42;
  v43 = v114;
  v114 = 0;
  v120 = v43;
  sub_16CCCB0(v139, (__int64)v141, (__int64)v127);
  v44 = v131;
  v45 = v130;
  v142 = 0;
  v143 = 0;
  v144 = 0;
  v46 = (char *)v131 - (char *)v130;
  if ( v131 == v130 )
  {
    v48 = 0;
  }
  else
  {
    if ( v46 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_152;
    v47 = sub_22077B0((char *)v131 - (char *)v130);
    v44 = v131;
    v45 = v130;
    v48 = (__m128i *)v47;
  }
  v142 = v48;
  v143 = v48;
  v144 = &v48->m128i_i8[v46];
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
  v143 = v48;
  v4 = v135;
  v5 = (__m128i *)v133;
  sub_16CCCB0(v133, (__int64)v135, (__int64)v115);
  v51 = v119;
  v52 = v118;
  v136 = 0;
  v137 = 0;
  v138 = 0;
  v53 = (char *)v119 - (char *)v118;
  if ( v119 == v118 )
  {
    v55 = 0;
  }
  else
  {
    if ( v53 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_152;
    v54 = sub_22077B0((char *)v119 - (char *)v118);
    v51 = v119;
    v52 = v118;
    v55 = (__m128i *)v54;
  }
  v136 = v55;
  v137 = v55;
  v138 = &v55->m128i_i8[v53];
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
  v137 = v55;
  v4 = v153;
  v5 = &v151;
  sub_16CCCB0(&v151, (__int64)v153, (__int64)v139);
  v58 = v143;
  v59 = v142;
  v154 = 0;
  v155 = 0;
  v156 = 0;
  v60 = (char *)v143 - (char *)v142;
  if ( v143 == v142 )
  {
    v62 = 0;
  }
  else
  {
    if ( v60 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_152;
    v61 = sub_22077B0((char *)v143 - (char *)v142);
    v58 = v143;
    v59 = v142;
    v62 = (__m128i *)v61;
  }
  v154 = v62;
  v155 = v62;
  v156 = &v62->m128i_i8[v60];
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
  v155 = v62;
  v4 = v147;
  v5 = (__m128i *)v145;
  sub_16CCCB0(v145, (__int64)v147, (__int64)v133);
  v65 = v137;
  v66 = v136;
  v148 = 0;
  v149 = 0;
  v150 = 0;
  v67 = (char *)v137 - (char *)v136;
  if ( v137 == v136 )
  {
    v69 = 0;
    goto LABEL_67;
  }
  if ( v67 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_152:
    sub_4261EA(v5, v4, v6);
  v68 = sub_22077B0((char *)v137 - (char *)v136);
  v65 = v137;
  v66 = v136;
  v69 = (__m128i *)v68;
LABEL_67:
  v148 = v69;
  v70 = v69;
  v149 = v69;
  v150 = &v69->m128i_i8[v67];
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
  v149 = v70;
  while ( 1 )
  {
    v72 = v154;
    if ( (char *)v70 - (char *)v69 != (char *)v155 - (char *)v154 )
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
      sub_1292090(a1, v73, &v70[-2].m128i_i64[1]);
      v70 = v149;
    }
    else
    {
      if ( v73 )
      {
        *(_QWORD *)v73 = v70[-2].m128i_i64[1];
        v73 = *(_BYTE **)(a1 + 8);
        v70 = v149;
      }
      *(_QWORD *)(a1 + 8) = v73 + 8;
    }
    v69 = v148;
    v70 = (__m128i *)((char *)v70 - 24);
    v149 = v70;
    if ( v70 != v148 )
    {
      sub_13FE0F0((__int64)v145);
      v69 = v148;
      v70 = v149;
    }
  }
LABEL_86:
  if ( v69 )
    j_j___libc_free_0(v69, v150 - (__int8 *)v69);
  if ( v146 != v145[1] )
    _libc_free(v146);
  if ( v154 )
    j_j___libc_free_0(v154, v156 - (__int8 *)v154);
  if ( v152 != v151.m128i_i64[1] )
    _libc_free(v152);
  if ( v136 )
    j_j___libc_free_0(v136, v138 - (__int8 *)v136);
  if ( v134 != v133[1] )
    _libc_free(v134);
  if ( v142 )
    j_j___libc_free_0(v142, v144 - (__int8 *)v142);
  if ( v140 != v139[1] )
    _libc_free(v140);
  if ( v118 )
    j_j___libc_free_0(v118, v120 - (__int8 *)v118);
  if ( v116 != v115[1] )
    _libc_free(v116);
  if ( v112 )
    j_j___libc_free_0(v112, v114 - (__int8 *)v112);
  if ( v110 != v109[1] )
    _libc_free(v110);
  if ( v130 )
    j_j___libc_free_0(v130, v132 - (__int8 *)v130);
  if ( v128 != v127[1] )
    _libc_free(v128);
  if ( v124 )
    j_j___libc_free_0(v124, v126 - (__int8 *)v124);
  if ( v122 != v121[1] )
    _libc_free(v122);
  if ( v94 )
    j_j___libc_free_0(v94, v96 - (__int8 *)v94);
  if ( v92 != v91[1] )
    _libc_free(v92);
  if ( v88 )
    j_j___libc_free_0(v88, v90 - (__int8 *)v88);
  if ( v86 != v85[1] )
    _libc_free(v86);
  if ( v106 )
    j_j___libc_free_0(v106, v108 - (__int8 *)v106);
  if ( v104 != v103[1] )
    _libc_free(v104);
  if ( v100 )
    j_j___libc_free_0(v100, v102 - (__int8 *)v100);
  if ( v98 != v97[1] )
    _libc_free(v98);
  if ( v81 )
    j_j___libc_free_0(v81, v83 - (_QWORD)v81);
  if ( v77 != v76 )
    _libc_free((unsigned __int64)v77);
  if ( v84[13] )
    j_j___libc_free_0(v84[13], v84[15] - v84[13]);
  if ( v84[2] != v84[1] )
    _libc_free(v84[2]);
}
