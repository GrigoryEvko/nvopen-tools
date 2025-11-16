// Function: sub_39659E0
// Address: 0x39659e0
//
void __fastcall sub_39659E0(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rbx
  unsigned __int64 v4; // rax
  __m128i *v5; // rsi
  const __m128i *v6; // rsi
  __m128i *v7; // rdi
  __int64 v8; // rdx
  const __m128i *v9; // rcx
  const __m128i *v10; // r8
  unsigned __int64 v11; // r14
  __int64 v12; // rax
  unsigned __int64 v13; // rdi
  __m128i *v14; // rdx
  const __m128i *v15; // rax
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rax
  __m128i *v19; // rcx
  const __m128i *v20; // r8
  unsigned __int64 v21; // rbx
  __int64 v22; // rax
  __m128i *v23; // rdi
  __m128i *v24; // rdx
  const __m128i *v25; // rax
  unsigned __int64 v26; // rax
  __m128i *v27; // rax
  __int8 *v28; // rax
  const __m128i *v29; // rcx
  unsigned __int64 v30; // r8
  unsigned __int64 v31; // r14
  __int64 v32; // rax
  __m128i *v33; // rdi
  __m128i *v34; // rdx
  const __m128i *v35; // rax
  unsigned __int64 v36; // r9
  unsigned __int64 v37; // rbx
  __int64 v38; // rax
  unsigned __int64 v39; // r8
  __m128i *v40; // rdx
  const __m128i *v41; // rax
  unsigned __int64 v42; // rax
  unsigned __int64 v43; // rax
  unsigned __int64 v44; // rax
  const __m128i *v45; // rcx
  unsigned __int64 v46; // r8
  unsigned __int64 v47; // r14
  __int64 v48; // rax
  __m128i *v49; // rdi
  __m128i *v50; // rdx
  const __m128i *v51; // rax
  const __m128i *v52; // rcx
  unsigned __int64 v53; // r8
  unsigned __int64 v54; // rbx
  __int64 v55; // rax
  __m128i *v56; // rdi
  __m128i *v57; // rdx
  const __m128i *v58; // rax
  const __m128i *v59; // rcx
  const __m128i *v60; // r8
  unsigned __int64 v61; // rbx
  __int64 v62; // rax
  unsigned __int64 v63; // rdi
  __m128i *v64; // rdx
  const __m128i *v65; // rax
  const __m128i *v66; // rcx
  const __m128i *v67; // r8
  unsigned __int64 v68; // r13
  __int64 v69; // rax
  const __m128i *v70; // rdi
  __m128i *v71; // rdx
  const __m128i *v72; // rax
  __m128i *v73; // r13
  unsigned __int64 v74; // rdx
  _BYTE *v75; // rsi
  unsigned __int64 v76; // rdi
  int v77; // eax
  unsigned int v78; // esi
  __int64 v79; // rdi
  __int64 v80; // r14
  __int64 *v81; // rax
  char v82; // dl
  unsigned __int64 v83; // rax
  __int64 *v84; // rcx
  __int64 *v85; // rsi
  __int64 *v86; // rdx
  const __m128i *v87; // rax
  __m128i v88; // [rsp+30h] [rbp-750h] BYREF
  __int64 v89; // [rsp+40h] [rbp-740h]
  __int64 v90; // [rsp+50h] [rbp-730h] BYREF
  _BYTE *v91; // [rsp+58h] [rbp-728h]
  _BYTE *v92; // [rsp+60h] [rbp-720h]
  __int64 v93; // [rsp+68h] [rbp-718h]
  int v94; // [rsp+70h] [rbp-710h]
  _BYTE v95[64]; // [rsp+78h] [rbp-708h] BYREF
  const __m128i *v96; // [rsp+B8h] [rbp-6C8h] BYREF
  __m128i *v97; // [rsp+C0h] [rbp-6C0h]
  const __m128i *v98; // [rsp+C8h] [rbp-6B8h]
  _QWORD v99[16]; // [rsp+D0h] [rbp-6B0h] BYREF
  _QWORD v100[5]; // [rsp+150h] [rbp-630h] BYREF
  _BYTE v101[64]; // [rsp+178h] [rbp-608h] BYREF
  unsigned __int64 v102; // [rsp+1B8h] [rbp-5C8h]
  __m128i *v103; // [rsp+1C0h] [rbp-5C0h]
  __int8 *v104; // [rsp+1C8h] [rbp-5B8h]
  _QWORD v105[2]; // [rsp+1D0h] [rbp-5B0h] BYREF
  unsigned __int64 v106; // [rsp+1E0h] [rbp-5A0h]
  char v107[64]; // [rsp+1F8h] [rbp-588h] BYREF
  const __m128i *v108; // [rsp+238h] [rbp-548h]
  const __m128i *v109; // [rsp+240h] [rbp-540h]
  __int8 *v110; // [rsp+248h] [rbp-538h]
  _QWORD v111[2]; // [rsp+250h] [rbp-530h] BYREF
  unsigned __int64 v112; // [rsp+260h] [rbp-520h]
  _BYTE v113[64]; // [rsp+278h] [rbp-508h] BYREF
  unsigned __int64 v114; // [rsp+2B8h] [rbp-4C8h]
  unsigned __int64 v115; // [rsp+2C0h] [rbp-4C0h]
  unsigned __int64 v116; // [rsp+2C8h] [rbp-4B8h]
  _QWORD v117[2]; // [rsp+2D0h] [rbp-4B0h] BYREF
  unsigned __int64 v118; // [rsp+2E0h] [rbp-4A0h]
  char v119[64]; // [rsp+2F8h] [rbp-488h] BYREF
  const __m128i *v120; // [rsp+338h] [rbp-448h]
  const __m128i *v121; // [rsp+340h] [rbp-440h]
  unsigned __int64 v122; // [rsp+348h] [rbp-438h]
  _QWORD v123[2]; // [rsp+350h] [rbp-430h] BYREF
  unsigned __int64 v124; // [rsp+360h] [rbp-420h]
  char v125[64]; // [rsp+378h] [rbp-408h] BYREF
  unsigned __int64 v126; // [rsp+3B8h] [rbp-3C8h]
  unsigned __int64 v127; // [rsp+3C0h] [rbp-3C0h]
  unsigned __int64 v128; // [rsp+3C8h] [rbp-3B8h]
  _QWORD v129[2]; // [rsp+3D0h] [rbp-3B0h] BYREF
  unsigned __int64 v130; // [rsp+3E0h] [rbp-3A0h]
  char v131[64]; // [rsp+3F8h] [rbp-388h] BYREF
  const __m128i *v132; // [rsp+438h] [rbp-348h]
  const __m128i *v133; // [rsp+440h] [rbp-340h]
  unsigned __int64 v134; // [rsp+448h] [rbp-338h]
  _QWORD v135[5]; // [rsp+450h] [rbp-330h] BYREF
  _BYTE v136[64]; // [rsp+478h] [rbp-308h] BYREF
  unsigned __int64 v137; // [rsp+4B8h] [rbp-2C8h]
  __m128i *v138; // [rsp+4C0h] [rbp-2C0h]
  __int8 *v139; // [rsp+4C8h] [rbp-2B8h]
  _QWORD v140[2]; // [rsp+4D0h] [rbp-2B0h] BYREF
  unsigned __int64 v141; // [rsp+4E0h] [rbp-2A0h]
  char v142[64]; // [rsp+4F8h] [rbp-288h] BYREF
  const __m128i *v143; // [rsp+538h] [rbp-248h]
  const __m128i *v144; // [rsp+540h] [rbp-240h]
  __int8 *v145; // [rsp+548h] [rbp-238h]
  _QWORD v146[5]; // [rsp+550h] [rbp-230h] BYREF
  _BYTE v147[64]; // [rsp+578h] [rbp-208h] BYREF
  __m128i *v148; // [rsp+5B8h] [rbp-1C8h]
  __m128i *v149; // [rsp+5C0h] [rbp-1C0h]
  __int8 *v150; // [rsp+5C8h] [rbp-1B8h]
  _QWORD v151[5]; // [rsp+5D0h] [rbp-1B0h] BYREF
  _BYTE v152[64]; // [rsp+5F8h] [rbp-188h] BYREF
  __m128i *v153; // [rsp+638h] [rbp-148h]
  __m128i *v154; // [rsp+640h] [rbp-140h]
  __int8 *v155; // [rsp+648h] [rbp-138h]
  __int64 v156; // [rsp+650h] [rbp-130h] BYREF
  __int64 *v157; // [rsp+658h] [rbp-128h]
  __int64 *v158; // [rsp+660h] [rbp-120h]
  unsigned int v159; // [rsp+668h] [rbp-118h]
  unsigned int v160; // [rsp+66Ch] [rbp-114h]
  int v161; // [rsp+670h] [rbp-110h]
  _BYTE v162[64]; // [rsp+678h] [rbp-108h] BYREF
  const __m128i *v163; // [rsp+6B8h] [rbp-C8h] BYREF
  __m128i *v164; // [rsp+6C0h] [rbp-C0h]
  __m128i *v165; // [rsp+6C8h] [rbp-B8h]
  __m128i v166; // [rsp+6D0h] [rbp-B0h] BYREF
  unsigned __int64 v167; // [rsp+6E0h] [rbp-A0h]
  _BYTE v168[64]; // [rsp+6F8h] [rbp-88h] BYREF
  unsigned __int64 v169; // [rsp+738h] [rbp-48h]
  unsigned __int64 v170; // [rsp+740h] [rbp-40h]
  unsigned __int64 v171; // [rsp+748h] [rbp-38h]

  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  v3 = *(_QWORD *)(a2 + 80);
  v90 = 0;
  v93 = 8;
  if ( v3 )
    v3 -= 24;
  memset(v99, 0, sizeof(v99));
  LODWORD(v99[3]) = 8;
  v99[1] = &v99[5];
  v99[2] = &v99[5];
  v91 = v95;
  v92 = v95;
  v94 = 0;
  v96 = 0;
  v97 = 0;
  v98 = 0;
  sub_1953970((__int64)&v166, (__int64)&v90, v3);
  v4 = sub_157EBA0(v3);
  v166.m128i_i64[0] = v3;
  v5 = v97;
  v166.m128i_i64[1] = v4;
  LODWORD(v167) = 0;
  if ( v97 == v98 )
  {
    sub_13FDF40(&v96, v97, &v166);
  }
  else
  {
    if ( v97 )
    {
      *v97 = _mm_loadu_si128(&v166);
      v5[1].m128i_i64[0] = v167;
      v5 = v97;
    }
    v97 = (__m128i *)((char *)v5 + 24);
  }
  sub_13FE0F0((__int64)&v90);
  v6 = (const __m128i *)v113;
  v7 = (__m128i *)v111;
  sub_16CCCB0(v111, (__int64)v113, (__int64)v99);
  v9 = (const __m128i *)v99[14];
  v10 = (const __m128i *)v99[13];
  v114 = 0;
  v115 = 0;
  v116 = 0;
  v11 = v99[14] - v99[13];
  if ( v99[14] == v99[13] )
  {
    v13 = 0;
  }
  else
  {
    if ( v11 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_164;
    v12 = sub_22077B0(v99[14] - v99[13]);
    v9 = (const __m128i *)v99[14];
    v10 = (const __m128i *)v99[13];
    v13 = v12;
  }
  v114 = v13;
  v115 = v13;
  v116 = v13 + v11;
  if ( v10 != v9 )
  {
    v14 = (__m128i *)v13;
    v15 = v10;
    do
    {
      if ( v14 )
      {
        *v14 = _mm_loadu_si128(v15);
        v14[1].m128i_i64[0] = v15[1].m128i_i64[0];
      }
      v15 = (const __m128i *)((char *)v15 + 24);
      v14 = (__m128i *)((char *)v14 + 24);
    }
    while ( v15 != v9 );
    v13 += 8 * ((unsigned __int64)((char *)&v15[-2].m128i_u64[1] - (char *)v10) >> 3) + 24;
  }
  v115 = v13;
  sub_16CCEE0(v117, (__int64)v119, 8, (__int64)v111);
  v16 = v114;
  v6 = (const __m128i *)v101;
  v114 = 0;
  v120 = (const __m128i *)v16;
  v17 = v115;
  v115 = 0;
  v121 = (const __m128i *)v17;
  v18 = v116;
  v116 = 0;
  v122 = v18;
  v7 = (__m128i *)v100;
  sub_16CCCB0(v100, (__int64)v101, (__int64)&v90);
  v19 = v97;
  v20 = v96;
  v102 = 0;
  v103 = 0;
  v104 = 0;
  v21 = (char *)v97 - (char *)v96;
  if ( v97 == v96 )
  {
    v23 = 0;
  }
  else
  {
    if ( v21 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_164;
    v22 = sub_22077B0((char *)v97 - (char *)v96);
    v19 = v97;
    v20 = v96;
    v23 = (__m128i *)v22;
  }
  v102 = (unsigned __int64)v23;
  v103 = v23;
  v104 = &v23->m128i_i8[v21];
  if ( v20 != v19 )
  {
    v24 = v23;
    v25 = v20;
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
    v23 = (__m128i *)((char *)v23 + 8 * ((unsigned __int64)((char *)&v25[-2].m128i_u64[1] - (char *)v20) >> 3) + 24);
  }
  v103 = v23;
  sub_16CCEE0(v105, (__int64)v107, 8, (__int64)v100);
  v26 = v102;
  v7 = (__m128i *)v135;
  v6 = (const __m128i *)v136;
  v102 = 0;
  v108 = (const __m128i *)v26;
  v27 = v103;
  v103 = 0;
  v109 = v27;
  v28 = v104;
  v104 = 0;
  v110 = v28;
  sub_16CCCB0(v135, (__int64)v136, (__int64)v117);
  v29 = v121;
  v30 = (unsigned __int64)v120;
  v137 = 0;
  v138 = 0;
  v139 = 0;
  v31 = (char *)v121 - (char *)v120;
  if ( v121 == v120 )
  {
    v33 = 0;
  }
  else
  {
    if ( v31 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_164;
    v32 = sub_22077B0((char *)v121 - (char *)v120);
    v29 = v121;
    v30 = (unsigned __int64)v120;
    v33 = (__m128i *)v32;
  }
  v137 = (unsigned __int64)v33;
  v138 = v33;
  v139 = &v33->m128i_i8[v31];
  if ( (const __m128i *)v30 != v29 )
  {
    v34 = v33;
    v35 = (const __m128i *)v30;
    do
    {
      if ( v34 )
      {
        *v34 = _mm_loadu_si128(v35);
        v34[1].m128i_i64[0] = v35[1].m128i_i64[0];
      }
      v35 = (const __m128i *)((char *)v35 + 24);
      v34 = (__m128i *)((char *)v34 + 24);
    }
    while ( v35 != v29 );
    v33 = (__m128i *)((char *)v33 + 8 * (((unsigned __int64)&v35[-2].m128i_u64[1] - v30) >> 3) + 24);
  }
  v138 = v33;
  sub_16CCEE0(v140, (__int64)v142, 8, (__int64)v135);
  v7 = (__m128i *)v123;
  v143 = (const __m128i *)v137;
  v137 = 0;
  v144 = v138;
  v138 = 0;
  v145 = v139;
  v139 = 0;
  sub_16CCCB0(v123, (__int64)v125, (__int64)v105);
  v6 = v109;
  v36 = (unsigned __int64)v108;
  v126 = 0;
  v127 = 0;
  v128 = 0;
  v37 = (char *)v109 - (char *)v108;
  if ( v109 == v108 )
  {
    v39 = 0;
  }
  else
  {
    if ( v37 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_164;
    v38 = sub_22077B0((char *)v109 - (char *)v108);
    v6 = v109;
    v36 = (unsigned __int64)v108;
    v39 = v38;
  }
  v126 = v39;
  v127 = v39;
  v128 = v39 + v37;
  if ( (const __m128i *)v36 != v6 )
  {
    v40 = (__m128i *)v39;
    v41 = (const __m128i *)v36;
    do
    {
      if ( v40 )
      {
        *v40 = _mm_loadu_si128(v41);
        v40[1].m128i_i64[0] = v41[1].m128i_i64[0];
      }
      v41 = (const __m128i *)((char *)v41 + 24);
      v40 = (__m128i *)((char *)v40 + 24);
    }
    while ( v41 != v6 );
    v39 += 8 * (((unsigned __int64)&v41[-2].m128i_u64[1] - v36) >> 3) + 24;
  }
  v127 = v39;
  sub_16CCEE0(v129, (__int64)v131, 8, (__int64)v123);
  v42 = v126;
  v6 = (const __m128i *)v152;
  v126 = 0;
  v132 = (const __m128i *)v42;
  v43 = v127;
  v127 = 0;
  v133 = (const __m128i *)v43;
  v44 = v128;
  v128 = 0;
  v134 = v44;
  v7 = (__m128i *)v151;
  sub_16CCCB0(v151, (__int64)v152, (__int64)v140);
  v45 = v144;
  v46 = (unsigned __int64)v143;
  v153 = 0;
  v154 = 0;
  v155 = 0;
  v47 = (char *)v144 - (char *)v143;
  if ( v144 == v143 )
  {
    v49 = 0;
  }
  else
  {
    if ( v47 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_164;
    v48 = sub_22077B0((char *)v144 - (char *)v143);
    v45 = v144;
    v46 = (unsigned __int64)v143;
    v49 = (__m128i *)v48;
  }
  v153 = v49;
  v154 = v49;
  v155 = &v49->m128i_i8[v47];
  if ( (const __m128i *)v46 != v45 )
  {
    v50 = v49;
    v51 = (const __m128i *)v46;
    do
    {
      if ( v50 )
      {
        *v50 = _mm_loadu_si128(v51);
        v50[1].m128i_i64[0] = v51[1].m128i_i64[0];
      }
      v51 = (const __m128i *)((char *)v51 + 24);
      v50 = (__m128i *)((char *)v50 + 24);
    }
    while ( v51 != v45 );
    v49 = (__m128i *)((char *)v49 + 8 * (((unsigned __int64)&v51[-2].m128i_u64[1] - v46) >> 3) + 24);
  }
  v154 = v49;
  v6 = (const __m128i *)v147;
  v7 = (__m128i *)v146;
  sub_16CCCB0(v146, (__int64)v147, (__int64)v129);
  v52 = v133;
  v53 = (unsigned __int64)v132;
  v148 = 0;
  v149 = 0;
  v150 = 0;
  v54 = (char *)v133 - (char *)v132;
  if ( v133 == v132 )
  {
    v56 = 0;
  }
  else
  {
    if ( v54 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_164;
    v55 = sub_22077B0((char *)v133 - (char *)v132);
    v52 = v133;
    v53 = (unsigned __int64)v132;
    v56 = (__m128i *)v55;
  }
  v148 = v56;
  v149 = v56;
  v150 = &v56->m128i_i8[v54];
  if ( (const __m128i *)v53 != v52 )
  {
    v57 = v56;
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
    while ( v58 != v52 );
    v56 = (__m128i *)((char *)v56 + 8 * (((unsigned __int64)&v58[-2].m128i_u64[1] - v53) >> 3) + 24);
  }
  v149 = v56;
  v6 = (const __m128i *)v168;
  v7 = &v166;
  sub_16CCCB0(&v166, (__int64)v168, (__int64)v151);
  v59 = v154;
  v60 = v153;
  v169 = 0;
  v170 = 0;
  v171 = 0;
  v61 = (char *)v154 - (char *)v153;
  if ( v154 == v153 )
  {
    v63 = 0;
  }
  else
  {
    if ( v61 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_164;
    v62 = sub_22077B0((char *)v154 - (char *)v153);
    v59 = v154;
    v60 = v153;
    v63 = v62;
  }
  v169 = v63;
  v170 = v63;
  v171 = v63 + v61;
  if ( v60 != v59 )
  {
    v64 = (__m128i *)v63;
    v65 = v60;
    do
    {
      if ( v64 )
      {
        *v64 = _mm_loadu_si128(v65);
        v64[1].m128i_i64[0] = v65[1].m128i_i64[0];
      }
      v65 = (const __m128i *)((char *)v65 + 24);
      v64 = (__m128i *)((char *)v64 + 24);
    }
    while ( v65 != v59 );
    v63 += 8 * ((unsigned __int64)((char *)&v65[-2].m128i_u64[1] - (char *)v60) >> 3) + 24;
  }
  v170 = v63;
  v6 = (const __m128i *)v162;
  v7 = (__m128i *)&v156;
  sub_16CCCB0(&v156, (__int64)v162, (__int64)v146);
  v66 = v149;
  v67 = v148;
  v163 = 0;
  v164 = 0;
  v165 = 0;
  v68 = (char *)v149 - (char *)v148;
  if ( v149 == v148 )
  {
    v70 = 0;
    goto LABEL_73;
  }
  if ( v68 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_164:
    sub_4261EA(v7, v6, v8);
  v69 = sub_22077B0((char *)v149 - (char *)v148);
  v66 = v149;
  v67 = v148;
  v70 = (const __m128i *)v69;
LABEL_73:
  v163 = v70;
  v164 = (__m128i *)v70;
  v165 = (__m128i *)((char *)v70 + v68);
  if ( v66 == v67 )
  {
    v73 = (__m128i *)v70;
  }
  else
  {
    v71 = (__m128i *)v70;
    v72 = v67;
    do
    {
      if ( v71 )
      {
        *v71 = _mm_loadu_si128(v72);
        v71[1].m128i_i64[0] = v72[1].m128i_i64[0];
      }
      v72 = (const __m128i *)((char *)v72 + 24);
      v71 = (__m128i *)((char *)v71 + 24);
    }
    while ( v66 != v72 );
    v73 = (__m128i *)&v70[1].m128i_u64[((unsigned __int64)((char *)&v66[-2].m128i_u64[1] - (char *)v67) >> 3) + 1];
  }
  v164 = v73;
LABEL_80:
  while ( 1 )
  {
    v74 = v169;
    if ( (char *)v73 - (char *)v70 == v170 - v169 )
      break;
LABEL_81:
    v75 = (_BYTE *)a1[1];
    if ( v75 == (_BYTE *)a1[2] )
    {
      sub_1292090((__int64)a1, v75, &v73[-2].m128i_i64[1]);
      v73 = v164;
    }
    else
    {
      if ( v75 )
      {
        *(_QWORD *)v75 = v73[-2].m128i_i64[1];
        v75 = (_BYTE *)a1[1];
        v73 = v164;
      }
      a1[1] = v75 + 8;
    }
    v70 = v163;
    v73 = (__m128i *)((char *)v73 - 24);
    v164 = v73;
    if ( v73 != v163 )
    {
      while ( 1 )
      {
LABEL_86:
        v76 = sub_157EBA0(v73[-2].m128i_i64[1]);
        v77 = 0;
        if ( v76 )
        {
          v77 = sub_15F4D60(v76);
          v73 = v164;
        }
        v78 = v73[-1].m128i_u32[2];
        if ( v78 == v77 )
        {
          v70 = v163;
          goto LABEL_80;
        }
        v79 = v73[-1].m128i_i64[0];
        v73[-1].m128i_i32[2] = v78 + 1;
        v80 = sub_15F4DF0(v79, v78);
        v81 = v157;
        if ( v158 != v157 )
          goto LABEL_90;
        v84 = &v157[v160];
        if ( v157 == v84 )
          goto LABEL_107;
        v85 = 0;
        while ( 1 )
        {
          if ( v80 == *v81 )
          {
            v73 = v164;
            goto LABEL_86;
          }
          v86 = v81 + 1;
          if ( *v81 != -2 )
            break;
          if ( v86 == v84 )
          {
            v85 = v81;
LABEL_104:
            *v85 = v80;
            v73 = v164;
            --v161;
            ++v156;
            goto LABEL_91;
          }
LABEL_99:
          v85 = v81;
          v81 = v86;
        }
        if ( v84 != v86 )
          break;
        if ( v85 )
          goto LABEL_104;
LABEL_107:
        if ( v160 < v159 )
        {
          ++v160;
          *v84 = v80;
          v73 = v164;
          ++v156;
          goto LABEL_91;
        }
LABEL_90:
        sub_16CCBA0((__int64)&v156, v80);
        v73 = v164;
        if ( v82 )
        {
LABEL_91:
          v83 = sub_157EBA0(v80);
          v88.m128i_i64[0] = v80;
          v88.m128i_i64[1] = v83;
          LODWORD(v89) = 0;
          if ( v165 == v73 )
          {
            sub_13FDF40(&v163, v73, &v88);
            v73 = v164;
          }
          else
          {
            if ( v73 )
            {
              *v73 = _mm_loadu_si128(&v88);
              v73[1].m128i_i64[0] = v89;
              v73 = v164;
            }
            v73 = (__m128i *)((char *)v73 + 24);
            v164 = v73;
          }
        }
      }
      v81 = v85;
      goto LABEL_99;
    }
  }
  if ( v70 != v73 )
  {
    v87 = v70;
    while ( v87->m128i_i64[0] == *(_QWORD *)v74 && v87[1].m128i_i32[0] == *(_DWORD *)(v74 + 16) )
    {
      v87 = (const __m128i *)((char *)v87 + 24);
      v74 += 24LL;
      if ( v73 == v87 )
        goto LABEL_115;
    }
    goto LABEL_81;
  }
LABEL_115:
  if ( v70 )
    j_j___libc_free_0((unsigned __int64)v70);
  if ( v158 != v157 )
    _libc_free((unsigned __int64)v158);
  if ( v169 )
    j_j___libc_free_0(v169);
  if ( v167 != v166.m128i_i64[1] )
    _libc_free(v167);
  sub_19E7650(v146);
  sub_19E7650(v151);
  if ( v132 )
    j_j___libc_free_0((unsigned __int64)v132);
  if ( v130 != v129[1] )
    _libc_free(v130);
  if ( v126 )
    j_j___libc_free_0(v126);
  if ( v124 != v123[1] )
    _libc_free(v124);
  if ( v143 )
    j_j___libc_free_0((unsigned __int64)v143);
  if ( v141 != v140[1] )
    _libc_free(v141);
  sub_19E7650(v135);
  if ( v108 )
    j_j___libc_free_0((unsigned __int64)v108);
  if ( v106 != v105[1] )
    _libc_free(v106);
  sub_19E7650(v100);
  if ( v120 )
    j_j___libc_free_0((unsigned __int64)v120);
  if ( v118 != v117[1] )
    _libc_free(v118);
  if ( v114 )
    j_j___libc_free_0(v114);
  if ( v112 != v111[1] )
    _libc_free(v112);
  if ( v96 )
    j_j___libc_free_0((unsigned __int64)v96);
  if ( v92 != v91 )
    _libc_free((unsigned __int64)v92);
  sub_19E7650(v99);
}
