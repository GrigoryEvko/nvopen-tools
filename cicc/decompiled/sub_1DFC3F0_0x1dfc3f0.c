// Function: sub_1DFC3F0
// Address: 0x1dfc3f0
//
void __fastcall sub_1DFC3F0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  const __m128i *v4; // rsi
  __m128i *v5; // rdi
  const __m128i *v6; // rcx
  const __m128i *v7; // rdx
  unsigned __int64 v8; // r13
  __m128i *v9; // rax
  __m128i *v10; // rcx
  unsigned __int64 v11; // r14
  __m128i *v12; // rax
  __m128i *v13; // rsi
  unsigned __int64 v14; // r13
  __m128i *v15; // rax
  __m128i *v16; // rsi
  __m128i *v17; // rax
  __m128i *v18; // rax
  __int8 *v19; // rax
  const __m128i *v20; // rcx
  unsigned __int64 v21; // r15
  __m128i *v22; // rax
  __m128i *v23; // rcx
  __m128i *v24; // rax
  __m128i *v25; // rax
  __int8 *v26; // rax
  const __m128i *v27; // rcx
  unsigned __int64 v28; // r14
  __m128i *v29; // rax
  __m128i *v30; // rcx
  const __m128i *v31; // rcx
  unsigned __int64 v32; // r15
  __m128i *v33; // rax
  __m128i *v34; // rcx
  const __m128i *v35; // rcx
  unsigned __int64 v36; // r12
  __m128i *v37; // rax
  __m128i *v38; // rcx
  const __m128i *v39; // rcx
  unsigned __int64 v40; // r13
  __int64 v41; // rax
  __m128i *v42; // rdi
  __m128i *v43; // rax
  __m128i *v44; // rdx
  __int64 *v45; // rcx
  _BYTE *v46; // rsi
  __m128i *v47; // rax
  __int64 v48; // [rsp+10h] [rbp-730h] BYREF
  _QWORD *v49; // [rsp+18h] [rbp-728h]
  _QWORD *v50; // [rsp+20h] [rbp-720h]
  __int64 v51; // [rsp+28h] [rbp-718h]
  int v52; // [rsp+30h] [rbp-710h]
  _QWORD v53[8]; // [rsp+38h] [rbp-708h] BYREF
  const __m128i *v54; // [rsp+78h] [rbp-6C8h] BYREF
  const __m128i *v55; // [rsp+80h] [rbp-6C0h]
  __int64 v56; // [rsp+88h] [rbp-6B8h]
  _QWORD v57[16]; // [rsp+90h] [rbp-6B0h] BYREF
  _QWORD v58[2]; // [rsp+110h] [rbp-630h] BYREF
  unsigned __int64 v59; // [rsp+120h] [rbp-620h]
  char v60[64]; // [rsp+138h] [rbp-608h] BYREF
  __m128i *v61; // [rsp+178h] [rbp-5C8h]
  __m128i *v62; // [rsp+180h] [rbp-5C0h]
  __int8 *v63; // [rsp+188h] [rbp-5B8h]
  _QWORD v64[2]; // [rsp+190h] [rbp-5B0h] BYREF
  unsigned __int64 v65; // [rsp+1A0h] [rbp-5A0h]
  char v66[64]; // [rsp+1B8h] [rbp-588h] BYREF
  const __m128i *v67; // [rsp+1F8h] [rbp-548h]
  __m128i *v68; // [rsp+200h] [rbp-540h]
  __int8 *v69; // [rsp+208h] [rbp-538h]
  _QWORD v70[2]; // [rsp+210h] [rbp-530h] BYREF
  unsigned __int64 v71; // [rsp+220h] [rbp-520h]
  _BYTE v72[64]; // [rsp+238h] [rbp-508h] BYREF
  __m128i *v73; // [rsp+278h] [rbp-4C8h]
  __m128i *v74; // [rsp+280h] [rbp-4C0h]
  __int8 *v75; // [rsp+288h] [rbp-4B8h]
  _QWORD v76[2]; // [rsp+290h] [rbp-4B0h] BYREF
  unsigned __int64 v77; // [rsp+2A0h] [rbp-4A0h]
  char v78[64]; // [rsp+2B8h] [rbp-488h] BYREF
  const __m128i *v79; // [rsp+2F8h] [rbp-448h]
  __m128i *v80; // [rsp+300h] [rbp-440h]
  __int8 *v81; // [rsp+308h] [rbp-438h]
  _QWORD v82[2]; // [rsp+310h] [rbp-430h] BYREF
  unsigned __int64 v83; // [rsp+320h] [rbp-420h]
  _BYTE v84[64]; // [rsp+338h] [rbp-408h] BYREF
  __m128i *v85; // [rsp+378h] [rbp-3C8h]
  __m128i *v86; // [rsp+380h] [rbp-3C0h]
  __int8 *v87; // [rsp+388h] [rbp-3B8h]
  _QWORD v88[2]; // [rsp+390h] [rbp-3B0h] BYREF
  unsigned __int64 v89; // [rsp+3A0h] [rbp-3A0h]
  char v90[64]; // [rsp+3B8h] [rbp-388h] BYREF
  const __m128i *v91; // [rsp+3F8h] [rbp-348h]
  const __m128i *v92; // [rsp+400h] [rbp-340h]
  __int8 *v93; // [rsp+408h] [rbp-338h]
  _QWORD v94[2]; // [rsp+410h] [rbp-330h] BYREF
  unsigned __int64 v95; // [rsp+420h] [rbp-320h]
  char v96[64]; // [rsp+438h] [rbp-308h] BYREF
  __m128i *v97; // [rsp+478h] [rbp-2C8h]
  __m128i *v98; // [rsp+480h] [rbp-2C0h]
  __int8 *v99; // [rsp+488h] [rbp-2B8h]
  _QWORD v100[2]; // [rsp+490h] [rbp-2B0h] BYREF
  unsigned __int64 v101; // [rsp+4A0h] [rbp-2A0h]
  char v102[64]; // [rsp+4B8h] [rbp-288h] BYREF
  const __m128i *v103; // [rsp+4F8h] [rbp-248h]
  const __m128i *v104; // [rsp+500h] [rbp-240h]
  __int8 *v105; // [rsp+508h] [rbp-238h]
  _QWORD v106[2]; // [rsp+510h] [rbp-230h] BYREF
  unsigned __int64 v107; // [rsp+520h] [rbp-220h]
  _BYTE v108[64]; // [rsp+538h] [rbp-208h] BYREF
  __m128i *v109; // [rsp+578h] [rbp-1C8h]
  __m128i *v110; // [rsp+580h] [rbp-1C0h]
  __int8 *v111; // [rsp+588h] [rbp-1B8h]
  _QWORD v112[2]; // [rsp+590h] [rbp-1B0h] BYREF
  unsigned __int64 v113; // [rsp+5A0h] [rbp-1A0h]
  _BYTE v114[64]; // [rsp+5B8h] [rbp-188h] BYREF
  __m128i *v115; // [rsp+5F8h] [rbp-148h]
  __m128i *v116; // [rsp+600h] [rbp-140h]
  __int8 *v117; // [rsp+608h] [rbp-138h]
  _QWORD v118[2]; // [rsp+610h] [rbp-130h] BYREF
  unsigned __int64 v119; // [rsp+620h] [rbp-120h]
  _BYTE v120[64]; // [rsp+638h] [rbp-108h] BYREF
  __m128i *v121; // [rsp+678h] [rbp-C8h]
  __m128i *v122; // [rsp+680h] [rbp-C0h]
  __int8 *v123; // [rsp+688h] [rbp-B8h]
  __m128i v124; // [rsp+690h] [rbp-B0h] BYREF
  unsigned __int64 v125; // [rsp+6A0h] [rbp-A0h]
  _BYTE v126[64]; // [rsp+6B8h] [rbp-88h] BYREF
  __m128i *v127; // [rsp+6F8h] [rbp-48h]
  __m128i *v128; // [rsp+700h] [rbp-40h]
  __int8 *v129; // [rsp+708h] [rbp-38h]

  v53[0] = a2;
  memset(v57, 0, sizeof(v57));
  v124.m128i_i64[0] = a2;
  v57[1] = &v57[5];
  v57[2] = &v57[5];
  v49 = v53;
  v50 = v53;
  v51 = 0x100000008LL;
  v3 = *(_QWORD *)(a2 + 88);
  LODWORD(v57[3]) = 8;
  v124.m128i_i64[1] = v3;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  v52 = 0;
  v48 = 1;
  sub_1D530F0(&v54, 0, &v124);
  sub_1D53270((__int64)&v48);
  v4 = (const __m128i *)v72;
  v5 = (__m128i *)v70;
  sub_16CCCB0(v70, (__int64)v72, (__int64)v57);
  v6 = (const __m128i *)v57[14];
  v7 = (const __m128i *)v57[13];
  v73 = 0;
  v74 = 0;
  v75 = 0;
  v8 = v57[14] - v57[13];
  if ( v57[14] == v57[13] )
  {
    v9 = 0;
  }
  else
  {
    if ( v8 > 0x7FFFFFFFFFFFFFF0LL )
      goto LABEL_152;
    v9 = (__m128i *)sub_22077B0(v57[14] - v57[13]);
    v6 = (const __m128i *)v57[14];
    v7 = (const __m128i *)v57[13];
  }
  v73 = v9;
  v74 = v9;
  v75 = &v9->m128i_i8[v8];
  if ( v6 == v7 )
  {
    v10 = v9;
  }
  else
  {
    v10 = (__m128i *)((char *)v9 + (char *)v6 - (char *)v7);
    do
    {
      if ( v9 )
        *v9 = _mm_loadu_si128(v7);
      ++v9;
      ++v7;
    }
    while ( v10 != v9 );
  }
  v74 = v10;
  sub_16CCEE0(v76, (__int64)v78, 8, (__int64)v70);
  v5 = (__m128i *)v58;
  v79 = v73;
  v73 = 0;
  v80 = v74;
  v74 = 0;
  v81 = v75;
  v75 = 0;
  sub_16CCCB0(v58, (__int64)v60, (__int64)&v48);
  v4 = v55;
  v7 = v54;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v11 = (char *)v55 - (char *)v54;
  if ( v55 == v54 )
  {
    v12 = 0;
  }
  else
  {
    if ( v11 > 0x7FFFFFFFFFFFFFF0LL )
      goto LABEL_152;
    v12 = (__m128i *)sub_22077B0((char *)v55 - (char *)v54);
    v4 = v55;
    v7 = v54;
  }
  v61 = v12;
  v62 = v12;
  v63 = &v12->m128i_i8[v11];
  if ( v4 == v7 )
  {
    v13 = v12;
  }
  else
  {
    v13 = (__m128i *)((char *)v12 + (char *)v4 - (char *)v7);
    do
    {
      if ( v12 )
        *v12 = _mm_loadu_si128(v7);
      ++v12;
      ++v7;
    }
    while ( v13 != v12 );
  }
  v62 = v13;
  sub_16CCEE0(v64, (__int64)v66, 8, (__int64)v58);
  v5 = (__m128i *)v94;
  v67 = v61;
  v61 = 0;
  v68 = v62;
  v62 = 0;
  v69 = v63;
  v63 = 0;
  sub_16CCCB0(v94, (__int64)v96, (__int64)v76);
  v4 = v80;
  v7 = v79;
  v97 = 0;
  v98 = 0;
  v99 = 0;
  v14 = (char *)v80 - (char *)v79;
  if ( v80 == v79 )
  {
    v15 = 0;
  }
  else
  {
    if ( v14 > 0x7FFFFFFFFFFFFFF0LL )
      goto LABEL_152;
    v15 = (__m128i *)sub_22077B0((char *)v80 - (char *)v79);
    v4 = v80;
    v7 = v79;
  }
  v97 = v15;
  v98 = v15;
  v99 = &v15->m128i_i8[v14];
  if ( v7 == v4 )
  {
    v16 = v15;
  }
  else
  {
    v16 = (__m128i *)((char *)v15 + (char *)v4 - (char *)v7);
    do
    {
      if ( v15 )
        *v15 = _mm_loadu_si128(v7);
      ++v15;
      ++v7;
    }
    while ( v15 != v16 );
  }
  v98 = v16;
  sub_16CCEE0(v100, (__int64)v102, 8, (__int64)v94);
  v17 = v97;
  v5 = (__m128i *)v82;
  v4 = (const __m128i *)v84;
  v97 = 0;
  v103 = v17;
  v18 = v98;
  v98 = 0;
  v104 = v18;
  v19 = v99;
  v99 = 0;
  v105 = v19;
  sub_16CCCB0(v82, (__int64)v84, (__int64)v64);
  v20 = v68;
  v7 = v67;
  v85 = 0;
  v86 = 0;
  v87 = 0;
  v21 = (char *)v68 - (char *)v67;
  if ( v68 == v67 )
  {
    v22 = 0;
  }
  else
  {
    if ( v21 > 0x7FFFFFFFFFFFFFF0LL )
      goto LABEL_152;
    v22 = (__m128i *)sub_22077B0((char *)v68 - (char *)v67);
    v20 = v68;
    v7 = v67;
  }
  v85 = v22;
  v86 = v22;
  v87 = &v22->m128i_i8[v21];
  if ( v20 == v7 )
  {
    v23 = v22;
  }
  else
  {
    v23 = (__m128i *)((char *)v22 + (char *)v20 - (char *)v7);
    do
    {
      if ( v22 )
        *v22 = _mm_loadu_si128(v7);
      ++v22;
      ++v7;
    }
    while ( v23 != v22 );
  }
  v86 = v23;
  sub_16CCEE0(v88, (__int64)v90, 8, (__int64)v82);
  v24 = v85;
  v5 = (__m128i *)v112;
  v4 = (const __m128i *)v114;
  v85 = 0;
  v91 = v24;
  v25 = v86;
  v86 = 0;
  v92 = v25;
  v26 = v87;
  v87 = 0;
  v93 = v26;
  sub_16CCCB0(v112, (__int64)v114, (__int64)v100);
  v27 = v104;
  v7 = v103;
  v115 = 0;
  v116 = 0;
  v117 = 0;
  v28 = (char *)v104 - (char *)v103;
  if ( v104 == v103 )
  {
    v29 = 0;
  }
  else
  {
    if ( v28 > 0x7FFFFFFFFFFFFFF0LL )
      goto LABEL_152;
    v29 = (__m128i *)sub_22077B0((char *)v104 - (char *)v103);
    v27 = v104;
    v7 = v103;
  }
  v115 = v29;
  v116 = v29;
  v117 = &v29->m128i_i8[v28];
  if ( v7 == v27 )
  {
    v30 = v29;
  }
  else
  {
    v30 = (__m128i *)((char *)v29 + (char *)v27 - (char *)v7);
    do
    {
      if ( v29 )
        *v29 = _mm_loadu_si128(v7);
      ++v29;
      ++v7;
    }
    while ( v30 != v29 );
  }
  v4 = (const __m128i *)v108;
  v116 = v30;
  v5 = (__m128i *)v106;
  sub_16CCCB0(v106, (__int64)v108, (__int64)v88);
  v31 = v92;
  v7 = v91;
  v109 = 0;
  v110 = 0;
  v111 = 0;
  v32 = (char *)v92 - (char *)v91;
  if ( v92 == v91 )
  {
    v33 = 0;
  }
  else
  {
    if ( v32 > 0x7FFFFFFFFFFFFFF0LL )
      goto LABEL_152;
    v33 = (__m128i *)sub_22077B0((char *)v92 - (char *)v91);
    v31 = v92;
    v7 = v91;
  }
  v109 = v33;
  v110 = v33;
  v111 = &v33->m128i_i8[v32];
  if ( v31 == v7 )
  {
    v34 = v33;
  }
  else
  {
    v34 = (__m128i *)((char *)v33 + (char *)v31 - (char *)v7);
    do
    {
      if ( v33 )
        *v33 = _mm_loadu_si128(v7);
      ++v33;
      ++v7;
    }
    while ( v34 != v33 );
  }
  v5 = &v124;
  v4 = (const __m128i *)v126;
  v110 = v34;
  sub_16CCCB0(&v124, (__int64)v126, (__int64)v112);
  v35 = v116;
  v7 = v115;
  v127 = 0;
  v128 = 0;
  v129 = 0;
  v36 = (char *)v116 - (char *)v115;
  if ( v116 == v115 )
  {
    v37 = 0;
  }
  else
  {
    if ( v36 > 0x7FFFFFFFFFFFFFF0LL )
      goto LABEL_152;
    v37 = (__m128i *)sub_22077B0((char *)v116 - (char *)v115);
    v35 = v116;
    v7 = v115;
  }
  v127 = v37;
  v128 = v37;
  v129 = &v37->m128i_i8[v36];
  if ( v35 == v7 )
  {
    v38 = v37;
  }
  else
  {
    v38 = (__m128i *)((char *)v37 + (char *)v35 - (char *)v7);
    do
    {
      if ( v37 )
        *v37 = _mm_loadu_si128(v7);
      ++v37;
      ++v7;
    }
    while ( v37 != v38 );
  }
  v4 = (const __m128i *)v120;
  v128 = v38;
  v5 = (__m128i *)v118;
  sub_16CCCB0(v118, (__int64)v120, (__int64)v106);
  v7 = v110;
  v39 = v109;
  v121 = 0;
  v122 = 0;
  v123 = 0;
  v40 = (char *)v110 - (char *)v109;
  if ( v110 == v109 )
  {
    v42 = 0;
    goto LABEL_60;
  }
  if ( v40 > 0x7FFFFFFFFFFFFFF0LL )
LABEL_152:
    sub_4261EA(v5, v4, v7);
  v41 = sub_22077B0((char *)v110 - (char *)v109);
  v7 = v110;
  v39 = v109;
  v42 = (__m128i *)v41;
LABEL_60:
  v121 = v42;
  v122 = v42;
  v123 = &v42->m128i_i8[v40];
  if ( v7 == v39 )
  {
    v44 = v42;
  }
  else
  {
    v43 = v42;
    v44 = (__m128i *)((char *)v42 + (char *)v7 - (char *)v39);
    do
    {
      if ( v43 )
        *v43 = _mm_loadu_si128(v39);
      ++v43;
      ++v39;
    }
    while ( v43 != v44 );
  }
  v122 = v44;
  while ( 1 )
  {
    v45 = (__int64 *)v127;
    if ( (char *)v44 - (char *)v42 != (char *)v128 - (char *)v127 )
      goto LABEL_67;
    if ( v42 == v44 )
      break;
    v47 = v42;
    while ( v47->m128i_i64[0] == *v45 && v47->m128i_i64[1] == v45[1] )
    {
      ++v47;
      v45 += 2;
      if ( v44 == v47 )
        goto LABEL_78;
    }
LABEL_67:
    v46 = *(_BYTE **)(a1 + 8);
    if ( v46 == *(_BYTE **)(a1 + 16) )
    {
      sub_1D4AF10(a1, v46, (__m128i *)v44[-1].m128i_i64);
      v44 = v122;
    }
    else
    {
      if ( v46 )
      {
        *(_QWORD *)v46 = v44[-1].m128i_i64[0];
        v46 = *(_BYTE **)(a1 + 8);
        v44 = v122;
      }
      *(_QWORD *)(a1 + 8) = v46 + 8;
    }
    v42 = v121;
    v122 = --v44;
    if ( v44 != v121 )
    {
      sub_1D53270((__int64)v118);
      v42 = v121;
      v44 = v122;
    }
  }
LABEL_78:
  if ( v42 )
    j_j___libc_free_0(v42, v123 - (__int8 *)v42);
  if ( v119 != v118[1] )
    _libc_free(v119);
  if ( v127 )
    j_j___libc_free_0(v127, v129 - (__int8 *)v127);
  if ( v125 != v124.m128i_i64[1] )
    _libc_free(v125);
  if ( v109 )
    j_j___libc_free_0(v109, v111 - (__int8 *)v109);
  if ( v107 != v106[1] )
    _libc_free(v107);
  if ( v115 )
    j_j___libc_free_0(v115, v117 - (__int8 *)v115);
  if ( v113 != v112[1] )
    _libc_free(v113);
  if ( v91 )
    j_j___libc_free_0(v91, v93 - (__int8 *)v91);
  if ( v89 != v88[1] )
    _libc_free(v89);
  if ( v85 )
    j_j___libc_free_0(v85, v87 - (__int8 *)v85);
  if ( v83 != v82[1] )
    _libc_free(v83);
  if ( v103 )
    j_j___libc_free_0(v103, v105 - (__int8 *)v103);
  if ( v101 != v100[1] )
    _libc_free(v101);
  if ( v97 )
    j_j___libc_free_0(v97, v99 - (__int8 *)v97);
  if ( v95 != v94[1] )
    _libc_free(v95);
  if ( v67 )
    j_j___libc_free_0(v67, v69 - (__int8 *)v67);
  if ( v65 != v64[1] )
    _libc_free(v65);
  if ( v61 )
    j_j___libc_free_0(v61, v63 - (__int8 *)v61);
  if ( v59 != v58[1] )
    _libc_free(v59);
  if ( v79 )
    j_j___libc_free_0(v79, v81 - (__int8 *)v79);
  if ( v77 != v76[1] )
    _libc_free(v77);
  if ( v73 )
    j_j___libc_free_0(v73, v75 - (__int8 *)v73);
  if ( v71 != v70[1] )
    _libc_free(v71);
  if ( v54 )
    j_j___libc_free_0(v54, v56 - (_QWORD)v54);
  if ( v50 != v49 )
    _libc_free((unsigned __int64)v50);
  if ( v57[13] )
    j_j___libc_free_0(v57[13], v57[15] - v57[13]);
  if ( v57[2] != v57[1] )
    _libc_free(v57[2]);
}
