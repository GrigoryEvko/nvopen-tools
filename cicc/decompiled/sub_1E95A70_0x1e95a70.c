// Function: sub_1E95A70
// Address: 0x1e95a70
//
_QWORD *__fastcall sub_1E95A70(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // rax
  const __m128i *v5; // rsi
  __m128i *v6; // rdi
  const __m128i *v7; // rcx
  const __m128i *v8; // rdx
  unsigned __int64 v9; // r14
  __m128i *v10; // rax
  __m128i *v11; // rcx
  unsigned __int64 v12; // rbx
  __m128i *v13; // rax
  __m128i *v14; // rsi
  unsigned __int64 v15; // r14
  __m128i *v16; // rax
  __m128i *v17; // rsi
  __m128i *v18; // rax
  __m128i *v19; // rax
  __int8 *v20; // rax
  const __m128i *v21; // rcx
  unsigned __int64 v22; // rbx
  __m128i *v23; // rax
  __m128i *v24; // rcx
  __m128i *v25; // rax
  __m128i *v26; // rax
  __int8 *v27; // rax
  const __m128i *v28; // rcx
  unsigned __int64 v29; // r15
  __m128i *v30; // rax
  __m128i *v31; // rcx
  const __m128i *v32; // rcx
  unsigned __int64 v33; // rbx
  __m128i *v34; // rax
  __m128i *v35; // rcx
  const __m128i *v36; // rcx
  unsigned __int64 v37; // rbx
  __m128i *v38; // rax
  __m128i *v39; // rcx
  const __m128i *v40; // rcx
  unsigned __int64 v41; // r14
  __int64 v42; // rax
  __m128i *v43; // rdi
  __m128i *v44; // rax
  __m128i *v45; // rdx
  __int64 *v46; // rcx
  _BYTE *v47; // rsi
  __m128i *v48; // rax
  __int64 *v49; // rdi
  __int64 *v50; // r14
  __int64 *v51; // rbx
  __int64 *v52; // rsi
  __int64 *v53; // rax
  __int64 v54; // rdx
  __int64 *v56; // [rsp+10h] [rbp-750h] BYREF
  _BYTE *v57; // [rsp+18h] [rbp-748h]
  _BYTE *v58; // [rsp+20h] [rbp-740h]
  __int64 v59; // [rsp+30h] [rbp-730h] BYREF
  _QWORD *v60; // [rsp+38h] [rbp-728h]
  _QWORD *v61; // [rsp+40h] [rbp-720h]
  __int64 v62; // [rsp+48h] [rbp-718h]
  int v63; // [rsp+50h] [rbp-710h]
  _QWORD v64[8]; // [rsp+58h] [rbp-708h] BYREF
  const __m128i *v65; // [rsp+98h] [rbp-6C8h] BYREF
  const __m128i *v66; // [rsp+A0h] [rbp-6C0h]
  __int64 v67; // [rsp+A8h] [rbp-6B8h]
  _QWORD v68[16]; // [rsp+B0h] [rbp-6B0h] BYREF
  _QWORD v69[2]; // [rsp+130h] [rbp-630h] BYREF
  unsigned __int64 v70; // [rsp+140h] [rbp-620h]
  char v71[64]; // [rsp+158h] [rbp-608h] BYREF
  __m128i *v72; // [rsp+198h] [rbp-5C8h]
  __m128i *v73; // [rsp+1A0h] [rbp-5C0h]
  __int8 *v74; // [rsp+1A8h] [rbp-5B8h]
  _QWORD v75[2]; // [rsp+1B0h] [rbp-5B0h] BYREF
  unsigned __int64 v76; // [rsp+1C0h] [rbp-5A0h]
  char v77[64]; // [rsp+1D8h] [rbp-588h] BYREF
  const __m128i *v78; // [rsp+218h] [rbp-548h]
  __m128i *v79; // [rsp+220h] [rbp-540h]
  __int8 *v80; // [rsp+228h] [rbp-538h]
  _QWORD v81[2]; // [rsp+230h] [rbp-530h] BYREF
  unsigned __int64 v82; // [rsp+240h] [rbp-520h]
  _BYTE v83[64]; // [rsp+258h] [rbp-508h] BYREF
  __m128i *v84; // [rsp+298h] [rbp-4C8h]
  __m128i *v85; // [rsp+2A0h] [rbp-4C0h]
  __int8 *v86; // [rsp+2A8h] [rbp-4B8h]
  _QWORD v87[2]; // [rsp+2B0h] [rbp-4B0h] BYREF
  unsigned __int64 v88; // [rsp+2C0h] [rbp-4A0h]
  char v89[64]; // [rsp+2D8h] [rbp-488h] BYREF
  const __m128i *v90; // [rsp+318h] [rbp-448h]
  __m128i *v91; // [rsp+320h] [rbp-440h]
  __int8 *v92; // [rsp+328h] [rbp-438h]
  _QWORD v93[2]; // [rsp+330h] [rbp-430h] BYREF
  unsigned __int64 v94; // [rsp+340h] [rbp-420h]
  _BYTE v95[64]; // [rsp+358h] [rbp-408h] BYREF
  __m128i *v96; // [rsp+398h] [rbp-3C8h]
  __m128i *v97; // [rsp+3A0h] [rbp-3C0h]
  __int8 *v98; // [rsp+3A8h] [rbp-3B8h]
  _QWORD v99[2]; // [rsp+3B0h] [rbp-3B0h] BYREF
  unsigned __int64 v100; // [rsp+3C0h] [rbp-3A0h]
  char v101[64]; // [rsp+3D8h] [rbp-388h] BYREF
  const __m128i *v102; // [rsp+418h] [rbp-348h]
  const __m128i *v103; // [rsp+420h] [rbp-340h]
  __int8 *v104; // [rsp+428h] [rbp-338h]
  _QWORD v105[2]; // [rsp+430h] [rbp-330h] BYREF
  unsigned __int64 v106; // [rsp+440h] [rbp-320h]
  char v107[64]; // [rsp+458h] [rbp-308h] BYREF
  __m128i *v108; // [rsp+498h] [rbp-2C8h]
  __m128i *v109; // [rsp+4A0h] [rbp-2C0h]
  __int8 *v110; // [rsp+4A8h] [rbp-2B8h]
  _QWORD v111[2]; // [rsp+4B0h] [rbp-2B0h] BYREF
  unsigned __int64 v112; // [rsp+4C0h] [rbp-2A0h]
  char v113[64]; // [rsp+4D8h] [rbp-288h] BYREF
  const __m128i *v114; // [rsp+518h] [rbp-248h]
  const __m128i *v115; // [rsp+520h] [rbp-240h]
  __int8 *v116; // [rsp+528h] [rbp-238h]
  _QWORD v117[2]; // [rsp+530h] [rbp-230h] BYREF
  unsigned __int64 v118; // [rsp+540h] [rbp-220h]
  _BYTE v119[64]; // [rsp+558h] [rbp-208h] BYREF
  __m128i *v120; // [rsp+598h] [rbp-1C8h]
  __m128i *v121; // [rsp+5A0h] [rbp-1C0h]
  __int8 *v122; // [rsp+5A8h] [rbp-1B8h]
  _QWORD v123[2]; // [rsp+5B0h] [rbp-1B0h] BYREF
  unsigned __int64 v124; // [rsp+5C0h] [rbp-1A0h]
  _BYTE v125[64]; // [rsp+5D8h] [rbp-188h] BYREF
  __m128i *v126; // [rsp+618h] [rbp-148h]
  __m128i *v127; // [rsp+620h] [rbp-140h]
  __int8 *v128; // [rsp+628h] [rbp-138h]
  _QWORD v129[2]; // [rsp+630h] [rbp-130h] BYREF
  unsigned __int64 v130; // [rsp+640h] [rbp-120h]
  _BYTE v131[64]; // [rsp+658h] [rbp-108h] BYREF
  __m128i *v132; // [rsp+698h] [rbp-C8h]
  __m128i *v133; // [rsp+6A0h] [rbp-C0h]
  __int8 *v134; // [rsp+6A8h] [rbp-B8h]
  __m128i v135; // [rsp+6B0h] [rbp-B0h] BYREF
  unsigned __int64 v136; // [rsp+6C0h] [rbp-A0h]
  _BYTE v137[64]; // [rsp+6D8h] [rbp-88h] BYREF
  __m128i *v138; // [rsp+718h] [rbp-48h]
  __m128i *v139; // [rsp+720h] [rbp-40h]
  __int8 *v140; // [rsp+728h] [rbp-38h]

  v3 = *(_QWORD *)(a2 + 328);
  v65 = 0;
  memset(v68, 0, sizeof(v68));
  v66 = 0;
  v68[1] = &v68[5];
  v68[2] = &v68[5];
  v60 = v64;
  v61 = v64;
  v67 = 0;
  v62 = 0x100000008LL;
  v4 = *(_QWORD *)(v3 + 88);
  v64[0] = v3;
  v135.m128i_i64[0] = v3;
  v135.m128i_i64[1] = v4;
  v56 = 0;
  v57 = 0;
  v58 = 0;
  LODWORD(v68[3]) = 8;
  v63 = 0;
  v59 = 1;
  sub_1D530F0(&v65, 0, &v135);
  sub_1D53270((__int64)&v59);
  v5 = (const __m128i *)v83;
  v6 = (__m128i *)v81;
  sub_16CCCB0(v81, (__int64)v83, (__int64)v68);
  v7 = (const __m128i *)v68[14];
  v8 = (const __m128i *)v68[13];
  v84 = 0;
  v85 = 0;
  v86 = 0;
  v9 = v68[14] - v68[13];
  if ( v68[14] == v68[13] )
  {
    v10 = 0;
  }
  else
  {
    if ( v9 > 0x7FFFFFFFFFFFFFF0LL )
      goto LABEL_163;
    v10 = (__m128i *)sub_22077B0(v68[14] - v68[13]);
    v7 = (const __m128i *)v68[14];
    v8 = (const __m128i *)v68[13];
  }
  v84 = v10;
  v85 = v10;
  v86 = &v10->m128i_i8[v9];
  if ( v8 == v7 )
  {
    v11 = v10;
  }
  else
  {
    v11 = (__m128i *)((char *)v10 + (char *)v7 - (char *)v8);
    do
    {
      if ( v10 )
        *v10 = _mm_loadu_si128(v8);
      ++v10;
      ++v8;
    }
    while ( v11 != v10 );
  }
  v85 = v11;
  sub_16CCEE0(v87, (__int64)v89, 8, (__int64)v81);
  v6 = (__m128i *)v69;
  v90 = v84;
  v84 = 0;
  v91 = v85;
  v85 = 0;
  v92 = v86;
  v86 = 0;
  sub_16CCCB0(v69, (__int64)v71, (__int64)&v59);
  v5 = v66;
  v8 = v65;
  v72 = 0;
  v73 = 0;
  v74 = 0;
  v12 = (char *)v66 - (char *)v65;
  if ( v66 == v65 )
  {
    v13 = 0;
  }
  else
  {
    if ( v12 > 0x7FFFFFFFFFFFFFF0LL )
      goto LABEL_163;
    v13 = (__m128i *)sub_22077B0((char *)v66 - (char *)v65);
    v5 = v66;
    v8 = v65;
  }
  v72 = v13;
  v73 = v13;
  v74 = &v13->m128i_i8[v12];
  if ( v5 == v8 )
  {
    v14 = v13;
  }
  else
  {
    v14 = (__m128i *)((char *)v13 + (char *)v5 - (char *)v8);
    do
    {
      if ( v13 )
        *v13 = _mm_loadu_si128(v8);
      ++v13;
      ++v8;
    }
    while ( v13 != v14 );
  }
  v73 = v14;
  sub_16CCEE0(v75, (__int64)v77, 8, (__int64)v69);
  v6 = (__m128i *)v105;
  v78 = v72;
  v72 = 0;
  v79 = v73;
  v73 = 0;
  v80 = v74;
  v74 = 0;
  sub_16CCCB0(v105, (__int64)v107, (__int64)v87);
  v5 = v91;
  v8 = v90;
  v108 = 0;
  v109 = 0;
  v110 = 0;
  v15 = (char *)v91 - (char *)v90;
  if ( v91 == v90 )
  {
    v16 = 0;
  }
  else
  {
    if ( v15 > 0x7FFFFFFFFFFFFFF0LL )
      goto LABEL_163;
    v16 = (__m128i *)sub_22077B0((char *)v91 - (char *)v90);
    v5 = v91;
    v8 = v90;
  }
  v108 = v16;
  v109 = v16;
  v110 = &v16->m128i_i8[v15];
  if ( v8 == v5 )
  {
    v17 = v16;
  }
  else
  {
    v17 = (__m128i *)((char *)v16 + (char *)v5 - (char *)v8);
    do
    {
      if ( v16 )
        *v16 = _mm_loadu_si128(v8);
      ++v16;
      ++v8;
    }
    while ( v17 != v16 );
  }
  v109 = v17;
  sub_16CCEE0(v111, (__int64)v113, 8, (__int64)v105);
  v18 = v108;
  v6 = (__m128i *)v93;
  v5 = (const __m128i *)v95;
  v108 = 0;
  v114 = v18;
  v19 = v109;
  v109 = 0;
  v115 = v19;
  v20 = v110;
  v110 = 0;
  v116 = v20;
  sub_16CCCB0(v93, (__int64)v95, (__int64)v75);
  v21 = v79;
  v8 = v78;
  v96 = 0;
  v97 = 0;
  v98 = 0;
  v22 = (char *)v79 - (char *)v78;
  if ( v79 == v78 )
  {
    v23 = 0;
  }
  else
  {
    if ( v22 > 0x7FFFFFFFFFFFFFF0LL )
      goto LABEL_163;
    v23 = (__m128i *)sub_22077B0((char *)v79 - (char *)v78);
    v21 = v79;
    v8 = v78;
  }
  v96 = v23;
  v97 = v23;
  v98 = &v23->m128i_i8[v22];
  if ( v8 == v21 )
  {
    v24 = v23;
  }
  else
  {
    v24 = (__m128i *)((char *)v23 + (char *)v21 - (char *)v8);
    do
    {
      if ( v23 )
        *v23 = _mm_loadu_si128(v8);
      ++v23;
      ++v8;
    }
    while ( v24 != v23 );
  }
  v97 = v24;
  sub_16CCEE0(v99, (__int64)v101, 8, (__int64)v93);
  v25 = v96;
  v6 = (__m128i *)v123;
  v5 = (const __m128i *)v125;
  v96 = 0;
  v102 = v25;
  v26 = v97;
  v97 = 0;
  v103 = v26;
  v27 = v98;
  v98 = 0;
  v104 = v27;
  sub_16CCCB0(v123, (__int64)v125, (__int64)v111);
  v28 = v115;
  v8 = v114;
  v126 = 0;
  v127 = 0;
  v128 = 0;
  v29 = (char *)v115 - (char *)v114;
  if ( v115 == v114 )
  {
    v30 = 0;
  }
  else
  {
    if ( v29 > 0x7FFFFFFFFFFFFFF0LL )
      goto LABEL_163;
    v30 = (__m128i *)sub_22077B0((char *)v115 - (char *)v114);
    v28 = v115;
    v8 = v114;
  }
  v126 = v30;
  v127 = v30;
  v128 = &v30->m128i_i8[v29];
  if ( v28 == v8 )
  {
    v31 = v30;
  }
  else
  {
    v31 = (__m128i *)((char *)v30 + (char *)v28 - (char *)v8);
    do
    {
      if ( v30 )
        *v30 = _mm_loadu_si128(v8);
      ++v30;
      ++v8;
    }
    while ( v31 != v30 );
  }
  v5 = (const __m128i *)v119;
  v127 = v31;
  v6 = (__m128i *)v117;
  sub_16CCCB0(v117, (__int64)v119, (__int64)v99);
  v32 = v103;
  v8 = v102;
  v120 = 0;
  v121 = 0;
  v122 = 0;
  v33 = (char *)v103 - (char *)v102;
  if ( v103 == v102 )
  {
    v34 = 0;
  }
  else
  {
    if ( v33 > 0x7FFFFFFFFFFFFFF0LL )
      goto LABEL_163;
    v34 = (__m128i *)sub_22077B0((char *)v103 - (char *)v102);
    v32 = v103;
    v8 = v102;
  }
  v120 = v34;
  v121 = v34;
  v122 = &v34->m128i_i8[v33];
  if ( v32 == v8 )
  {
    v35 = v34;
  }
  else
  {
    v35 = (__m128i *)((char *)v34 + (char *)v32 - (char *)v8);
    do
    {
      if ( v34 )
        *v34 = _mm_loadu_si128(v8);
      ++v34;
      ++v8;
    }
    while ( v35 != v34 );
  }
  v5 = (const __m128i *)v137;
  v6 = &v135;
  v121 = v35;
  sub_16CCCB0(&v135, (__int64)v137, (__int64)v123);
  v36 = v127;
  v8 = v126;
  v138 = 0;
  v139 = 0;
  v140 = 0;
  v37 = (char *)v127 - (char *)v126;
  if ( v127 == v126 )
  {
    v38 = 0;
  }
  else
  {
    if ( v37 > 0x7FFFFFFFFFFFFFF0LL )
      goto LABEL_163;
    v38 = (__m128i *)sub_22077B0((char *)v127 - (char *)v126);
    v36 = v127;
    v8 = v126;
  }
  v138 = v38;
  v139 = v38;
  v140 = &v38->m128i_i8[v37];
  if ( v8 == v36 )
  {
    v39 = v38;
  }
  else
  {
    v39 = (__m128i *)((char *)v38 + (char *)v36 - (char *)v8);
    do
    {
      if ( v38 )
        *v38 = _mm_loadu_si128(v8);
      ++v38;
      ++v8;
    }
    while ( v38 != v39 );
  }
  v5 = (const __m128i *)v131;
  v139 = v39;
  v6 = (__m128i *)v129;
  sub_16CCCB0(v129, (__int64)v131, (__int64)v117);
  v8 = v121;
  v40 = v120;
  v132 = 0;
  v133 = 0;
  v134 = 0;
  v41 = (char *)v121 - (char *)v120;
  if ( v121 == v120 )
  {
    v43 = 0;
    goto LABEL_60;
  }
  if ( v41 > 0x7FFFFFFFFFFFFFF0LL )
LABEL_163:
    sub_4261EA(v6, v5, v8);
  v42 = sub_22077B0((char *)v121 - (char *)v120);
  v8 = v121;
  v40 = v120;
  v43 = (__m128i *)v42;
LABEL_60:
  v132 = v43;
  v133 = v43;
  v134 = &v43->m128i_i8[v41];
  if ( v40 == v8 )
  {
    v45 = v43;
  }
  else
  {
    v44 = v43;
    v45 = (__m128i *)((char *)v43 + (char *)v8 - (char *)v40);
    do
    {
      if ( v44 )
        *v44 = _mm_loadu_si128(v40);
      ++v44;
      ++v40;
    }
    while ( v44 != v45 );
  }
  v133 = v45;
  while ( 1 )
  {
    v46 = (__int64 *)v138;
    if ( (char *)v45 - (char *)v43 != (char *)v139 - (char *)v138 )
      goto LABEL_67;
    if ( v45 == v43 )
      break;
    v48 = v43;
    while ( v48->m128i_i64[0] == *v46 && v48->m128i_i64[1] == v46[1] )
    {
      ++v48;
      v46 += 2;
      if ( v45 == v48 )
        goto LABEL_78;
    }
LABEL_67:
    v47 = v57;
    if ( v57 == v58 )
    {
      sub_1D4AF10((__int64)&v56, v57, (__m128i *)v45[-1].m128i_i64);
      v45 = v133;
    }
    else
    {
      if ( v57 )
      {
        *(_QWORD *)v57 = v45[-1].m128i_i64[0];
        v47 = v57;
        v45 = v133;
      }
      v57 = v47 + 8;
    }
    v43 = v132;
    v133 = --v45;
    if ( v45 != v132 )
    {
      sub_1D53270((__int64)v129);
      v43 = v132;
      v45 = v133;
    }
  }
LABEL_78:
  if ( v43 )
    j_j___libc_free_0(v43, v134 - (__int8 *)v43);
  if ( v130 != v129[1] )
    _libc_free(v130);
  if ( v138 )
    j_j___libc_free_0(v138, v140 - (__int8 *)v138);
  if ( v136 != v135.m128i_i64[1] )
    _libc_free(v136);
  if ( v120 )
    j_j___libc_free_0(v120, v122 - (__int8 *)v120);
  if ( v118 != v117[1] )
    _libc_free(v118);
  if ( v126 )
    j_j___libc_free_0(v126, v128 - (__int8 *)v126);
  if ( v124 != v123[1] )
    _libc_free(v124);
  if ( v102 )
    j_j___libc_free_0(v102, v104 - (__int8 *)v102);
  if ( v100 != v99[1] )
    _libc_free(v100);
  if ( v96 )
    j_j___libc_free_0(v96, v98 - (__int8 *)v96);
  if ( v94 != v93[1] )
    _libc_free(v94);
  if ( v114 )
    j_j___libc_free_0(v114, v116 - (__int8 *)v114);
  if ( v112 != v111[1] )
    _libc_free(v112);
  if ( v108 )
    j_j___libc_free_0(v108, v110 - (__int8 *)v108);
  if ( v106 != v105[1] )
    _libc_free(v106);
  if ( v78 )
    j_j___libc_free_0(v78, v80 - (__int8 *)v78);
  if ( v76 != v75[1] )
    _libc_free(v76);
  if ( v72 )
    j_j___libc_free_0(v72, v74 - (__int8 *)v72);
  if ( v70 != v69[1] )
    _libc_free(v70);
  if ( v90 )
    j_j___libc_free_0(v90, v92 - (__int8 *)v90);
  if ( v88 != v87[1] )
    _libc_free(v88);
  if ( v84 )
    j_j___libc_free_0(v84, v86 - (__int8 *)v84);
  if ( v82 != v81[1] )
    _libc_free(v82);
  if ( v65 )
    j_j___libc_free_0(v65, v67 - (_QWORD)v65);
  if ( v61 != v60 )
    _libc_free((unsigned __int64)v61);
  if ( v68[13] )
    j_j___libc_free_0(v68[13], v68[15] - v68[13]);
  if ( v68[2] != v68[1] )
    _libc_free(v68[2]);
  v49 = (__int64 *)v57;
  v50 = v56;
  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  if ( v50 != v49 )
  {
    v51 = v49 - 1;
    v52 = 0;
    v53 = 0;
    while ( 1 )
    {
      v54 = *v51;
      v135.m128i_i64[0] = *v51;
      if ( v52 == v53 )
      {
        sub_1D4AF10((__int64)a1, v52, &v135);
        if ( v50 == v51 )
          goto LABEL_142;
      }
      else
      {
        if ( v53 )
        {
          *v53 = v54;
          v53 = (__int64 *)a1[1];
        }
        a1[1] = v53 + 1;
        if ( v50 == v51 )
        {
LABEL_142:
          v49 = v56;
          break;
        }
      }
      v53 = (__int64 *)a1[1];
      v52 = (__int64 *)a1[2];
      --v51;
    }
  }
  if ( v49 )
    j_j___libc_free_0(v49, v58 - (_BYTE *)v49);
  return a1;
}
