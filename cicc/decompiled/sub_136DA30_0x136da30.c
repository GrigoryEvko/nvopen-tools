// Function: sub_136DA30
// Address: 0x136da30
//
__int64 __fastcall sub_136DA30(__int64 a1, __int64 a2, __int64 a3)
{
  _BYTE *v4; // rsi
  char *v7; // rdi
  __int64 v8; // rdx
  const __m128i *v9; // rcx
  const __m128i *v10; // r8
  unsigned __int64 v11; // r15
  __int64 v12; // rax
  __m128i *v13; // rdi
  __m128i *v14; // rdx
  const __m128i *v15; // rax
  __m128i *v16; // rax
  __m128i *v17; // rax
  __int8 *v18; // rax
  const __m128i *v19; // rcx
  const __m128i *v20; // r8
  unsigned __int64 v21; // r15
  __int64 v22; // rax
  __m128i *v23; // rdi
  __m128i *v24; // rdx
  const __m128i *v25; // rax
  __m128i *v26; // rax
  __m128i *v27; // rax
  __int8 *v28; // rax
  const __m128i *v29; // rcx
  const __m128i *v30; // r8
  unsigned __int64 v31; // rbx
  __int64 v32; // rax
  __m128i *v33; // rdi
  __m128i *v34; // rdx
  const __m128i *v35; // rax
  const __m128i *v36; // rcx
  const __m128i *v37; // r8
  unsigned __int64 v38; // rbx
  __int64 v39; // rax
  __m128i *v40; // rdi
  __m128i *v41; // rdx
  const __m128i *v42; // rax
  const __m128i *v43; // rcx
  const __m128i *v44; // r8
  unsigned __int64 v45; // rbx
  __int64 v46; // rax
  __m128i *v47; // rdi
  __m128i *v48; // rdx
  const __m128i *v49; // rax
  const __m128i *v50; // rcx
  const __m128i *v51; // r8
  unsigned __int64 v52; // r13
  __int64 v53; // rax
  __m128i *v54; // rdi
  __m128i *v55; // rdx
  const __m128i *v56; // rax
  __m128i *v57; // rcx
  _BYTE *v58; // rsi
  __m128i *v59; // rax
  char v61[8]; // [rsp+0h] [rbp-430h] BYREF
  __int64 v62; // [rsp+8h] [rbp-428h]
  unsigned __int64 v63; // [rsp+10h] [rbp-420h]
  _BYTE v64[64]; // [rsp+28h] [rbp-408h] BYREF
  __m128i *v65; // [rsp+68h] [rbp-3C8h]
  __m128i *v66; // [rsp+70h] [rbp-3C0h]
  __int8 *v67; // [rsp+78h] [rbp-3B8h]
  char v68[8]; // [rsp+80h] [rbp-3B0h] BYREF
  __int64 v69; // [rsp+88h] [rbp-3A8h]
  unsigned __int64 v70; // [rsp+90h] [rbp-3A0h]
  char v71[64]; // [rsp+A8h] [rbp-388h] BYREF
  const __m128i *v72; // [rsp+E8h] [rbp-348h]
  const __m128i *v73; // [rsp+F0h] [rbp-340h]
  __int8 *v74; // [rsp+F8h] [rbp-338h]
  char v75[8]; // [rsp+100h] [rbp-330h] BYREF
  __int64 v76; // [rsp+108h] [rbp-328h]
  unsigned __int64 v77; // [rsp+110h] [rbp-320h]
  _BYTE v78[64]; // [rsp+128h] [rbp-308h] BYREF
  __m128i *v79; // [rsp+168h] [rbp-2C8h]
  __m128i *v80; // [rsp+170h] [rbp-2C0h]
  __int8 *v81; // [rsp+178h] [rbp-2B8h]
  char v82[8]; // [rsp+180h] [rbp-2B0h] BYREF
  __int64 v83; // [rsp+188h] [rbp-2A8h]
  unsigned __int64 v84; // [rsp+190h] [rbp-2A0h]
  char v85[64]; // [rsp+1A8h] [rbp-288h] BYREF
  const __m128i *v86; // [rsp+1E8h] [rbp-248h]
  const __m128i *v87; // [rsp+1F0h] [rbp-240h]
  __int8 *v88; // [rsp+1F8h] [rbp-238h]
  char v89[8]; // [rsp+200h] [rbp-230h] BYREF
  __int64 v90; // [rsp+208h] [rbp-228h]
  unsigned __int64 v91; // [rsp+210h] [rbp-220h]
  _BYTE v92[64]; // [rsp+228h] [rbp-208h] BYREF
  __m128i *v93; // [rsp+268h] [rbp-1C8h]
  __m128i *v94; // [rsp+270h] [rbp-1C0h]
  __int8 *v95; // [rsp+278h] [rbp-1B8h]
  char v96[8]; // [rsp+280h] [rbp-1B0h] BYREF
  __int64 v97; // [rsp+288h] [rbp-1A8h]
  unsigned __int64 v98; // [rsp+290h] [rbp-1A0h]
  _BYTE v99[64]; // [rsp+2A8h] [rbp-188h] BYREF
  __m128i *v100; // [rsp+2E8h] [rbp-148h]
  __m128i *v101; // [rsp+2F0h] [rbp-140h]
  __int8 *v102; // [rsp+2F8h] [rbp-138h]
  char v103[8]; // [rsp+300h] [rbp-130h] BYREF
  __int64 v104; // [rsp+308h] [rbp-128h]
  unsigned __int64 v105; // [rsp+310h] [rbp-120h]
  _BYTE v106[64]; // [rsp+328h] [rbp-108h] BYREF
  __m128i *v107; // [rsp+368h] [rbp-C8h]
  __m128i *v108; // [rsp+370h] [rbp-C0h]
  __int8 *v109; // [rsp+378h] [rbp-B8h]
  char v110[8]; // [rsp+380h] [rbp-B0h] BYREF
  __int64 v111; // [rsp+388h] [rbp-A8h]
  unsigned __int64 v112; // [rsp+390h] [rbp-A0h]
  _BYTE v113[64]; // [rsp+3A8h] [rbp-88h] BYREF
  __m128i *v114; // [rsp+3E8h] [rbp-48h]
  __m128i *v115; // [rsp+3F0h] [rbp-40h]
  __int8 *v116; // [rsp+3F8h] [rbp-38h]

  v4 = v78;
  v7 = v75;
  sub_16CCCB0(v75, v78, a2);
  v9 = *(const __m128i **)(a2 + 112);
  v10 = *(const __m128i **)(a2 + 104);
  v79 = 0;
  v80 = 0;
  v81 = 0;
  v11 = (char *)v9 - (char *)v10;
  if ( v9 == v10 )
  {
    v13 = 0;
  }
  else
  {
    if ( v11 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_108;
    v12 = sub_22077B0((char *)v9 - (char *)v10);
    v9 = *(const __m128i **)(a2 + 112);
    v10 = *(const __m128i **)(a2 + 104);
    v13 = (__m128i *)v12;
  }
  v79 = v13;
  v80 = v13;
  v81 = &v13->m128i_i8[v11];
  if ( v10 != v9 )
  {
    v14 = v13;
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
    v13 = (__m128i *)((char *)v13 + 8 * ((unsigned __int64)((char *)&v15[-2].m128i_u64[1] - (char *)v10) >> 3) + 24);
  }
  v80 = v13;
  sub_16CCEE0(v82, v85, 8, v75);
  v16 = v79;
  v7 = v61;
  v4 = v64;
  v79 = 0;
  v86 = v16;
  v17 = v80;
  v80 = 0;
  v87 = v17;
  v18 = v81;
  v81 = 0;
  v88 = v18;
  sub_16CCCB0(v61, v64, a1);
  v19 = *(const __m128i **)(a1 + 112);
  v20 = *(const __m128i **)(a1 + 104);
  v65 = 0;
  v66 = 0;
  v67 = 0;
  v21 = (char *)v19 - (char *)v20;
  if ( v19 == v20 )
  {
    v23 = 0;
  }
  else
  {
    if ( v21 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_108;
    v22 = sub_22077B0((char *)v19 - (char *)v20);
    v19 = *(const __m128i **)(a1 + 112);
    v20 = *(const __m128i **)(a1 + 104);
    v23 = (__m128i *)v22;
  }
  v65 = v23;
  v66 = v23;
  v67 = &v23->m128i_i8[v21];
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
  v66 = v23;
  sub_16CCEE0(v68, v71, 8, v61);
  v26 = v65;
  v7 = v96;
  v4 = v99;
  v65 = 0;
  v72 = v26;
  v27 = v66;
  v66 = 0;
  v73 = v27;
  v28 = v67;
  v67 = 0;
  v74 = v28;
  sub_16CCCB0(v96, v99, v82);
  v29 = v87;
  v30 = v86;
  v100 = 0;
  v101 = 0;
  v102 = 0;
  v31 = (char *)v87 - (char *)v86;
  if ( v87 == v86 )
  {
    v33 = 0;
  }
  else
  {
    if ( v31 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_108;
    v32 = sub_22077B0((char *)v87 - (char *)v86);
    v29 = v87;
    v30 = v86;
    v33 = (__m128i *)v32;
  }
  v100 = v33;
  v101 = v33;
  v102 = &v33->m128i_i8[v31];
  if ( v30 != v29 )
  {
    v34 = v33;
    v35 = v30;
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
    while ( v29 != v35 );
    v33 = (__m128i *)((char *)v33 + 8 * ((unsigned __int64)((char *)&v29[-2].m128i_u64[1] - (char *)v30) >> 3) + 24);
  }
  v101 = v33;
  v4 = v92;
  v7 = v89;
  sub_16CCCB0(v89, v92, v68);
  v36 = v73;
  v37 = v72;
  v93 = 0;
  v94 = 0;
  v95 = 0;
  v38 = (char *)v73 - (char *)v72;
  if ( v73 == v72 )
  {
    v38 = 0;
    v40 = 0;
  }
  else
  {
    if ( v38 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_108;
    v39 = sub_22077B0((char *)v73 - (char *)v72);
    v36 = v73;
    v37 = v72;
    v40 = (__m128i *)v39;
  }
  v93 = v40;
  v94 = v40;
  v95 = &v40->m128i_i8[v38];
  if ( v36 != v37 )
  {
    v41 = v40;
    v42 = v37;
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
    while ( v36 != v42 );
    v40 = (__m128i *)((char *)v40 + 8 * ((unsigned __int64)((char *)&v36[-2].m128i_u64[1] - (char *)v37) >> 3) + 24);
  }
  v94 = v40;
  v4 = v113;
  v7 = v110;
  sub_16CCCB0(v110, v113, v96);
  v43 = v101;
  v44 = v100;
  v114 = 0;
  v115 = 0;
  v116 = 0;
  v45 = (char *)v101 - (char *)v100;
  if ( v101 == v100 )
  {
    v47 = 0;
  }
  else
  {
    if ( v45 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_108;
    v46 = sub_22077B0((char *)v101 - (char *)v100);
    v43 = v101;
    v44 = v100;
    v47 = (__m128i *)v46;
  }
  v114 = v47;
  v115 = v47;
  v116 = &v47->m128i_i8[v45];
  if ( v44 != v43 )
  {
    v48 = v47;
    v49 = v44;
    do
    {
      if ( v48 )
      {
        *v48 = _mm_loadu_si128(v49);
        v48[1].m128i_i64[0] = v49[1].m128i_i64[0];
      }
      v49 = (const __m128i *)((char *)v49 + 24);
      v48 = (__m128i *)((char *)v48 + 24);
    }
    while ( v43 != v49 );
    v47 = (__m128i *)((char *)v47 + 8 * ((unsigned __int64)((char *)&v43[-2].m128i_u64[1] - (char *)v44) >> 3) + 24);
  }
  v115 = v47;
  v4 = v106;
  v7 = v103;
  sub_16CCCB0(v103, v106, v89);
  v50 = v94;
  v51 = v93;
  v107 = 0;
  v108 = 0;
  v109 = 0;
  v52 = (char *)v94 - (char *)v93;
  if ( v94 == v93 )
  {
    v54 = 0;
    goto LABEL_49;
  }
  if ( v52 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_108:
    sub_4261EA(v7, v4, v8);
  v53 = sub_22077B0((char *)v94 - (char *)v93);
  v50 = v94;
  v51 = v93;
  v54 = (__m128i *)v53;
LABEL_49:
  v107 = v54;
  v55 = v54;
  v108 = v54;
  v109 = &v54->m128i_i8[v52];
  if ( v51 != v50 )
  {
    v56 = v51;
    do
    {
      if ( v55 )
      {
        *v55 = _mm_loadu_si128(v56);
        v55[1].m128i_i64[0] = v56[1].m128i_i64[0];
      }
      v56 = (const __m128i *)((char *)v56 + 24);
      v55 = (__m128i *)((char *)v55 + 24);
    }
    while ( v50 != v56 );
    v55 = (__m128i *)((char *)v54 + 8 * ((unsigned __int64)((char *)&v50[-2].m128i_u64[1] - (char *)v51) >> 3) + 24);
  }
  v108 = v55;
  while ( 1 )
  {
    v57 = v114;
    if ( (char *)v55 - (char *)v54 != (char *)v115 - (char *)v114 )
      goto LABEL_57;
    if ( v54 == v55 )
      break;
    v59 = v54;
    while ( v59->m128i_i64[0] == v57->m128i_i64[0] && v59[1].m128i_i32[0] == v57[1].m128i_i32[0] )
    {
      v59 = (__m128i *)((char *)v59 + 24);
      v57 = (__m128i *)((char *)v57 + 24);
      if ( v55 == v59 )
        goto LABEL_68;
    }
LABEL_57:
    v58 = *(_BYTE **)(a3 + 8);
    if ( v58 == *(_BYTE **)(a3 + 16) )
    {
      sub_136D8A0(a3, v58, &v55[-2].m128i_i64[1]);
      v55 = v108;
    }
    else
    {
      if ( v58 )
      {
        *(_QWORD *)v58 = v55[-2].m128i_i64[1];
        v58 = *(_BYTE **)(a3 + 8);
        v55 = v108;
      }
      *(_QWORD *)(a3 + 8) = v58 + 8;
    }
    v54 = v107;
    v55 = (__m128i *)((char *)v55 - 24);
    v108 = v55;
    if ( v55 != v107 )
    {
      sub_136D710((__int64)v103);
      v54 = v107;
      v55 = v108;
    }
  }
LABEL_68:
  if ( v54 )
    j_j___libc_free_0(v54, v109 - (__int8 *)v54);
  if ( v105 != v104 )
    _libc_free(v105);
  if ( v114 )
    j_j___libc_free_0(v114, v116 - (__int8 *)v114);
  if ( v112 != v111 )
    _libc_free(v112);
  if ( v93 )
    j_j___libc_free_0(v93, v95 - (__int8 *)v93);
  if ( v91 != v90 )
    _libc_free(v91);
  if ( v100 )
    j_j___libc_free_0(v100, v102 - (__int8 *)v100);
  if ( v98 != v97 )
    _libc_free(v98);
  if ( v72 )
    j_j___libc_free_0(v72, v74 - (__int8 *)v72);
  if ( v70 != v69 )
    _libc_free(v70);
  if ( v65 )
    j_j___libc_free_0(v65, v67 - (__int8 *)v65);
  if ( v63 != v62 )
    _libc_free(v63);
  if ( v86 )
    j_j___libc_free_0(v86, v88 - (__int8 *)v86);
  if ( v84 != v83 )
    _libc_free(v84);
  if ( v79 )
    j_j___libc_free_0(v79, v81 - (__int8 *)v79);
  if ( v77 != v76 )
    _libc_free(v77);
  return a3;
}
