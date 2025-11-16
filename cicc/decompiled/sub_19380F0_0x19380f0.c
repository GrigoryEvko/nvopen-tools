// Function: sub_19380F0
// Address: 0x19380f0
//
__int64 __fastcall sub_19380F0(__int64 a1, __int64 a2, __int64 a3)
{
  _BYTE *v4; // rsi
  _QWORD *v7; // rdi
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
  _QWORD v61[2]; // [rsp+0h] [rbp-430h] BYREF
  unsigned __int64 v62; // [rsp+10h] [rbp-420h]
  _BYTE v63[64]; // [rsp+28h] [rbp-408h] BYREF
  __m128i *v64; // [rsp+68h] [rbp-3C8h]
  __m128i *v65; // [rsp+70h] [rbp-3C0h]
  __int8 *v66; // [rsp+78h] [rbp-3B8h]
  _QWORD v67[2]; // [rsp+80h] [rbp-3B0h] BYREF
  unsigned __int64 v68; // [rsp+90h] [rbp-3A0h]
  char v69[64]; // [rsp+A8h] [rbp-388h] BYREF
  const __m128i *v70; // [rsp+E8h] [rbp-348h]
  const __m128i *v71; // [rsp+F0h] [rbp-340h]
  __int8 *v72; // [rsp+F8h] [rbp-338h]
  _QWORD v73[2]; // [rsp+100h] [rbp-330h] BYREF
  unsigned __int64 v74; // [rsp+110h] [rbp-320h]
  _BYTE v75[64]; // [rsp+128h] [rbp-308h] BYREF
  __m128i *v76; // [rsp+168h] [rbp-2C8h]
  __m128i *v77; // [rsp+170h] [rbp-2C0h]
  __int8 *v78; // [rsp+178h] [rbp-2B8h]
  _QWORD v79[2]; // [rsp+180h] [rbp-2B0h] BYREF
  unsigned __int64 v80; // [rsp+190h] [rbp-2A0h]
  char v81[64]; // [rsp+1A8h] [rbp-288h] BYREF
  const __m128i *v82; // [rsp+1E8h] [rbp-248h]
  const __m128i *v83; // [rsp+1F0h] [rbp-240h]
  __int8 *v84; // [rsp+1F8h] [rbp-238h]
  _QWORD v85[2]; // [rsp+200h] [rbp-230h] BYREF
  unsigned __int64 v86; // [rsp+210h] [rbp-220h]
  _BYTE v87[64]; // [rsp+228h] [rbp-208h] BYREF
  __m128i *v88; // [rsp+268h] [rbp-1C8h]
  __m128i *v89; // [rsp+270h] [rbp-1C0h]
  __int8 *v90; // [rsp+278h] [rbp-1B8h]
  _QWORD v91[2]; // [rsp+280h] [rbp-1B0h] BYREF
  unsigned __int64 v92; // [rsp+290h] [rbp-1A0h]
  _BYTE v93[64]; // [rsp+2A8h] [rbp-188h] BYREF
  __m128i *v94; // [rsp+2E8h] [rbp-148h]
  __m128i *v95; // [rsp+2F0h] [rbp-140h]
  __int8 *v96; // [rsp+2F8h] [rbp-138h]
  _QWORD v97[2]; // [rsp+300h] [rbp-130h] BYREF
  unsigned __int64 v98; // [rsp+310h] [rbp-120h]
  _BYTE v99[64]; // [rsp+328h] [rbp-108h] BYREF
  __m128i *v100; // [rsp+368h] [rbp-C8h]
  __m128i *v101; // [rsp+370h] [rbp-C0h]
  __int8 *v102; // [rsp+378h] [rbp-B8h]
  _QWORD v103[2]; // [rsp+380h] [rbp-B0h] BYREF
  unsigned __int64 v104; // [rsp+390h] [rbp-A0h]
  _BYTE v105[64]; // [rsp+3A8h] [rbp-88h] BYREF
  __m128i *v106; // [rsp+3E8h] [rbp-48h]
  __m128i *v107; // [rsp+3F0h] [rbp-40h]
  __int8 *v108; // [rsp+3F8h] [rbp-38h]

  v4 = v75;
  v7 = v73;
  sub_16CCCB0(v73, (__int64)v75, a2);
  v9 = *(const __m128i **)(a2 + 112);
  v10 = *(const __m128i **)(a2 + 104);
  v76 = 0;
  v77 = 0;
  v78 = 0;
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
  v76 = v13;
  v77 = v13;
  v78 = &v13->m128i_i8[v11];
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
  v77 = v13;
  sub_16CCEE0(v79, (__int64)v81, 8, (__int64)v73);
  v16 = v76;
  v7 = v61;
  v4 = v63;
  v76 = 0;
  v82 = v16;
  v17 = v77;
  v77 = 0;
  v83 = v17;
  v18 = v78;
  v78 = 0;
  v84 = v18;
  sub_16CCCB0(v61, (__int64)v63, a1);
  v19 = *(const __m128i **)(a1 + 112);
  v20 = *(const __m128i **)(a1 + 104);
  v64 = 0;
  v65 = 0;
  v66 = 0;
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
  v64 = v23;
  v65 = v23;
  v66 = &v23->m128i_i8[v21];
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
  v65 = v23;
  sub_16CCEE0(v67, (__int64)v69, 8, (__int64)v61);
  v26 = v64;
  v7 = v91;
  v4 = v93;
  v64 = 0;
  v70 = v26;
  v27 = v65;
  v65 = 0;
  v71 = v27;
  v28 = v66;
  v66 = 0;
  v72 = v28;
  sub_16CCCB0(v91, (__int64)v93, (__int64)v79);
  v29 = v83;
  v30 = v82;
  v94 = 0;
  v95 = 0;
  v96 = 0;
  v31 = (char *)v83 - (char *)v82;
  if ( v83 == v82 )
  {
    v33 = 0;
  }
  else
  {
    if ( v31 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_108;
    v32 = sub_22077B0((char *)v83 - (char *)v82);
    v29 = v83;
    v30 = v82;
    v33 = (__m128i *)v32;
  }
  v94 = v33;
  v95 = v33;
  v96 = &v33->m128i_i8[v31];
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
  v95 = v33;
  v4 = v87;
  v7 = v85;
  sub_16CCCB0(v85, (__int64)v87, (__int64)v67);
  v36 = v71;
  v37 = v70;
  v88 = 0;
  v89 = 0;
  v90 = 0;
  v38 = (char *)v71 - (char *)v70;
  if ( v71 == v70 )
  {
    v38 = 0;
    v40 = 0;
  }
  else
  {
    if ( v38 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_108;
    v39 = sub_22077B0((char *)v71 - (char *)v70);
    v36 = v71;
    v37 = v70;
    v40 = (__m128i *)v39;
  }
  v88 = v40;
  v89 = v40;
  v90 = &v40->m128i_i8[v38];
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
  v89 = v40;
  v4 = v105;
  v7 = v103;
  sub_16CCCB0(v103, (__int64)v105, (__int64)v91);
  v43 = v95;
  v44 = v94;
  v106 = 0;
  v107 = 0;
  v108 = 0;
  v45 = (char *)v95 - (char *)v94;
  if ( v95 == v94 )
  {
    v47 = 0;
  }
  else
  {
    if ( v45 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_108;
    v46 = sub_22077B0((char *)v95 - (char *)v94);
    v43 = v95;
    v44 = v94;
    v47 = (__m128i *)v46;
  }
  v106 = v47;
  v107 = v47;
  v108 = &v47->m128i_i8[v45];
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
  v107 = v47;
  v4 = v99;
  v7 = v97;
  sub_16CCCB0(v97, (__int64)v99, (__int64)v85);
  v50 = v89;
  v51 = v88;
  v100 = 0;
  v101 = 0;
  v102 = 0;
  v52 = (char *)v89 - (char *)v88;
  if ( v89 == v88 )
  {
    v54 = 0;
    goto LABEL_49;
  }
  if ( v52 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_108:
    sub_4261EA(v7, v4, v8);
  v53 = sub_22077B0((char *)v89 - (char *)v88);
  v50 = v89;
  v51 = v88;
  v54 = (__m128i *)v53;
LABEL_49:
  v100 = v54;
  v55 = v54;
  v101 = v54;
  v102 = &v54->m128i_i8[v52];
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
  v101 = v55;
  while ( 1 )
  {
    v57 = v106;
    if ( (char *)v55 - (char *)v54 != (char *)v107 - (char *)v106 )
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
      sub_1292090(a3, v58, &v55[-2].m128i_i64[1]);
      v55 = v101;
    }
    else
    {
      if ( v58 )
      {
        *(_QWORD *)v58 = v55[-2].m128i_i64[1];
        v58 = *(_BYTE **)(a3 + 8);
        v55 = v101;
      }
      *(_QWORD *)(a3 + 8) = v58 + 8;
    }
    v54 = v100;
    v55 = (__m128i *)((char *)v55 - 24);
    v101 = v55;
    if ( v55 != v100 )
    {
      sub_13FE0F0((__int64)v97);
      v54 = v100;
      v55 = v101;
    }
  }
LABEL_68:
  if ( v54 )
    j_j___libc_free_0(v54, v102 - (__int8 *)v54);
  if ( v98 != v97[1] )
    _libc_free(v98);
  if ( v106 )
    j_j___libc_free_0(v106, v108 - (__int8 *)v106);
  if ( v104 != v103[1] )
    _libc_free(v104);
  if ( v88 )
    j_j___libc_free_0(v88, v90 - (__int8 *)v88);
  if ( v86 != v85[1] )
    _libc_free(v86);
  if ( v94 )
    j_j___libc_free_0(v94, v96 - (__int8 *)v94);
  if ( v92 != v91[1] )
    _libc_free(v92);
  if ( v70 )
    j_j___libc_free_0(v70, v72 - (__int8 *)v70);
  if ( v68 != v67[1] )
    _libc_free(v68);
  if ( v64 )
    j_j___libc_free_0(v64, v66 - (__int8 *)v64);
  if ( v62 != v61[1] )
    _libc_free(v62);
  if ( v82 )
    j_j___libc_free_0(v82, v84 - (__int8 *)v82);
  if ( v80 != v79[1] )
    _libc_free(v80);
  if ( v76 )
    j_j___libc_free_0(v76, v78 - (__int8 *)v76);
  if ( v74 != v73[1] )
    _libc_free(v74);
  return a3;
}
