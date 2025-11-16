// Function: sub_1C2CBE0
// Address: 0x1c2cbe0
//
void __fastcall sub_1C2CBE0(__int64 *a1)
{
  __int64 v2; // rdx
  __int64 v3; // r12
  __m128i *v4; // rax
  const __m128i *v5; // rax
  __m128i *v6; // rax
  __int8 *v7; // rax
  const __m128i *v8; // rax
  __m128i *v9; // rax
  __int8 *v10; // rax
  __m128i *v11; // rax
  unsigned __int64 v12; // rax
  __int8 *v13; // rax
  _BYTE *v14; // rsi
  __int64 *v15; // rdi
  __int64 v16; // rdx
  const __m128i *v17; // rcx
  const __m128i *v18; // r8
  unsigned __int64 v19; // r15
  __int64 v20; // rax
  __m128i *v21; // rdi
  __m128i *v22; // rdx
  const __m128i *v23; // rax
  const __m128i *v24; // rcx
  const __m128i *v25; // r8
  unsigned __int64 v26; // r14
  __int64 v27; // rax
  __m128i *v28; // rdi
  __m128i *v29; // rdx
  const __m128i *v30; // rax
  unsigned __int64 v31; // rcx
  __m128i *v32; // rax
  const __m128i *v33; // rdx
  __m128i *v34; // rcx
  __int64 *v35; // r12
  unsigned __int64 *v36; // r13
  __int64 v37; // rsi
  __int64 v38; // r12
  __int64 i; // r13
  __int64 v40; // rsi
  _QWORD v41[16]; // [rsp+10h] [rbp-330h] BYREF
  __int64 v42; // [rsp+90h] [rbp-2B0h] BYREF
  _QWORD *v43; // [rsp+98h] [rbp-2A8h]
  _QWORD *v44; // [rsp+A0h] [rbp-2A0h]
  __int64 v45; // [rsp+A8h] [rbp-298h]
  int v46; // [rsp+B0h] [rbp-290h]
  _QWORD v47[8]; // [rsp+B8h] [rbp-288h] BYREF
  const __m128i *v48; // [rsp+F8h] [rbp-248h] BYREF
  __m128i *v49; // [rsp+100h] [rbp-240h]
  char *v50; // [rsp+108h] [rbp-238h]
  __int64 v51; // [rsp+110h] [rbp-230h] BYREF
  __int64 v52; // [rsp+118h] [rbp-228h]
  unsigned __int64 v53; // [rsp+120h] [rbp-220h]
  _BYTE v54[64]; // [rsp+138h] [rbp-208h] BYREF
  const __m128i *v55; // [rsp+178h] [rbp-1C8h]
  __m128i *v56; // [rsp+180h] [rbp-1C0h]
  __int8 *v57; // [rsp+188h] [rbp-1B8h]
  __int64 v58; // [rsp+190h] [rbp-1B0h] BYREF
  __int64 v59; // [rsp+198h] [rbp-1A8h]
  unsigned __int64 v60; // [rsp+1A0h] [rbp-1A0h]
  _BYTE v61[64]; // [rsp+1B8h] [rbp-188h] BYREF
  __m128i *v62; // [rsp+1F8h] [rbp-148h]
  unsigned __int64 v63; // [rsp+200h] [rbp-140h]
  __int8 *v64; // [rsp+208h] [rbp-138h]
  __m128i v65; // [rsp+210h] [rbp-130h] BYREF
  unsigned __int64 v66; // [rsp+220h] [rbp-120h] BYREF
  char v67[64]; // [rsp+238h] [rbp-108h] BYREF
  const __m128i *v68; // [rsp+278h] [rbp-C8h]
  const __m128i *v69; // [rsp+280h] [rbp-C0h]
  __int8 *v70; // [rsp+288h] [rbp-B8h]
  _QWORD v71[2]; // [rsp+290h] [rbp-B0h] BYREF
  unsigned __int64 v72; // [rsp+2A0h] [rbp-A0h]
  char v73[64]; // [rsp+2B8h] [rbp-88h] BYREF
  const __m128i *v74; // [rsp+2F8h] [rbp-48h]
  const __m128i *v75; // [rsp+300h] [rbp-40h]
  __int8 *v76; // [rsp+308h] [rbp-38h]

  v2 = *a1;
  memset(v41, 0, sizeof(v41));
  LODWORD(v41[3]) = 8;
  v41[1] = &v41[5];
  v3 = *(_QWORD *)(v2 + 80);
  v41[2] = &v41[5];
  v48 = 0;
  v49 = 0;
  v50 = 0;
  if ( v3 )
    v3 -= 24;
  v43 = v47;
  v44 = v47;
  v47[0] = v3;
  v45 = 0x100000008LL;
  v46 = 0;
  v42 = 1;
  v65.m128i_i64[1] = sub_157EBA0(v3);
  v65.m128i_i64[0] = v3;
  LODWORD(v66) = 0;
  sub_13FDF40(&v48, 0, &v65);
  sub_1B88860((__int64)&v42);
  sub_16CCEE0(&v58, (__int64)v61, 8, (__int64)v41);
  v4 = (__m128i *)v41[13];
  memset(&v41[13], 0, 24);
  v62 = v4;
  v63 = v41[14];
  v64 = (__int8 *)v41[15];
  sub_16CCEE0(&v51, (__int64)v54, 8, (__int64)&v42);
  v5 = v48;
  v48 = 0;
  v55 = v5;
  v6 = v49;
  v49 = 0;
  v56 = v6;
  v7 = v50;
  v50 = 0;
  v57 = v7;
  sub_16CCEE0(&v65, (__int64)v67, 8, (__int64)&v51);
  v8 = v55;
  v55 = 0;
  v68 = v8;
  v9 = v56;
  v56 = 0;
  v69 = v9;
  v10 = v57;
  v57 = 0;
  v70 = v10;
  sub_16CCEE0(v71, (__int64)v73, 8, (__int64)&v58);
  v11 = v62;
  v62 = 0;
  v74 = v11;
  v12 = v63;
  v63 = 0;
  v75 = (const __m128i *)v12;
  v13 = v64;
  v64 = 0;
  v76 = v13;
  if ( v55 )
    j_j___libc_free_0(v55, v57 - (__int8 *)v55);
  if ( v53 != v52 )
    _libc_free(v53);
  if ( v62 )
    j_j___libc_free_0(v62, v64 - (__int8 *)v62);
  if ( v60 != v59 )
    _libc_free(v60);
  if ( v48 )
    j_j___libc_free_0(v48, v50 - (char *)v48);
  if ( v44 != v43 )
    _libc_free((unsigned __int64)v44);
  if ( v41[13] )
    j_j___libc_free_0(v41[13], v41[15] - v41[13]);
  if ( v41[2] != v41[1] )
    _libc_free(v41[2]);
  v14 = v54;
  v15 = &v51;
  sub_16CCCB0(&v51, (__int64)v54, (__int64)&v65);
  v17 = v69;
  v18 = v68;
  v55 = 0;
  v56 = 0;
  v57 = 0;
  v19 = (char *)v69 - (char *)v68;
  if ( v69 == v68 )
  {
    v19 = 0;
    v21 = 0;
  }
  else
  {
    if ( v19 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_76;
    v20 = sub_22077B0((char *)v69 - (char *)v68);
    v17 = v69;
    v18 = v68;
    v21 = (__m128i *)v20;
  }
  v55 = v21;
  v56 = v21;
  v57 = &v21->m128i_i8[v19];
  if ( v18 != v17 )
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
  v14 = v61;
  v56 = v21;
  v15 = &v58;
  sub_16CCCB0(&v58, (__int64)v61, (__int64)v71);
  v24 = v75;
  v25 = v74;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  v26 = (char *)v75 - (char *)v74;
  if ( v75 != v74 )
  {
    if ( v26 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v27 = sub_22077B0((char *)v75 - (char *)v74);
      v24 = v75;
      v25 = v74;
      v28 = (__m128i *)v27;
      goto LABEL_31;
    }
LABEL_76:
    sub_4261EA(v15, v14, v16);
  }
  v28 = 0;
LABEL_31:
  v62 = v28;
  v63 = (unsigned __int64)v28;
  v64 = &v28->m128i_i8[v26];
  if ( v24 == v25 )
  {
    v31 = (unsigned __int64)v28;
  }
  else
  {
    v29 = v28;
    v30 = v25;
    do
    {
      if ( v29 )
      {
        *v29 = _mm_loadu_si128(v30);
        v29[1].m128i_i64[0] = v30[1].m128i_i64[0];
      }
      v30 = (const __m128i *)((char *)v30 + 24);
      v29 = (__m128i *)((char *)v29 + 24);
    }
    while ( v24 != v30 );
    v31 = (unsigned __int64)&v28[1].m128i_u64[((unsigned __int64)((char *)&v24[-2].m128i_u64[1] - (char *)v25) >> 3) + 1];
  }
  v63 = v31;
LABEL_39:
  v32 = v56;
  v33 = v55;
  if ( (char *)v56 - (char *)v55 == v31 - (_QWORD)v28 )
    goto LABEL_42;
  while ( 1 )
  {
    do
    {
      sub_1C2B230((__int64)a1, v32[-2].m128i_i64[1]);
      v56 = (__m128i *)((char *)v56 - 24);
      v33 = v55;
      v32 = v56;
      if ( v56 != v55 )
      {
        sub_1B88860((__int64)&v51);
        v28 = v62;
        v31 = v63;
        goto LABEL_39;
      }
      v28 = v62;
    }
    while ( (char *)v56 - (char *)v55 != v63 - (_QWORD)v62 );
LABEL_42:
    if ( v33 == v32 )
      break;
    v34 = v28;
    while ( v33->m128i_i64[0] == v34->m128i_i64[0] && v33[1].m128i_i32[0] == v34[1].m128i_i32[0] )
    {
      v33 = (const __m128i *)((char *)v33 + 24);
      v34 = (__m128i *)((char *)v34 + 24);
      if ( v33 == v32 )
        goto LABEL_47;
    }
  }
LABEL_47:
  if ( v28 )
    j_j___libc_free_0(v28, v64 - (__int8 *)v28);
  if ( v60 != v59 )
    _libc_free(v60);
  if ( v55 )
    j_j___libc_free_0(v55, v57 - (__int8 *)v55);
  if ( v53 != v52 )
    _libc_free(v53);
  if ( v74 )
    j_j___libc_free_0(v74, v76 - (__int8 *)v74);
  if ( v72 != v71[1] )
    _libc_free(v72);
  if ( v68 )
    j_j___libc_free_0(v68, v70 - (__int8 *)v68);
  if ( v66 != v65.m128i_i64[1] )
    _libc_free(v66);
  sub_13FB8B0((__int64)&v65, a1[1]);
  v35 = (__int64 *)v65.m128i_i64[0];
  v36 = (unsigned __int64 *)(v65.m128i_i64[0] + 8LL * v65.m128i_u32[2]);
  if ( (unsigned __int64 *)v65.m128i_i64[0] != v36 )
  {
    do
    {
      v37 = *v35++;
      sub_1C2BFA0((__int64)a1, v37);
    }
    while ( v36 != (unsigned __int64 *)v35 );
    v36 = (unsigned __int64 *)v65.m128i_i64[0];
  }
  if ( v36 != &v66 )
    _libc_free((unsigned __int64)v36);
  v38 = *(_QWORD *)(*a1 + 80);
  for ( i = *a1 + 72; i != v38; v38 = *(_QWORD *)(v38 + 8) )
  {
    v40 = v38 - 24;
    if ( !v38 )
      v40 = 0;
    sub_1C2AD00((__int64)a1, v40);
  }
}
