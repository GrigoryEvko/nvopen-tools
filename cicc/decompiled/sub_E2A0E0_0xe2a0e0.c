// Function: sub_E2A0E0
// Address: 0xe2a0e0
//
__int64 __fastcall sub_E2A0E0(unsigned int *a1, __int64 *a2)
{
  char *v4; // rsi
  unsigned __int64 v5; // rax
  char *v6; // rdi
  unsigned __int64 v7; // rsi
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  __m128i *v10; // rdi
  _BYTE *v11; // r12
  __int64 v12; // r8
  unsigned __int64 v13; // rcx
  unsigned __int64 v14; // rax
  _BYTE *v15; // r14
  unsigned __int64 v16; // rax
  char *v17; // rdi
  unsigned __int64 v18; // rax
  __int64 v19; // rax
  _BYTE *v20; // r12
  __int64 v21; // r8
  int v22; // r10d
  unsigned __int64 v23; // rcx
  _BYTE *v24; // rdi
  unsigned __int64 v25; // rax
  _BYTE *v26; // r14
  unsigned __int64 v27; // rax
  char *v28; // rdi
  unsigned __int64 v29; // rax
  __int64 v30; // rax
  _BYTE *v31; // r12
  __int64 v32; // r8
  unsigned __int64 v33; // rcx
  unsigned __int64 v34; // rax
  _BYTE *v35; // r14
  unsigned __int64 v36; // rax
  char *v37; // rdi
  unsigned __int64 v38; // rax
  __int64 v39; // rax
  _BYTE *v40; // r12
  signed __int64 v41; // r8
  unsigned __int64 v42; // rcx
  unsigned __int64 v43; // rax
  _BYTE *v44; // r9
  unsigned __int64 v45; // rax
  __int64 v46; // rdi
  unsigned __int64 v47; // rax
  __int64 v48; // rax
  char *v50; // rax
  char *v51; // rdi
  unsigned __int64 v52; // rsi
  unsigned __int64 v53; // rax
  __int64 v54; // rax
  unsigned __int64 v55; // rax
  char *v56; // rdi
  unsigned __int64 v57; // rsi
  unsigned __int64 v58; // rax
  __int64 v59; // rax
  char *v60; // r14
  unsigned __int64 v61; // rax
  char *v62; // rdi
  unsigned __int64 v63; // rsi
  unsigned __int64 v64; // rax
  __int64 v65; // rax
  char *v66; // r14
  unsigned __int64 v67; // rax
  char *v68; // rdi
  unsigned __int64 v69; // rsi
  unsigned __int64 v70; // rax
  __int64 v71; // rax
  char *v72; // r14
  _BYTE v73[32]; // [rsp+15h] [rbp-8Bh] BYREF
  _BYTE v74[32]; // [rsp+35h] [rbp-6Bh] BYREF
  _BYTE v75[32]; // [rsp+55h] [rbp-4Bh] BYREF
  _BYTE v76[43]; // [rsp+75h] [rbp-2Bh] BYREF

  v4 = (char *)a2[1];
  v5 = a2[2];
  v6 = (char *)*a2;
  if ( (unsigned __int64)(v4 + 32) > v5 )
  {
    v7 = (unsigned __int64)(v4 + 1024);
    v8 = 2 * v5;
    if ( v7 <= v8 )
      a2[2] = v8;
    else
      a2[2] = v7;
    v9 = realloc(v6);
    *a2 = v9;
    v6 = (char *)v9;
    if ( !v9 )
      goto LABEL_74;
    v4 = (char *)a2[1];
  }
  v10 = (__m128i *)&v6[(_QWORD)v4];
  v11 = v76;
  *v10 = _mm_load_si128((const __m128i *)&xmmword_3F7CA00);
  v10[1] = _mm_load_si128((const __m128i *)&xmmword_3F7CA10);
  v12 = a2[1] + 32;
  a2[1] = v12;
  v13 = a1[6];
  do
  {
    *--v11 = v13 % 0xA + 48;
    v14 = v13;
    v13 /= 0xAu;
  }
  while ( v14 > 9 );
  v15 = (_BYTE *)(v76 - v11);
  if ( v76 != v11 )
  {
    v67 = a2[2];
    v68 = (char *)*a2;
    if ( (unsigned __int64)&v15[v12] > v67 )
    {
      v69 = (unsigned __int64)&v15[v12 + 992];
      v70 = 2 * v67;
      if ( v69 > v70 )
        a2[2] = v69;
      else
        a2[2] = v70;
      v71 = realloc(v68);
      *a2 = v71;
      v68 = (char *)v71;
      if ( !v71 )
        goto LABEL_74;
      v12 = a2[1];
    }
    memcpy(&v68[v12], v11, (size_t)v15);
    v72 = &v15[a2[1]];
    a2[1] = (__int64)v72;
    v12 = (__int64)v72;
  }
  v16 = a2[2];
  v17 = (char *)*a2;
  if ( v12 + 2 > v16 )
  {
    v18 = 2 * v16;
    if ( v12 + 994 <= v18 )
      a2[2] = v18;
    else
      a2[2] = v12 + 994;
    v19 = realloc(v17);
    *a2 = v19;
    v17 = (char *)v19;
    if ( !v19 )
      goto LABEL_74;
    v12 = a2[1];
  }
  *(_WORD *)&v17[v12] = 8236;
  v20 = v75;
  v21 = a2[1] + 2;
  a2[1] = v21;
  v22 = a1[7];
  v23 = abs32(v22);
  do
  {
    v24 = v20--;
    *v20 = v23 % 0xA + 48;
    v25 = v23;
    v23 /= 0xAu;
  }
  while ( v25 > 9 );
  if ( v22 < 0 )
  {
    *(v20 - 1) = 45;
    v20 = v24 - 2;
  }
  v26 = (_BYTE *)(v75 - v20);
  if ( v75 != v20 )
  {
    v61 = a2[2];
    v62 = (char *)*a2;
    if ( (unsigned __int64)&v26[v21] > v61 )
    {
      v63 = (unsigned __int64)&v26[v21 + 992];
      v64 = 2 * v61;
      if ( v63 > v64 )
        a2[2] = v63;
      else
        a2[2] = v64;
      v65 = realloc(v62);
      *a2 = v65;
      v62 = (char *)v65;
      if ( !v65 )
        goto LABEL_74;
      v21 = a2[1];
    }
    memcpy(&v62[v21], v20, v75 - v20);
    v66 = &v26[a2[1]];
    a2[1] = (__int64)v66;
    v21 = (__int64)v66;
  }
  v27 = a2[2];
  v28 = (char *)*a2;
  if ( v21 + 2 > v27 )
  {
    v29 = 2 * v27;
    if ( v21 + 994 <= v29 )
      a2[2] = v29;
    else
      a2[2] = v21 + 994;
    v30 = realloc(v28);
    *a2 = v30;
    v28 = (char *)v30;
    if ( !v30 )
      goto LABEL_74;
    v21 = a2[1];
  }
  *(_WORD *)&v28[v21] = 8236;
  v31 = v74;
  v32 = a2[1] + 2;
  a2[1] = v32;
  v33 = a1[8];
  do
  {
    *--v31 = v33 % 0xA + 48;
    v34 = v33;
    v33 /= 0xAu;
  }
  while ( v34 > 9 );
  v35 = (_BYTE *)(v74 - v31);
  if ( v74 != v31 )
  {
    v55 = a2[2];
    v56 = (char *)*a2;
    if ( (unsigned __int64)&v35[v32] > v55 )
    {
      v57 = (unsigned __int64)&v35[v32 + 992];
      v58 = 2 * v55;
      if ( v57 > v58 )
        a2[2] = v57;
      else
        a2[2] = v58;
      v59 = realloc(v56);
      *a2 = v59;
      v56 = (char *)v59;
      if ( !v59 )
        goto LABEL_74;
      v32 = a2[1];
    }
    memcpy(&v56[v32], v31, v74 - v31);
    v60 = &v35[a2[1]];
    a2[1] = (__int64)v60;
    v32 = (__int64)v60;
  }
  v36 = a2[2];
  v37 = (char *)*a2;
  if ( v32 + 2 > v36 )
  {
    v38 = 2 * v36;
    if ( v32 + 994 <= v38 )
      a2[2] = v38;
    else
      a2[2] = v32 + 994;
    v39 = realloc(v37);
    *a2 = v39;
    v37 = (char *)v39;
    if ( !v39 )
      goto LABEL_74;
    v32 = a2[1];
  }
  *(_WORD *)&v37[v32] = 8236;
  v40 = v73;
  v41 = a2[1] + 2;
  a2[1] = v41;
  v42 = a1[9];
  do
  {
    *--v40 = v42 % 0xA + 48;
    v43 = v42;
    v42 /= 0xAu;
  }
  while ( v43 > 9 );
  v44 = (_BYTE *)(v73 - v40);
  if ( v73 != v40 )
  {
    v50 = (char *)a2[2];
    v51 = (char *)*a2;
    if ( &v44[v41] > v50 )
    {
      v52 = (unsigned __int64)&v44[v41 + 992];
      v53 = 2LL * (_QWORD)v50;
      if ( v52 > v53 )
        a2[2] = v52;
      else
        a2[2] = v53;
      v54 = realloc(v51);
      *a2 = v54;
      v51 = (char *)v54;
      if ( !v54 )
        goto LABEL_74;
      v41 = a2[1];
    }
    memcpy(&v51[v41], v40, v73 - v40);
    v41 = v73 - v40 + a2[1];
    a2[1] = v41;
  }
  v45 = a2[2];
  v46 = *a2;
  if ( v41 + 2 > v45 )
  {
    v47 = 2 * v45;
    if ( v41 + 994 <= v47 )
      a2[2] = v47;
    else
      a2[2] = v41 + 994;
    v48 = realloc((void *)v46);
    *a2 = v48;
    v46 = v48;
    if ( v48 )
    {
      v41 = a2[1];
      goto LABEL_40;
    }
LABEL_74:
    abort();
  }
LABEL_40:
  *(_WORD *)(v46 + v41) = 10025;
  a2[1] += 2;
  return 10025;
}
