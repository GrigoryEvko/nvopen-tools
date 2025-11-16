// Function: sub_23FCF20
// Address: 0x23fcf20
//
void sub_23FCF20()
{
  __int64 v0; // r9
  char *v1; // rdx
  __int64 v2; // rax
  const __m128i *v3; // rbx
  __int64 v4; // r15
  unsigned __int64 v5; // r15
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rcx
  unsigned __int64 v8; // rax
  size_t v9; // rcx
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rdx
  size_t v12; // r15
  int v13; // eax
  unsigned int v14; // r9d
  _QWORD *v15; // r10
  __int64 v16; // r9
  char *v17; // rdx
  __int64 v18; // rax
  __m128i *v19; // rdi
  __int64 v20; // r15
  const __m128i *v21; // rbx
  unsigned __int64 v22; // r15
  unsigned __int64 v23; // rax
  unsigned __int64 v24; // rcx
  unsigned __int64 v25; // rax
  size_t v26; // rcx
  unsigned __int64 v27; // rdx
  unsigned __int64 v28; // rdx
  size_t v29; // r15
  int v30; // eax
  unsigned int v31; // r9d
  _QWORD *v32; // r10
  void *v33; // rax
  __int64 v34; // rax
  size_t v35; // rdx
  unsigned __int8 *v36; // rsi
  __int64 v37; // rdi
  __int64 v38; // rax
  void *v39; // rax
  __int64 v40; // rax
  _QWORD *v41; // [rsp+8h] [rbp-D8h]
  _QWORD *v42; // [rsp+8h] [rbp-D8h]
  unsigned int v43; // [rsp+14h] [rbp-CCh]
  unsigned int v44; // [rsp+14h] [rbp-CCh]
  __int64 *src; // [rsp+18h] [rbp-C8h]
  __int64 *srca; // [rsp+18h] [rbp-C8h]
  __int64 *v47; // [rsp+20h] [rbp-C0h]
  _QWORD *v48; // [rsp+20h] [rbp-C0h]
  __int64 *v49; // [rsp+20h] [rbp-C0h]
  _QWORD *v50; // [rsp+20h] [rbp-C0h]
  __m128i *v51; // [rsp+28h] [rbp-B8h]
  const __m128i *v52; // [rsp+28h] [rbp-B8h]
  char *v53; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v54; // [rsp+38h] [rbp-A8h]
  __m128i *v55; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v56; // [rsp+48h] [rbp-98h]
  __m128i v57; // [rsp+50h] [rbp-90h] BYREF
  _QWORD v58[2]; // [rsp+60h] [rbp-80h] BYREF
  char v59; // [rsp+70h] [rbp-70h]
  __int64 *v60; // [rsp+80h] [rbp-60h] BYREF
  unsigned __int64 v61; // [rsp+88h] [rbp-58h]
  __int16 v62; // [rsp+A0h] [rbp-40h]

  if ( !qword_4FE2A50 )
    goto LABEL_2;
  v62 = 260;
  v60 = &qword_4FE2A48;
  sub_C7EA90((__int64)v58, (__int64 *)&v60, 0, 1u, 0, 0);
  if ( (v59 & 1) != 0 )
  {
    v39 = sub_CB72A0();
    v40 = sub_904010((__int64)v39, "Error: Couldn't read the chr-module-list file ");
    v35 = qword_4FE2A50;
    v36 = (unsigned __int8 *)qword_4FE2A48;
    v37 = v40;
LABEL_47:
    v38 = sub_CB6200(v37, v36, v35);
    sub_904010(v38, "\n");
    exit(1);
  }
  v1 = *(char **)(v58[0] + 8LL);
  v2 = *(_QWORD *)(v58[0] + 16LL);
  v55 = &v57;
  v56 = 0;
  v53 = v1;
  v54 = v2 - (_QWORD)v1;
  sub_C93960(&v53, (__int64)&v55, 10, -1, 1, v0);
  v3 = v55;
  v4 = (unsigned int)v56;
  v51 = &v55[v4];
  if ( v55 == &v55[v4] )
    goto LABEL_20;
  do
  {
    while ( 1 )
    {
      v5 = 0;
      v57 = _mm_loadu_si128(v3);
      v6 = sub_C935B0(&v57, byte_3F15413, 6, 0);
      v7 = v57.m128i_u64[1];
      if ( v6 < v57.m128i_i64[1] )
      {
        v5 = v57.m128i_i64[1] - v6;
        v7 = v6;
      }
      v60 = (__int64 *)(v7 + v57.m128i_i64[0]);
      v61 = v5;
      v8 = sub_C93740((__int64 *)&v60, byte_3F15413, 6, 0xFFFFFFFFFFFFFFFFLL);
      v9 = v61;
      v10 = v8 + 1;
      v57.m128i_i64[0] = (__int64)v60;
      if ( v8 + 1 > v61 )
        v10 = v61;
      v11 = v61 - v5 + v10;
      if ( v11 <= v61 )
        v9 = v11;
      v57.m128i_i64[1] = v9;
      v12 = v9;
      if ( v9 )
      {
        v47 = v60;
        v13 = sub_C92610();
        v14 = sub_C92740((__int64)&qword_4FE27B0, v47, v12, v13);
        v15 = (_QWORD *)(qword_4FE27B0 + 8LL * v14);
        if ( !*v15 )
          goto LABEL_18;
        if ( *v15 == -8 )
          break;
      }
      if ( v51 == ++v3 )
        goto LABEL_19;
    }
    LODWORD(qword_4FE27C0) = qword_4FE27C0 - 1;
LABEL_18:
    v41 = (_QWORD *)(qword_4FE27B0 + 8LL * v14);
    ++v3;
    v43 = v14;
    src = v47;
    v48 = (_QWORD *)sub_C7D670(v12 + 9, 8);
    memcpy(v48 + 1, src, v12);
    *((_BYTE *)v48 + v12 + 8) = 0;
    *v48 = v12;
    *v41 = v48;
    ++HIDWORD(qword_4FE27B8);
    sub_C929D0(&qword_4FE27B0, v43);
  }
  while ( v51 != v3 );
LABEL_19:
  v51 = v55;
LABEL_20:
  if ( v51 != &v57 )
    _libc_free((unsigned __int64)v51);
  if ( (v59 & 1) == 0 && v58[0] )
    (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v58[0] + 8LL))(v58[0]);
LABEL_2:
  if ( !qword_4FE2950 )
    return;
  v62 = 260;
  v60 = &qword_4FE2948;
  sub_C7EA90((__int64)v58, (__int64 *)&v60, 0, 1u, 0, 0);
  if ( (v59 & 1) != 0 )
  {
    v33 = sub_CB72A0();
    v34 = sub_904010((__int64)v33, "Error: Couldn't read the chr-function-list file ");
    v35 = qword_4FE2950;
    v36 = (unsigned __int8 *)qword_4FE2948;
    v37 = v34;
    goto LABEL_47;
  }
  v17 = *(char **)(v58[0] + 8LL);
  v18 = *(_QWORD *)(v58[0] + 16LL);
  v55 = &v57;
  v56 = 0;
  v53 = v17;
  v54 = v18 - (_QWORD)v17;
  sub_C93960(&v53, (__int64)&v55, 10, -1, 1, v16);
  v19 = v55;
  v20 = (unsigned int)v56;
  v52 = &v55[v20];
  if ( &v55[v20] == v55 )
    goto LABEL_41;
  v21 = v55;
  while ( 1 )
  {
LABEL_29:
    v22 = 0;
    v57 = _mm_loadu_si128(v21);
    v23 = sub_C935B0(&v57, byte_3F15413, 6, 0);
    v24 = v57.m128i_u64[1];
    if ( v23 < v57.m128i_i64[1] )
    {
      v22 = v57.m128i_i64[1] - v23;
      v24 = v23;
    }
    v60 = (__int64 *)(v24 + v57.m128i_i64[0]);
    v61 = v22;
    v25 = sub_C93740((__int64 *)&v60, byte_3F15413, 6, 0xFFFFFFFFFFFFFFFFLL);
    v26 = v61;
    v27 = v25 + 1;
    v57.m128i_i64[0] = (__int64)v60;
    if ( v25 + 1 > v61 )
      v27 = v61;
    v28 = v61 - v22 + v27;
    if ( v28 <= v61 )
      v26 = v28;
    v57.m128i_i64[1] = v26;
    v29 = v26;
    if ( v26 )
    {
      v49 = v60;
      v30 = sub_C92610();
      v31 = sub_C92740((__int64)&qword_4FE2790, v49, v29, v30);
      v32 = (_QWORD *)(qword_4FE2790 + 8LL * v31);
      if ( !*v32 )
        goto LABEL_39;
      if ( *v32 == -8 )
        break;
    }
    if ( v52 == ++v21 )
      goto LABEL_40;
  }
  LODWORD(qword_4FE27A0) = qword_4FE27A0 - 1;
LABEL_39:
  v42 = (_QWORD *)(qword_4FE2790 + 8LL * v31);
  ++v21;
  v44 = v31;
  srca = v49;
  v50 = (_QWORD *)sub_C7D670(v29 + 9, 8);
  memcpy(v50 + 1, srca, v29);
  *((_BYTE *)v50 + v29 + 8) = 0;
  *v50 = v29;
  *v42 = v50;
  ++HIDWORD(qword_4FE2798);
  sub_C929D0(&qword_4FE2790, v44);
  if ( v52 != v21 )
    goto LABEL_29;
LABEL_40:
  v19 = v55;
LABEL_41:
  if ( v19 != &v57 )
    _libc_free((unsigned __int64)v19);
  if ( (v59 & 1) == 0 )
  {
    if ( v58[0] )
      (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v58[0] + 8LL))(v58[0]);
  }
}
