// Function: sub_23BE720
// Address: 0x23be720
//
void __fastcall sub_23BE720(__m128i **a1, __int64 a2)
{
  _BYTE *v2; // rax
  __int64 v3; // rdx
  void *v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rbx
  __m128i *v7; // rdi
  __int64 v8; // rdx
  int v9; // eax
  size_t v10; // r15
  unsigned int v11; // r8d
  _QWORD *v12; // r9
  size_t v13; // rdx
  size_t v14; // rdx
  char *v15; // rax
  size_t v16; // rdx
  __m128i *v17; // r14
  unsigned __int64 v18; // r13
  unsigned __int64 v19; // r15
  __int64 v20; // rax
  __int64 *v21; // rbx
  __int64 v22; // r13
  __int64 v23; // r8
  __int64 v24; // r12
  __int64 v25; // r14
  __int64 v26; // rax
  _QWORD *v27; // rax
  size_t v28; // rcx
  size_t *v29; // rbx
  size_t **v30; // r15
  char *v31; // rdx
  __int64 v32; // rax
  int v33; // eax
  unsigned __int64 v34; // rdi
  __int64 v35; // r14
  __int64 v36; // rbx
  _QWORD *v37; // r12
  unsigned __int64 v38; // rdi
  __int64 v39; // r13
  unsigned __int64 v40; // rdi
  __int64 v41; // rbx
  unsigned __int64 *v42; // r12
  unsigned __int64 v43; // rdi
  __int64 v44; // r14
  __int64 v45; // rbx
  _QWORD *v46; // r12
  unsigned __int64 v47; // rdi
  __int64 v48; // r13
  unsigned __int64 v49; // rdi
  unsigned __int64 *v50; // rbx
  unsigned __int64 *v51; // r12
  __int64 *v53; // [rsp+50h] [rbp-180h]
  size_t v54; // [rsp+58h] [rbp-178h]
  __int64 v55; // [rsp+68h] [rbp-168h]
  char *v56; // [rsp+70h] [rbp-160h]
  size_t v57; // [rsp+70h] [rbp-160h]
  _QWORD *v58; // [rsp+70h] [rbp-160h]
  unsigned int v59; // [rsp+70h] [rbp-160h]
  __int64 v60; // [rsp+78h] [rbp-158h]
  int v61; // [rsp+8Ch] [rbp-144h] BYREF
  char *v62; // [rsp+90h] [rbp-140h] BYREF
  size_t v63; // [rsp+98h] [rbp-138h]
  __int64 v64; // [rsp+A0h] [rbp-130h] BYREF
  void *v65; // [rsp+B0h] [rbp-120h] BYREF
  __int64 v66; // [rsp+B8h] [rbp-118h]
  __int64 v67; // [rsp+C0h] [rbp-110h]
  __int64 v68; // [rsp+C8h] [rbp-108h]
  __int64 v69; // [rsp+D0h] [rbp-100h]
  __int64 v70; // [rsp+D8h] [rbp-F8h]
  __int128 *v71; // [rsp+E0h] [rbp-F0h]
  unsigned __int64 *v72; // [rsp+F0h] [rbp-E0h] BYREF
  __m128i *v73; // [rsp+F8h] [rbp-D8h]
  __m128i *v74; // [rsp+100h] [rbp-D0h]
  unsigned __int64 v75; // [rsp+108h] [rbp-C8h] BYREF
  __int128 v76; // [rsp+110h] [rbp-C0h]
  __int64 v77[2]; // [rsp+120h] [rbp-B0h] BYREF
  __int64 v78; // [rsp+130h] [rbp-A0h] BYREF
  char *v79; // [rsp+140h] [rbp-90h] BYREF
  size_t v80; // [rsp+148h] [rbp-88h]
  __m128i v81; // [rsp+150h] [rbp-80h] BYREF
  __m128i v82; // [rsp+160h] [rbp-70h] BYREF
  __int128 v83; // [rsp+170h] [rbp-60h] BYREF
  __m128i v84; // [rsp+180h] [rbp-50h] BYREF
  __int64 v85; // [rsp+190h] [rbp-40h] BYREF

  v2 = (_BYTE *)sub_2E791E0(a2);
  if ( !sub_BC63A0(v2, v3) )
    return;
  v4 = (void *)sub_2E31BC0(*(_QWORD *)(a2 + 328));
  v66 = v5;
  v65 = v4;
  sub_95CA80((__int64 *)&v79, (__int64)&v65);
  v72 = 0;
  *((_QWORD *)&v76 + 1) = 0x5000000000LL;
  v73 = 0;
  v74 = 0;
  v75 = 0;
  *(_QWORD *)&v76 = 0;
  sub_2241BD0(v77, (__int64)&v79);
  sub_2240A30((unsigned __int64 *)&v79);
  v6 = *(_QWORD *)(a2 + 328);
  v61 = 0;
  while ( a2 + 320 != v6 )
  {
    v79 = (char *)sub_2E31BC0(v6);
    v80 = v13;
    sub_95CA80((__int64 *)&v62, (__int64)&v79);
    if ( v63 )
    {
      v7 = v73;
      if ( v73 == v74 )
        goto LABEL_20;
    }
    else
    {
      v80 = 3;
      v79 = "{0}";
      v81.m128i_i64[1] = 1;
      v81.m128i_i64[0] = (__int64)&v83 + 8;
      v82.m128i_i8[0] = 1;
      v82.m128i_i64[1] = (__int64)&unk_4A15FA0;
      *(_QWORD *)&v83 = &v61;
      *((_QWORD *)&v83 + 1) = &v82.m128i_i64[1];
      sub_23328D0((__int64)&v65, (__int64)&v79);
      sub_23AEBB0((__int64)&v62, (__int64)&v65);
      sub_2240A30((unsigned __int64 *)&v65);
      ++v61;
      v7 = v73;
      if ( v73 == v74 )
      {
LABEL_20:
        sub_23BB3E0((unsigned __int64 *)&v72, v7, (__int64)&v62);
        goto LABEL_9;
      }
    }
    if ( v7 )
    {
      v7->m128i_i64[0] = (__int64)v7[1].m128i_i64;
      sub_23AEDD0(v7->m128i_i64, v62, (__int64)&v62[v63]);
      v7 = v73;
    }
    v73 = v7 + 2;
LABEL_9:
    v79 = v62;
    v80 = v63;
    v65 = (void *)sub_2E31BC0(v6);
    v66 = v8;
    sub_95CA80(v81.m128i_i64, (__int64)&v65);
    *((_QWORD *)&v83 + 1) = 0;
    *(_QWORD *)&v83 = &v84;
    v70 = 0x100000000LL;
    v84.m128i_i8[0] = 0;
    v66 = 0;
    v65 = &unk_49DD210;
    v71 = &v83;
    v67 = 0;
    v68 = 0;
    v69 = 0;
    sub_CB5980((__int64)&v65, 0, 0, 0);
    sub_2E393D0(v6, &v65, 0, 1);
    v65 = &unk_49DD210;
    sub_CB5840((__int64)&v65);
    v9 = sub_C92610();
    v10 = v80;
    v56 = v79;
    v54 = v80;
    v11 = sub_C92740((__int64)&v75, v79, v80, v9);
    v12 = (_QWORD *)(v75 + 8LL * v11);
    if ( !*v12 )
      goto LABEL_41;
    if ( *v12 == -8 )
    {
      --DWORD2(v76);
LABEL_41:
      v31 = v56;
      v53 = (__int64 *)(v75 + 8LL * v11);
      v59 = v11;
      v32 = sub_23AE710(80, 8, v31, v10);
      if ( v32 )
      {
        *(_QWORD *)(v32 + 8) = v32 + 24;
        *(_QWORD *)v32 = v54;
        if ( (__m128i *)v81.m128i_i64[0] == &v82 )
        {
          *(__m128i *)(v32 + 24) = _mm_load_si128(&v82);
        }
        else
        {
          *(_QWORD *)(v32 + 8) = v81.m128i_i64[0];
          *(_QWORD *)(v32 + 24) = v82.m128i_i64[0];
        }
        *(_QWORD *)(v32 + 16) = v81.m128i_i64[1];
        v81.m128i_i64[0] = (__int64)&v82;
        v81.m128i_i64[1] = 0;
        v82.m128i_i8[0] = 0;
        *(_QWORD *)(v32 + 40) = v32 + 56;
        if ( (__m128i *)v83 == &v84 )
        {
          *(__m128i *)(v32 + 56) = _mm_load_si128(&v84);
        }
        else
        {
          *(_QWORD *)(v32 + 40) = v83;
          *(_QWORD *)(v32 + 56) = v84.m128i_i64[0];
        }
        *(_QWORD *)(v32 + 48) = *((_QWORD *)&v83 + 1);
        *(_QWORD *)&v83 = &v84;
        *((_QWORD *)&v83 + 1) = 0;
        v84.m128i_i8[0] = 0;
      }
      *v53 = v32;
      ++DWORD1(v76);
      sub_C929D0((__int64 *)&v75, v59);
    }
    if ( (__m128i *)v83 != &v84 )
      j_j___libc_free_0(v83);
    if ( (__m128i *)v81.m128i_i64[0] != &v82 )
      j_j___libc_free_0(v81.m128i_u64[0]);
    if ( v62 != (char *)&v64 )
      j_j___libc_free_0((unsigned __int64)v62);
    v6 = *(_QWORD *)(v6 + 8);
  }
  v79 = (char *)sub_2E791E0(a2);
  v80 = v14;
  sub_23BB830(a1, &v79);
  v15 = (char *)sub_2E791E0(a2);
  v17 = v73;
  v18 = (unsigned __int64)v72;
  v81 = 0u;
  v79 = v15;
  v80 = v16;
  v82.m128i_i64[0] = 0;
  v19 = (char *)v73 - (char *)v72;
  if ( v73 == (__m128i *)v72 )
  {
    v19 = 0;
    v21 = 0;
  }
  else
  {
    if ( v19 > 0x7FFFFFFFFFFFFFE0LL )
      sub_4261EA(a2, &v79, v16);
    v20 = sub_22077B0((char *)v73 - (char *)v72);
    v17 = v73;
    v18 = (unsigned __int64)v72;
    v21 = (__int64 *)v20;
  }
  v81.m128i_i64[0] = (__int64)v21;
  v81.m128i_i64[1] = (__int64)v21;
  for ( v82.m128i_i64[0] = (__int64)v21 + v19; v17 != (__m128i *)v18; v21 += 4 )
  {
    if ( v21 )
    {
      *v21 = (__int64)(v21 + 2);
      sub_23AEDD0(v21, *(_BYTE **)v18, *(_QWORD *)v18 + *(_QWORD *)(v18 + 8));
    }
    v18 += 32LL;
  }
  v81.m128i_i64[1] = (__int64)v21;
  v82.m128i_i64[1] = 0;
  *(_QWORD *)&v83 = 0;
  *((_QWORD *)&v83 + 1) = 0x5000000000LL;
  if ( DWORD1(v76) )
  {
    sub_C92620((__int64)&v82.m128i_i64[1], v76);
    v22 = v75;
    v55 = v82.m128i_i64[1];
    *(_QWORD *)((char *)&v83 + 4) = *(_QWORD *)((char *)&v76 + 4);
    if ( (_DWORD)v83 )
    {
      v23 = v82.m128i_i64[1];
      v24 = 8LL * (unsigned int)v83 + 8;
      v60 = 8LL * (unsigned int)(v83 - 1);
      v25 = 0;
      v26 = v75;
      while ( 1 )
      {
        v29 = *(size_t **)(v26 + v25);
        v30 = (size_t **)(v23 + v25);
        if ( v29 == (size_t *)-8LL || !v29 )
        {
          *v30 = v29;
        }
        else
        {
          v57 = *v29;
          v27 = (_QWORD *)sub_23AE710(80, 8, v29 + 10, *v29);
          if ( v27 )
          {
            v28 = v57;
            v58 = v27;
            *v27 = v28;
            sub_2241BD0(v27 + 1, (__int64)(v29 + 1));
            sub_2241BD0(v58 + 5, (__int64)(v29 + 5));
            v27 = v58;
          }
          *v30 = v27;
          *(_DWORD *)(v55 + v24) = *(_DWORD *)(v22 + v24);
        }
        v24 += 4;
        if ( v60 == v25 )
          break;
        v26 = v75;
        v23 = v82.m128i_i64[1];
        v25 += 8;
      }
    }
  }
  sub_2241BD0(v84.m128i_i64, (__int64)v77);
  v33 = sub_C92610();
  sub_23BE540((__int64)(a1 + 3), v79, v80, v33, &v81);
  if ( (__int64 *)v84.m128i_i64[0] != &v85 )
    j_j___libc_free_0(v84.m128i_u64[0]);
  v34 = v82.m128i_u64[1];
  if ( DWORD1(v83) && (_DWORD)v83 )
  {
    v35 = 8LL * (unsigned int)v83;
    v36 = 0;
    do
    {
      v37 = *(_QWORD **)(v34 + v36);
      if ( v37 != (_QWORD *)-8LL && v37 )
      {
        v38 = v37[5];
        v39 = *v37 + 81LL;
        if ( (_QWORD *)v38 != v37 + 7 )
          j_j___libc_free_0(v38);
        v40 = v37[1];
        if ( (_QWORD *)v40 != v37 + 3 )
          j_j___libc_free_0(v40);
        sub_C7D6A0((__int64)v37, v39, 8);
        v34 = v82.m128i_u64[1];
      }
      v36 += 8;
    }
    while ( v36 != v35 );
  }
  _libc_free(v34);
  v41 = v81.m128i_i64[1];
  v42 = (unsigned __int64 *)v81.m128i_i64[0];
  if ( v81.m128i_i64[1] != v81.m128i_i64[0] )
  {
    do
    {
      if ( (unsigned __int64 *)*v42 != v42 + 2 )
        j_j___libc_free_0(*v42);
      v42 += 4;
    }
    while ( (unsigned __int64 *)v41 != v42 );
    v42 = (unsigned __int64 *)v81.m128i_i64[0];
  }
  if ( v42 )
    j_j___libc_free_0((unsigned __int64)v42);
  if ( (__int64 *)v77[0] != &v78 )
    j_j___libc_free_0(v77[0]);
  v43 = v75;
  if ( DWORD1(v76) && (_DWORD)v76 )
  {
    v44 = 8LL * (unsigned int)v76;
    v45 = 0;
    do
    {
      v46 = *(_QWORD **)(v43 + v45);
      if ( v46 != (_QWORD *)-8LL && v46 )
      {
        v47 = v46[5];
        v48 = *v46 + 81LL;
        if ( (_QWORD *)v47 != v46 + 7 )
          j_j___libc_free_0(v47);
        v49 = v46[1];
        if ( (_QWORD *)v49 != v46 + 3 )
          j_j___libc_free_0(v49);
        sub_C7D6A0((__int64)v46, v48, 8);
        v43 = v75;
      }
      v45 += 8;
    }
    while ( v45 != v44 );
  }
  _libc_free(v43);
  v50 = (unsigned __int64 *)v73;
  v51 = v72;
  if ( v73 != (__m128i *)v72 )
  {
    do
    {
      if ( (unsigned __int64 *)*v51 != v51 + 2 )
        j_j___libc_free_0(*v51);
      v51 += 4;
    }
    while ( v50 != v51 );
    v51 = v72;
  }
  if ( v51 )
    j_j___libc_free_0((unsigned __int64)v51);
}
