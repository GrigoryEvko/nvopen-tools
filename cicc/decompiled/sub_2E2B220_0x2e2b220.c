// Function: sub_2E2B220
// Address: 0x2e2b220
//
void __fastcall sub_2E2B220(__m128i *a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 (__fastcall *v6)(__int64); // rax
  __int64 v7; // r8
  __int64 v8; // r9
  unsigned __int64 *v9; // rsi
  __int64 v10; // rdx
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rcx
  __int64 *v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __m128i *v18; // rdx
  const __m128i *v19; // rcx
  unsigned __int64 v20; // rsi
  unsigned __int64 v21; // r13
  __int64 v22; // rax
  const __m128i *v23; // rax
  __int64 v24; // rax
  const __m128i *v25; // rsi
  unsigned __int64 v26; // r8
  __m128i *v27; // rcx
  const __m128i *v28; // rax
  __int64 v29; // r8
  __int64 v30; // r9
  __m128i *v31; // r14
  __int64 v32; // rbx
  __int64 *v33; // rax
  __int64 v34; // rcx
  __int64 *v35; // rdx
  _QWORD *v36; // rdi
  __int64 v37; // r12
  __int64 *v38; // rax
  __int64 v39; // rsi
  unsigned __int64 v40; // rcx
  __m128i *v41; // rax
  __int8 v42; // si
  unsigned int v43; // r12d
  __int64 v44; // rbx
  __int64 v45; // rdx
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // r14
  __int64 v49; // r13
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rdi
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // r12
  unsigned __int64 *v56; // r13
  unsigned __int64 *v57; // rbx
  char v58; // dl
  unsigned __int64 *v59; // rax
  unsigned __int64 *v60; // rbx
  __int64 v61; // r12
  unsigned __int64 v62; // [rsp+28h] [rbp-178h]
  __int32 v63; // [rsp+28h] [rbp-178h]
  unsigned __int64 v64; // [rsp+30h] [rbp-170h]
  int v65; // [rsp+30h] [rbp-170h]
  unsigned int v66; // [rsp+38h] [rbp-168h]
  __int64 v67; // [rsp+38h] [rbp-168h]
  __m128i v68; // [rsp+48h] [rbp-158h] BYREF
  char v69; // [rsp+60h] [rbp-140h]
  __int64 v70; // [rsp+70h] [rbp-130h]
  __m128i *v71; // [rsp+78h] [rbp-128h] BYREF
  __m128i *v72; // [rsp+80h] [rbp-120h]
  char *v73; // [rsp+88h] [rbp-118h]
  __int64 v74; // [rsp+90h] [rbp-110h] BYREF
  const __m128i *v75; // [rsp+98h] [rbp-108h]
  const __m128i *v76; // [rsp+A0h] [rbp-100h]
  const __m128i *v77; // [rsp+B8h] [rbp-E8h]
  unsigned __int64 v78; // [rsp+C0h] [rbp-E0h]
  __int64 v79; // [rsp+D0h] [rbp-D0h] BYREF
  char *v80; // [rsp+D8h] [rbp-C8h]
  __int64 v81; // [rsp+E0h] [rbp-C0h]
  int v82; // [rsp+E8h] [rbp-B8h]
  char v83; // [rsp+ECh] [rbp-B4h]
  char v84; // [rsp+F0h] [rbp-B0h] BYREF

  a1[5].m128i_i64[0] = a2;
  a1[5].m128i_i64[1] = *(_QWORD *)(a2 + 32);
  v4 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 200LL))(*(_QWORD *)(a2 + 16));
  a1[6].m128i_i64[0] = v4;
  v5 = v4;
  v6 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v4 + 24LL);
  if ( v6 == sub_2E241E0 )
    v66 = *(_DWORD *)(v5 + 16);
  else
    v66 = ((__int64 (__fastcall *)(__int64, __int64))v6)(v5, a2);
  v79 = 0;
  sub_2E25C40((__int64)&a1[6].m128i_i64[1], v66, &v79);
  v79 = 0;
  sub_2E25C40((__int64)a1[8].m128i_i64, v66, &v79);
  v9 = (unsigned __int64 *)a1[10].m128i_i64[0];
  v10 = a1[9].m128i_i64[1];
  v11 = (unsigned int)((__int64)(*(_QWORD *)(a1[5].m128i_i64[0] + 104) - *(_QWORD *)(a1[5].m128i_i64[0] + 96)) >> 3);
  v12 = ((__int64)v9 - v10) >> 5;
  if ( v11 > v12 )
  {
    sub_2E25DB0(&a1[9].m128i_u64[1], v11 - v12, v10, v12, v7, v8);
  }
  else if ( v11 < v12 )
  {
    v59 = (unsigned __int64 *)(v10 + 32 * v11);
    if ( v9 != v59 )
    {
      v60 = v59;
      v61 = (__int64)v59;
      do
      {
        if ( (unsigned __int64 *)*v60 != v60 + 2 )
          _libc_free(*v60);
        v60 += 4;
      }
      while ( v9 != v60 );
      a1[10].m128i_i64[0] = v61;
    }
  }
  if ( (*(_BYTE *)(*(_QWORD *)a1[5].m128i_i64[1] + 344LL) & 1) == 0 )
    sub_C64ED0("regalloc=... not currently supported with -O0", 1u);
  sub_2E256C0((__int64)a1, a2);
  v13 = &v74;
  v14 = *(_QWORD *)(a1[5].m128i_i64[0] + 328);
  v83 = 1;
  v79 = 0;
  v68.m128i_i64[0] = v14;
  v80 = &v84;
  v81 = 16;
  v82 = 0;
  sub_2E27000(&v74, &v68, (__int64)&v79, v15, v16, v17);
  v19 = v76;
  v71 = 0;
  v20 = (unsigned __int64)v75;
  v72 = 0;
  v70 = v74;
  v73 = 0;
  v21 = (char *)v76 - (char *)v75;
  if ( v76 == v75 )
  {
    v13 = 0;
  }
  else
  {
    if ( v21 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_87;
    v22 = sub_22077B0((char *)v76 - (char *)v75);
    v19 = v76;
    v20 = (unsigned __int64)v75;
    v13 = (__int64 *)v22;
  }
  v71 = (__m128i *)v13;
  v18 = (__m128i *)v13;
  v72 = (__m128i *)v13;
  v73 = (char *)v13 + v21;
  if ( v19 != (const __m128i *)v20 )
  {
    v23 = (const __m128i *)v20;
    do
    {
      if ( v18 )
      {
        *v18 = _mm_loadu_si128(v23);
        v18[1].m128i_i64[0] = v23[1].m128i_i64[0];
      }
      v23 = (const __m128i *)((char *)v23 + 24);
      v18 = (__m128i *)((char *)v18 + 24);
    }
    while ( v23 != v19 );
    v18 = (__m128i *)&v13[(((unsigned __int64)&v23[-2].m128i_u64[1] - v20) >> 3) + 3];
  }
  v20 = v78;
  v72 = v18;
  if ( (const __m128i *)v78 == v77 )
  {
    v62 = 0;
    goto LABEL_81;
  }
  if ( v78 - (unsigned __int64)v77 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_87:
    sub_4261EA(v13, v20, v18);
  v24 = sub_22077B0(v78 - (_QWORD)v77);
  v25 = (const __m128i *)v78;
  v26 = (unsigned __int64)v77;
  v62 = v24;
  v13 = (__int64 *)v71;
  v18 = v72;
  if ( (const __m128i *)v78 == v77 )
  {
LABEL_81:
    v64 = 0;
    goto LABEL_23;
  }
  v27 = (__m128i *)v24;
  v28 = v77;
  do
  {
    if ( v27 )
    {
      *v27 = _mm_loadu_si128(v28);
      v27[1].m128i_i64[0] = v28[1].m128i_i64[0];
    }
    v28 = (const __m128i *)((char *)v28 + 24);
    v27 = (__m128i *)((char *)v27 + 24);
  }
  while ( v28 != v25 );
  v64 = 8 * (((unsigned __int64)&v28[-2].m128i_u64[1] - v26) >> 3) + 24;
LABEL_23:
  if ( (char *)v18 - (char *)v13 == v64 )
    goto LABEL_35;
  while ( 2 )
  {
    while ( 2 )
    {
      sub_2E2A740(a1, v18[-2].m128i_i64[1], v66);
      v68.m128i_i64[1] = 0;
      sub_2E25C40((__int64)&a1[6].m128i_i64[1], v66, &v68.m128i_i64[1]);
      v68.m128i_i64[1] = 0;
      sub_2E25C40((__int64)a1[8].m128i_i64, v66, &v68.m128i_i64[1]);
      v31 = v72;
      do
      {
        v32 = v31[-2].m128i_i64[1];
        if ( !v31[-1].m128i_i8[8] )
        {
          v33 = *(__int64 **)(v32 + 112);
          v31[-1].m128i_i8[8] = 1;
          v31[-1].m128i_i64[0] = (__int64)v33;
          v34 = *(unsigned int *)(v32 + 120);
          if ( v33 == (__int64 *)(*(_QWORD *)(v32 + 112) + 8 * v34) )
            goto LABEL_33;
          goto LABEL_27;
        }
        while ( 1 )
        {
          v34 = *(unsigned int *)(v32 + 120);
          v33 = (__int64 *)v31[-1].m128i_i64[0];
          if ( v33 == (__int64 *)(*(_QWORD *)(v32 + 112) + 8 * v34) )
            break;
LABEL_27:
          v35 = v33 + 1;
          v31[-1].m128i_i64[0] = (__int64)(v33 + 1);
          v36 = (_QWORD *)v70;
          v37 = *v33;
          if ( !*(_BYTE *)(v70 + 28) )
            goto LABEL_70;
          v38 = *(__int64 **)(v70 + 8);
          v39 = *(unsigned int *)(v70 + 20);
          v35 = &v38[v39];
          if ( v38 == v35 )
          {
LABEL_72:
            if ( (unsigned int)v39 < *(_DWORD *)(v70 + 16) )
            {
              *(_DWORD *)(v70 + 20) = v39 + 1;
              *v35 = v37;
              ++*v36;
LABEL_71:
              v68.m128i_i64[1] = v37;
              v69 = 0;
              sub_2E26FC0((unsigned __int64 *)&v71, (const __m128i *)&v68.m128i_u64[1]);
              v13 = (__int64 *)v71;
              v18 = v72;
              goto LABEL_23;
            }
LABEL_70:
            sub_C8CC70(v70, v37, (__int64)v35, v34, v29, v30);
            if ( v58 )
              goto LABEL_71;
          }
          else
          {
            while ( v37 != *v38 )
            {
              if ( v35 == ++v38 )
                goto LABEL_72;
            }
          }
        }
LABEL_33:
        v72 = (__m128i *)((char *)v72 - 24);
        v13 = (__int64 *)v71;
        v31 = v72;
      }
      while ( v72 != v71 );
      v18 = v71;
      if ( v64 )
        continue;
      break;
    }
LABEL_35:
    if ( v18 != (__m128i *)v13 )
    {
      v40 = v62;
      v41 = (__m128i *)v13;
      while ( v41->m128i_i64[0] == *(_QWORD *)v40 )
      {
        v42 = v41[1].m128i_i8[0];
        if ( v42 != *(_BYTE *)(v40 + 16) || v42 && v41->m128i_i64[1] != *(_QWORD *)(v40 + 8) )
          break;
        v41 = (__m128i *)((char *)v41 + 24);
        v40 += 24LL;
        if ( v18 == v41 )
          goto LABEL_42;
      }
      continue;
    }
    break;
  }
LABEL_42:
  if ( v62 )
  {
    j_j___libc_free_0(v62);
    v13 = (__int64 *)v71;
  }
  if ( v13 )
    j_j___libc_free_0((unsigned __int64)v13);
  if ( v77 )
    j_j___libc_free_0((unsigned __int64)v77);
  if ( v75 )
    j_j___libc_free_0((unsigned __int64)v75);
  v65 = 0;
  v63 = a1->m128i_i32[2];
  if ( v63 )
  {
    do
    {
      v43 = v65 | 0x80000000;
      v44 = 56LL * (v65 & 0x7FFFFFFF);
      v45 = v44 + a1->m128i_i64[0];
      v46 = *(_QWORD *)(v45 + 32);
      v47 = (*(_QWORD *)(v45 + 40) - v46) >> 3;
      if ( (_DWORD)v47 )
      {
        v48 = 0;
        v67 = 8LL * (unsigned int)(v47 - 1);
        while ( 1 )
        {
          v49 = *(_QWORD *)(v46 + v48);
          v50 = sub_2EBEE10(a1[5].m128i_i64[1], v43);
          v51 = a1[6].m128i_i64[0];
          v52 = *(_QWORD *)(*(_QWORD *)(a1->m128i_i64[0] + v44 + 32) + v48);
          if ( v49 == v50 )
          {
            sub_2E8F690(v52, v43, v51, 0);
            if ( v67 == v48 )
              break;
          }
          else
          {
            sub_2E8F280(v52, v43, v51, 0);
            if ( v67 == v48 )
              break;
          }
          v48 += 8;
          v46 = *(_QWORD *)(a1->m128i_i64[0] + v44 + 32);
        }
      }
      ++v65;
    }
    while ( v65 != v63 );
  }
  v53 = a1[6].m128i_i64[1];
  if ( v53 != a1[7].m128i_i64[0] )
    a1[7].m128i_i64[0] = v53;
  v54 = a1[8].m128i_i64[0];
  if ( v54 != a1[8].m128i_i64[1] )
    a1[8].m128i_i64[1] = v54;
  v55 = a1[9].m128i_i64[1];
  v56 = (unsigned __int64 *)a1[10].m128i_i64[0];
  if ( (unsigned __int64 *)v55 != v56 )
  {
    v57 = (unsigned __int64 *)a1[9].m128i_i64[1];
    do
    {
      if ( (unsigned __int64 *)*v57 != v57 + 2 )
        _libc_free(*v57);
      v57 += 4;
    }
    while ( v56 != v57 );
    a1[10].m128i_i64[0] = v55;
  }
  if ( !v83 )
    _libc_free((unsigned __int64)v80);
}
