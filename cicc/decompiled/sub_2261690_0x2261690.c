// Function: sub_2261690
// Address: 0x2261690
//
void __fastcall sub_2261690(__m128i *a1, __m128i *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r12
  size_t v6; // r13
  const void *v7; // r14
  int v8; // eax
  __m128i *v9; // r12
  int v10; // eax
  __int64 v11; // rax
  size_t v12; // r14
  const void *v13; // r13
  int v14; // eax
  int v15; // eax
  __int64 v16; // rax
  unsigned int v17; // eax
  size_t v18; // r13
  const void *v19; // r14
  int v20; // eax
  int v21; // eax
  __int64 v22; // rax
  unsigned int v23; // r13d
  size_t v24; // r14
  int v25; // eax
  int v26; // eax
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  size_t v30; // rdx
  const void *v31; // r12
  int v32; // eax
  int v33; // eax
  __int64 v34; // rax
  unsigned int v35; // r13d
  size_t v36; // r14
  const void *v37; // r12
  int v38; // eax
  int v39; // eax
  __int64 v40; // rax
  unsigned int v41; // eax
  __m128i *v42; // r14
  __int64 v43; // rax
  size_t v44; // r12
  const void *v45; // r13
  int v46; // eax
  int v47; // eax
  __int64 v48; // rax
  unsigned int v49; // r12d
  const void *v50; // r13
  int v51; // eax
  int v52; // eax
  __int64 v53; // rdx
  __int64 v54; // rax
  size_t v55; // r13
  const void *v56; // r14
  int v57; // eax
  int v58; // eax
  __int64 v59; // rax
  unsigned int v60; // r13d
  size_t v61; // r14
  int v62; // eax
  int v63; // eax
  __int64 v64; // rax
  __m128i v65; // xmm4
  __int64 v66; // r12
  __int64 i; // r13
  __m128i *v68; // r13
  const void *v69; // rcx
  __int64 v70; // r12
  size_t v71; // r8
  __int64 v72; // rdx
  size_t v73; // r13
  const void *v74; // r14
  int v75; // eax
  int v76; // eax
  __int64 v77; // rax
  unsigned int v78; // r13d
  size_t v79; // r14
  int v80; // eax
  int v81; // eax
  __int64 v82; // rax
  unsigned int v83; // eax
  __int64 v84; // rcx
  __int64 v85; // rdx
  size_t v86; // r12
  const void *v87; // r13
  int v88; // eax
  int v89; // eax
  __int64 v90; // rax
  unsigned int v91; // r13d
  size_t v92; // r12
  const void *v93; // r14
  int v94; // eax
  int v95; // eax
  __int64 v96; // rax
  unsigned int v97; // eax
  __m128i v98; // xmm7
  __int64 v99; // [rsp+8h] [rbp-68h]
  __m128i *v100; // [rsp+10h] [rbp-60h]
  __m128i *v101; // [rsp+18h] [rbp-58h]
  __m128i *v102; // [rsp+20h] [rbp-50h]
  __m128i *v103; // [rsp+28h] [rbp-48h]
  unsigned int v104; // [rsp+30h] [rbp-40h]
  const void *v105; // [rsp+30h] [rbp-40h]
  size_t v106; // [rsp+30h] [rbp-40h]
  size_t v107; // [rsp+30h] [rbp-40h]
  const void *v108; // [rsp+30h] [rbp-40h]
  const void *v109; // [rsp+30h] [rbp-40h]

  v4 = (char *)a2 - (char *)a1;
  v99 = a3;
  v100 = a2;
  if ( (char *)a2 - (char *)a1 <= 256 )
    return;
  if ( !a3 )
  {
    v101 = a2;
    goto LABEL_52;
  }
  while ( 2 )
  {
    v6 = a1[1].m128i_u64[1];
    v7 = (const void *)a1[1].m128i_i64[0];
    --v99;
    v8 = sub_C92610();
    v9 = &a1[(__int64)(((unsigned __int64)((char *)v100 - (char *)a1) >> 63) + v100 - a1) >> 1];
    v10 = sub_C92860((__int64 *)a4, v7, v6, v8);
    if ( v10 == -1 || (v11 = *(_QWORD *)a4 + 8LL * v10, v11 == *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8)) )
      v104 = 0;
    else
      v104 = *(_DWORD *)(*(_QWORD *)v11 + 8LL);
    v12 = v9->m128i_u64[1];
    v13 = (const void *)v9->m128i_i64[0];
    v14 = sub_C92610();
    v15 = sub_C92860((__int64 *)a4, v13, v12, v14);
    if ( v15 == -1 || (v16 = *(_QWORD *)a4 + 8LL * v15, v16 == *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8)) )
      v17 = 0;
    else
      v17 = *(_DWORD *)(*(_QWORD *)v16 + 8LL);
    if ( v17 >= v104 )
    {
      v55 = a1[1].m128i_u64[1];
      v56 = (const void *)a1[1].m128i_i64[0];
      v57 = sub_C92610();
      v58 = sub_C92860((__int64 *)a4, v56, v55, v57);
      if ( v58 == -1 || (v59 = *(_QWORD *)a4 + 8LL * v58, v59 == *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8)) )
        v60 = 0;
      else
        v60 = *(_DWORD *)(*(_QWORD *)v59 + 8LL);
      v61 = v100[-1].m128i_u64[1];
      v108 = (const void *)v100[-1].m128i_i64[0];
      v62 = sub_C92610();
      v63 = sub_C92860((__int64 *)a4, v108, v61, v62);
      if ( v63 == -1 || (v64 = *(_QWORD *)a4 + 8LL * v63, v64 == *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8)) )
      {
        if ( v60 )
          goto LABEL_48;
      }
      else if ( *(_DWORD *)(*(_QWORD *)v64 + 8LL) < v60 )
      {
LABEL_48:
        v65 = _mm_loadu_si128(a1 + 1);
        v31 = (const void *)a1->m128i_i64[0];
        v30 = a1->m128i_u64[1];
        a1[1].m128i_i64[0] = a1->m128i_i64[0];
        a1[1].m128i_i64[1] = v30;
        *a1 = v65;
        goto LABEL_17;
      }
      v73 = v9->m128i_u64[1];
      v74 = (const void *)v9->m128i_i64[0];
      v75 = sub_C92610();
      v76 = sub_C92860((__int64 *)a4, v74, v73, v75);
      if ( v76 == -1 || (v77 = *(_QWORD *)a4 + 8LL * v76, v77 == *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8)) )
        v78 = 0;
      else
        v78 = *(_DWORD *)(*(_QWORD *)v77 + 8LL);
      v79 = v100[-1].m128i_u64[1];
      v109 = (const void *)v100[-1].m128i_i64[0];
      v80 = sub_C92610();
      v81 = sub_C92860((__int64 *)a4, v109, v79, v80);
      if ( v81 == -1 || (v82 = *(_QWORD *)a4 + 8LL * v81, v82 == *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8)) )
        v83 = 0;
      else
        v83 = *(_DWORD *)(*(_QWORD *)v82 + 8LL);
      v84 = a1->m128i_i64[0];
      v85 = a1->m128i_i64[1];
      if ( v83 >= v78 )
      {
        *a1 = _mm_loadu_si128(v9);
        v9->m128i_i64[0] = v84;
        v9->m128i_i64[1] = v85;
      }
      else
      {
        *a1 = _mm_loadu_si128(v100 - 1);
        v100[-1].m128i_i64[0] = v84;
        v100[-1].m128i_i64[1] = v85;
      }
      v30 = a1[1].m128i_u64[1];
      v31 = (const void *)a1[1].m128i_i64[0];
      goto LABEL_17;
    }
    v18 = v9->m128i_u64[1];
    v19 = (const void *)v9->m128i_i64[0];
    v20 = sub_C92610();
    v21 = sub_C92860((__int64 *)a4, v19, v18, v20);
    if ( v21 == -1 || (v22 = *(_QWORD *)a4 + 8LL * v21, v22 == *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8)) )
      v23 = 0;
    else
      v23 = *(_DWORD *)(*(_QWORD *)v22 + 8LL);
    v24 = v100[-1].m128i_u64[1];
    v105 = (const void *)v100[-1].m128i_i64[0];
    v25 = sub_C92610();
    v26 = sub_C92860((__int64 *)a4, v105, v24, v25);
    if ( v26 == -1 || (v27 = *(_QWORD *)a4 + 8LL * v26, v27 == *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8)) )
    {
      if ( v23 )
        goto LABEL_16;
LABEL_71:
      v86 = a1[1].m128i_u64[1];
      v87 = (const void *)a1[1].m128i_i64[0];
      v88 = sub_C92610();
      v89 = sub_C92860((__int64 *)a4, v87, v86, v88);
      if ( v89 == -1 || (v90 = *(_QWORD *)a4 + 8LL * v89, v90 == *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8)) )
        v91 = 0;
      else
        v91 = *(_DWORD *)(*(_QWORD *)v90 + 8LL);
      v92 = v100[-1].m128i_u64[1];
      v93 = (const void *)v100[-1].m128i_i64[0];
      v94 = sub_C92610();
      v95 = sub_C92860((__int64 *)a4, v93, v92, v94);
      if ( v95 == -1 || (v96 = *(_QWORD *)a4 + 8LL * v95, v96 == *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8)) )
        v97 = 0;
      else
        v97 = *(_DWORD *)(*(_QWORD *)v96 + 8LL);
      v31 = (const void *)a1->m128i_i64[0];
      v30 = a1->m128i_u64[1];
      if ( v97 >= v91 )
      {
        v98 = _mm_loadu_si128(a1 + 1);
        a1[1].m128i_i64[0] = (__int64)v31;
        a1[1].m128i_i64[1] = v30;
        *a1 = v98;
      }
      else
      {
        *a1 = _mm_loadu_si128(v100 - 1);
        v100[-1].m128i_i64[0] = (__int64)v31;
        v100[-1].m128i_i64[1] = v30;
        v30 = a1[1].m128i_u64[1];
        v31 = (const void *)a1[1].m128i_i64[0];
      }
      goto LABEL_17;
    }
    if ( *(_DWORD *)(*(_QWORD *)v27 + 8LL) >= v23 )
      goto LABEL_71;
LABEL_16:
    v28 = a1->m128i_i64[0];
    v29 = a1->m128i_i64[1];
    *a1 = _mm_loadu_si128(v9);
    v9->m128i_i64[0] = v28;
    v9->m128i_i64[1] = v29;
    v30 = a1[1].m128i_u64[1];
    v31 = (const void *)a1[1].m128i_i64[0];
LABEL_17:
    v102 = a1 + 1;
    v103 = v100;
    while ( 1 )
    {
      v106 = v30;
      v101 = v102;
      v32 = sub_C92610();
      v33 = sub_C92860((__int64 *)a4, v31, v106, v32);
      if ( v33 == -1 || (v34 = *(_QWORD *)a4 + 8LL * v33, v34 == *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8)) )
        v35 = 0;
      else
        v35 = *(_DWORD *)(*(_QWORD *)v34 + 8LL);
      v36 = a1->m128i_u64[1];
      v37 = (const void *)a1->m128i_i64[0];
      v38 = sub_C92610();
      v39 = sub_C92860((__int64 *)a4, v37, v36, v38);
      if ( v39 == -1 || (v40 = *(_QWORD *)a4 + 8LL * v39, v40 == *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8)) )
        v41 = 0;
      else
        v41 = *(_DWORD *)(*(_QWORD *)v40 + 8LL);
      if ( v41 >= v35 )
        break;
LABEL_35:
      v30 = v102[1].m128i_u64[1];
      v31 = (const void *)v102[1].m128i_i64[0];
      ++v102;
    }
    v42 = v103 - 1;
    do
    {
      while ( 1 )
      {
        v44 = a1->m128i_u64[1];
        v45 = (const void *)a1->m128i_i64[0];
        v103 = v42;
        v46 = sub_C92610();
        v47 = sub_C92860((__int64 *)a4, v45, v44, v46);
        if ( v47 == -1 || (v48 = *(_QWORD *)a4 + 8LL * v47, v48 == *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8)) )
          v49 = 0;
        else
          v49 = *(_DWORD *)(*(_QWORD *)v48 + 8LL);
        v50 = (const void *)v42->m128i_i64[0];
        v107 = v42->m128i_u64[1];
        v51 = sub_C92610();
        v52 = sub_C92860((__int64 *)a4, v50, v107, v51);
        if ( v52 == -1 )
          break;
        v43 = *(_QWORD *)a4 + 8LL * v52;
        if ( v43 == *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8) )
          break;
        --v42;
        if ( *(_DWORD *)(*(_QWORD *)v43 + 8LL) >= v49 )
          goto LABEL_33;
      }
      --v42;
    }
    while ( v49 );
LABEL_33:
    if ( v102 < v103 )
    {
      v53 = v102->m128i_i64[0];
      v54 = v102->m128i_i64[1];
      *v102 = _mm_loadu_si128(v103);
      v103->m128i_i64[0] = v53;
      v103->m128i_i64[1] = v54;
      goto LABEL_35;
    }
    v4 = (char *)v102 - (char *)a1;
    sub_2261690(v102, v100, v99, a4);
    if ( (char *)v102 - (char *)a1 > 256 )
    {
      if ( v99 )
      {
        v100 = v102;
        continue;
      }
LABEL_52:
      v66 = v4 >> 4;
      for ( i = (v66 - 2) >> 1; ; --i )
      {
        sub_2260950((__int64)a1, i, v66, (const void *)a1[i].m128i_i64[0], a1[i].m128i_u64[1], a4);
        if ( !i )
          break;
      }
      v68 = v101 - 1;
      do
      {
        v69 = (const void *)v68->m128i_i64[0];
        v70 = (char *)v68 - (char *)a1;
        v71 = v68->m128i_u64[1];
        v72 = (char *)v68 - (char *)a1;
        *v68-- = _mm_loadu_si128(a1);
        sub_2260950((__int64)a1, 0, v72 >> 4, v69, v71, a4);
      }
      while ( v70 > 16 );
    }
    break;
  }
}
