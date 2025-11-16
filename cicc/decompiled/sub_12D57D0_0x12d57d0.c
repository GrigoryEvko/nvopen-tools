// Function: sub_12D57D0
// Address: 0x12d57d0
//
void __fastcall sub_12D57D0(__m128i *a1, __m128i *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rbx
  __m128i *v6; // r12
  int v7; // eax
  __int64 v8; // rax
  unsigned int v9; // ebx
  int v10; // eax
  __int64 v11; // rax
  unsigned int v12; // eax
  int v13; // eax
  __int64 v14; // rax
  unsigned int v15; // ebx
  int v16; // eax
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rsi
  __m128i *v22; // r12
  int v23; // eax
  __int64 v24; // rax
  unsigned int v25; // ebx
  int v26; // eax
  __int64 v27; // rax
  unsigned int v28; // eax
  __m128i *v29; // r13
  __int64 v30; // rax
  int v31; // eax
  __int64 v32; // rax
  unsigned int v33; // ebx
  int v34; // eax
  __int64 v35; // rdx
  __int64 v36; // rax
  int v37; // eax
  __int64 v38; // rax
  unsigned int v39; // ebx
  int v40; // eax
  __int64 v41; // rax
  __m128i v42; // xmm3
  __int64 v43; // rbx
  __int64 j; // r12
  __int64 *m128i_i64; // r12
  __int64 v46; // rcx
  __int64 v47; // rbx
  __int64 v48; // r8
  __int64 v49; // rdx
  int v50; // eax
  __int64 v51; // rax
  unsigned int v52; // ebx
  int v53; // eax
  __int64 v54; // rax
  unsigned int v55; // eax
  __int64 v56; // rcx
  __int64 v57; // rdx
  int v58; // eax
  __int64 v59; // rax
  unsigned int v60; // ebx
  int v61; // eax
  __int64 v62; // rax
  unsigned int v63; // eax
  __m128i v64; // xmm6
  __int64 v65; // [rsp+8h] [rbp-58h]
  __m128i *v66; // [rsp+10h] [rbp-50h]
  __m128i *v67; // [rsp+18h] [rbp-48h]
  __m128i *i; // [rsp+20h] [rbp-40h]

  v4 = (char *)a2 - (char *)a1;
  v65 = a3;
  v66 = a2;
  if ( (char *)a2 - (char *)a1 <= 256 )
    return;
  if ( !a3 )
  {
    v67 = a2;
    goto LABEL_52;
  }
  while ( 2 )
  {
    --v65;
    v6 = &a1[(__int64)(((unsigned __int64)((char *)v66 - (char *)a1) >> 63) + v66 - a1) >> 1];
    v7 = sub_16D1B30(a4, a1[1].m128i_i64[0], a1[1].m128i_i64[1]);
    if ( v7 == -1 || (v8 = *(_QWORD *)a4 + 8LL * v7, v8 == *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8)) )
      v9 = 0;
    else
      v9 = *(_DWORD *)(*(_QWORD *)v8 + 8LL);
    v10 = sub_16D1B30(a4, v6->m128i_i64[0], v6->m128i_i64[1]);
    if ( v10 == -1 || (v11 = *(_QWORD *)a4 + 8LL * v10, v11 == *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8)) )
      v12 = 0;
    else
      v12 = *(_DWORD *)(*(_QWORD *)v11 + 8LL);
    if ( v12 >= v9 )
    {
      v37 = sub_16D1B30(a4, a1[1].m128i_i64[0], a1[1].m128i_i64[1]);
      if ( v37 == -1 || (v38 = *(_QWORD *)a4 + 8LL * v37, v38 == *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8)) )
        v39 = 0;
      else
        v39 = *(_DWORD *)(*(_QWORD *)v38 + 8LL);
      v40 = sub_16D1B30(a4, v66[-1].m128i_i64[0], v66[-1].m128i_i64[1]);
      if ( v40 == -1 || (v41 = *(_QWORD *)a4 + 8LL * v40, v41 == *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8)) )
      {
        if ( v39 )
          goto LABEL_48;
      }
      else if ( *(_DWORD *)(*(_QWORD *)v41 + 8LL) < v39 )
      {
LABEL_48:
        v42 = _mm_loadu_si128(a1 + 1);
        v21 = a1->m128i_i64[0];
        v20 = a1->m128i_i64[1];
        a1[1].m128i_i64[0] = a1->m128i_i64[0];
        a1[1].m128i_i64[1] = v20;
        *a1 = v42;
        goto LABEL_17;
      }
      v50 = sub_16D1B30(a4, v6->m128i_i64[0], v6->m128i_i64[1]);
      if ( v50 == -1 || (v51 = *(_QWORD *)a4 + 8LL * v50, v51 == *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8)) )
        v52 = 0;
      else
        v52 = *(_DWORD *)(*(_QWORD *)v51 + 8LL);
      v53 = sub_16D1B30(a4, v66[-1].m128i_i64[0], v66[-1].m128i_i64[1]);
      if ( v53 == -1 || (v54 = *(_QWORD *)a4 + 8LL * v53, v54 == *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8)) )
        v55 = 0;
      else
        v55 = *(_DWORD *)(*(_QWORD *)v54 + 8LL);
      v56 = a1->m128i_i64[0];
      v57 = a1->m128i_i64[1];
      if ( v55 >= v52 )
      {
        *a1 = _mm_loadu_si128(v6);
        v6->m128i_i64[0] = v56;
        v6->m128i_i64[1] = v57;
      }
      else
      {
        *a1 = _mm_loadu_si128(v66 - 1);
        v66[-1].m128i_i64[0] = v56;
        v66[-1].m128i_i64[1] = v57;
      }
      v20 = a1[1].m128i_i64[1];
      v21 = a1[1].m128i_i64[0];
      goto LABEL_17;
    }
    v13 = sub_16D1B30(a4, v6->m128i_i64[0], v6->m128i_i64[1]);
    if ( v13 == -1 || (v14 = *(_QWORD *)a4 + 8LL * v13, v14 == *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8)) )
      v15 = 0;
    else
      v15 = *(_DWORD *)(*(_QWORD *)v14 + 8LL);
    v16 = sub_16D1B30(a4, v66[-1].m128i_i64[0], v66[-1].m128i_i64[1]);
    if ( v16 == -1 || (v17 = *(_QWORD *)a4 + 8LL * v16, v17 == *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8)) )
    {
      if ( v15 )
        goto LABEL_16;
LABEL_71:
      v58 = sub_16D1B30(a4, a1[1].m128i_i64[0], a1[1].m128i_i64[1]);
      if ( v58 == -1 || (v59 = *(_QWORD *)a4 + 8LL * v58, v59 == *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8)) )
        v60 = 0;
      else
        v60 = *(_DWORD *)(*(_QWORD *)v59 + 8LL);
      v61 = sub_16D1B30(a4, v66[-1].m128i_i64[0], v66[-1].m128i_i64[1]);
      if ( v61 == -1 || (v62 = *(_QWORD *)a4 + 8LL * v61, v62 == *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8)) )
        v63 = 0;
      else
        v63 = *(_DWORD *)(*(_QWORD *)v62 + 8LL);
      v21 = a1->m128i_i64[0];
      v20 = a1->m128i_i64[1];
      if ( v63 >= v60 )
      {
        v64 = _mm_loadu_si128(a1 + 1);
        a1[1].m128i_i64[0] = v21;
        a1[1].m128i_i64[1] = v20;
        *a1 = v64;
      }
      else
      {
        *a1 = _mm_loadu_si128(v66 - 1);
        v66[-1].m128i_i64[0] = v21;
        v66[-1].m128i_i64[1] = v20;
        v20 = a1[1].m128i_i64[1];
        v21 = a1[1].m128i_i64[0];
      }
      goto LABEL_17;
    }
    if ( *(_DWORD *)(*(_QWORD *)v17 + 8LL) >= v15 )
      goto LABEL_71;
LABEL_16:
    v18 = a1->m128i_i64[0];
    v19 = a1->m128i_i64[1];
    *a1 = _mm_loadu_si128(v6);
    v6->m128i_i64[0] = v18;
    v6->m128i_i64[1] = v19;
    v20 = a1[1].m128i_i64[1];
    v21 = a1[1].m128i_i64[0];
LABEL_17:
    v22 = v66;
    for ( i = a1 + 1; ; ++i )
    {
      v67 = i;
      v23 = sub_16D1B30(a4, v21, v20);
      if ( v23 == -1 || (v24 = *(_QWORD *)a4 + 8LL * v23, v24 == *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8)) )
        v25 = 0;
      else
        v25 = *(_DWORD *)(*(_QWORD *)v24 + 8LL);
      v26 = sub_16D1B30(a4, a1->m128i_i64[0], a1->m128i_i64[1]);
      if ( v26 == -1 || (v27 = *(_QWORD *)a4 + 8LL * v26, v27 == *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8)) )
        v28 = 0;
      else
        v28 = *(_DWORD *)(*(_QWORD *)v27 + 8LL);
      if ( v28 >= v25 )
        break;
LABEL_35:
      v20 = i[1].m128i_i64[1];
      v21 = i[1].m128i_i64[0];
    }
    v29 = v22 - 1;
    do
    {
      while ( 1 )
      {
        v22 = v29;
        v31 = sub_16D1B30(a4, a1->m128i_i64[0], a1->m128i_i64[1]);
        if ( v31 == -1 || (v32 = *(_QWORD *)a4 + 8LL * v31, v32 == *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8)) )
          v33 = 0;
        else
          v33 = *(_DWORD *)(*(_QWORD *)v32 + 8LL);
        v34 = sub_16D1B30(a4, v29->m128i_i64[0], v29->m128i_i64[1]);
        if ( v34 == -1 )
          break;
        v30 = *(_QWORD *)a4 + 8LL * v34;
        if ( v30 == *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8) )
          break;
        --v29;
        if ( *(_DWORD *)(*(_QWORD *)v30 + 8LL) >= v33 )
          goto LABEL_33;
      }
      --v29;
    }
    while ( v33 );
LABEL_33:
    if ( i < v22 )
    {
      v35 = i->m128i_i64[0];
      v36 = i->m128i_i64[1];
      *i = _mm_loadu_si128(v22);
      v22->m128i_i64[0] = v35;
      v22->m128i_i64[1] = v36;
      goto LABEL_35;
    }
    v4 = (char *)i - (char *)a1;
    sub_12D57D0(i, v66, v65, a4);
    if ( (char *)i - (char *)a1 > 256 )
    {
      if ( v65 )
      {
        v66 = i;
        continue;
      }
LABEL_52:
      v43 = v4 >> 4;
      for ( j = (v43 - 2) >> 1; ; --j )
      {
        sub_12D45C0((__int64)a1, j, v43, a1[j].m128i_i64[0], a1[j].m128i_i64[1], a4);
        if ( !j )
          break;
      }
      m128i_i64 = v67[-1].m128i_i64;
      do
      {
        v46 = *m128i_i64;
        v47 = (char *)m128i_i64 - (char *)a1;
        v48 = m128i_i64[1];
        v49 = (char *)m128i_i64 - (char *)a1;
        *(__m128i *)m128i_i64 = _mm_loadu_si128(a1);
        m128i_i64 -= 2;
        sub_12D45C0((__int64)a1, 0, v49 >> 4, v46, v48, a4);
      }
      while ( v47 > 16 );
    }
    break;
  }
}
