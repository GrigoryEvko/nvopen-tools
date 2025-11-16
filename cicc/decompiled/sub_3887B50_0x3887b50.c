// Function: sub_3887B50
// Address: 0x3887b50
//
_DWORD *__fastcall sub_3887B50(int *a1, char *a2, _QWORD *a3)
{
  char *v3; // r15
  _QWORD *v4; // r12
  _DWORD *v6; // r13
  __int64 v7; // rax
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // r14
  char *v10; // rcx
  __int64 v11; // r14
  __m128i v12; // xmm2
  int v13; // eax
  __int64 v14; // rdi
  int *v15; // r15
  char *v16; // r14
  char *v17; // rbx
  __int64 v18; // rax
  __int64 v19; // rax
  _QWORD *v20; // rdx
  char *v21; // rdi
  size_t v22; // rdx
  char *v23; // rax
  __m128i v24; // xmm0
  int v25; // eax
  __int64 v26; // rdi
  _QWORD *v27; // rax
  _QWORD *v28; // r8
  __int64 v29; // rax
  size_t v30; // rdx
  char *v31; // rdx
  __int64 v32; // rax
  __int64 v34; // rdx
  __int64 j; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // r14
  _BYTE *v39; // rax
  __int64 v40; // rax
  size_t v41; // rcx
  __m128i v42; // xmm3
  __int64 v43; // rax
  char *v44; // rcx
  _QWORD *i; // rax
  __int64 v46; // rax
  _QWORD *v47; // [rsp+0h] [rbp-40h]
  _QWORD *v48; // [rsp+0h] [rbp-40h]
  _QWORD *v49; // [rsp+8h] [rbp-38h]
  size_t v50; // [rsp+8h] [rbp-38h]
  _QWORD *v51; // [rsp+8h] [rbp-38h]
  size_t v52; // [rsp+8h] [rbp-38h]
  _QWORD *v53; // [rsp+8h] [rbp-38h]
  size_t v54; // [rsp+8h] [rbp-38h]

  v3 = a2;
  v4 = a3;
  v6 = (_DWORD *)a3[1];
  if ( v6 )
  {
    v7 = *((_QWORD *)v6 + 1);
    a3[1] = v7;
    if ( v7 )
    {
      if ( v6 == *(_DWORD **)(v7 + 24) )
      {
        *(_QWORD *)(v7 + 24) = 0;
        a3 = *(_QWORD **)(a3[1] + 16LL);
        if ( a3 )
        {
          v4[1] = a3;
          for ( i = (_QWORD *)a3[3]; i; i = (_QWORD *)i[3] )
          {
            v4[1] = i;
            a3 = i;
          }
          v46 = a3[2];
          if ( v46 )
            v4[1] = v46;
        }
      }
      else
      {
        *(_QWORD *)(v7 + 16) = 0;
      }
    }
    else
    {
      *a3 = 0;
    }
    v8 = *((_QWORD *)v6 + 4);
    if ( v8 )
    {
      a2 = (char *)(*((_QWORD *)v6 + 6) - v8);
      j_j___libc_free_0(v8);
    }
    v9 = *((_QWORD *)a1 + 5) - *((_QWORD *)a1 + 4);
    *((_QWORD *)v6 + 4) = 0;
    *((_QWORD *)v6 + 5) = 0;
    *((_QWORD *)v6 + 6) = 0;
    if ( v9 )
    {
      if ( v9 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_60;
      v10 = (char *)sub_22077B0(v9);
    }
    else
    {
      v9 = 0;
      v10 = 0;
    }
    *((_QWORD *)v6 + 4) = v10;
    *((_QWORD *)v6 + 6) = &v10[v9];
    *((_QWORD *)v6 + 5) = v10;
    a2 = (char *)*((_QWORD *)a1 + 4);
    v11 = *((_QWORD *)a1 + 5) - (_QWORD)a2;
    if ( *((char **)a1 + 5) != a2 )
      v10 = (char *)memmove(v10, a2, *((_QWORD *)a1 + 5) - (_QWORD)a2);
    *((_QWORD *)v6 + 5) = &v10[v11];
    v12 = _mm_loadu_si128((const __m128i *)(a1 + 14));
    *((_QWORD *)v6 + 9) = *((_QWORD *)a1 + 9);
    *(__m128i *)(v6 + 14) = v12;
  }
  else
  {
    v8 = 80;
    v37 = sub_22077B0(0x50u);
    a2 = (char *)*((_QWORD *)a1 + 4);
    v38 = v37;
    v39 = (_BYTE *)*((_QWORD *)a1 + 5);
    *(_QWORD *)(v38 + 32) = 0;
    *(_QWORD *)(v38 + 40) = 0;
    *(_QWORD *)(v38 + 48) = 0;
    a3 = (_QWORD *)(v39 - a2);
    if ( v39 == a2 )
    {
      v41 = 0;
    }
    else
    {
      if ( (unsigned __int64)a3 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_60;
      v53 = (_QWORD *)(v39 - a2);
      v40 = sub_22077B0((unsigned __int64)a3);
      a2 = (char *)*((_QWORD *)a1 + 4);
      a3 = v53;
      v6 = (_DWORD *)v40;
      v39 = (_BYTE *)*((_QWORD *)a1 + 5);
      v41 = v39 - a2;
    }
    *(_QWORD *)(v38 + 32) = v6;
    *(_QWORD *)(v38 + 40) = v6;
    *(_QWORD *)(v38 + 48) = (char *)a3 + (_QWORD)v6;
    if ( v39 != a2 )
    {
      v54 = v41;
      memmove(v6, a2, v41);
      v41 = v54;
    }
    v42 = _mm_loadu_si128((const __m128i *)(a1 + 14));
    v43 = *((_QWORD *)a1 + 9);
    v44 = (char *)v6 + v41;
    v6 = (_DWORD *)v38;
    *(_QWORD *)(v38 + 40) = v44;
    *(_QWORD *)(v38 + 72) = v43;
    *(__m128i *)(v38 + 56) = v42;
  }
  v13 = *a1;
  *((_QWORD *)v6 + 2) = 0;
  *((_QWORD *)v6 + 3) = 0;
  *v6 = v13;
  *((_QWORD *)v6 + 1) = v3;
  v14 = *((_QWORD *)a1 + 3);
  if ( v14 )
  {
    a2 = (char *)v6;
    *((_QWORD *)v6 + 3) = sub_3887B50(v14, v6, v4);
  }
  v15 = (int *)*((_QWORD *)a1 + 2);
  v16 = (char *)v6;
  if ( !v15 )
    return v6;
  v17 = (char *)v4[1];
  if ( !v17 )
    goto LABEL_32;
LABEL_17:
  v18 = *((_QWORD *)v17 + 1);
  v4[1] = v18;
  if ( v18 )
  {
    if ( v17 == *(char **)(v18 + 24) )
    {
      *(_QWORD *)(v18 + 24) = 0;
      v34 = *(_QWORD *)(v4[1] + 16LL);
      if ( v34 )
      {
        v4[1] = v34;
        for ( j = *(_QWORD *)(v34 + 24); j; j = *(_QWORD *)(j + 24) )
        {
          v4[1] = j;
          v34 = j;
        }
        v36 = *(_QWORD *)(v34 + 16);
        if ( v36 )
          v4[1] = v36;
      }
    }
    else
    {
      *(_QWORD *)(v18 + 16) = 0;
    }
  }
  else
  {
    *v4 = 0;
  }
  v8 = *((_QWORD *)v17 + 4);
  if ( v8 )
  {
    a2 = (char *)(*((_QWORD *)v17 + 6) - v8);
    j_j___libc_free_0(v8);
  }
  a3 = (_QWORD *)(*((_QWORD *)v15 + 5) - *((_QWORD *)v15 + 4));
  *((_QWORD *)v17 + 4) = 0;
  *((_QWORD *)v17 + 5) = 0;
  *((_QWORD *)v17 + 6) = 0;
  if ( a3 )
  {
    if ( (unsigned __int64)a3 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v49 = a3;
      v19 = sub_22077B0((unsigned __int64)a3);
      v20 = v49;
      v21 = (char *)v19;
      goto LABEL_25;
    }
LABEL_60:
    sub_4261EA(v8, a2, a3);
  }
  v20 = 0;
  v21 = 0;
LABEL_25:
  *((_QWORD *)v17 + 4) = v21;
  *((_QWORD *)v17 + 6) = (char *)v20 + (_QWORD)v21;
  *((_QWORD *)v17 + 5) = v21;
  a2 = (char *)*((_QWORD *)v15 + 4);
  v22 = *((_QWORD *)v15 + 5) - (_QWORD)a2;
  if ( *((char **)v15 + 5) != a2 )
  {
    v50 = *((_QWORD *)v15 + 5) - (_QWORD)a2;
    v23 = (char *)memmove(v21, a2, v22);
    v22 = v50;
    v21 = v23;
  }
  *((_QWORD *)v17 + 5) = &v21[v22];
  v24 = _mm_loadu_si128((const __m128i *)(v15 + 14));
  *((_QWORD *)v17 + 9) = *((_QWORD *)v15 + 9);
  *(__m128i *)(v17 + 56) = v24;
  while ( 1 )
  {
    v25 = *v15;
    *((_QWORD *)v17 + 2) = 0;
    *((_QWORD *)v17 + 3) = 0;
    *(_DWORD *)v17 = v25;
    *((_QWORD *)v16 + 2) = v17;
    *((_QWORD *)v17 + 1) = v16;
    v26 = *((_QWORD *)v15 + 3);
    if ( v26 )
    {
      a2 = v17;
      *((_QWORD *)v17 + 3) = sub_3887B50(v26, v17, v4);
    }
    v15 = (int *)*((_QWORD *)v15 + 2);
    if ( !v15 )
      return v6;
    v16 = v17;
    v17 = (char *)v4[1];
    if ( v17 )
      goto LABEL_17;
LABEL_32:
    v8 = 80;
    v27 = (_QWORD *)sub_22077B0(0x50u);
    a3 = (_QWORD *)(*((_QWORD *)v15 + 5) - *((_QWORD *)v15 + 4));
    v28 = v27;
    v27[4] = 0;
    v27[5] = 0;
    v27[6] = 0;
    if ( a3 )
    {
      v47 = v27;
      if ( (unsigned __int64)a3 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_60;
      v51 = a3;
      v29 = sub_22077B0((unsigned __int64)a3);
      v28 = v47;
      a3 = v51;
      v17 = (char *)v29;
    }
    v28[4] = v17;
    v28[6] = (char *)a3 + (_QWORD)v17;
    v28[5] = v17;
    a2 = (char *)*((_QWORD *)v15 + 4);
    v30 = *((_QWORD *)v15 + 5) - (_QWORD)a2;
    if ( *((char **)v15 + 5) != a2 )
    {
      v48 = v28;
      v52 = *((_QWORD *)v15 + 5) - (_QWORD)a2;
      memmove(v17, a2, v30);
      v28 = v48;
      v30 = v52;
    }
    v31 = &v17[v30];
    v17 = (char *)v28;
    v28[5] = v31;
    v32 = *((_QWORD *)v15 + 9);
    *(__m128i *)(v28 + 7) = _mm_loadu_si128((const __m128i *)(v15 + 14));
    v28[9] = v32;
  }
}
