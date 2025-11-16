// Function: sub_35BC090
// Address: 0x35bc090
//
_QWORD *__fastcall sub_35BC090(_QWORD *a1, __int64 a2, _QWORD *a3)
{
  __int64 v3; // rax
  unsigned int v4; // ebx
  __int64 v5; // rbx
  const void **v6; // r12
  __int64 v7; // r15
  float *v8; // rax
  float *v9; // r13
  float *v10; // r15
  __int64 v11; // rbx
  unsigned int *v12; // rax
  unsigned int *v13; // rbx
  __int64 v14; // rdx
  unsigned int *v15; // r13
  unsigned int v16; // edi
  __int64 v17; // rax
  _QWORD *v18; // r9
  __int64 v19; // rcx
  __int64 v20; // rdx
  int v21; // r12d
  float *v22; // rax
  float *v23; // rdi
  unsigned int v24; // eax
  unsigned int i; // edx
  __int64 v26; // rcx
  float *v27; // rdx
  float *j; // rax
  float v29; // xmm0_4
  float *v30; // rax
  float *v31; // rbx
  __int64 v32; // rbx
  __int64 v33; // r12
  __int64 v34; // rax
  __int64 v35; // rcx
  __int64 v36; // rdx
  __int64 v37; // rsi
  __int64 v38; // rax
  __int64 v39; // rdx
  char v40; // di
  unsigned int v42; // edi
  _QWORD *v43; // r9
  __int64 v44; // rcx
  __int64 v45; // rdx
  unsigned __int64 v46; // r12
  float *v47; // rax
  unsigned int v48; // eax
  float v49; // xmm0_4
  __int64 v50; // rdx
  float *v51; // rdx
  float *v52; // rax
  float v53; // xmm0_4
  __int64 v54; // rbx
  unsigned __int64 v55; // rdi
  _QWORD *v57; // [rsp+8h] [rbp-78h]
  __int64 n; // [rsp+10h] [rbp-70h]
  unsigned int *v59; // [rsp+18h] [rbp-68h]
  float *v60; // [rsp+20h] [rbp-60h]
  __int64 v63; // [rsp+38h] [rbp-48h]
  unsigned int v64; // [rsp+38h] [rbp-48h]
  size_t v65; // [rsp+40h] [rbp-40h]
  unsigned int v66; // [rsp+4Ch] [rbp-34h]

  *((_DWORD *)a1 + 2) = 0;
  a1[2] = 0;
  a1[3] = a1 + 1;
  a1[4] = a1 + 1;
  a1[5] = 0;
  v3 = a3[1];
  v57 = a1 + 1;
  while ( *a3 != v3 )
  {
    v4 = *(_DWORD *)(v3 - 4);
    a3[1] = v3 - 4;
    v66 = v4;
    v5 = 96LL * v4;
    v6 = *(const void ***)(*(_QWORD *)(a2 + 160) + v5);
    v7 = *(unsigned int *)v6;
    n = 4 * v7;
    v8 = (float *)sub_2207820(4 * v7);
    v9 = v8;
    if ( v8 && v7 )
      memset(v8, 0, n);
    if ( n )
    {
      v10 = &v9[(unsigned __int64)n / 4];
      memmove(v9, v6[1], n);
      v11 = *(_QWORD *)(a2 + 160) + v5;
      v12 = *(unsigned int **)(v11 + 72);
      v59 = *(unsigned int **)(v11 + 80);
      if ( v12 == v59 )
        goto LABEL_29;
    }
    else
    {
      v10 = v9;
      v54 = *(_QWORD *)(a2 + 160) + v5;
      v12 = *(unsigned int **)(v54 + 72);
      v59 = *(unsigned int **)(v54 + 80);
      if ( v12 == v59 )
        goto LABEL_69;
    }
    v60 = v9;
    v13 = v12;
    do
    {
      v14 = *(_QWORD *)(a2 + 208) + 48LL * *v13;
      v15 = *(unsigned int **)v14;
      v16 = *(_DWORD *)(v14 + 20);
      v17 = a1[2];
      if ( v66 == v16 )
      {
        v42 = *(_DWORD *)(v14 + 24);
        v43 = v57;
        if ( v17 )
        {
          do
          {
            while ( 1 )
            {
              v44 = *(_QWORD *)(v17 + 16);
              v45 = *(_QWORD *)(v17 + 24);
              if ( v42 <= *(_DWORD *)(v17 + 32) )
                break;
              v17 = *(_QWORD *)(v17 + 24);
              if ( !v45 )
                goto LABEL_56;
            }
            v43 = (_QWORD *)v17;
            v17 = *(_QWORD *)(v17 + 16);
          }
          while ( v44 );
LABEL_56:
          if ( v57 != v43 && v42 < *((_DWORD *)v43 + 8) )
            v43 = v57;
        }
        v46 = 4LL * *v15;
        v65 = *v15;
        v64 = *((_DWORD *)v43 + 9);
        v47 = (float *)sub_2207820(v46);
        v23 = v47;
        if ( v47 && v65 )
          v23 = (float *)memset(v47, 0, v46);
        if ( *v15 )
        {
          v48 = 0;
          do
          {
            v49 = *(float *)(*((_QWORD *)v15 + 1) + 4 * (v64 + (unsigned __int64)(v15[1] * v48)));
            v50 = v48++;
            v23[v50] = v49;
          }
          while ( *v15 > v48 );
        }
        v51 = v23;
        v52 = v60;
        if ( v60 != v10 )
        {
          do
          {
            v53 = *v52 + *v51++;
            *v52++ = v53;
          }
          while ( v51 != &v23[(unsigned __int64)n / 4] );
        }
      }
      else
      {
        v18 = v57;
        if ( v17 )
        {
          do
          {
            while ( 1 )
            {
              v19 = *(_QWORD *)(v17 + 16);
              v20 = *(_QWORD *)(v17 + 24);
              if ( v16 <= *(_DWORD *)(v17 + 32) )
                break;
              v17 = *(_QWORD *)(v17 + 24);
              if ( !v20 )
                goto LABEL_14;
            }
            v18 = (_QWORD *)v17;
            v17 = *(_QWORD *)(v17 + 16);
          }
          while ( v19 );
LABEL_14:
          if ( v57 != v18 && v16 < *((_DWORD *)v18 + 8) )
            v18 = v57;
        }
        v21 = *((_DWORD *)v18 + 9);
        v63 = v15[1];
        v22 = (float *)sub_2207820(4 * v63);
        v23 = v22;
        if ( v22 && v63 )
          v23 = (float *)memset(v22, 0, 4 * v63);
        v24 = v15[1];
        if ( v24 )
        {
          for ( i = 0; i < v24; ++i )
          {
            v26 = i;
            v23[v26] = *(float *)(*((_QWORD *)v15 + 1) + 4 * (v26 + v21 * v24));
            v24 = v15[1];
          }
        }
        v27 = v23;
        for ( j = v60; j != v10; *(j - 1) = v29 )
          v29 = *j++ + *v27++;
      }
      if ( v23 )
        j_j___libc_free_0_0((unsigned __int64)v23);
      ++v13;
    }
    while ( v59 != v13 );
    v9 = v60;
LABEL_29:
    if ( v10 == v9 || (v30 = v9 + 1, v31 = v9, v10 == v9 + 1) )
    {
LABEL_69:
      LODWORD(v32) = 0;
      goto LABEL_35;
    }
    do
    {
      if ( *v31 > *v30 )
        v31 = v30;
      ++v30;
    }
    while ( v10 != v30 );
    v32 = v31 - v9;
LABEL_35:
    v33 = (__int64)v57;
    v34 = a1[2];
    if ( !v34 )
      goto LABEL_42;
    do
    {
      while ( 1 )
      {
        v35 = *(_QWORD *)(v34 + 16);
        v36 = *(_QWORD *)(v34 + 24);
        if ( v66 <= *(_DWORD *)(v34 + 32) )
          break;
        v34 = *(_QWORD *)(v34 + 24);
        if ( !v36 )
          goto LABEL_40;
      }
      v33 = v34;
      v34 = *(_QWORD *)(v34 + 16);
    }
    while ( v35 );
LABEL_40:
    if ( v57 == (_QWORD *)v33 || v66 < *(_DWORD *)(v33 + 32) )
    {
LABEL_42:
      v37 = v33;
      v33 = sub_22077B0(0x28u);
      *(_DWORD *)(v33 + 36) = 0;
      *(_DWORD *)(v33 + 32) = v66;
      v38 = sub_609E00(a1, v37, (unsigned int *)(v33 + 32));
      if ( v39 )
      {
        v40 = v38 || v57 == (_QWORD *)v39 || v66 < *(_DWORD *)(v39 + 32);
        sub_220F040(v40, v33, (_QWORD *)v39, v57);
        ++a1[5];
      }
      else
      {
        v55 = v33;
        v33 = v38;
        j_j___libc_free_0(v55);
      }
    }
    *(_DWORD *)(v33 + 36) = v32;
    if ( v9 )
      j_j___libc_free_0_0((unsigned __int64)v9);
    v3 = a3[1];
  }
  return a1;
}
