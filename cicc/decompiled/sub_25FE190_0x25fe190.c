// Function: sub_25FE190
// Address: 0x25fe190
//
unsigned __int64 *__fastcall sub_25FE190(unsigned __int64 *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v7; // rdx
  __int64 v8; // r14
  bool v9; // cf
  unsigned __int64 v10; // rax
  __int64 v11; // r9
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // rdi
  int v15; // edx
  __int64 v16; // rdi
  unsigned __int64 v17; // r13
  unsigned __int64 i; // r15
  void *v19; // rax
  __int64 v20; // rdx
  unsigned __int64 v21; // rax
  __int64 v22; // rdi
  void *v23; // rax
  __int64 v24; // rdx
  const void *v25; // rsi
  __int64 v26; // rdi
  unsigned __int64 j; // r12
  __int64 v28; // rsi
  __int64 v29; // rdi
  unsigned __int64 v31; // r15
  __int64 v32; // rax
  unsigned __int64 v33; // [rsp+10h] [rbp-50h]
  unsigned __int64 v35; // [rsp+20h] [rbp-40h]
  unsigned __int64 v36; // [rsp+28h] [rbp-38h]

  v3 = a1[1];
  v36 = *a1;
  v4 = (__int64)(v3 - *a1) >> 5;
  if ( v4 == 0x3FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  if ( v4 )
    v7 = (__int64)(v3 - *a1) >> 5;
  v8 = a2;
  v9 = __CFADD__(v7, v4);
  v10 = v7 + v4;
  v11 = a2 - v36;
  if ( v9 )
  {
    v31 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v10 )
    {
      v33 = 0;
      v12 = 32;
      v35 = 0;
      goto LABEL_7;
    }
    if ( v10 > 0x3FFFFFFFFFFFFFFLL )
      v10 = 0x3FFFFFFFFFFFFFFLL;
    v31 = 32 * v10;
  }
  v32 = sub_22077B0(v31);
  v11 = a2 - v36;
  v35 = v32;
  v33 = v32 + v31;
  v12 = v32 + 32;
LABEL_7:
  v13 = v35 + v11;
  if ( v35 + v11 )
  {
    v14 = *(_QWORD *)(a3 + 8);
    v15 = *(_DWORD *)(a3 + 24);
    *(_QWORD *)v13 = 1;
    ++*(_QWORD *)a3;
    *(_QWORD *)(v13 + 8) = v14;
    v16 = *(_QWORD *)(a3 + 16);
    *(_DWORD *)(v13 + 24) = v15;
    *(_QWORD *)(v13 + 16) = v16;
    *(_QWORD *)(a3 + 8) = 0;
    *(_QWORD *)(a3 + 16) = 0;
    *(_DWORD *)(a3 + 24) = 0;
  }
  v17 = v36;
  if ( a2 != v36 )
  {
    for ( i = v35; ; i = v21 )
    {
      if ( !i )
        goto LABEL_12;
      *(_QWORD *)i = 0;
      *(_DWORD *)(i + 24) = 0;
      *(_QWORD *)(i + 8) = 0;
      *(_DWORD *)(i + 16) = 0;
      *(_DWORD *)(i + 20) = 0;
      sub_C7D6A0(0, 0, 8);
      v22 = *(unsigned int *)(v17 + 24);
      *(_DWORD *)(i + 24) = v22;
      if ( (_DWORD)v22 )
        break;
      v17 += 32LL;
      *(_QWORD *)(i + 8) = 0;
      v21 = i + 32;
      *(_DWORD *)(i + 16) = 0;
      *(_DWORD *)(i + 20) = 0;
      if ( a2 == v17 )
      {
LABEL_17:
        v12 = i + 64;
        goto LABEL_18;
      }
LABEL_13:
      ;
    }
    v19 = (void *)sub_C7D670(16 * v22, 8);
    v20 = *(unsigned int *)(i + 24);
    *(_QWORD *)(i + 8) = v19;
    *(_DWORD *)(i + 16) = *(_DWORD *)(v17 + 16);
    *(_DWORD *)(i + 20) = *(_DWORD *)(v17 + 20);
    memcpy(v19, *(const void **)(v17 + 8), 16 * v20);
LABEL_12:
    v17 += 32LL;
    v21 = i + 32;
    if ( a2 == v17 )
      goto LABEL_17;
    goto LABEL_13;
  }
LABEL_18:
  if ( a2 != v3 )
  {
    do
    {
      while ( 1 )
      {
        *(_QWORD *)v12 = 0;
        *(_DWORD *)(v12 + 24) = 0;
        *(_QWORD *)(v12 + 8) = 0;
        *(_DWORD *)(v12 + 16) = 0;
        *(_DWORD *)(v12 + 20) = 0;
        sub_C7D6A0(0, 0, 8);
        v26 = *(unsigned int *)(v8 + 24);
        *(_DWORD *)(v12 + 24) = v26;
        if ( !(_DWORD)v26 )
          break;
        v8 += 32;
        v12 += 32;
        v23 = (void *)sub_C7D670(16 * v26, 8);
        v24 = *(unsigned int *)(v12 - 8);
        v25 = *(const void **)(v8 - 24);
        *(_QWORD *)(v12 - 24) = v23;
        *(_DWORD *)(v12 - 16) = *(_DWORD *)(v8 - 16);
        *(_DWORD *)(v12 - 12) = *(_DWORD *)(v8 - 12);
        memcpy(v23, v25, 16 * v24);
        if ( v3 == v8 )
          goto LABEL_23;
      }
      v8 += 32;
      *(_QWORD *)(v12 + 8) = 0;
      v12 += 32;
      *(_DWORD *)(v12 - 16) = 0;
      *(_DWORD *)(v12 - 12) = 0;
    }
    while ( v3 != v8 );
  }
LABEL_23:
  for ( j = v36; j != v3; j += 32LL )
  {
    v28 = *(unsigned int *)(j + 24);
    v29 = *(_QWORD *)(j + 8);
    sub_C7D6A0(v29, 16 * v28, 8);
  }
  if ( v36 )
    j_j___libc_free_0(v36);
  *a1 = v35;
  a1[1] = v12;
  a1[2] = v33;
  return a1;
}
