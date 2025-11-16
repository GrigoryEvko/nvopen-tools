// Function: sub_12DD390
// Address: 0x12dd390
//
__int64 *__fastcall sub_12DD390(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v7; // r14
  bool v8; // cf
  unsigned __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // r15
  bool v12; // zf
  __int64 v13; // rsi
  __int64 v14; // r13
  unsigned int v15; // eax
  __int64 v16; // r13
  __int64 i; // r15
  void *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdi
  void *v22; // rax
  __int64 v23; // rdx
  const void *v24; // rsi
  __int64 v25; // rdi
  __int64 j; // r12
  __int64 v27; // rdi
  void *v29; // rax
  __int64 v30; // rdx
  const void *v31; // rsi
  __int64 v32; // r15
  __int64 v34; // [rsp+10h] [rbp-50h]
  __int64 v36; // [rsp+20h] [rbp-40h]
  __int64 v37; // [rsp+28h] [rbp-38h]

  v3 = a1[1];
  v37 = *a1;
  v4 = (v3 - *a1) >> 5;
  if ( v4 == 0x3FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v5 = 1;
  v7 = a2;
  if ( v4 )
    v5 = (v3 - *a1) >> 5;
  v8 = __CFADD__(v5, v4);
  v9 = v5 + v4;
  v10 = a2 - v37;
  if ( v8 )
  {
    v32 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v9 )
    {
      v34 = 0;
      v11 = 32;
      v36 = 0;
      goto LABEL_7;
    }
    if ( v9 > 0x3FFFFFFFFFFFFFFLL )
      v9 = 0x3FFFFFFFFFFFFFFLL;
    v32 = 32 * v9;
  }
  v36 = sub_22077B0(v32);
  v34 = v36 + v32;
  v11 = v36 + 32;
LABEL_7:
  v12 = v36 + v10 == 0;
  v13 = v36 + v10;
  v14 = v13;
  if ( !v12 )
  {
    *(_QWORD *)v13 = 0;
    *(_QWORD *)(v13 + 8) = 0;
    *(_QWORD *)(v13 + 16) = 0;
    *(_DWORD *)(v13 + 24) = 0;
    j___libc_free_0(0);
    v15 = *(_DWORD *)(a3 + 24);
    *(_DWORD *)(v13 + 24) = v15;
    if ( v15 )
    {
      v29 = (void *)sub_22077B0(16LL * v15);
      v30 = *(unsigned int *)(v13 + 24);
      *(_QWORD *)(v13 + 8) = v29;
      v31 = *(const void **)(a3 + 8);
      *(_QWORD *)(v14 + 16) = *(_QWORD *)(a3 + 16);
      memcpy(v29, v31, 16 * v30);
    }
    else
    {
      *(_QWORD *)(v13 + 8) = 0;
      *(_QWORD *)(v13 + 16) = 0;
    }
  }
  v16 = v37;
  if ( a2 != v37 )
  {
    for ( i = v36; ; i = v20 )
    {
      if ( !i )
        goto LABEL_13;
      *(_QWORD *)i = 0;
      *(_DWORD *)(i + 24) = 0;
      *(_QWORD *)(i + 8) = 0;
      *(_DWORD *)(i + 16) = 0;
      *(_DWORD *)(i + 20) = 0;
      j___libc_free_0(0);
      v21 = *(unsigned int *)(v16 + 24);
      *(_DWORD *)(i + 24) = v21;
      if ( (_DWORD)v21 )
        break;
      v16 += 32;
      *(_QWORD *)(i + 8) = 0;
      v20 = i + 32;
      *(_DWORD *)(i + 16) = 0;
      *(_DWORD *)(i + 20) = 0;
      if ( a2 == v16 )
      {
LABEL_18:
        v11 = i + 64;
        goto LABEL_19;
      }
LABEL_14:
      ;
    }
    v18 = (void *)sub_22077B0(16 * v21);
    v19 = *(unsigned int *)(i + 24);
    *(_QWORD *)(i + 8) = v18;
    *(_DWORD *)(i + 16) = *(_DWORD *)(v16 + 16);
    *(_DWORD *)(i + 20) = *(_DWORD *)(v16 + 20);
    memcpy(v18, *(const void **)(v16 + 8), 16 * v19);
LABEL_13:
    v16 += 32;
    v20 = i + 32;
    if ( a2 == v16 )
      goto LABEL_18;
    goto LABEL_14;
  }
LABEL_19:
  if ( a2 != v3 )
  {
    do
    {
      while ( 1 )
      {
        *(_QWORD *)v11 = 0;
        *(_DWORD *)(v11 + 24) = 0;
        *(_QWORD *)(v11 + 8) = 0;
        *(_DWORD *)(v11 + 16) = 0;
        *(_DWORD *)(v11 + 20) = 0;
        j___libc_free_0(0);
        v25 = *(unsigned int *)(v7 + 24);
        *(_DWORD *)(v11 + 24) = v25;
        if ( !(_DWORD)v25 )
          break;
        v7 += 32;
        v11 += 32;
        v22 = (void *)sub_22077B0(16 * v25);
        v23 = *(unsigned int *)(v11 - 8);
        v24 = *(const void **)(v7 - 24);
        *(_QWORD *)(v11 - 24) = v22;
        *(_DWORD *)(v11 - 16) = *(_DWORD *)(v7 - 16);
        *(_DWORD *)(v11 - 12) = *(_DWORD *)(v7 - 12);
        memcpy(v22, v24, 16 * v23);
        if ( v3 == v7 )
          goto LABEL_24;
      }
      v7 += 32;
      *(_QWORD *)(v11 + 8) = 0;
      v11 += 32;
      *(_DWORD *)(v11 - 16) = 0;
      *(_DWORD *)(v11 - 12) = 0;
    }
    while ( v3 != v7 );
  }
LABEL_24:
  for ( j = v37; j != v3; j += 32 )
  {
    v27 = *(_QWORD *)(j + 8);
    j___libc_free_0(v27);
  }
  if ( v37 )
    j_j___libc_free_0(v37, a1[2] - v37);
  *a1 = v36;
  a1[1] = v11;
  a1[2] = v34;
  return a1;
}
