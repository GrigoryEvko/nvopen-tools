// Function: sub_2265400
// Address: 0x2265400
//
unsigned __int64 *__fastcall sub_2265400(unsigned __int64 *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v7; // r14
  bool v8; // cf
  unsigned __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // r15
  bool v12; // zf
  __int64 v13; // r8
  __int64 v14; // r13
  unsigned int v15; // eax
  unsigned __int64 v16; // r13
  unsigned __int64 i; // r15
  void *v18; // rax
  __int64 v19; // rdx
  unsigned __int64 v20; // rax
  __int64 v21; // rdi
  void *v22; // rax
  __int64 v23; // rdx
  const void *v24; // rsi
  __int64 v25; // rdi
  unsigned __int64 j; // r12
  __int64 v27; // rsi
  __int64 v28; // rdi
  void *v30; // rax
  __int64 v31; // rdx
  const void *v32; // rsi
  unsigned __int64 v33; // r15
  __int64 v34; // rax
  unsigned __int64 v36; // [rsp+10h] [rbp-50h]
  unsigned __int64 v38; // [rsp+20h] [rbp-40h]
  unsigned __int64 v39; // [rsp+28h] [rbp-38h]

  v3 = a1[1];
  v39 = *a1;
  v4 = (__int64)(v3 - *a1) >> 5;
  if ( v4 == 0x3FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v5 = 1;
  if ( v4 )
    v5 = (__int64)(v3 - *a1) >> 5;
  v7 = a2;
  v8 = __CFADD__(v5, v4);
  v9 = v5 + v4;
  v10 = a2 - v39;
  if ( v8 )
  {
    v33 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v9 )
    {
      v36 = 0;
      v11 = 32;
      v38 = 0;
      goto LABEL_7;
    }
    if ( v9 > 0x3FFFFFFFFFFFFFFLL )
      v9 = 0x3FFFFFFFFFFFFFFLL;
    v33 = 32 * v9;
  }
  v34 = sub_22077B0(v33);
  v10 = a2 - v39;
  v38 = v34;
  v36 = v34 + v33;
  v11 = v34 + 32;
LABEL_7:
  v12 = v38 + v10 == 0;
  v13 = v38 + v10;
  v14 = v13;
  if ( !v12 )
  {
    *(_QWORD *)v13 = 0;
    *(_QWORD *)(v13 + 8) = 0;
    *(_QWORD *)(v13 + 16) = 0;
    *(_DWORD *)(v13 + 24) = 0;
    sub_C7D6A0(0, 0, 8);
    v15 = *(_DWORD *)(a3 + 24);
    *(_DWORD *)(v14 + 24) = v15;
    if ( v15 )
    {
      v30 = (void *)sub_C7D670(16LL * v15, 8);
      v31 = *(unsigned int *)(v14 + 24);
      *(_QWORD *)(v14 + 8) = v30;
      v32 = *(const void **)(a3 + 8);
      *(_QWORD *)(v14 + 16) = *(_QWORD *)(a3 + 16);
      memcpy(v30, v32, 16 * v31);
    }
    else
    {
      *(_QWORD *)(v14 + 8) = 0;
      *(_QWORD *)(v14 + 16) = 0;
    }
  }
  v16 = v39;
  if ( a2 != v39 )
  {
    for ( i = v38; ; i = v20 )
    {
      if ( !i )
        goto LABEL_13;
      *(_QWORD *)i = 0;
      *(_DWORD *)(i + 24) = 0;
      *(_QWORD *)(i + 8) = 0;
      *(_DWORD *)(i + 16) = 0;
      *(_DWORD *)(i + 20) = 0;
      sub_C7D6A0(0, 0, 8);
      v21 = *(unsigned int *)(v16 + 24);
      *(_DWORD *)(i + 24) = v21;
      if ( (_DWORD)v21 )
        break;
      v16 += 32LL;
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
    v18 = (void *)sub_C7D670(16 * v21, 8);
    v19 = *(unsigned int *)(i + 24);
    *(_QWORD *)(i + 8) = v18;
    *(_DWORD *)(i + 16) = *(_DWORD *)(v16 + 16);
    *(_DWORD *)(i + 20) = *(_DWORD *)(v16 + 20);
    memcpy(v18, *(const void **)(v16 + 8), 16 * v19);
LABEL_13:
    v16 += 32LL;
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
        sub_C7D6A0(0, 0, 8);
        v25 = *(unsigned int *)(v7 + 24);
        *(_DWORD *)(v11 + 24) = v25;
        if ( !(_DWORD)v25 )
          break;
        v7 += 32;
        v11 += 32;
        v22 = (void *)sub_C7D670(16 * v25, 8);
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
  for ( j = v39; j != v3; j += 32LL )
  {
    v27 = *(unsigned int *)(j + 24);
    v28 = *(_QWORD *)(j + 8);
    sub_C7D6A0(v28, 16 * v27, 8);
  }
  if ( v39 )
    j_j___libc_free_0(v39);
  *a1 = v38;
  a1[1] = v11;
  a1[2] = v36;
  return a1;
}
