// Function: sub_1DB9980
// Address: 0x1db9980
//
void __fastcall sub_1DB9980(__int64 a1)
{
  int i; // ebx
  __int64 v3; // rax
  __int64 v4; // rdx
  unsigned __int64 *v5; // r12
  unsigned __int64 v6; // r14
  __int64 v7; // r15
  __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  unsigned __int64 **v10; // rbx
  __int64 v11; // rax
  unsigned __int64 *v12; // r15
  unsigned __int64 v13; // r14
  __int64 v14; // r12
  __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  __int64 v17; // r12
  unsigned __int64 *v18; // rbx
  unsigned __int64 *v19; // r12
  unsigned __int64 v20; // rdi
  __int64 v21; // rax
  _QWORD *v22; // rbx
  __int64 v23; // rdx
  unsigned __int64 *v24; // r12
  unsigned __int64 *v25; // rbx
  unsigned __int64 v26; // rdi
  int v27; // [rsp+8h] [rbp-38h]
  unsigned __int64 **j; // [rsp+8h] [rbp-38h]

  v27 = *(_DWORD *)(a1 + 408);
  if ( v27 )
  {
    for ( i = 0; i != v27; ++i )
    {
      v3 = *(_QWORD *)(a1 + 400);
      v4 = i & 0x7FFFFFFF;
      v5 = *(unsigned __int64 **)(v3 + 8 * v4);
      if ( v5 )
      {
        sub_1DB4CE0(*(_QWORD *)(v3 + 8 * v4));
        v6 = v5[12];
        if ( v6 )
        {
          v7 = *(_QWORD *)(v6 + 16);
          while ( v7 )
          {
            sub_1DB97B0(*(_QWORD *)(v7 + 24));
            v8 = v7;
            v7 = *(_QWORD *)(v7 + 16);
            j_j___libc_free_0(v8, 56);
          }
          j_j___libc_free_0(v6, 48);
        }
        v9 = v5[8];
        if ( (unsigned __int64 *)v9 != v5 + 10 )
          _libc_free(v9);
        if ( (unsigned __int64 *)*v5 != v5 + 2 )
          _libc_free(*v5);
        j_j___libc_free_0(v5, 120);
      }
    }
  }
  v10 = *(unsigned __int64 ***)(a1 + 672);
  v11 = *(unsigned int *)(a1 + 680);
  *(_DWORD *)(a1 + 408) = 0;
  *(_DWORD *)(a1 + 440) = 0;
  *(_DWORD *)(a1 + 520) = 0;
  *(_DWORD *)(a1 + 600) = 0;
  for ( j = &v10[v11]; j != v10; ++v10 )
  {
    v12 = *v10;
    if ( *v10 )
    {
      v13 = v12[12];
      if ( v13 )
      {
        v14 = *(_QWORD *)(v13 + 16);
        while ( v14 )
        {
          sub_1DB97B0(*(_QWORD *)(v14 + 24));
          v15 = v14;
          v14 = *(_QWORD *)(v14 + 16);
          j_j___libc_free_0(v15, 56);
        }
        j_j___libc_free_0(v13, 48);
      }
      v16 = v12[8];
      if ( (unsigned __int64 *)v16 != v12 + 10 )
        _libc_free(v16);
      if ( (unsigned __int64 *)*v12 != v12 + 2 )
        _libc_free(*v12);
      j_j___libc_free_0(v12, 104);
    }
  }
  v17 = *(unsigned int *)(a1 + 368);
  v18 = *(unsigned __int64 **)(a1 + 360);
  *(_DWORD *)(a1 + 680) = 0;
  v19 = &v18[2 * v17];
  while ( v19 != v18 )
  {
    v20 = *v18;
    v18 += 2;
    _libc_free(v20);
  }
  v21 = *(unsigned int *)(a1 + 320);
  *(_DWORD *)(a1 + 368) = 0;
  if ( (_DWORD)v21 )
  {
    v22 = *(_QWORD **)(a1 + 312);
    *(_QWORD *)(a1 + 376) = 0;
    v23 = *v22;
    v24 = &v22[v21];
    v25 = v22 + 1;
    *(_QWORD *)(a1 + 296) = v23;
    *(_QWORD *)(a1 + 304) = v23 + 4096;
    while ( v24 != v25 )
    {
      v26 = *v25++;
      _libc_free(v26);
    }
    *(_DWORD *)(a1 + 320) = 1;
  }
}
