// Function: sub_2E10910
// Address: 0x2e10910
//
void __fastcall sub_2E10910(__int64 a1)
{
  int v2; // ebx
  __int64 v3; // rax
  __int64 v4; // rdx
  unsigned __int64 *v5; // r12
  unsigned __int64 v6; // r14
  unsigned __int64 v7; // r15
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  unsigned __int64 **v10; // rbx
  __int64 v11; // rax
  unsigned __int64 *v12; // r15
  unsigned __int64 v13; // r14
  unsigned __int64 v14; // r12
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  __int64 v17; // r12
  __int64 *v18; // rbx
  __int64 *v19; // r12
  __int64 v20; // rsi
  __int64 v21; // rdi
  __int64 v22; // rdx
  __int64 *v23; // rax
  __int64 v24; // rcx
  __int64 *v25; // rbx
  __int64 *v26; // r14
  __int64 v27; // rdi
  unsigned int v28; // ecx
  __int64 v29; // rsi
  int v30; // [rsp+8h] [rbp-38h]
  unsigned __int64 **i; // [rsp+8h] [rbp-38h]

  v2 = 0;
  v30 = *(_DWORD *)(a1 + 160);
  if ( v30 )
  {
    do
    {
      v3 = *(_QWORD *)(a1 + 152);
      v4 = v2 & 0x7FFFFFFF;
      v5 = *(unsigned __int64 **)(v3 + 8 * v4);
      if ( v5 )
      {
        sub_2E0AFD0(*(_QWORD *)(v3 + 8 * v4));
        v6 = v5[12];
        if ( v6 )
        {
          v7 = *(_QWORD *)(v6 + 16);
          while ( v7 )
          {
            sub_2E10270(*(_QWORD *)(v7 + 24));
            v8 = v7;
            v7 = *(_QWORD *)(v7 + 16);
            j_j___libc_free_0(v8);
          }
          j_j___libc_free_0(v6);
        }
        v9 = v5[8];
        if ( (unsigned __int64 *)v9 != v5 + 10 )
          _libc_free(v9);
        if ( (unsigned __int64 *)*v5 != v5 + 2 )
          _libc_free(*v5);
        j_j___libc_free_0((unsigned __int64)v5);
      }
      ++v2;
    }
    while ( v2 != v30 );
  }
  v10 = *(unsigned __int64 ***)(a1 + 424);
  v11 = *(unsigned int *)(a1 + 432);
  *(_DWORD *)(a1 + 160) = 0;
  *(_DWORD *)(a1 + 192) = 0;
  *(_DWORD *)(a1 + 272) = 0;
  *(_DWORD *)(a1 + 352) = 0;
  for ( i = &v10[v11]; i != v10; ++v10 )
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
          sub_2E10270(*(_QWORD *)(v14 + 24));
          v15 = v14;
          v14 = *(_QWORD *)(v14 + 16);
          j_j___libc_free_0(v15);
        }
        j_j___libc_free_0(v13);
      }
      v16 = v12[8];
      if ( (unsigned __int64 *)v16 != v12 + 10 )
        _libc_free(v16);
      if ( (unsigned __int64 *)*v12 != v12 + 2 )
        _libc_free(*v12);
      j_j___libc_free_0((unsigned __int64)v12);
    }
  }
  v17 = *(unsigned int *)(a1 + 128);
  v18 = *(__int64 **)(a1 + 120);
  *(_DWORD *)(a1 + 432) = 0;
  v19 = &v18[2 * v17];
  while ( v19 != v18 )
  {
    v20 = v18[1];
    v21 = *v18;
    v18 += 2;
    sub_C7D6A0(v21, v20, 16);
  }
  *(_DWORD *)(a1 + 128) = 0;
  v22 = *(unsigned int *)(a1 + 80);
  if ( (_DWORD)v22 )
  {
    *(_QWORD *)(a1 + 136) = 0;
    v23 = *(__int64 **)(a1 + 72);
    v24 = *v23;
    v25 = &v23[v22];
    v26 = v23 + 1;
    *(_QWORD *)(a1 + 56) = *v23;
    *(_QWORD *)(a1 + 64) = v24 + 4096;
    if ( v25 != v23 + 1 )
    {
      while ( 1 )
      {
        v27 = *v26;
        v28 = (unsigned int)(v26 - v23) >> 7;
        v29 = 4096LL << v28;
        if ( v28 >= 0x1E )
          v29 = 0x40000000000LL;
        ++v26;
        sub_C7D6A0(v27, v29, 16);
        if ( v25 == v26 )
          break;
        v23 = *(__int64 **)(a1 + 72);
      }
    }
    *(_DWORD *)(a1 + 80) = 1;
  }
}
