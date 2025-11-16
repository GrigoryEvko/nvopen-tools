// Function: sub_163BFE0
// Address: 0x163bfe0
//
void __fastcall sub_163BFE0(__int64 a1)
{
  __int64 *v2; // r12
  __int64 v3; // rsi
  __int64 v4; // rdx
  unsigned int v5; // ecx
  __int64 v6; // rbx
  bool v7; // cf
  __int64 v8; // rcx
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rbx
  unsigned __int64 i; // r15
  __int64 v12; // rdi
  _QWORD *v13; // r13
  __int64 v14; // r15
  unsigned __int64 v15; // r12
  unsigned __int64 j; // rbx
  __int64 v17; // rdi
  unsigned __int64 *v18; // rbx
  unsigned __int64 *v19; // r12
  unsigned __int64 v20; // rdi
  __int64 v21; // rax
  _QWORD *v22; // rbx
  __int64 v23; // rdx
  unsigned __int64 *v24; // r12
  unsigned __int64 *v25; // rbx
  unsigned __int64 v26; // rdi
  __int64 *v27; // [rsp+8h] [rbp-38h]
  _QWORD *v28; // [rsp+8h] [rbp-38h]

  v2 = *(__int64 **)(a1 + 16);
  v3 = *(unsigned int *)(a1 + 24);
  v27 = &v2[v3];
  if ( v2 != v27 )
  {
    v4 = *(_QWORD *)(a1 + 16);
    while ( 1 )
    {
      v5 = (unsigned int)(((__int64)v2 - v4) >> 3) >> 7;
      v6 = 4096LL << v5;
      v7 = v5 < 0x1E;
      v8 = *v2;
      if ( !v7 )
        v6 = 0x40000000000LL;
      v9 = (v8 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      v10 = v8 + v6;
      if ( v8 == *(_QWORD *)(v4 + 8 * v3 - 8) )
        v10 = *(_QWORD *)a1;
      for ( i = v9 + 104; v10 >= i; j___libc_free_0(*(_QWORD *)(i - 200)) )
      {
        v12 = *(_QWORD *)(i - 32);
        i += 104LL;
        j___libc_free_0(v12);
        j___libc_free_0(*(_QWORD *)(i - 168));
      }
      if ( v27 == ++v2 )
        break;
      v4 = *(_QWORD *)(a1 + 16);
      v3 = *(unsigned int *)(a1 + 24);
    }
  }
  v13 = *(_QWORD **)(a1 + 64);
  v14 = 2LL * *(unsigned int *)(a1 + 72);
  v28 = &v13[v14];
  if ( &v13[v14] != v13 )
  {
    do
    {
      v15 = *v13 + v13[1];
      for ( j = ((*v13 + 7LL) & 0xFFFFFFFFFFFFFFF8LL) + 104; v15 >= j; j___libc_free_0(*(_QWORD *)(j - 200)) )
      {
        v17 = *(_QWORD *)(j - 32);
        j += 104LL;
        j___libc_free_0(v17);
        j___libc_free_0(*(_QWORD *)(j - 168));
      }
      v13 += 2;
    }
    while ( v28 != v13 );
    v18 = *(unsigned __int64 **)(a1 + 64);
    v19 = &v18[2 * *(unsigned int *)(a1 + 72)];
    while ( v18 != v19 )
    {
      v20 = *v18;
      v18 += 2;
      _libc_free(v20);
    }
  }
  v21 = *(unsigned int *)(a1 + 24);
  *(_DWORD *)(a1 + 72) = 0;
  if ( (_DWORD)v21 )
  {
    v22 = *(_QWORD **)(a1 + 16);
    *(_QWORD *)(a1 + 80) = 0;
    v23 = *v22;
    v24 = &v22[v21];
    v25 = v22 + 1;
    *(_QWORD *)a1 = v23;
    *(_QWORD *)(a1 + 8) = v23 + 4096;
    while ( v24 != v25 )
    {
      v26 = *v25++;
      _libc_free(v26);
    }
    *(_DWORD *)(a1 + 24) = 1;
  }
}
