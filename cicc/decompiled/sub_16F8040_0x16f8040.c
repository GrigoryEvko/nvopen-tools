// Function: sub_16F8040
// Address: 0x16f8040
//
void __fastcall sub_16F8040(__int64 *a1)
{
  __int64 v2; // r12
  __int64 v3; // rbx
  __int64 v4; // rdi
  unsigned __int64 *v5; // rbx
  unsigned __int64 *v6; // r14
  unsigned __int64 v7; // rdi
  unsigned __int64 *v8; // rbx
  unsigned __int64 v9; // r14
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  __int64 v12; // r12
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  unsigned __int64 *v15; // rbx
  unsigned __int64 *v16; // rax
  unsigned __int64 v17; // rcx
  unsigned __int64 *v18; // rdi
  unsigned __int64 *v19; // rbx
  unsigned __int64 *v20; // r13
  unsigned __int64 v21; // rdi
  unsigned __int64 *v22; // rbx
  unsigned __int64 v23; // r13
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rdi

  v2 = a1[1];
  if ( v2 )
  {
    v3 = *(_QWORD *)(v2 + 136);
    while ( v3 )
    {
      sub_16F67F0(*(_QWORD *)(v3 + 24));
      v4 = v3;
      v3 = *(_QWORD *)(v3 + 16);
      j_j___libc_free_0(v4, 64);
    }
    v5 = *(unsigned __int64 **)(v2 + 24);
    v6 = &v5[*(unsigned int *)(v2 + 32)];
    while ( v6 != v5 )
    {
      v7 = *v5++;
      _libc_free(v7);
    }
    v8 = *(unsigned __int64 **)(v2 + 72);
    v9 = (unsigned __int64)&v8[2 * *(unsigned int *)(v2 + 80)];
    if ( v8 != (unsigned __int64 *)v9 )
    {
      do
      {
        v10 = *v8;
        v8 += 2;
        _libc_free(v10);
      }
      while ( (unsigned __int64 *)v9 != v8 );
      v9 = *(_QWORD *)(v2 + 72);
    }
    if ( v9 != v2 + 88 )
      _libc_free(v9);
    v11 = *(_QWORD *)(v2 + 24);
    if ( v11 != v2 + 40 )
      _libc_free(v11);
    j_j___libc_free_0(v2, 168);
  }
  v12 = *a1;
  if ( *a1 )
  {
    v13 = *(_QWORD *)(v12 + 232);
    if ( v13 != v12 + 248 )
      _libc_free(v13);
    v14 = *(_QWORD *)(v12 + 200);
    if ( v14 != v12 + 216 )
      _libc_free(v14);
    v15 = *(unsigned __int64 **)(v12 + 192);
    if ( (unsigned __int64 *)(v12 + 184) != v15 )
    {
      do
      {
        v16 = v15;
        v15 = (unsigned __int64 *)v15[1];
        v17 = *v16 & 0xFFFFFFFFFFFFFFF8LL;
        *v15 = v17 | *v15 & 7;
        *(_QWORD *)(v17 + 8) = v15;
        v18 = (unsigned __int64 *)v16[5];
        *v16 &= 7u;
        v16[1] = 0;
        if ( v18 != v16 + 7 )
          j_j___libc_free_0(v18, v16[7] + 1);
      }
      while ( v15 != (unsigned __int64 *)(v12 + 184) );
    }
    v19 = *(unsigned __int64 **)(v12 + 96);
    v20 = &v19[*(unsigned int *)(v12 + 104)];
    while ( v20 != v19 )
    {
      v21 = *v19++;
      _libc_free(v21);
    }
    v22 = *(unsigned __int64 **)(v12 + 144);
    v23 = (unsigned __int64)&v22[2 * *(unsigned int *)(v12 + 152)];
    if ( v22 != (unsigned __int64 *)v23 )
    {
      do
      {
        v24 = *v22;
        v22 += 2;
        _libc_free(v24);
      }
      while ( v22 != (unsigned __int64 *)v23 );
      v23 = *(_QWORD *)(v12 + 144);
    }
    if ( v23 != v12 + 160 )
      _libc_free(v23);
    v25 = *(_QWORD *)(v12 + 96);
    if ( v25 != v12 + 112 )
      _libc_free(v25);
    j_j___libc_free_0(v12, 352);
  }
}
