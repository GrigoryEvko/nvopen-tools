// Function: sub_2102890
// Address: 0x2102890
//
void *__fastcall sub_2102890(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  unsigned __int64 v6; // r9
  __int64 v7; // r13
  __int64 v8; // rsi
  __int64 v9; // rbx
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  unsigned __int64 *v12; // rbx
  __int64 v13; // rax
  unsigned __int64 *v14; // r13
  unsigned __int64 v15; // rdi
  unsigned __int64 *v16; // rbx
  unsigned __int64 v17; // r13
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi

  *(_QWORD *)a1 = &unk_4A00C40;
  _libc_free(*(_QWORD *)(a1 + 408));
  v7 = *(_QWORD *)(a1 + 392);
  if ( v7 )
  {
    v8 = 176LL * *(_QWORD *)(v7 - 8);
    v9 = v7 + v8;
    if ( v7 != v7 + v8 )
    {
      do
      {
        v9 -= 176;
        v10 = *(_QWORD *)(v9 + 112);
        if ( v10 != v9 + 128 )
          _libc_free(v10);
        v11 = *(_QWORD *)(v9 + 32);
        if ( v11 != v9 + 48 )
          _libc_free(v11);
      }
      while ( v7 != v9 );
      v8 = 176LL * *(_QWORD *)(v7 - 8);
    }
    a2 = v8 + 8;
    j_j_j___libc_free_0_0(v7 - 8);
  }
  sub_20FC940(a1 + 376, a2, v3, v4, v5, v6);
  v12 = *(unsigned __int64 **)(a1 + 288);
  v13 = *(unsigned int *)(a1 + 296);
  *(_QWORD *)(a1 + 264) = 0;
  v14 = &v12[v13];
  while ( v14 != v12 )
  {
    v15 = *v12++;
    _libc_free(v15);
  }
  v16 = *(unsigned __int64 **)(a1 + 336);
  v17 = (unsigned __int64)&v16[2 * *(unsigned int *)(a1 + 344)];
  if ( v16 != (unsigned __int64 *)v17 )
  {
    do
    {
      v18 = *v16;
      v16 += 2;
      _libc_free(v18);
    }
    while ( (unsigned __int64 *)v17 != v16 );
    v17 = *(_QWORD *)(a1 + 336);
  }
  if ( v17 != a1 + 352 )
    _libc_free(v17);
  v19 = *(_QWORD *)(a1 + 288);
  if ( v19 != a1 + 304 )
    _libc_free(v19);
  _libc_free(*(_QWORD *)(a1 + 208));
  _libc_free(*(_QWORD *)(a1 + 184));
  _libc_free(*(_QWORD *)(a1 + 160));
  *(_QWORD *)a1 = &unk_49EE078;
  return sub_16366C0((_QWORD *)a1);
}
