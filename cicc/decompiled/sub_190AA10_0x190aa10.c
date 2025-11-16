// Function: sub_190AA10
// Address: 0x190aa10
//
void *__fastcall sub_190AA10(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 *v5; // rbx
  unsigned __int64 *v6; // r13
  unsigned __int64 v7; // rdi
  unsigned __int64 *v8; // rbx
  unsigned __int64 v9; // r13
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  __int64 v12; // rdi

  *(_QWORD *)a1 = off_49F33A8;
  v2 = *(_QWORD *)(a1 + 952);
  if ( v2 != a1 + 968 )
    _libc_free(v2);
  j___libc_free_0(*(_QWORD *)(a1 + 920));
  v3 = *(_QWORD *)(a1 + 832);
  if ( v3 != a1 + 848 )
    _libc_free(v3);
  v4 = *(_QWORD *)(a1 + 752);
  if ( v4 != a1 + 768 )
    _libc_free(v4);
  if ( (*(_BYTE *)(a1 + 680) & 1) == 0 )
    j___libc_free_0(*(_QWORD *)(a1 + 688));
  v5 = *(unsigned __int64 **)(a1 + 584);
  v6 = &v5[*(unsigned int *)(a1 + 592)];
  while ( v6 != v5 )
  {
    v7 = *v5++;
    _libc_free(v7);
  }
  v8 = *(unsigned __int64 **)(a1 + 632);
  v9 = (unsigned __int64)&v8[2 * *(unsigned int *)(a1 + 640)];
  if ( v8 != (unsigned __int64 *)v9 )
  {
    do
    {
      v10 = *v8;
      v8 += 2;
      _libc_free(v10);
    }
    while ( (unsigned __int64 *)v9 != v8 );
    v9 = *(_QWORD *)(a1 + 632);
  }
  if ( v9 != a1 + 648 )
    _libc_free(v9);
  v11 = *(_QWORD *)(a1 + 584);
  if ( v11 != a1 + 600 )
    _libc_free(v11);
  j___libc_free_0(*(_QWORD *)(a1 + 544));
  sub_190A790(a1 + 312);
  j___libc_free_0(*(_QWORD *)(a1 + 280));
  v12 = *(_QWORD *)(a1 + 240);
  if ( v12 )
    j_j___libc_free_0(v12, *(_QWORD *)(a1 + 256) - v12);
  j___libc_free_0(*(_QWORD *)(a1 + 216));
  *(_QWORD *)a1 = &unk_49EE078;
  return sub_16366C0((_QWORD *)a1);
}
