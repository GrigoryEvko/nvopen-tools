// Function: sub_1F2F180
// Address: 0x1f2f180
//
void *__fastcall sub_1F2F180(__int64 a1)
{
  unsigned __int64 *v2; // rbx
  unsigned __int64 *v3; // r12
  __int64 v4; // rbx
  unsigned __int64 v5; // r12
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  __int64 v8; // rbx
  unsigned __int64 v9; // r12
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  unsigned __int64 *v13; // rbx
  unsigned __int64 *v14; // r12
  __int64 v15; // rdi

  *(_QWORD *)a1 = off_49FEB28;
  v2 = *(unsigned __int64 **)(a1 + 1896);
  v3 = &v2[6 * *(unsigned int *)(a1 + 1904)];
  if ( v2 != v3 )
  {
    do
    {
      v3 -= 6;
      if ( (unsigned __int64 *)*v3 != v3 + 2 )
        _libc_free(*v3);
    }
    while ( v2 != v3 );
    v3 = *(unsigned __int64 **)(a1 + 1896);
  }
  if ( v3 != (unsigned __int64 *)(a1 + 1912) )
    _libc_free((unsigned __int64)v3);
  v4 = *(_QWORD *)(a1 + 1832);
  v5 = v4 + 24LL * *(unsigned int *)(a1 + 1840);
  if ( v4 != v5 )
  {
    do
    {
      v6 = *(_QWORD *)(v5 - 24);
      v5 -= 24LL;
      _libc_free(v6);
    }
    while ( v4 != v5 );
    v5 = *(_QWORD *)(a1 + 1832);
  }
  if ( v5 != a1 + 1848 )
    _libc_free(v5);
  v7 = *(_QWORD *)(a1 + 1808);
  if ( v7 != a1 + 1824 )
    _libc_free(v7);
  v8 = *(_QWORD *)(a1 + 1744);
  v9 = v8 + 24LL * *(unsigned int *)(a1 + 1752);
  if ( v8 != v9 )
  {
    do
    {
      v10 = *(_QWORD *)(v9 - 24);
      v9 -= 24LL;
      _libc_free(v10);
    }
    while ( v8 != v9 );
    v9 = *(_QWORD *)(a1 + 1744);
  }
  if ( v9 != a1 + 1760 )
    _libc_free(v9);
  v11 = *(_QWORD *)(a1 + 1664);
  if ( v11 != a1 + 1680 )
    _libc_free(v11);
  v12 = *(_QWORD *)(a1 + 1584);
  if ( v12 != a1 + 1600 )
    _libc_free(v12);
  v13 = *(unsigned __int64 **)(a1 + 288);
  v14 = &v13[10 * *(unsigned int *)(a1 + 296)];
  if ( v13 != v14 )
  {
    do
    {
      v14 -= 10;
      if ( (unsigned __int64 *)*v14 != v14 + 2 )
        _libc_free(*v14);
    }
    while ( v13 != v14 );
    v14 = *(unsigned __int64 **)(a1 + 288);
  }
  if ( v14 != (unsigned __int64 *)(a1 + 304) )
    _libc_free((unsigned __int64)v14);
  v15 = *(_QWORD *)(a1 + 264);
  if ( v15 )
    j_j___libc_free_0(v15, *(_QWORD *)(a1 + 280) - v15);
  _libc_free(*(_QWORD *)(a1 + 208));
  _libc_free(*(_QWORD *)(a1 + 184));
  _libc_free(*(_QWORD *)(a1 + 160));
  *(_QWORD *)a1 = &unk_49EE078;
  return sub_16366C0((_QWORD *)a1);
}
