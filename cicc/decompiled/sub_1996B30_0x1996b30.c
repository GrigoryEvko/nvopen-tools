// Function: sub_1996B30
// Address: 0x1996b30
//
__int64 __fastcall sub_1996B30(__int64 a1)
{
  unsigned __int64 v2; // rdi
  __int64 v3; // r13
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // rdi
  __int64 v6; // r13
  unsigned __int64 v7; // r12
  unsigned __int64 v8; // rdi
  __int64 v9; // rax
  unsigned __int64 *v11; // r12
  unsigned __int64 *v12; // r13

  v2 = *(_QWORD *)(a1 + 1928);
  if ( v2 != *(_QWORD *)(a1 + 1920) )
    _libc_free(v2);
  v3 = *(_QWORD *)(a1 + 744);
  v4 = v3 + 96LL * *(unsigned int *)(a1 + 752);
  if ( v3 != v4 )
  {
    do
    {
      v4 -= 96LL;
      v5 = *(_QWORD *)(v4 + 32);
      if ( v5 != v4 + 48 )
        _libc_free(v5);
    }
    while ( v3 != v4 );
    v4 = *(_QWORD *)(a1 + 744);
  }
  if ( v4 != a1 + 760 )
    _libc_free(v4);
  v6 = *(_QWORD *)(a1 + 56);
  v7 = v6 + 80LL * *(unsigned int *)(a1 + 64);
  if ( v6 != v7 )
  {
    do
    {
      v7 -= 80LL;
      v8 = *(_QWORD *)(v7 + 32);
      if ( v8 != *(_QWORD *)(v7 + 24) )
        _libc_free(v8);
    }
    while ( v6 != v7 );
    v7 = *(_QWORD *)(a1 + 56);
  }
  if ( v7 != a1 + 72 )
    _libc_free(v7);
  v9 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v9 )
  {
    v11 = *(unsigned __int64 **)(a1 + 8);
    v12 = &v11[6 * v9];
    do
    {
      if ( (unsigned __int64 *)*v11 != v11 + 2 )
        _libc_free(*v11);
      v11 += 6;
    }
    while ( v12 != v11 );
  }
  return j___libc_free_0(*(_QWORD *)(a1 + 8));
}
