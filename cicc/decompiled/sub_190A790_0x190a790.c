// Function: sub_190A790
// Address: 0x190a790
//
__int64 __fastcall sub_190A790(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // r13
  __int64 v4; // r12
  unsigned __int64 v5; // rdi
  __int64 v6; // r13
  __int64 v7; // r12
  __int64 v8; // r13
  unsigned __int64 v9; // rdi

  j___libc_free_0(*(_QWORD *)(a1 + 160));
  j___libc_free_0(*(_QWORD *)(a1 + 128));
  v2 = *(_QWORD *)(a1 + 96);
  if ( v2 )
    j_j___libc_free_0(v2, *(_QWORD *)(a1 + 112) - v2);
  v3 = *(_QWORD *)(a1 + 80);
  v4 = *(_QWORD *)(a1 + 72);
  if ( v3 != v4 )
  {
    do
    {
      v5 = *(_QWORD *)(v4 + 24);
      if ( v5 != v4 + 40 )
        _libc_free(v5);
      v4 += 56;
    }
    while ( v3 != v4 );
    v4 = *(_QWORD *)(a1 + 72);
  }
  if ( v4 )
    j_j___libc_free_0(v4, *(_QWORD *)(a1 + 88) - v4);
  v6 = *(unsigned int *)(a1 + 56);
  if ( (_DWORD)v6 )
  {
    v7 = *(_QWORD *)(a1 + 40);
    v8 = v7 + (v6 << 6);
    do
    {
      v9 = *(_QWORD *)(v7 + 24);
      if ( v9 != v7 + 40 )
        _libc_free(v9);
      v7 += 64;
    }
    while ( v8 != v7 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 40));
  return j___libc_free_0(*(_QWORD *)(a1 + 8));
}
