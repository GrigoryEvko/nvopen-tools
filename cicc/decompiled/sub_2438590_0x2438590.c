// Function: sub_2438590
// Address: 0x2438590
//
__int64 __fastcall sub_2438590(__int64 a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  _QWORD *v5; // r13
  _QWORD *v6; // r12
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi

  v2 = a1 + 112;
  v3 = *(_QWORD *)(a1 + 96);
  if ( v3 != v2 )
    _libc_free(v3);
  v4 = *(_QWORD *)(a1 + 48);
  if ( v4 != a1 + 64 )
    _libc_free(v4);
  v5 = *(_QWORD **)(a1 + 32);
  v6 = &v5[18 * *(unsigned int *)(a1 + 40)];
  if ( v5 != v6 )
  {
    do
    {
      v6 -= 18;
      v7 = v6[14];
      if ( (_QWORD *)v7 != v6 + 16 )
        _libc_free(v7);
      v8 = v6[10];
      if ( (_QWORD *)v8 != v6 + 12 )
        _libc_free(v8);
      v9 = v6[6];
      if ( (_QWORD *)v9 != v6 + 8 )
        _libc_free(v9);
      v10 = v6[2];
      if ( (_QWORD *)v10 != v6 + 4 )
        _libc_free(v10);
    }
    while ( v5 != v6 );
    v6 = *(_QWORD **)(a1 + 32);
  }
  if ( v6 != (_QWORD *)(a1 + 48) )
    _libc_free((unsigned __int64)v6);
  return sub_C7D6A0(*(_QWORD *)(a1 + 8), 16LL * *(unsigned int *)(a1 + 24), 8);
}
