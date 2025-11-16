// Function: sub_233BBD0
// Address: 0x233bbd0
//
__int64 __fastcall sub_233BBD0(__int64 a1)
{
  __int64 v2; // r13
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rdi
  __int64 v5; // r14
  unsigned __int64 v6; // r12
  unsigned __int64 v7; // rdi

  v2 = *(_QWORD *)(a1 + 152);
  v3 = v2 + 88LL * *(unsigned int *)(a1 + 160);
  if ( v2 != v3 )
  {
    do
    {
      v3 -= 88LL;
      v4 = *(_QWORD *)(v3 + 8);
      if ( v4 != v3 + 24 )
        _libc_free(v4);
    }
    while ( v2 != v3 );
    v3 = *(_QWORD *)(a1 + 152);
  }
  if ( v3 != a1 + 168 )
    _libc_free(v3);
  sub_C7D6A0(*(_QWORD *)(a1 + 128), 16LL * *(unsigned int *)(a1 + 144), 8);
  v5 = *(_QWORD *)(a1 + 104);
  v6 = v5 + 88LL * *(unsigned int *)(a1 + 112);
  if ( v5 != v6 )
  {
    do
    {
      v6 -= 88LL;
      v7 = *(_QWORD *)(v6 + 8);
      if ( v7 != v6 + 24 )
        _libc_free(v7);
    }
    while ( v5 != v6 );
    v6 = *(_QWORD *)(a1 + 104);
  }
  if ( a1 + 120 != v6 )
    _libc_free(v6);
  return sub_C7D6A0(*(_QWORD *)(a1 + 80), 16LL * *(unsigned int *)(a1 + 96), 8);
}
