// Function: sub_2855330
// Address: 0x2855330
//
__int64 __fastcall sub_2855330(__int64 a1)
{
  __int64 v2; // r13
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rdi
  __int64 v5; // r13
  unsigned __int64 v6; // r12
  __int64 v7; // rax
  unsigned __int64 *v9; // r12
  unsigned __int64 *v10; // r13

  if ( !*(_BYTE *)(a1 + 2148) )
    _libc_free(*(_QWORD *)(a1 + 2128));
  v2 = *(_QWORD *)(a1 + 760);
  v3 = v2 + 112LL * *(unsigned int *)(a1 + 768);
  if ( v2 != v3 )
  {
    do
    {
      v3 -= 112LL;
      v4 = *(_QWORD *)(v3 + 40);
      if ( v4 != v3 + 56 )
        _libc_free(v4);
    }
    while ( v2 != v3 );
    v3 = *(_QWORD *)(a1 + 760);
  }
  if ( v3 != a1 + 776 )
    _libc_free(v3);
  v5 = *(_QWORD *)(a1 + 56);
  v6 = v5 + 80LL * *(unsigned int *)(a1 + 64);
  if ( v5 != v6 )
  {
    do
    {
      while ( 1 )
      {
        v6 -= 80LL;
        if ( !*(_BYTE *)(v6 + 44) )
          break;
        if ( v5 == v6 )
          goto LABEL_15;
      }
      _libc_free(*(_QWORD *)(v6 + 24));
    }
    while ( v5 != v6 );
LABEL_15:
    v6 = *(_QWORD *)(a1 + 56);
  }
  if ( v6 != a1 + 72 )
    _libc_free(v6);
  v7 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v7 )
  {
    v9 = *(unsigned __int64 **)(a1 + 8);
    v10 = &v9[6 * v7];
    do
    {
      if ( (unsigned __int64 *)*v9 != v9 + 2 )
        _libc_free(*v9);
      v9 += 6;
    }
    while ( v10 != v9 );
    v7 = *(unsigned int *)(a1 + 24);
  }
  return sub_C7D6A0(*(_QWORD *)(a1 + 8), 48 * v7, 8);
}
