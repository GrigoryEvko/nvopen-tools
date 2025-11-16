// Function: sub_35469B0
// Address: 0x35469b0
//
void __fastcall sub_35469B0(__int64 a1)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rdi

  v2 = *(_QWORD *)a1;
  v3 = *(_QWORD *)a1 + 88LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v3 )
  {
    do
    {
      v3 -= 88LL;
      v4 = *(_QWORD *)(v3 + 32);
      if ( v4 != v3 + 48 )
        _libc_free(v4);
      sub_C7D6A0(*(_QWORD *)(v3 + 8), 8LL * *(unsigned int *)(v3 + 24), 8);
    }
    while ( v3 != v2 );
    v3 = *(_QWORD *)a1;
  }
  if ( v3 != a1 + 16 )
    _libc_free(v3);
}
