// Function: sub_2EB40F0
// Address: 0x2eb40f0
//
void __fastcall sub_2EB40F0(__int64 a1)
{
  __int64 v2; // rbx
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rdi

  v2 = *(_QWORD *)(a1 + 528);
  v3 = v2 + 56LL * *(unsigned int *)(a1 + 536);
  if ( v2 != v3 )
  {
    do
    {
      v3 -= 56LL;
      v4 = *(_QWORD *)(v3 + 24);
      if ( v4 != v3 + 40 )
        _libc_free(v4);
    }
    while ( v2 != v3 );
    v3 = *(_QWORD *)(a1 + 528);
  }
  if ( v3 != a1 + 544 )
    _libc_free(v3);
  if ( *(_QWORD *)a1 != a1 + 16 )
    _libc_free(*(_QWORD *)a1);
}
