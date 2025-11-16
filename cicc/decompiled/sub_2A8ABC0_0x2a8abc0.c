// Function: sub_2A8ABC0
// Address: 0x2a8abc0
//
void __fastcall sub_2A8ABC0(__int64 a1)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rdi

  v2 = *(_QWORD *)a1;
  v3 = *(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v3 )
  {
    do
    {
      v3 -= 24LL;
      if ( *(_DWORD *)(v3 + 16) > 0x40u )
      {
        v4 = *(_QWORD *)(v3 + 8);
        if ( v4 )
          j_j___libc_free_0_0(v4);
      }
    }
    while ( v3 != v2 );
    v3 = *(_QWORD *)a1;
  }
  if ( v3 != a1 + 16 )
    _libc_free(v3);
}
