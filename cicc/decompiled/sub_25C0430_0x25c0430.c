// Function: sub_25C0430
// Address: 0x25c0430
//
void __fastcall sub_25C0430(__int64 a1)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rdi

  v2 = *(_QWORD *)a1;
  v3 = *(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v3 )
  {
    do
    {
      v3 -= 32LL;
      if ( *(_DWORD *)(v3 + 24) > 0x40u )
      {
        v4 = *(_QWORD *)(v3 + 16);
        if ( v4 )
          j_j___libc_free_0_0(v4);
      }
      if ( *(_DWORD *)(v3 + 8) > 0x40u && *(_QWORD *)v3 )
        j_j___libc_free_0_0(*(_QWORD *)v3);
    }
    while ( v2 != v3 );
    v3 = *(_QWORD *)a1;
  }
  if ( v3 != a1 + 16 )
    _libc_free(v3);
}
