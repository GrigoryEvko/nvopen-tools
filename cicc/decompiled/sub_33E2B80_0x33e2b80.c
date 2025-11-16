// Function: sub_33E2B80
// Address: 0x33e2b80
//
void __fastcall sub_33E2B80(__int64 a1)
{
  unsigned __int64 v1; // rbx
  unsigned __int64 v2; // r12

  v1 = *(_QWORD *)a1;
  v2 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v2 )
  {
    do
    {
      v2 -= 16LL;
      if ( *(_DWORD *)(v2 + 8) > 0x40u && *(_QWORD *)v2 )
        j_j___libc_free_0_0(*(_QWORD *)v2);
    }
    while ( v2 != v1 );
    v2 = *(_QWORD *)a1;
  }
  if ( v2 != a1 + 16 )
    _libc_free(v2);
}
