// Function: sub_39091F0
// Address: 0x39091f0
//
void __fastcall sub_39091F0(__int64 a1)
{
  unsigned __int64 v2; // rdi
  __int64 v3; // r13
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // rdi

  *(_QWORD *)a1 = &unk_4A3EAA8;
  v2 = *(_QWORD *)(a1 + 72);
  if ( v2 != a1 + 88 )
    j_j___libc_free_0(v2);
  v3 = *(_QWORD *)(a1 + 8);
  v4 = v3 + 40LL * *(unsigned int *)(a1 + 16);
  if ( v3 != v4 )
  {
    do
    {
      v4 -= 40LL;
      if ( *(_DWORD *)(v4 + 32) > 0x40u )
      {
        v5 = *(_QWORD *)(v4 + 24);
        if ( v5 )
          j_j___libc_free_0_0(v5);
      }
    }
    while ( v3 != v4 );
    v4 = *(_QWORD *)(a1 + 8);
  }
  if ( v4 != a1 + 24 )
    _libc_free(v4);
}
