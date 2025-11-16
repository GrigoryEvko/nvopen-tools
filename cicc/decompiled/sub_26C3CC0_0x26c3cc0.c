// Function: sub_26C3CC0
// Address: 0x26c3cc0
//
void __fastcall sub_26C3CC0(__int64 a1)
{
  bool v2; // cc
  unsigned __int64 v3; // rdi

  v2 = *(_DWORD *)(a1 + 24) <= 0x40u;
  *(_BYTE *)(a1 + 32) = 0;
  if ( !v2 )
  {
    v3 = *(_QWORD *)(a1 + 16);
    if ( v3 )
      j_j___libc_free_0_0(v3);
  }
  if ( *(_DWORD *)(a1 + 8) > 0x40u )
  {
    if ( *(_QWORD *)a1 )
      j_j___libc_free_0_0(*(_QWORD *)a1);
  }
}
