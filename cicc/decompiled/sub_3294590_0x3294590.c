// Function: sub_3294590
// Address: 0x3294590
//
void __fastcall sub_3294590(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi

  if ( *(_DWORD *)(a1 + 184) > 0x40u )
  {
    v2 = *(_QWORD *)(a1 + 176);
    if ( v2 )
      j_j___libc_free_0_0(v2);
  }
  if ( *(_DWORD *)(a1 + 112) > 0x40u )
  {
    v3 = *(_QWORD *)(a1 + 104);
    if ( v3 )
      j_j___libc_free_0_0(v3);
  }
  if ( *(_DWORD *)(a1 + 40) > 0x40u )
  {
    v4 = *(_QWORD *)(a1 + 32);
    if ( v4 )
      j_j___libc_free_0_0(v4);
  }
}
