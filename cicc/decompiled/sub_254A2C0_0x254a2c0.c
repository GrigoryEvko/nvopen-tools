// Function: sub_254A2C0
// Address: 0x254a2c0
//
__int64 __fastcall sub_254A2C0(__int64 a1)
{
  bool v2; // cc
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi

  *(_QWORD *)(a1 - 88) = &unk_4A16FF8;
  v2 = *(_DWORD *)(a1 + 72) <= 0x40u;
  *(_QWORD *)a1 = &unk_4A16D38;
  if ( !v2 )
  {
    v3 = *(_QWORD *)(a1 + 64);
    if ( v3 )
      j_j___libc_free_0_0(v3);
  }
  if ( *(_DWORD *)(a1 + 56) > 0x40u )
  {
    v4 = *(_QWORD *)(a1 + 48);
    if ( v4 )
      j_j___libc_free_0_0(v4);
  }
  if ( *(_DWORD *)(a1 + 40) > 0x40u )
  {
    v5 = *(_QWORD *)(a1 + 32);
    if ( v5 )
      j_j___libc_free_0_0(v5);
  }
  if ( *(_DWORD *)(a1 + 24) > 0x40u )
  {
    v6 = *(_QWORD *)(a1 + 16);
    if ( v6 )
      j_j___libc_free_0_0(v6);
  }
  v7 = *(_QWORD *)(a1 - 48);
  *(_QWORD *)(a1 - 88) = &unk_4A16C00;
  if ( v7 != a1 - 32 )
    _libc_free(v7);
  return sub_C7D6A0(*(_QWORD *)(a1 - 72), 8LL * *(unsigned int *)(a1 - 56), 8);
}
