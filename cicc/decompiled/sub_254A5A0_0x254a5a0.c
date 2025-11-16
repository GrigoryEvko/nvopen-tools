// Function: sub_254A5A0
// Address: 0x254a5a0
//
void __fastcall sub_254A5A0(__int64 a1)
{
  unsigned __int64 v1; // r12
  bool v3; // cf
  bool v4; // zf
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi

  v1 = a1 - 88;
  *(_QWORD *)(a1 - 88) = &unk_4A16FF8;
  v3 = *(_DWORD *)(a1 + 72) < 0x40u;
  v4 = *(_DWORD *)(a1 + 72) == 64;
  *(_QWORD *)a1 = &unk_4A16D38;
  if ( !v3 && !v4 )
  {
    v5 = *(_QWORD *)(a1 + 64);
    if ( v5 )
      j_j___libc_free_0_0(v5);
  }
  if ( *(_DWORD *)(a1 + 56) > 0x40u )
  {
    v6 = *(_QWORD *)(a1 + 48);
    if ( v6 )
      j_j___libc_free_0_0(v6);
  }
  if ( *(_DWORD *)(a1 + 40) > 0x40u )
  {
    v7 = *(_QWORD *)(a1 + 32);
    if ( v7 )
      j_j___libc_free_0_0(v7);
  }
  if ( *(_DWORD *)(a1 + 24) > 0x40u )
  {
    v8 = *(_QWORD *)(a1 + 16);
    if ( v8 )
      j_j___libc_free_0_0(v8);
  }
  v9 = *(_QWORD *)(a1 - 48);
  *(_QWORD *)(a1 - 88) = &unk_4A16C00;
  if ( v9 != a1 - 32 )
    _libc_free(v9);
  sub_C7D6A0(*(_QWORD *)(a1 - 72), 8LL * *(unsigned int *)(a1 - 56), 8);
  j_j___libc_free_0(v1);
}
