// Function: sub_26709C0
// Address: 0x26709c0
//
__int64 __fastcall sub_26709C0(__int64 a1)
{
  bool v2; // zf
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi

  *(_QWORD *)a1 = off_4A20228;
  v2 = *(_BYTE *)(a1 + 212) == 0;
  *(_QWORD *)(a1 + 88) = &unk_4A202B8;
  if ( v2 )
    _libc_free(*(_QWORD *)(a1 + 192));
  v3 = *(_QWORD *)(a1 + 136);
  if ( v3 != a1 + 152 )
    _libc_free(v3);
  sub_C7D6A0(*(_QWORD *)(a1 + 112), 8LL * *(unsigned int *)(a1 + 128), 8);
  v4 = *(_QWORD *)(a1 + 40);
  *(_QWORD *)a1 = &unk_4A16C00;
  if ( v4 != a1 + 56 )
    _libc_free(v4);
  return sub_C7D6A0(*(_QWORD *)(a1 + 16), 8LL * *(unsigned int *)(a1 + 32), 8);
}
