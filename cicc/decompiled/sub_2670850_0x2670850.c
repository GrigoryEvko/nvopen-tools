// Function: sub_2670850
// Address: 0x2670850
//
__int64 __fastcall sub_2670850(__int64 a1)
{
  bool v2; // zf
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi

  *(_QWORD *)(a1 - 88) = off_4A20228;
  v2 = *(_BYTE *)(a1 + 124) == 0;
  *(_QWORD *)a1 = &unk_4A202B8;
  if ( v2 )
    _libc_free(*(_QWORD *)(a1 + 104));
  v3 = *(_QWORD *)(a1 + 48);
  if ( v3 != a1 + 64 )
    _libc_free(v3);
  sub_C7D6A0(*(_QWORD *)(a1 + 24), 8LL * *(unsigned int *)(a1 + 40), 8);
  v4 = *(_QWORD *)(a1 - 48);
  *(_QWORD *)(a1 - 88) = &unk_4A16C00;
  if ( v4 != a1 - 32 )
    _libc_free(v4);
  return sub_C7D6A0(*(_QWORD *)(a1 - 72), 8LL * *(unsigned int *)(a1 - 56), 8);
}
