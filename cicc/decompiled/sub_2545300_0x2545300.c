// Function: sub_2545300
// Address: 0x2545300
//
__int64 __fastcall sub_2545300(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi

  *(_QWORD *)(a1 - 88) = &unk_4A171F8;
  *(_QWORD *)a1 = &unk_4A171B8;
  v2 = *(_QWORD *)(a1 + 56);
  if ( v2 != a1 + 72 )
    _libc_free(v2);
  sub_C7D6A0(*(_QWORD *)(a1 + 32), 24LL * *(unsigned int *)(a1 + 48), 8);
  v3 = *(_QWORD *)(a1 - 48);
  *(_QWORD *)(a1 - 88) = &unk_4A16C00;
  if ( v3 != a1 - 32 )
    _libc_free(v3);
  return sub_C7D6A0(*(_QWORD *)(a1 - 72), 8LL * *(unsigned int *)(a1 - 56), 8);
}
