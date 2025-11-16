// Function: sub_2544410
// Address: 0x2544410
//
__int64 __fastcall sub_2544410(__int64 a1)
{
  unsigned __int64 v2; // rdi

  if ( !*(_BYTE *)(a1 + 44) )
    _libc_free(*(_QWORD *)(a1 + 24));
  v2 = *(_QWORD *)(a1 - 48);
  *(_QWORD *)(a1 - 88) = &unk_4A16C00;
  if ( v2 != a1 - 32 )
    _libc_free(v2);
  return sub_C7D6A0(*(_QWORD *)(a1 - 72), 8LL * *(unsigned int *)(a1 - 56), 8);
}
