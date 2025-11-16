// Function: sub_2544470
// Address: 0x2544470
//
__int64 __fastcall sub_2544470(__int64 a1)
{
  unsigned __int64 v2; // rdi

  if ( !*(_BYTE *)(a1 + 132) )
    _libc_free(*(_QWORD *)(a1 + 112));
  v2 = *(_QWORD *)(a1 + 40);
  *(_QWORD *)a1 = &unk_4A16C00;
  if ( v2 != a1 + 56 )
    _libc_free(v2);
  return sub_C7D6A0(*(_QWORD *)(a1 + 16), 8LL * *(unsigned int *)(a1 + 32), 8);
}
