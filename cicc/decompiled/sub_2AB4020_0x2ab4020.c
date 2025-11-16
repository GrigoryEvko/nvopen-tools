// Function: sub_2AB4020
// Address: 0x2ab4020
//
__int64 __fastcall sub_2AB4020(__int64 a1)
{
  if ( !*(_BYTE *)(a1 + 108) )
    _libc_free(*(_QWORD *)(a1 + 88));
  return sub_C7D6A0(*(_QWORD *)(a1 + 24), 16LL * *(unsigned int *)(a1 + 40), 8);
}
