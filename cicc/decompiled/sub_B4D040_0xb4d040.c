// Function: sub_B4D040
// Address: 0xb4d040
//
_BOOL8 __fastcall sub_B4D040(__int64 a1)
{
  if ( **(_BYTE **)(a1 - 32) != 17 )
    return 0;
  if ( sub_AA5B70(*(_QWORD *)(a1 + 40)) )
    return (*(_WORD *)(a1 + 2) & 0x40) == 0;
  return 0;
}
