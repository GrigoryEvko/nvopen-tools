// Function: sub_8D23B0
// Address: 0x8d23b0
//
_BOOL8 __fastcall sub_8D23B0(__int64 a1)
{
  while ( *(_BYTE *)(a1 + 140) == 12 )
    a1 = *(_QWORD *)(a1 + 160);
  return (*(_BYTE *)(a1 + 141) & 0x20) != 0;
}
