// Function: sub_8D3F30
// Address: 0x8d3f30
//
_BOOL8 __fastcall sub_8D3F30(__int64 a1)
{
  _BOOL8 result; // rax

  result = 0;
  if ( *(_BYTE *)(a1 + 140) == 14 )
    return (*(_WORD *)(a1 + 160) & 0x8FF) == 2048;
  return result;
}
