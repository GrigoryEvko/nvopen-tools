// Function: sub_8D5780
// Address: 0x8d5780
//
_BOOL8 __fastcall sub_8D5780(__int64 a1, __int64 a2)
{
  int v2; // ebx
  _BOOL8 result; // rax

  v2 = sub_8D4C10(a2, 0);
  result = 0;
  if ( v2 )
  {
    if ( (*(_BYTE *)(a1 + 140) & 0xFB) == 8 )
      v2 &= ~(unsigned int)sub_8D4C10(a1, dword_4F077C4 != 2);
    return v2 != 0;
  }
  return result;
}
