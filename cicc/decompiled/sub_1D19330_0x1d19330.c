// Function: sub_1D19330
// Address: 0x1d19330
//
__int64 __fastcall sub_1D19330(__int64 a1, __int16 a2)
{
  __int64 result; // rax

  if ( (a2 & 1) != 0 )
  {
    result = *(_WORD *)(a1 + 80) & 0xF001;
    *(_WORD *)(a1 + 80) = result | *(_WORD *)(a1 + 80) & a2 & 0xFFE;
  }
  return result;
}
