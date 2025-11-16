// Function: sub_EA1850
// Address: 0xea1850
//
__int16 __fastcall sub_EA1850(__int64 a1, char a2)
{
  __int16 result; // ax
  __int16 v3; // dx

  result = *(_WORD *)(a1 + 12);
  HIBYTE(result) &= ~0x20u;
  if ( a2 )
  {
    HIBYTE(v3) = HIBYTE(result) | 0x20;
    LOBYTE(v3) = *(_WORD *)(a1 + 12);
    result = v3;
  }
  *(_WORD *)(a1 + 12) = result;
  return result;
}
