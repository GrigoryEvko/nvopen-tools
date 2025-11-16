// Function: sub_D974D0
// Address: 0xd974d0
//
_BYTE *__fastcall sub_D974D0(__int64 a1, __int64 a2)
{
  __int16 v2; // dx
  _BYTE *result; // rax

  v2 = *(_WORD *)(a2 + 24);
  if ( v2 == 8 )
  {
    result = *(_BYTE **)(**(_QWORD **)(*(_QWORD *)(a2 + 48) + 32LL) + 56LL);
    if ( result )
      result -= 24;
  }
  else
  {
    result = 0;
    if ( v2 == 15 )
    {
      result = *(_BYTE **)(a2 - 8);
      if ( *result <= 0x1Cu )
        return 0;
    }
  }
  return result;
}
