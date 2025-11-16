// Function: sub_25F0850
// Address: 0x25f0850
//
bool __fastcall sub_25F0850(__int64 a1, __int64 a2)
{
  bool result; // al
  __int64 v3; // rax
  int v4; // eax

  if ( (unsigned __int8)sub_B2D610(a2, 3)
    || (unsigned __int8)sub_B2D610(a2, 31)
    || (unsigned __int8)sub_B2D610(a2, 36)
    || (unsigned __int8)sub_B2D610(a2, 56)
    || (unsigned __int8)sub_B2D610(a2, 57)
    || (unsigned __int8)sub_B2D610(a2, 63)
    || (unsigned __int8)sub_B2D610(a2, 59) )
  {
    return 0;
  }
  result = 1;
  if ( (*(_BYTE *)(a2 + 2) & 8) != 0 )
  {
    v3 = sub_B2E500(a2);
    v4 = sub_B2A630(v3);
    if ( v4 > 10 )
      return v4 != 12;
    else
      return v4 <= 6;
  }
  return result;
}
