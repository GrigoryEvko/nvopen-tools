// Function: sub_3440750
// Address: 0x3440750
//
char *__fastcall sub_3440750(__int64 a1, __int64 a2, __int64 a3)
{
  char *v3; // r8
  bool v5; // al
  char v6; // al
  char *v7; // r8
  _QWORD v8[3]; // [rsp+0h] [rbp-20h] BYREF

  v8[0] = a2;
  v8[1] = a3;
  if ( !(_WORD)a2 )
  {
    v5 = sub_3007070((__int64)v8);
    v3 = "r";
    if ( !v5 )
    {
      v6 = sub_3007030((__int64)v8);
      v7 = "f";
      if ( !v6 )
        return 0;
      return v7;
    }
    return v3;
  }
  if ( (unsigned __int16)(a2 - 2) <= 7u || (unsigned __int16)(a2 - 17) <= 0x6Cu )
    return "r";
  v3 = "r";
  if ( (unsigned __int16)(a2 - 176) <= 0x1Fu )
    return v3;
  if ( (unsigned __int16)(a2 - 10) > 6u && (unsigned __int16)(a2 - 126) > 0x31u )
  {
    v3 = "f";
    if ( (unsigned __int16)(a2 - 208) > 0x14u )
      return 0;
    return v3;
  }
  return "f";
}
