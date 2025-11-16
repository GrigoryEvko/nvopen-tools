// Function: sub_D90F00
// Address: 0xd90f00
//
char __fastcall sub_D90F00(__int64 a1, __int64 a2)
{
  char result; // al
  unsigned __int8 *v3; // rdi
  unsigned __int8 v4; // bl
  _BYTE *v5; // rsi

  if ( a1 == a2 )
    return 1;
  result = 0;
  if ( *(_WORD *)(a1 + 24) == 15 && *(_WORD *)(a2 + 24) == 15 )
  {
    v3 = *(unsigned __int8 **)(a1 - 8);
    v4 = *v3;
    if ( *v3 > 0x1Cu )
    {
      v5 = *(_BYTE **)(a2 - 8);
      if ( *v5 > 0x1Cu )
      {
        result = sub_B46220((__int64)v3, (__int64)v5);
        if ( result )
          return v4 == 63 || (unsigned int)v4 - 42 <= 0x11;
      }
    }
  }
  return result;
}
