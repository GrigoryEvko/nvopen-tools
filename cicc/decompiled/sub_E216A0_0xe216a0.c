// Function: sub_E216A0
// Address: 0xe216a0
//
__int64 __fastcall sub_E216A0(__int64 a1, char a2, int a3)
{
  unsigned __int8 v3; // al
  __int64 v5; // rsi

  v3 = a2 - 48;
  if ( (unsigned __int8)(a2 - 65) > 0x19u )
  {
    if ( v3 > 9u )
    {
      *(_BYTE *)(a1 + 8) = 1;
      return 0;
    }
LABEL_4:
    v5 = (char)(a2 - 48);
    if ( a3 != 1 )
      goto LABEL_5;
    return (unsigned __int8)asc_3F7C7E0[v5];
  }
  if ( v3 <= 9u )
    goto LABEL_4;
  v5 = a2 - 55;
  if ( a3 == 1 )
    return (unsigned __int8)asc_3F7C7E0[v5];
LABEL_5:
  if ( a3 == 2 )
    return byte_3F7C7A0[v5];
  else
    return byte_3F7C820[v5];
}
