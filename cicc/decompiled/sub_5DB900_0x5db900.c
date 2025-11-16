// Function: sub_5DB900
// Address: 0x5db900
//
int __fastcall sub_5DB900(__int64 a1, char a2)
{
  int result; // eax
  char v3; // al

  switch ( a2 )
  {
    case 2:
    case 12:
      result = sub_5D5A80(a1, 0);
      break;
    case 6:
      v3 = *(_BYTE *)(a1 + 140);
      if ( (unsigned __int8)(v3 - 9) <= 2u || v3 == 2 && (*(_BYTE *)(a1 + 161) & 8) != 0 )
        result = sub_5DB710(a1);
      else
        result = sub_5D71E0(a1);
      break;
    case 7:
      result = sub_5D6390(a1);
      break;
    case 11:
      result = sub_5D7720(a1);
      break;
    default:
      sub_721090(a1);
  }
  return result;
}
