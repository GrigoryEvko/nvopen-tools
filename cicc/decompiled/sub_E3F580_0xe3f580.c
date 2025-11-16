// Function: sub_E3F580
// Address: 0xe3f580
//
char __fastcall sub_E3F580(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  switch ( a2 )
  {
    case 13LL:
      if ( *(_QWORD *)a1 == 0x79642E646E756F72LL && *(_DWORD *)(a1 + 8) == 1768776046 && *(_BYTE *)(a1 + 12) == 99 )
        return 7;
      return a4;
    case 15LL:
      if ( *(_QWORD *)a1 != 0x6F742E646E756F72LL
        || *(_DWORD *)(a1 + 8) != 1918985582
        || *(_WORD *)(a1 + 12) != 29541
        || *(_BYTE *)(a1 + 14) != 116 )
      {
        return a4;
      }
      return 1;
    case 19LL:
      if ( *(_QWORD *)a1 ^ 0x6F742E646E756F72LL | *(_QWORD *)(a1 + 8) ^ 0x617473657261656ELL
        || *(_WORD *)(a1 + 16) != 24951
        || *(_BYTE *)(a1 + 18) != 121 )
      {
        return a4;
      }
      return 4;
    case 14LL:
      if ( *(_QWORD *)a1 != 0x6F642E646E756F72LL || *(_DWORD *)(a1 + 8) != 1635217015 || *(_WORD *)(a1 + 12) != 25714 )
        return a4;
      return 3;
    case 12LL:
      if ( *(_QWORD *)a1 != 0x70752E646E756F72LL || *(_DWORD *)(a1 + 8) != 1685217655 )
        return a4;
      return 2;
    default:
      if ( a2 != 16 || *(_QWORD *)a1 ^ 0x6F742E646E756F72LL | *(_QWORD *)(a1 + 8) ^ 0x6F72657A64726177LL )
        return a4;
      return 0;
  }
}
