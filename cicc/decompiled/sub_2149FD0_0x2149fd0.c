// Function: sub_2149FD0
// Address: 0x2149fd0
//
bool __fastcall sub_2149FD0(_QWORD *a1, __int64 a2)
{
  char v2; // al
  bool result; // al

  if ( !a2 )
    return 0;
  v2 = *(_BYTE *)(a2 + 148);
  if ( (unsigned __int8)(v2 - 1) <= 1u || (unsigned __int8)(v2 - 11) <= 7u )
    return 0;
  result = 1;
  if ( a2 != a1[9] && a2 != a1[10] && a2 != a1[20] && a2 != a1[13] && a2 != a1[35] && a2 != a1[19] && a2 != a1[18] )
  {
    result = 1;
    if ( a2 != a1[17]
      && a2 != a1[16]
      && a2 != a1[11]
      && a2 != a1[34]
      && a2 != a1[12]
      && a2 != a1[21]
      && a2 != a1[14]
      && a2 != a1[42]
      && a2 != a1[28]
      && a2 != a1[29] )
    {
      return a2 == a1[24]
          || a2 == a1[23]
          || a2 == a1[26]
          || a2 == a1[25]
          || a2 == a1[32]
          || a2 == a1[30]
          || a2 == a1[38]
          || a2 == a1[27]
          || a2 == a1[31]
          || a2 == a1[39]
          || a2 == a1[33]
          || a2 == a1[22]
          || a2 == a1[15]
          || a2 == a1[40]
          || a1[41] == a2;
    }
  }
  return result;
}
