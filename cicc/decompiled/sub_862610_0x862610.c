// Function: sub_862610
// Address: 0x862610
//
_BOOL8 __fastcall sub_862610(_BYTE *a1)
{
  _BOOL8 result; // rax
  char v2; // al
  __int64 i; // rax
  __int64 v4; // rax
  __int64 v5; // rcx

  if ( (a1[194] & 2) != 0 && (a1[206] & 8) == 0 )
    return 1;
  v2 = a1[195];
  if ( (v2 & 8) == 0 && (a1[207] & 0x40) == 0 && ((a1[206] & 2) == 0 || (v2 & 1) == 0) )
  {
    if ( (unsigned int)sub_89A120() )
    {
      if ( !sub_8625D0((__int64)a1) )
        return 1;
    }
    else if ( (a1[201] & 1) != 0 )
    {
      return 1;
    }
  }
  result = 0;
  if ( (a1[89] & 1) != 0 )
  {
    for ( i = dword_4F04C64; ; i = *(int *)(v4 + 552) )
    {
      v4 = qword_4F04C68[0] + 776 * i;
      if ( *(_BYTE *)(v4 + 4) == 17 && *(_BYTE **)(v4 + 216) == a1 )
        break;
    }
    v5 = *(int *)(qword_4F04C68[0] + 776LL * *(int *)(v4 + 552) + 400);
    result = 0;
    if ( (_DWORD)v5 != -1 )
      return (unsigned int)sub_862610(*(_QWORD *)(qword_4F04C68[0] + 776 * v5 + 216)) != 0;
  }
  return result;
}
