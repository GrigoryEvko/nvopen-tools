// Function: sub_CF7060
// Address: 0xcf7060
//
__int64 __fastcall sub_CF7060(unsigned __int8 *a1)
{
  unsigned __int8 v1; // al
  __int64 result; // rax

  v1 = *a1;
  if ( *a1 <= 0x1Cu )
  {
    if ( v1 == 3 )
      return 1;
    if ( v1 <= 2u )
    {
      if ( v1 != 1 )
        return 1;
      result = sub_CF6FD0(a1);
      if ( (_BYTE)result )
        return 1;
      goto LABEL_10;
    }
  }
  else if ( v1 == 60 )
  {
    return 1;
  }
  result = sub_CF6FD0(a1);
  if ( (_BYTE)result )
    return 1;
LABEL_10:
  if ( *a1 == 22 )
  {
    if ( !(unsigned __int8)sub_B2D700((__int64)a1) )
      return sub_B2D680((__int64)a1);
    return 1;
  }
  return result;
}
