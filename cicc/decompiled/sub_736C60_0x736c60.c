// Function: sub_736C60
// Address: 0x736c60
//
__int64 *__fastcall sub_736C60(char a1, __int64 *a2)
{
  while ( 1 )
  {
    if ( !a2 )
      return 0;
    if ( *((_BYTE *)a2 + 8) == a1 )
      break;
    a2 = (__int64 *)*a2;
  }
  return a2;
}
