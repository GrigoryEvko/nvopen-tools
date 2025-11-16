// Function: sub_CF7590
// Address: 0xcf7590
//
__int64 __fastcall sub_CF7590(unsigned __int8 *a1, _BYTE *a2)
{
  unsigned __int8 v2; // al
  __int64 result; // rax

  *a2 = 0;
  v2 = *a1;
  if ( *a1 > 0x1Cu )
  {
    if ( v2 != 60 )
      goto LABEL_3;
    return 1;
  }
  if ( v2 == 22 )
  {
    if ( !(unsigned __int8)sub_B2D680((__int64)a1) )
      return sub_B2D670((__int64)a1, 9);
    return 1;
  }
LABEL_3:
  result = sub_CF6FD0(a1);
  if ( (_BYTE)result )
    *a2 = 1;
  return result;
}
