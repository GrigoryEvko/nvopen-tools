// Function: sub_CF7600
// Address: 0xcf7600
//
__int64 __fastcall sub_CF7600(unsigned __int8 *a1, _BYTE *a2)
{
  unsigned __int8 v2; // al
  __int64 result; // rax

  *a2 = 0;
  v2 = *a1;
  if ( *a1 <= 0x1Cu )
  {
    if ( v2 != 22 )
      return sub_CF6FD0(a1);
    if ( (unsigned __int8)sub_B2D670((__int64)a1, 77) && (result = sub_B2D700((__int64)a1), (_BYTE)result) )
      *a2 = 1;
    else
      return sub_B2D680((__int64)a1);
  }
  else
  {
    if ( v2 != 60 )
      return sub_CF6FD0(a1);
    return 1;
  }
  return result;
}
