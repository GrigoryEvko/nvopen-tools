// Function: sub_6E83C0
// Address: 0x6e83c0
//
__int64 __fastcall sub_6E83C0(__int64 *a1, char a2)
{
  __int64 result; // rax

  result = *a1;
  if ( *a1 )
    a1 = *(__int64 **)(result + 88);
  if ( *((_BYTE *)a1 + 173) == 12 )
  {
    result = *((unsigned __int8 *)a1 + 176);
    if ( (_BYTE)result == 11 )
    {
      a1 = (__int64 *)a1[23];
      if ( *((_BYTE *)a1 + 173) != 12 )
        return result;
      result = *((unsigned __int8 *)a1 + 176);
    }
    if ( (_BYTE)result == 3 )
    {
      result = *((_BYTE *)a1 + 177) & 0xFD;
      *((_BYTE *)a1 + 177) = *((_BYTE *)a1 + 177) & 0xFD | (2 * (a2 & 1));
    }
  }
  return result;
}
