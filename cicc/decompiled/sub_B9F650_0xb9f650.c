// Function: sub_B9F650
// Address: 0xb9f650
//
_BYTE *__fastcall sub_B9F650(__int64 *a1, _BYTE *a2)
{
  int v2; // eax
  _BYTE *result; // rax

  if ( !a2 )
    return (_BYTE *)sub_B9C770(a1, 0, 0, 0, 1);
  if ( (unsigned __int8)(*a2 - 5) <= 0x1Fu )
  {
    v2 = (*(a2 - 16) & 2) != 0 ? *((_DWORD *)a2 - 6) : (*((_WORD *)a2 - 8) >> 6) & 0xF;
    if ( v2 == 1 )
    {
      if ( *(_QWORD *)sub_A17150(a2 - 16) )
      {
        result = *(_BYTE **)sub_A17150(a2 - 16);
        if ( *result == 1 )
          return result;
        return a2;
      }
      return (_BYTE *)sub_B9C770(a1, 0, 0, 0, 1);
    }
  }
  return a2;
}
