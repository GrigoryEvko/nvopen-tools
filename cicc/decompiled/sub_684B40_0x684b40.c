// Function: sub_684B40
// Address: 0x684b40
//
_BYTE *__fastcall sub_684B40(_DWORD *a1, unsigned int a2)
{
  _BYTE *result; // rax

  result = byte_4CFFE80;
  if ( (byte_4CFFE80[4 * a2 + 2] & 2) == 0 )
  {
    result = (_BYTE *)sub_67D400((int *)a2, 5u, a1);
    if ( (_DWORD)result )
    {
      result = (_BYTE *)sub_617600(a2);
      if ( !(_DWORD)result )
      {
        result = (_BYTE *)sub_729F80((unsigned int)*a1);
        if ( !(_DWORD)result )
          return (_BYTE *)sub_684B30(a2, a1);
      }
    }
  }
  return result;
}
