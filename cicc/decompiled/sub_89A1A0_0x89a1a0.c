// Function: sub_89A1A0
// Address: 0x89a1a0
//
__int64 *__fastcall sub_89A1A0(_QWORD *a1, __int64 *a2, _QWORD **a3, __int64 **a4)
{
  __int64 *result; // rax

  if ( a3 )
    *a3 = a1;
  *a4 = a2;
LABEL_8:
  result = *a4;
  if ( *a4 )
  {
    while ( *((_BYTE *)result + 8) == 3 )
    {
      result = (__int64 *)*result;
      *a4 = result;
      if ( !result )
        break;
      if ( (result[3] & 0x18) == 0 && a3 && *a3 )
      {
        *a3 = (_QWORD *)**a3;
        goto LABEL_8;
      }
    }
  }
  return result;
}
