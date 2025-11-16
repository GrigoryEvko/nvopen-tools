// Function: sub_72F9B0
// Address: 0x72f9b0
//
_QWORD *__fastcall sub_72F9B0(__int64 a1, __int64 a2)
{
  _QWORD *result; // rax

  if ( !a2 )
    return 0;
  while ( 1 )
  {
    result = *(_QWORD **)(a2 + 200);
    if ( result )
      break;
LABEL_7:
    a2 = *(_QWORD *)(a2 + 16);
    if ( !a2 )
      return 0;
  }
  while ( result[1] != a1 )
  {
    result = (_QWORD *)*result;
    if ( !result )
      goto LABEL_7;
  }
  return result;
}
