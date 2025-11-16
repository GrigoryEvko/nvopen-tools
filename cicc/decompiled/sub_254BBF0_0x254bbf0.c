// Function: sub_254BBF0
// Address: 0x254bbf0
//
_QWORD *__fastcall sub_254BBF0(__int64 a1)
{
  _QWORD *result; // rax
  _QWORD *v2; // rdx

  result = *(_QWORD **)a1;
  v2 = *(_QWORD **)(a1 + 8);
  if ( *(_QWORD **)a1 != v2 )
  {
    do
    {
      if ( *result < 0xFFFFFFFFFFFFFFFELL )
        break;
      *(_QWORD *)a1 = ++result;
    }
    while ( result != v2 );
  }
  return result;
}
