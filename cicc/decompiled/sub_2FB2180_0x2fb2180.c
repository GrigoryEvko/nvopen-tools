// Function: sub_2FB2180
// Address: 0x2fb2180
//
_QWORD *__fastcall sub_2FB2180(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *result; // rax

  result = *(_QWORD **)(a3 + 104);
  if ( !result )
LABEL_5:
    BUG();
  while ( result[14] != a1 || result[15] != a2 )
  {
    result = (_QWORD *)result[13];
    if ( !result )
      goto LABEL_5;
  }
  return result;
}
