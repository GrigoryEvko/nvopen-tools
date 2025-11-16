// Function: sub_2FB21B0
// Address: 0x2fb21b0
//
_QWORD *__fastcall sub_2FB21B0(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *result; // rax

  result = *(_QWORD **)(a3 + 104);
  if ( !result )
LABEL_5:
    BUG();
  while ( a1 != (a1 & result[14]) || a2 != (a2 & result[15]) )
  {
    result = (_QWORD *)result[13];
    if ( !result )
      goto LABEL_5;
  }
  return result;
}
