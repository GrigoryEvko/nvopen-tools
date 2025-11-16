// Function: sub_1E64130
// Address: 0x1e64130
//
_QWORD *__fastcall sub_1E64130(_QWORD *a1, unsigned __int64 a2)
{
  _QWORD *result; // rax

  result = sub_1E63450(a1, a2);
  if ( !result )
    return (_QWORD *)sub_1E64080(a1, a2);
  return result;
}
