// Function: sub_1648700
// Address: 0x1648700
//
_QWORD *__fastcall sub_1648700(__int64 a1)
{
  _QWORD *result; // rax

  result = (_QWORD *)sub_1648690(a1);
  if ( (*result & 1) != 0 )
    return (_QWORD *)(*result & 0xFFFFFFFFFFFFFFFELL);
  return result;
}
