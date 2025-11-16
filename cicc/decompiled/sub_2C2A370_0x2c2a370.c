// Function: sub_2C2A370
// Address: 0x2c2a370
//
_QWORD *__fastcall sub_2C2A370(_QWORD *a1, unsigned int a2)
{
  _QWORD *result; // rax

  result = (_QWORD *)(*a1 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*a1 & 4) != 0 )
    return *(_QWORD **)(*result + 8LL * a2);
  return result;
}
