// Function: sub_730690
// Address: 0x730690
//
_QWORD *__fastcall sub_730690(__int64 a1)
{
  _QWORD *result; // rax

  result = sub_726700(2);
  result[7] = a1;
  *result = *(_QWORD *)(a1 + 128);
  return result;
}
