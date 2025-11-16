// Function: sub_24DAB80
// Address: 0x24dab80
//
_QWORD *__fastcall sub_24DAB80(_QWORD *a1, __int64 a2, __int64 a3)
{
  _QWORD *result; // rax

  result = (_QWORD *)sub_22077B0(0x28u);
  if ( result )
  {
    *result = a2;
    result[1] = a2;
    result[2] = a2;
    result[3] = a3;
    result[4] = 0;
  }
  *a1 = result;
  return result;
}
