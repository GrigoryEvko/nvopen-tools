// Function: sub_7E71B0
// Address: 0x7e71b0
//
_QWORD **__fastcall sub_7E71B0(_QWORD *a1)
{
  _QWORD **result; // rax

  result = &qword_4D03F68;
  do
    result = (_QWORD **)*result;
  while ( result && (!*((_BYTE *)result + 24) || result[2] != a1) );
  return result;
}
