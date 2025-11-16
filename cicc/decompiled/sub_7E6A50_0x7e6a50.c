// Function: sub_7E6A50
// Address: 0x7e6a50
//
_QWORD *__fastcall sub_7E6A50(_QWORD *a1, int *a2)
{
  _QWORD *result; // rax

  result = sub_7E69E0(a1, a2);
  if ( result )
  {
    *result = *(_QWORD *)dword_4D03F38;
    result[1] = *(_QWORD *)dword_4D03F38;
  }
  return result;
}
