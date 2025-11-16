// Function: sub_26E11A0
// Address: 0x26e11a0
//
_QWORD *__fastcall sub_26E11A0(__int64 a1)
{
  _QWORD *result; // rax
  _QWORD *i; // rbx

  result = *(_QWORD **)(a1 + 8);
  for ( i = (_QWORD *)result[3]; i; i = (_QWORD *)*i )
    result = sub_26E0FF0(a1, i + 2);
  return result;
}
