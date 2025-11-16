// Function: sub_2EC5970
// Address: 0x2ec5970
//
_QWORD *__fastcall sub_2EC5970(_QWORD *a1, __int64 a2)
{
  _QWORD *result; // rax
  _QWORD *v3; // rbx

  result = (_QWORD *)sub_22077B0(0x50u);
  v3 = result;
  if ( result )
  {
    sub_2EC5460(result);
    v3[8] = 0;
    v3[9] = 0;
    result = &unk_4A29AD0;
    *v3 = &unk_4A29AD0;
  }
  *a1 = v3;
  a1[1] = a2;
  return result;
}
