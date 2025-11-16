// Function: sub_7E1C50
// Address: 0x7e1c50
//
__int64 sub_7E1C50()
{
  __int64 result; // rax
  _QWORD *v1; // r12

  result = qword_4F18A08;
  if ( !qword_4F18A08 )
  {
    v1 = sub_7259C0(7);
    v1[20] = sub_72CBE0();
    result = sub_72D2E0(v1);
    qword_4F18A08 = result;
  }
  return result;
}
