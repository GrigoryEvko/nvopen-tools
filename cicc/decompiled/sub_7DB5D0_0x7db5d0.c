// Function: sub_7DB5D0
// Address: 0x7db5d0
//
__int64 sub_7DB5D0()
{
  __int64 result; // rax
  _QWORD *v1; // rsi

  result = qword_4F18910;
  if ( !qword_4F18910 )
  {
    qword_4F18910 = sub_7E16B0(10);
    sub_7E1CA0(qword_4F18910);
    sub_7E1C50();
    sub_7E1B70("dtor");
    sub_72BA30(unk_4F0694C);
    sub_7E1B70("handle");
    sub_72BA30(unk_4F06871);
    sub_7E1B70("next");
    v1 = sub_72BA30(2u);
    sub_7E1B70("flags");
    sub_7E1C00(qword_4F18910, (unsigned __int64)v1);
    return qword_4F18910;
  }
  return result;
}
