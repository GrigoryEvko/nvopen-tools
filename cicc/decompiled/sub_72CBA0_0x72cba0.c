// Function: sub_72CBA0
// Address: 0x72cba0
//
__int64 sub_72CBA0()
{
  __int64 result; // rax

  result = qword_4F07BA0;
  if ( !qword_4F07BA0 )
  {
    qword_4F07BA0 = (__int64)sub_7259C0(21);
    sub_8D6090(qword_4F07BA0);
    return qword_4F07BA0;
  }
  return result;
}
