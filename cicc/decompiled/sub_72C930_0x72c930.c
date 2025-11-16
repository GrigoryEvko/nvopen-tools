// Function: sub_72C930
// Address: 0x72c930
//
__int64 sub_72C930()
{
  __int64 result; // rax

  result = qword_4F07BA8;
  if ( !qword_4F07BA8 )
  {
    qword_4F07BA8 = (__int64)sub_7259C0(0);
    sub_8D6090(qword_4F07BA8);
    return qword_4F07BA8;
  }
  return result;
}
