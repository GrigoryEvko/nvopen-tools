// Function: sub_72CCD0
// Address: 0x72ccd0
//
__int64 sub_72CCD0()
{
  __int64 result; // rax

  result = qword_4F07AB8;
  if ( !qword_4F07AB8 )
  {
    result = sub_72CAB0("partial_ordering");
    qword_4F07AB8 = result;
  }
  return result;
}
