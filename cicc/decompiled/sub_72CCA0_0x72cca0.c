// Function: sub_72CCA0
// Address: 0x72cca0
//
__int64 sub_72CCA0()
{
  __int64 result; // rax

  result = qword_4F07AC0;
  if ( !qword_4F07AC0 )
  {
    result = sub_72CAB0("weak_ordering");
    qword_4F07AC0 = result;
  }
  return result;
}
