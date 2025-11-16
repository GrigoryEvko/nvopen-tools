// Function: sub_72CD30
// Address: 0x72cd30
//
__int64 sub_72CD30()
{
  __int64 result; // rax

  result = qword_4F07AA8;
  if ( !qword_4F07AA8 )
  {
    result = sub_72CAB0("weak_equality");
    qword_4F07AA8 = result;
  }
  return result;
}
