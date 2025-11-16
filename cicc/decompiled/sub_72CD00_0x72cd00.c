// Function: sub_72CD00
// Address: 0x72cd00
//
__int64 sub_72CD00()
{
  __int64 result; // rax

  result = qword_4F07AB0;
  if ( !qword_4F07AB0 )
  {
    result = sub_72CAB0("strong_equality");
    qword_4F07AB0 = result;
  }
  return result;
}
