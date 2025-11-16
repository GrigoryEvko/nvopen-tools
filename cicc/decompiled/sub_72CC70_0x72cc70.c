// Function: sub_72CC70
// Address: 0x72cc70
//
__int64 sub_72CC70()
{
  __int64 result; // rax

  result = qword_4F07AC8;
  if ( !qword_4F07AC8 )
  {
    result = sub_72CAB0("strong_ordering");
    qword_4F07AC8 = result;
  }
  return result;
}
