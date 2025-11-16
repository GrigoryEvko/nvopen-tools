// Function: sub_8C73D0
// Address: 0x8c73d0
//
__int64 sub_8C73D0()
{
  __int64 result; // rax
  __int64 v1; // rdx

  result = qword_4F60288;
  if ( qword_4F60288 )
  {
    v1 = *(_QWORD *)(qword_4F60288 + 32);
    if ( v1 )
      return *(_QWORD *)v1;
  }
  return result;
}
