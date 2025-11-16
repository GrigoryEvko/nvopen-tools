// Function: sub_8C7430
// Address: 0x8c7430
//
__int64 sub_8C7430()
{
  __int64 result; // rax
  __int64 v1; // rdx

  result = qword_4F60260;
  if ( qword_4F60260 )
  {
    v1 = *(_QWORD *)(qword_4F60260 + 32);
    if ( v1 )
      return *(_QWORD *)v1;
  }
  return result;
}
