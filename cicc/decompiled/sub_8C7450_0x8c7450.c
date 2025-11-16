// Function: sub_8C7450
// Address: 0x8c7450
//
__int64 sub_8C7450()
{
  __int64 result; // rax
  __int64 v1; // rdx

  result = qword_4F60268;
  if ( qword_4F60268 )
  {
    v1 = *(_QWORD *)(qword_4F60268 + 32);
    if ( v1 )
      return *(_QWORD *)v1;
  }
  return result;
}
