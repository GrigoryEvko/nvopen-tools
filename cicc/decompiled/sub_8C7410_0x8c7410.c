// Function: sub_8C7410
// Address: 0x8c7410
//
__int64 sub_8C7410()
{
  __int64 result; // rax
  __int64 v1; // rdx

  result = qword_4F60278;
  if ( qword_4F60278 )
  {
    v1 = *(_QWORD *)(qword_4F60278 + 32);
    if ( v1 )
      return *(_QWORD *)v1;
  }
  return result;
}
