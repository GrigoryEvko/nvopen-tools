// Function: sub_8C74A0
// Address: 0x8c74a0
//
__int64 sub_8C74A0()
{
  __int64 result; // rax
  __int64 v1; // rdx

  result = qword_4F60270;
  if ( qword_4F60270 )
  {
    v1 = *(_QWORD *)(qword_4F60270 + 32);
    if ( v1 )
      return *(_QWORD *)v1;
  }
  return result;
}
