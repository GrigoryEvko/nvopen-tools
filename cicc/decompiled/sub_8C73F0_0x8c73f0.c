// Function: sub_8C73F0
// Address: 0x8c73f0
//
__int64 sub_8C73F0()
{
  __int64 result; // rax
  __int64 v1; // rdx

  result = qword_4F60280;
  if ( qword_4F60280 )
  {
    v1 = *(_QWORD *)(qword_4F60280 + 32);
    if ( v1 )
      return *(_QWORD *)v1;
  }
  return result;
}
