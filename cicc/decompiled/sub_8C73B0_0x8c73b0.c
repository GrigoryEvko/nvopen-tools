// Function: sub_8C73B0
// Address: 0x8c73b0
//
__int64 sub_8C73B0()
{
  __int64 result; // rax
  __int64 v1; // rdx

  result = qword_4F60290;
  if ( qword_4F60290 )
  {
    v1 = *(_QWORD *)(qword_4F60290 + 32);
    if ( v1 )
      return *(_QWORD *)v1;
  }
  return result;
}
