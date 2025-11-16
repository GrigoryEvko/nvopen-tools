// Function: sub_B3AE10
// Address: 0xb3ae10
//
__int64 sub_B3AE10()
{
  __int64 result; // rax

  result = sub_22077B0(40);
  if ( result )
  {
    *(_QWORD *)(result + 32) = 0;
    *(_OWORD *)result = 0;
    *(_OWORD *)(result + 16) = 0;
  }
  return result;
}
