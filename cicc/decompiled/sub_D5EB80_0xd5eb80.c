// Function: sub_D5EB80
// Address: 0xd5eb80
//
__int64 __fastcall sub_D5EB80(_QWORD *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdx

  result = *(_QWORD *)(a2 + 40);
  v3 = *(_QWORD *)(a2 + 16);
  a1[1] = result;
  *a1 = v3;
  return result;
}
