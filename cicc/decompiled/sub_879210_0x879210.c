// Function: sub_879210
// Address: 0x879210
//
__int64 __fastcall sub_879210(_QWORD *a1)
{
  __int64 result; // rax

  result = *a1;
  a1[1] = *(_QWORD *)(*a1 + 32LL);
  *(_QWORD *)(result + 32) = a1;
  return result;
}
