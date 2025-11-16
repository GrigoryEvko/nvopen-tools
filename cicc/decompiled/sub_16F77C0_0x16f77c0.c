// Function: sub_16F77C0
// Address: 0x16f77c0
//
__int64 __fastcall sub_16F77C0(__int64 a1, char *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // rdx

  result = sub_16F7770(a1, a2, a3, *(_QWORD *)(a1 + 40));
  v4 = result - *(_QWORD *)(a1 + 40);
  *(_QWORD *)(a1 + 40) = result;
  *(_DWORD *)(a1 + 60) += v4;
  return result;
}
