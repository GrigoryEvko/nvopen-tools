// Function: sub_CA3AB0
// Address: 0xca3ab0
//
__int64 __fastcall sub_CA3AB0(__int64 a1, __int64 a2, int a3)
{
  __int64 v3; // rax

  v3 = *(_QWORD *)(a1 + 32);
  LOBYTE(v3) = *(_QWORD *)(a2 + 32) == v3;
  LOBYTE(a3) = *(_QWORD *)(a2 + 40) == *(_QWORD *)(a1 + 40);
  return a3 & (unsigned int)v3;
}
