// Function: sub_16A70F0
// Address: 0x16a70f0
//
__int64 __fastcall sub_16A70F0(__int64 a1, unsigned int a2)
{
  __int64 result; // rax

  result = ~(1LL << a2);
  *(_QWORD *)(a1 + 8LL * (a2 >> 6)) &= result;
  return result;
}
