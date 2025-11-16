// Function: sub_16A70D0
// Address: 0x16a70d0
//
__int64 __fastcall sub_16A70D0(__int64 a1, unsigned int a2)
{
  __int64 result; // rax

  result = a2 >> 6;
  *(_QWORD *)(a1 + 8 * result) |= 1LL << a2;
  return result;
}
