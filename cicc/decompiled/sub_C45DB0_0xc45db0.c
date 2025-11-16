// Function: sub_C45DB0
// Address: 0xc45db0
//
__int64 __fastcall sub_C45DB0(__int64 a1, unsigned int a2)
{
  __int64 result; // rax

  result = a2 >> 6;
  *(_QWORD *)(a1 + 8 * result) |= 1LL << a2;
  return result;
}
