// Function: sub_1E163F0
// Address: 0x1e163f0
//
unsigned __int64 __fastcall sub_1E163F0(__int64 *a1)
{
  __int64 v1; // rax
  unsigned __int64 result; // rax

  v1 = *a1;
  *((_WORD *)a1 + 23) |= 4u;
  result = v1 & 0xFFFFFFFFFFFFFFF8LL;
  *(_WORD *)(result + 46) |= 8u;
  return result;
}
