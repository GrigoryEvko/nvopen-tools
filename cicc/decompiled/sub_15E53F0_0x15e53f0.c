// Function: sub_15E53F0
// Address: 0x15e53f0
//
__int64 __fastcall sub_15E53F0(_QWORD *a1)
{
  __int64 *v1; // rcx
  unsigned __int64 v2; // rdx
  __int64 result; // rax

  sub_1631C10(a1[5] + 8LL, a1);
  v1 = (__int64 *)a1[8];
  v2 = a1[7] & 0xFFFFFFFFFFFFFFF8LL;
  result = v2 | *v1 & 7;
  *v1 = result;
  *(_QWORD *)(v2 + 8) = v1;
  a1[8] = 0;
  a1[7] &= 7uLL;
  return result;
}
