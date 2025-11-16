// Function: sub_B2C2B0
// Address: 0xb2c2b0
//
__int64 __fastcall sub_B2C2B0(_QWORD *a1)
{
  __int64 *v1; // rcx
  unsigned __int64 v2; // rdx
  __int64 result; // rax

  sub_BA8570(a1[5] + 24LL, a1);
  v1 = (__int64 *)a1[8];
  v2 = a1[7] & 0xFFFFFFFFFFFFFFF8LL;
  result = v2 | *v1 & 7;
  *v1 = result;
  *(_QWORD *)(v2 + 8) = v1;
  a1[8] = 0;
  a1[7] &= 7uLL;
  return result;
}
