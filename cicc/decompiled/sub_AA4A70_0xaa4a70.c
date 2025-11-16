// Function: sub_AA4A70
// Address: 0xaa4a70
//
__int64 __fastcall sub_AA4A70(_QWORD *a1)
{
  __int64 *v1; // rcx
  unsigned __int64 v2; // rdx
  __int64 result; // rax

  sub_B2B7E0(a1[9] + 72LL, a1);
  v1 = (__int64 *)a1[4];
  v2 = a1[3] & 0xFFFFFFFFFFFFFFF8LL;
  result = v2 | *v1 & 7;
  *v1 = result;
  *(_QWORD *)(v2 + 8) = v1;
  a1[4] = 0;
  a1[3] &= 7uLL;
  return result;
}
