// Function: sub_B14260
// Address: 0xb14260
//
__int64 __fastcall sub_B14260(_QWORD *a1)
{
  __int64 *v1; // rcx
  unsigned __int64 v2; // rdx
  __int64 result; // rax

  v1 = (__int64 *)a1[1];
  v2 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  result = v2 | *v1 & 7;
  *v1 = result;
  *(_QWORD *)(v2 + 8) = v1;
  a1[1] = 0;
  *a1 &= 7uLL;
  a1[2] = 0;
  return result;
}
