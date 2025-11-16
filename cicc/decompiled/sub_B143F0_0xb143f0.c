// Function: sub_B143F0
// Address: 0xb143f0
//
__int64 __fastcall sub_B143F0(__int64 *a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // rax
  __int64 result; // rax
  __int64 v7; // rdi

  v7 = *(_QWORD *)(a2 + 16);
  v2 = *(__int64 **)(a2 + 8);
  v3 = *v2;
  v4 = *a1 & 7;
  a1[1] = (__int64)v2;
  v3 &= 0xFFFFFFFFFFFFFFF8LL;
  *a1 = v3 | v4;
  *(_QWORD *)(v3 + 8) = a1;
  result = (unsigned __int64)a1 | *v2 & 7;
  *v2 = result;
  a1[2] = v7;
  return result;
}
