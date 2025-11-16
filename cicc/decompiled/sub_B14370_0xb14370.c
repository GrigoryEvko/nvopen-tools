// Function: sub_B14370
// Address: 0xb14370
//
__int64 __fastcall sub_B14370(unsigned __int64 a1, __int64 *a2)
{
  __int64 v2; // rcx
  __int64 v3; // rax
  __int64 result; // rax
  __int64 v6; // rdi

  v6 = a2[2];
  v2 = *a2;
  v3 = *(_QWORD *)a1;
  *(_QWORD *)(a1 + 8) = a2;
  v2 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)a1 = v2 | v3 & 7;
  *(_QWORD *)(v2 + 8) = a1;
  result = a1 | *a2 & 7;
  *a2 = result;
  *(_QWORD *)(a1 + 16) = v6;
  return result;
}
