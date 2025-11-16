// Function: sub_2E32650
// Address: 0x2e32650
//
__int64 __fastcall sub_2E32650(__int64 a1, __int64 a2)
{
  unsigned __int64 *v2; // rcx
  unsigned __int64 v3; // rdx

  sub_2E2F9E0(a2);
  *(_DWORD *)(a2 + 44) &= 0xFFFFFFF3;
  sub_2E31080(a1 + 40, a2);
  v2 = *(unsigned __int64 **)(a2 + 8);
  v3 = *(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL;
  *v2 = v3 | *v2 & 7;
  *(_QWORD *)(v3 + 8) = v2;
  *(_QWORD *)a2 &= 7uLL;
  *(_QWORD *)(a2 + 8) = 0;
  return a2;
}
