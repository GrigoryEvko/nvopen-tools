// Function: sub_15E55B0
// Address: 0x15e55b0
//
__int64 __fastcall sub_15E55B0(__int64 a1)
{
  unsigned __int64 *v1; // rcx
  unsigned __int64 v2; // rdx

  sub_1631C10(*(_QWORD *)(a1 + 40) + 8LL, a1);
  v1 = *(unsigned __int64 **)(a1 + 64);
  v2 = *(_QWORD *)(a1 + 56) & 0xFFFFFFFFFFFFFFF8LL;
  *v1 = v2 | *v1 & 7;
  *(_QWORD *)(v2 + 8) = v1;
  *(_QWORD *)(a1 + 56) &= 7uLL;
  *(_QWORD *)(a1 + 64) = 0;
  sub_15E5530(a1);
  sub_159D9E0(a1);
  sub_164BE60(a1);
  *(_DWORD *)(a1 + 20) = *(_DWORD *)(a1 + 20) & 0xF0000000 | 1;
  return sub_1648B90(a1);
}
