// Function: sub_B30290
// Address: 0xb30290
//
__int64 __fastcall sub_B30290(__int64 a1)
{
  unsigned __int64 *v1; // rcx
  unsigned __int64 v2; // rdx
  __int64 v3; // rdx
  __int64 v4; // rcx

  sub_BA85F0(*(_QWORD *)(a1 + 40) + 8LL, a1);
  v1 = *(unsigned __int64 **)(a1 + 64);
  v2 = *(_QWORD *)(a1 + 56) & 0xFFFFFFFFFFFFFFF8LL;
  *v1 = v2 | *v1 & 7;
  *(_QWORD *)(v2 + 8) = v1;
  *(_QWORD *)(a1 + 56) &= 7uLL;
  *(_QWORD *)(a1 + 64) = 0;
  sub_B30220(a1);
  *(_DWORD *)(a1 + 4) = *(_DWORD *)(a1 + 4) & 0xF8000000 | 1;
  sub_B2F9E0(a1, a1, v3, v4);
  return sub_BD2DD0(a1);
}
