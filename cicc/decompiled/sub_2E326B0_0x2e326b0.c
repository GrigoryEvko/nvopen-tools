// Function: sub_2E326B0
// Address: 0x2e326b0
//
__int64 __fastcall sub_2E326B0(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 v5; // rax

  if ( a2 != (__int64 *)(a1 + 48) && (*((_BYTE *)a2 + 44) & 4) != 0 )
    *(_DWORD *)(a3 + 44) |= 0xCu;
  sub_2E31040((__int64 *)(a1 + 40), a3);
  v4 = *a2;
  v5 = *(_QWORD *)a3;
  *(_QWORD *)(a3 + 8) = a2;
  v4 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)a3 = v4 | v5 & 7;
  *(_QWORD *)(v4 + 8) = a3;
  *a2 = a3 | *a2 & 7;
  return a3;
}
