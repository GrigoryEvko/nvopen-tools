// Function: sub_14F2510
// Address: 0x14f2510
//
__int64 __fastcall sub_14F2510(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 v5; // rax

  sub_157E9D0(a1, a3);
  v4 = *a2;
  v5 = *(_QWORD *)(a3 + 24);
  *(_QWORD *)(a3 + 32) = a2;
  v4 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)(a3 + 24) = v4 | v5 & 7;
  *(_QWORD *)(v4 + 8) = a3 + 24;
  *a2 = (a3 + 24) | *a2 & 7;
  return a3 + 24;
}
