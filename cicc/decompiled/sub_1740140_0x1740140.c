// Function: sub_1740140
// Address: 0x1740140
//
unsigned __int64 __fastcall sub_1740140(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rcx

  sub_157E9D0(*(_QWORD *)(a3 + 40) + 40LL, a2);
  v4 = *(_QWORD *)(a3 + 24);
  *(_QWORD *)(a2 + 32) = a3 + 24;
  v4 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)(a2 + 24) = v4 | *(_QWORD *)(a2 + 24) & 7LL;
  *(_QWORD *)(v4 + 8) = a2 + 24;
  *(_QWORD *)(a3 + 24) = *(_QWORD *)(a3 + 24) & 7LL | (a2 + 24);
  return sub_170B990(*a1, a2);
}
