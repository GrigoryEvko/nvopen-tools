// Function: sub_1BEFB10
// Address: 0x1befb10
//
void __fastcall sub_1BEFB10(__int64 a1)
{
  __int64 v1; // r12

  v1 = sub_1BEC610((__int64 *)a1);
  **(_QWORD **)(a1 + 16) = v1;
  sub_1BF09A0(a1 + 24, v1);
  *(_QWORD *)(a1 + 96) = v1;
  sub_1BEAFC0(a1 + 32);
  sub_1BEE1B0(*(_QWORD *)(a1 + 16) + 312LL, (_QWORD *)(a1 + 32));
}
