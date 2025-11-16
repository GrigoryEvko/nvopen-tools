// Function: sub_398B4A0
// Address: 0x398b4a0
//
void __fastcall sub_398B4A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r14
  unsigned int v6; // eax

  sub_397C0C0(*(_QWORD *)(a1 + 8), 3u, 0);
  sub_397C0C0(*(_QWORD *)(a1 + 8), *(unsigned int *)(a2 + 24), 0);
  v5 = *(_QWORD *)(a1 + 8);
  v6 = sub_39CC330(a3, *(_QWORD *)(a2 - 8LL * *(unsigned int *)(a2 + 8)));
  sub_397C0C0(v5, v6, 0);
  sub_398B530(a1, *(_QWORD *)(a2 + 8 * (1LL - *(unsigned int *)(a2 + 8))), a3);
  sub_397C0C0(*(_QWORD *)(a1 + 8), 4u, 0);
}
