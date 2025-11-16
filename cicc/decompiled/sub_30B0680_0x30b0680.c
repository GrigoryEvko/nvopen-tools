// Function: sub_30B0680
// Address: 0x30b0680
//
__int64 __fastcall sub_30B0680(__int64 a1)
{
  __int64 v2; // rsi

  v2 = 16LL * *(unsigned int *)(a1 + 120);
  *(_QWORD *)a1 = &unk_4A323A8;
  sub_C7D6A0(*(_QWORD *)(a1 + 104), v2, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 72), 16LL * *(unsigned int *)(a1 + 88), 8);
  return sub_C7D6A0(*(_QWORD *)(a1 + 40), 16LL * *(unsigned int *)(a1 + 56), 8);
}
