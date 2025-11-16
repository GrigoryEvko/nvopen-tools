// Function: sub_35315E0
// Address: 0x35315e0
//
void __fastcall sub_35315E0(__int64 a1)
{
  __int64 v1; // rsi

  v1 = 16LL * *(unsigned int *)(a1 + 64);
  *(_QWORD *)a1 = &unk_4A38F28;
  sub_C7D6A0(*(_QWORD *)(a1 + 48), v1, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 16), 16LL * *(unsigned int *)(a1 + 32), 8);
  nullsub_1605();
}
