// Function: sub_31B8060
// Address: 0x31b8060
//
__int64 __fastcall sub_31B8060(__int64 a1)
{
  __int64 v1; // rsi

  v1 = 8LL * *(unsigned int *)(a1 + 112);
  *(_QWORD *)a1 = &unk_4A349D0;
  sub_C7D6A0(*(_QWORD *)(a1 + 96), v1, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 64), 8LL * *(unsigned int *)(a1 + 80), 8);
  return sub_31B7F60((_QWORD *)a1);
}
