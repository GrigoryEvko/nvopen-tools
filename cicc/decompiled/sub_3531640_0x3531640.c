// Function: sub_3531640
// Address: 0x3531640
//
void __fastcall sub_3531640(unsigned __int64 a1)
{
  __int64 v1; // rsi

  v1 = 16LL * *(unsigned int *)(a1 + 64);
  *(_QWORD *)a1 = &unk_4A38F28;
  sub_C7D6A0(*(_QWORD *)(a1 + 48), v1, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 16), 16LL * *(unsigned int *)(a1 + 32), 8);
  nullsub_1605();
  j_j___libc_free_0(a1);
}
