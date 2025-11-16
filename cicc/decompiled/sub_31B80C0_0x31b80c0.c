// Function: sub_31B80C0
// Address: 0x31b80c0
//
void __fastcall sub_31B80C0(unsigned __int64 a1)
{
  __int64 v1; // rsi

  v1 = 8LL * *(unsigned int *)(a1 + 112);
  *(_QWORD *)a1 = &unk_4A349D0;
  sub_C7D6A0(*(_QWORD *)(a1 + 96), v1, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 64), 8LL * *(unsigned int *)(a1 + 80), 8);
  sub_31B7F60((_QWORD *)a1);
  j_j___libc_free_0(a1);
}
