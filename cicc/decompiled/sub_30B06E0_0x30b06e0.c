// Function: sub_30B06E0
// Address: 0x30b06e0
//
void __fastcall sub_30B06E0(unsigned __int64 a1)
{
  __int64 v2; // rsi

  v2 = 16LL * *(unsigned int *)(a1 + 120);
  *(_QWORD *)a1 = &unk_4A323A8;
  sub_C7D6A0(*(_QWORD *)(a1 + 104), v2, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 72), 16LL * *(unsigned int *)(a1 + 88), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 40), 16LL * *(unsigned int *)(a1 + 56), 8);
  j_j___libc_free_0(a1);
}
