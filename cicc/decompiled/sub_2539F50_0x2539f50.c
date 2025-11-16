// Function: sub_2539F50
// Address: 0x2539f50
//
void __fastcall sub_2539F50(unsigned __int64 a1)
{
  __int64 v1; // rsi

  v1 = 16LL * *(unsigned int *)(a1 + 80);
  *(_QWORD *)a1 = &unk_4A17318;
  sub_C7D6A0(*(_QWORD *)(a1 + 64), v1, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 24), 16LL * *(unsigned int *)(a1 + 40), 8);
  j_j___libc_free_0(a1);
}
