// Function: sub_22B1E80
// Address: 0x22b1e80
//
void __fastcall sub_22B1E80(unsigned __int64 a1)
{
  __int64 v1; // rsi

  v1 = 16LL * *(unsigned int *)(a1 + 64);
  *(_QWORD *)a1 = &unk_4A16248;
  sub_C7D6A0(*(_QWORD *)(a1 + 48), v1, 8);
  j_j___libc_free_0(a1);
}
