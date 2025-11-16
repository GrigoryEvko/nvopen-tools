// Function: sub_2306D40
// Address: 0x2306d40
//
void __fastcall sub_2306D40(unsigned __int64 a1)
{
  __int64 v1; // rsi

  v1 = 4LL * *(unsigned int *)(a1 + 40);
  *(_QWORD *)a1 = &unk_4A11838;
  sub_C7D6A0(*(_QWORD *)(a1 + 24), v1, 4);
  j_j___libc_free_0(a1);
}
