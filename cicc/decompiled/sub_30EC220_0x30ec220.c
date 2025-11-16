// Function: sub_30EC220
// Address: 0x30ec220
//
void __fastcall sub_30EC220(unsigned __int64 a1)
{
  __int64 v1; // rsi

  v1 = 16LL * *(unsigned int *)(a1 + 32);
  *(_QWORD *)a1 = &unk_4A20C88;
  sub_C7D6A0(*(_QWORD *)(a1 + 16), v1, 8);
  j_j___libc_free_0(a1);
}
