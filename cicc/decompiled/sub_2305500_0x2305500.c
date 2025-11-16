// Function: sub_2305500
// Address: 0x2305500
//
void __fastcall sub_2305500(unsigned __int64 a1)
{
  __int64 v1; // rsi

  v1 = 16LL * *(unsigned int *)(a1 + 32);
  *(_QWORD *)a1 = &unk_4A0ADB8;
  sub_C7D6A0(*(_QWORD *)(a1 + 16), v1, 8);
  j_j___libc_free_0(a1);
}
