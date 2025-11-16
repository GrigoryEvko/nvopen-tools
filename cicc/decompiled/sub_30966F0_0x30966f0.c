// Function: sub_30966F0
// Address: 0x30966f0
//
void __fastcall sub_30966F0(unsigned __int64 a1)
{
  __int64 v2; // rsi

  v2 = 8LL * *(unsigned int *)(a1 + 336);
  *(_QWORD *)a1 = &unk_4A3C3B8;
  sub_C7D6A0(*(_QWORD *)(a1 + 320), v2, 4);
  sub_C7D6A0(*(_QWORD *)(a1 + 288), 4LL * *(unsigned int *)(a1 + 304), 4);
  sub_C7D6A0(*(_QWORD *)(a1 + 256), 8LL * *(unsigned int *)(a1 + 272), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 224), 8LL * *(unsigned int *)(a1 + 240), 8);
  *(_QWORD *)a1 = &unk_49DAF80;
  sub_BB9100(a1);
  j_j___libc_free_0(a1);
}
