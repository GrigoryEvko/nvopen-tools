// Function: sub_222CBB0
// Address: 0x222cbb0
//
void __fastcall sub_222CBB0(unsigned __int64 a1, __int64 a2)
{
  _QWORD *v3; // rdi

  v3 = (_QWORD *)(a1 + 8);
  v3[30] = off_4A06628;
  *(v3 - 1) = off_4A06600;
  *v3 = off_4A06448;
  sub_222C7F0((__int64)v3, a2);
  sub_2207D90(a1 + 112);
  *(_QWORD *)(a1 + 8) = off_4A07480;
  sub_2209150((volatile signed __int32 **)(a1 + 64));
  *(_QWORD *)a1 = qword_4A06590;
  *(_QWORD *)(a1 + 248) = off_4A06798;
  sub_222E050(a1 + 248);
  j___libc_free_0(a1);
}
