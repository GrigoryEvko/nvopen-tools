// Function: sub_222CCF0
// Address: 0x222ccf0
//
void __fastcall sub_222CCF0(unsigned __int64 a1, __int64 a2)
{
  _QWORD *v3; // rdi

  v3 = (_QWORD *)(a1 + 16);
  v3[30] = off_4A06568;
  *(v3 - 2) = off_4A06540;
  *v3 = off_4A06448;
  sub_222C7F0((__int64)v3, a2);
  sub_2207D90(a1 + 120);
  *(_QWORD *)(a1 + 16) = off_4A07480;
  sub_2209150((volatile signed __int32 **)(a1 + 72));
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)a1 = qword_4A064D0;
  *(_QWORD *)(a1 + 256) = off_4A06798;
  sub_222E050(a1 + 256);
  j___libc_free_0(a1);
}
