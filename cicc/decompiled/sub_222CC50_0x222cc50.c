// Function: sub_222CC50
// Address: 0x222cc50
//
void __fastcall sub_222CC50(_QWORD *a1, __int64 a2)
{
  unsigned __int64 v2; // rbp

  v2 = (unsigned __int64)a1 + *(_QWORD *)(*a1 - 24LL);
  *(_QWORD *)(v2 + 248) = off_4A06628;
  *(_QWORD *)v2 = off_4A06600;
  *(_QWORD *)(v2 + 8) = off_4A06448;
  sub_222C7F0(v2 + 8, a2);
  sub_2207D90(v2 + 112);
  *(_QWORD *)(v2 + 8) = off_4A07480;
  sub_2209150((volatile signed __int32 **)(v2 + 64));
  *(_QWORD *)v2 = qword_4A06590;
  *(_QWORD *)(v2 + 248) = off_4A06798;
  sub_222E050(v2 + 248);
  j___libc_free_0(v2);
}
