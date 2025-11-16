// Function: sub_222CD90
// Address: 0x222cd90
//
void __fastcall sub_222CD90(_QWORD *a1, __int64 a2)
{
  unsigned __int64 v2; // rbp

  v2 = (unsigned __int64)a1 + *(_QWORD *)(*a1 - 24LL);
  *(_QWORD *)(v2 + 256) = off_4A06568;
  *(_QWORD *)v2 = off_4A06540;
  *(_QWORD *)(v2 + 16) = off_4A06448;
  sub_222C7F0(v2 + 16, a2);
  sub_2207D90(v2 + 120);
  *(_QWORD *)(v2 + 16) = off_4A07480;
  sub_2209150((volatile signed __int32 **)(v2 + 72));
  *(_QWORD *)(v2 + 8) = 0;
  *(_QWORD *)v2 = qword_4A064D0;
  *(_QWORD *)(v2 + 256) = off_4A06798;
  sub_222E050(v2 + 256);
  j___libc_free_0(v2);
}
