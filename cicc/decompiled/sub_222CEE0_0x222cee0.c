// Function: sub_222CEE0
// Address: 0x222cee0
//
__int64 __fastcall sub_222CEE0(_QWORD *a1, __int64 a2)
{
  char *v2; // rbx

  v2 = (char *)a1 + *(_QWORD *)(*a1 - 24LL);
  *((_QWORD *)v2 + 32) = off_4A06568;
  *(_QWORD *)v2 = off_4A06540;
  *((_QWORD *)v2 + 2) = off_4A06448;
  sub_222C7F0((__int64)(v2 + 16), a2);
  sub_2207D90((__int64)(v2 + 120));
  *((_QWORD *)v2 + 2) = off_4A07480;
  sub_2209150((volatile signed __int32 **)v2 + 9);
  *((_QWORD *)v2 + 1) = 0;
  *(_QWORD *)v2 = qword_4A064D0;
  *((_QWORD *)v2 + 32) = off_4A06798;
  return sub_222E050(v2 + 256);
}
