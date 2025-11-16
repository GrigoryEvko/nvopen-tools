// Function: sub_222CB10
// Address: 0x222cb10
//
__int64 __fastcall sub_222CB10(_QWORD *a1, __int64 a2)
{
  char *v2; // rbx

  v2 = (char *)a1 + *(_QWORD *)(*a1 - 24LL);
  *((_QWORD *)v2 + 31) = off_4A06628;
  *(_QWORD *)v2 = off_4A06600;
  *((_QWORD *)v2 + 1) = off_4A06448;
  sub_222C7F0((__int64)(v2 + 8), a2);
  sub_2207D90((__int64)(v2 + 112));
  *((_QWORD *)v2 + 1) = off_4A07480;
  sub_2209150((volatile signed __int32 **)v2 + 8);
  *(_QWORD *)v2 = qword_4A06590;
  *((_QWORD *)v2 + 31) = off_4A06798;
  return sub_222E050(v2 + 248);
}
