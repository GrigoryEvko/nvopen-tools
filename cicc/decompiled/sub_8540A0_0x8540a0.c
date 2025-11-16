// Function: sub_8540A0
// Address: 0x8540a0
//
__int64 __fastcall sub_8540A0(unsigned __int8 a1, _QWORD *a2)
{
  __int64 v2; // r12

  v2 = sub_853DF0(qword_4D03D40[a1]);
  *(_QWORD *)(v2 + 48) = *a2;
  *(_QWORD *)(v2 + 56) = *a2;
  sub_854040(v2);
  return v2;
}
