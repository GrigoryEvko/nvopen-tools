// Function: sub_130B7F0
// Address: 0x130b7f0
//
__int64 __fastcall sub_130B7F0(__int64 a1, __int64 a2, unsigned __int64 *a3, __int64 a4)
{
  unsigned __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rsi

  sub_1342130(a1, *(_QWORD *)(a2 + 68264), a3, 232, 0);
  v6 = *a3;
  if ( (*a3 & 0x1000) != 0 )
  {
    sub_1341F30(a1, *(_QWORD *)(a2 + 68264), a3);
    v6 = *a3;
  }
  a3[1] &= 0xFFFFFFFFFFFFF000LL;
  *a3 = v6 & 0xFFFFFFFFF00FFFFFLL | 0xE800000;
  _InterlockedSub64((volatile signed __int64 *)(a2 + 8), a3[2] >> 12);
  v7 = a2 + 24;
  v8 = a2 + 62264;
  if ( (*a3 & 0x4000) == 0 )
    v8 = v7;
  return (*(__int64 (__fastcall **)(__int64, __int64, unsigned __int64 *, __int64))(v8 + 32))(a1, v8, a3, a4);
}
