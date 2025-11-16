// Function: sub_131B1D0
// Address: 0x131b1d0
//
__int64 __fastcall sub_131B1D0(unsigned __int64 *a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v4; // rax
  __int64 v5; // rcx
  __int64 v6; // rsi
  unsigned __int64 v7; // rax

  v4 = a1[1];
  v5 = (v4 + a4 - 1) & -a4;
  *a2 = v5 - v4;
  v6 = a1[1] + v5 - v4;
  a1[2] = a1[2] + v4 - a3 - v5;
  v7 = *a1;
  a1[1] = v6 + a3;
  *a1 = v7 & 0xFFFFFFFFF0000000LL | 0xE80AFFF;
  return v6;
}
