// Function: sub_1D29190
// Address: 0x1d29190
//
__int64 __fastcall sub_1D29190(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r8
  __int128 v7; // rdi

  v6 = a2;
  *((_QWORD *)&v7 + 1) = a3;
  *(_QWORD *)&v7 = v6;
  return sub_1D274F0(v7, a3, a4, v6, a6);
}
