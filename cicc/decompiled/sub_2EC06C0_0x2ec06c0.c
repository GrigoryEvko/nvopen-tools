// Function: sub_2EC06C0
// Address: 0x2ec06c0
//
__int64 __fastcall sub_2EC06C0(__int64 a1, __int64 a2, _BYTE *a3, unsigned __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r12d

  v6 = sub_2EC0130(a1, a3, a4, a4, a5, a6);
  *(_QWORD *)(*(_QWORD *)(a1 + 56) + 16LL * (v6 & 0x7FFFFFFF)) = a2 & 0xFFFFFFFFFFFFFFFBLL;
  sub_2EBDFC0(a1, v6);
  return v6;
}
