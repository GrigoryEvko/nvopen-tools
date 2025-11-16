// Function: sub_2F65720
// Address: 0x2f65720
//
__int64 __fastcall sub_2F65720(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v3; // rcx
  __int64 v4; // rsi
  unsigned int v6; // edx

  v3 = *(_QWORD *)(a2 + 8);
  v4 = *(_QWORD *)(a2 + 56);
  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  v6 = *(_DWORD *)(v3 + 24LL * a3 + 16);
  *(_DWORD *)(a1 + 40) = 0;
  LODWORD(v3) = v6 & 0xFFF;
  *(_DWORD *)a1 = v3;
  *(_QWORD *)(a1 + 8) = v4 + 2LL * (v6 >> 12);
  *(_DWORD *)(a1 + 16) = v3;
  return a1;
}
