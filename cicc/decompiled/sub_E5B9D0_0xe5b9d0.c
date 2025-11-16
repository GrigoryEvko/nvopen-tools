// Function: sub_E5B9D0
// Address: 0xe5b9d0
//
__int64 __fastcall sub_E5B9D0(__int64 a1, __int64 a2, _QWORD *a3, _QWORD *a4, _QWORD *a5)
{
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = *a3;
  *a3 = 0;
  *(_QWORD *)(a1 + 16) = *a4;
  *a4 = 0;
  *(_QWORD *)(a1 + 24) = *a5;
  *a5 = 0;
  *(_WORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = a1 + 56;
  *(_QWORD *)(a1 + 56) = a1 + 72;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_WORD *)(a1 + 72) = -1267;
  *(_BYTE *)(a1 + 74) = 14;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = a1 + 112;
  *(_QWORD *)(a1 + 96) = 32;
  *(_DWORD *)(a1 + 104) = 0;
  *(_BYTE *)(a1 + 108) = 1;
  *(_DWORD *)(a1 + 368) = 0;
  return a1 + 112;
}
