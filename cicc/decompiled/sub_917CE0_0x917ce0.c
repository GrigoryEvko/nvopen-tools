// Function: sub_917CE0
// Address: 0x917ce0
//
__int64 __fastcall sub_917CE0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v5; // rdi

  v3 = a1 + 64;
  v5 = a1 + 136;
  *(_QWORD *)(v5 - 136) = a2;
  *(_QWORD *)(v5 - 56) = v3;
  *(_QWORD *)(v5 - 48) = v3;
  *(_QWORD *)(v5 - 128) = 0;
  *(_QWORD *)(v5 - 120) = a3;
  *(_QWORD *)(v5 - 112) = 0;
  *(_QWORD *)(v5 - 104) = 0;
  *(_QWORD *)(v5 - 96) = 0;
  *(_DWORD *)(v5 - 88) = 0;
  *(_DWORD *)(v5 - 72) = 0;
  *(_QWORD *)(v5 - 64) = 0;
  *(_QWORD *)(v5 - 40) = 0;
  *(_QWORD *)(v5 - 32) = 0;
  *(_QWORD *)(v5 - 24) = 0;
  *(_QWORD *)(v5 - 16) = 0;
  *(_DWORD *)(v5 - 8) = 0;
  sub_C656D0(v5, 6);
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = a1 + 184;
  *(_QWORD *)(a1 + 216) = a1 + 232;
  *(_QWORD *)(a1 + 168) = 4;
  *(_DWORD *)(a1 + 176) = 0;
  *(_BYTE *)(a1 + 180) = 1;
  *(_QWORD *)(a1 + 224) = 0x800000000LL;
  *(_QWORD *)(a1 + 296) = 0;
  *(_QWORD *)(a1 + 304) = 0;
  *(_QWORD *)(a1 + 312) = 0;
  *(_DWORD *)(a1 + 320) = 0;
  *(_BYTE *)(a1 + 328) = 1;
  return 0x800000000LL;
}
