// Function: sub_12778D0
// Address: 0x12778d0
//
__int64 __fastcall sub_12778D0(__int64 a1, __int64 a2, __int64 a3)
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
  sub_16BD940(v5, 6);
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 184) = 4;
  *(_DWORD *)(a1 + 192) = 0;
  *(_QWORD *)(a1 + 136) = &unk_49E69C0;
  *(_QWORD *)(a1 + 168) = a1 + 200;
  *(_QWORD *)(a1 + 176) = a1 + 200;
  *(_QWORD *)(a1 + 232) = a1 + 248;
  *(_QWORD *)(a1 + 240) = 0x800000000LL;
  *(_QWORD *)(a1 + 312) = 0;
  *(_QWORD *)(a1 + 320) = 0;
  *(_QWORD *)(a1 + 328) = 0;
  *(_DWORD *)(a1 + 336) = 0;
  *(_BYTE *)(a1 + 344) = 1;
  return 0x800000000LL;
}
