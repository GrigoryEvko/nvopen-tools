// Function: sub_1F010B0
// Address: 0x1f010b0
//
__int64 __fastcall sub_1F010B0(__int64 a1, _QWORD *a2)
{
  __int64 v3; // rdi
  __int64 (*v4)(void); // rdx
  __int64 v5; // rax
  __int64 (*v6)(void); // rdx
  __int64 v7; // rax
  __int64 v8; // rax

  *(_QWORD *)a1 = &unk_49FE548;
  *(_QWORD *)(a1 + 8) = a2[1];
  v3 = a2[2];
  v4 = *(__int64 (**)(void))(*(_QWORD *)v3 + 40LL);
  v5 = 0;
  if ( v4 != sub_1D00B00 )
  {
    v5 = v4();
    v3 = a2[2];
  }
  *(_QWORD *)(a1 + 16) = v5;
  v6 = *(__int64 (**)(void))(*(_QWORD *)v3 + 112LL);
  v7 = 0;
  if ( v6 != sub_1D00B10 )
    v7 = v6();
  *(_QWORD *)(a1 + 24) = v7;
  v8 = a2[5];
  *(_QWORD *)(a1 + 184) = a1 + 200;
  *(_QWORD *)(a1 + 40) = v8;
  *(_QWORD *)(a1 + 104) = a1 + 120;
  *(_WORD *)(a1 + 300) = 0;
  *(_QWORD *)(a1 + 32) = a2;
  *(_QWORD *)(a1 + 264) = 0xFFFFFFFFLL;
  *(_BYTE *)(a1 + 308) &= 0xFCu;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 112) = 0x400000000LL;
  *(_QWORD *)(a1 + 192) = 0x400000000LL;
  *(_QWORD *)(a1 + 272) = 0;
  *(_QWORD *)(a1 + 280) = 0;
  *(_QWORD *)(a1 + 288) = 0;
  *(_DWORD *)(a1 + 296) = 0;
  *(_DWORD *)(a1 + 304) = 0;
  *(_QWORD *)(a1 + 312) = 0;
  *(_QWORD *)(a1 + 320) = 0;
  *(_QWORD *)(a1 + 328) = 0;
  *(_QWORD *)(a1 + 336) = 0;
  *(_QWORD *)(a1 + 344) = 0;
  *(_QWORD *)(a1 + 352) = 0;
  *(_QWORD *)(a1 + 360) = 0;
  *(_QWORD *)(a1 + 368) = 0;
  *(_QWORD *)(a1 + 376) = a1 + 392;
  *(_QWORD *)(a1 + 536) = 0xFFFFFFFFLL;
  *(_BYTE *)(a1 + 580) &= 0xFCu;
  *(_QWORD *)(a1 + 384) = 0x400000000LL;
  *(_QWORD *)(a1 + 456) = a1 + 472;
  *(_QWORD *)(a1 + 464) = 0x400000000LL;
  *(_QWORD *)(a1 + 544) = 0;
  *(_QWORD *)(a1 + 552) = 0;
  *(_QWORD *)(a1 + 560) = 0;
  *(_DWORD *)(a1 + 568) = 0;
  *(_WORD *)(a1 + 572) = 0;
  *(_DWORD *)(a1 + 576) = 0;
  *(_QWORD *)(a1 + 584) = 0;
  *(_QWORD *)(a1 + 592) = 0;
  *(_QWORD *)(a1 + 600) = 0;
  *(_QWORD *)(a1 + 608) = 0;
  return 0x400000000LL;
}
