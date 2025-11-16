// Function: sub_2F8E8D0
// Address: 0x2f8e8d0
//
__int64 __fastcall sub_2F8E8D0(__int64 a1, _QWORD *a2)
{
  __int64 v3; // rdi
  __int64 (*v4)(void); // rdx
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax

  *(_QWORD *)a1 = &unk_4A2BC80;
  *(_QWORD *)(a1 + 8) = a2[1];
  v3 = a2[2];
  v4 = *(__int64 (**)(void))(*(_QWORD *)v3 + 128LL);
  v5 = 0;
  if ( v4 != sub_2DAC790 )
  {
    v5 = v4();
    v3 = a2[2];
  }
  *(_QWORD *)(a1 + 16) = v5;
  v6 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v3 + 200LL))(v3);
  *(_QWORD *)(a1 + 32) = a2;
  *(_QWORD *)(a1 + 24) = v6;
  v7 = a2[4];
  *(_QWORD *)(a1 + 192) = a1 + 208;
  *(_QWORD *)(a1 + 40) = v7;
  *(_WORD *)(a1 + 324) = 0;
  *(_QWORD *)(a1 + 112) = a1 + 128;
  *(_QWORD *)(a1 + 368) = a1 + 384;
  *(_QWORD *)(a1 + 272) = 0xFFFFFFFFLL;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 120) = 0x400000000LL;
  *(_QWORD *)(a1 + 200) = 0x400000000LL;
  *(_QWORD *)(a1 + 280) = 0;
  *(_QWORD *)(a1 + 288) = 0;
  *(_QWORD *)(a1 + 296) = 0;
  *(_QWORD *)(a1 + 304) = 0;
  *(_QWORD *)(a1 + 312) = 0;
  *(_DWORD *)(a1 + 320) = 0;
  *(_BYTE *)(a1 + 326) = 0;
  *(_QWORD *)(a1 + 328) = 0;
  *(_QWORD *)(a1 + 336) = 0;
  *(_QWORD *)(a1 + 344) = 0;
  *(_QWORD *)(a1 + 352) = 0;
  *(_QWORD *)(a1 + 360) = 0;
  *(_QWORD *)(a1 + 376) = 0x400000000LL;
  *(_QWORD *)(a1 + 448) = a1 + 464;
  *(_QWORD *)(a1 + 528) = 0xFFFFFFFFLL;
  *(_QWORD *)(a1 + 456) = 0x400000000LL;
  *(_QWORD *)(a1 + 536) = 0;
  *(_QWORD *)(a1 + 544) = 0;
  *(_QWORD *)(a1 + 552) = 0;
  *(_QWORD *)(a1 + 560) = 0;
  *(_QWORD *)(a1 + 568) = 0;
  *(_DWORD *)(a1 + 576) = 0;
  *(_WORD *)(a1 + 580) = 0;
  *(_BYTE *)(a1 + 582) = 0;
  return 0x400000000LL;
}
