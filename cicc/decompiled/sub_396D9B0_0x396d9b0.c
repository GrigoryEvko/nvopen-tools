// Function: sub_396D9B0
// Address: 0x396d9b0
//
__int64 __fastcall sub_396D9B0(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v5; // rax
  _QWORD *v6; // rax
  _QWORD *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rdi
  __int64 (*v10)(void); // rdx
  __int64 result; // rax

  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 24) = 3;
  *(_QWORD *)(a1 + 16) = &unk_5056088;
  *(_QWORD *)(a1 + 80) = a1 + 64;
  *(_QWORD *)(a1 + 88) = a1 + 64;
  *(_QWORD *)(a1 + 128) = a1 + 112;
  *(_QWORD *)(a1 + 136) = a1 + 112;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_DWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_DWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_BYTE *)(a1 + 152) = 0;
  *(_QWORD *)a1 = &unk_49FB790;
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 168) = 0;
  *(_DWORD *)(a1 + 176) = 8;
  v5 = (_QWORD *)malloc(8u);
  if ( !v5 )
  {
    sub_16BD1C0("Allocation failed", 1u);
    v5 = 0;
  }
  *(_QWORD *)(a1 + 160) = v5;
  *(_QWORD *)(a1 + 168) = 1;
  *v5 = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_DWORD *)(a1 + 200) = 8;
  v6 = (_QWORD *)malloc(8u);
  if ( !v6 )
  {
    sub_16BD1C0("Allocation failed", 1u);
    v6 = 0;
  }
  *(_QWORD *)(a1 + 184) = v6;
  *(_QWORD *)(a1 + 192) = 1;
  *v6 = 0;
  *(_QWORD *)(a1 + 208) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_DWORD *)(a1 + 224) = 8;
  v7 = (_QWORD *)malloc(8u);
  if ( !v7 )
  {
    sub_16BD1C0("Allocation failed", 1u);
    v7 = 0;
  }
  *(_QWORD *)(a1 + 208) = v7;
  *v7 = 0;
  *(_QWORD *)(a1 + 216) = 1;
  *(_QWORD *)(a1 + 232) = a2;
  *(_QWORD *)a1 = &unk_4A3F580;
  *(_QWORD *)(a1 + 240) = *(_QWORD *)(a2 + 608);
  v8 = *(_QWORD *)(*(_QWORD *)a3 + 8LL);
  *(_QWORD *)(a1 + 256) = *(_QWORD *)a3;
  *(_QWORD *)(a1 + 248) = v8;
  *(_QWORD *)a3 = 0;
  *(_QWORD *)(a1 + 424) = a1 + 440;
  *(_QWORD *)(a1 + 264) = 0;
  *(_QWORD *)(a1 + 272) = 0;
  *(_QWORD *)(a1 + 280) = 0;
  *(_QWORD *)(a1 + 288) = 0;
  *(_QWORD *)(a1 + 304) = 0;
  *(_QWORD *)(a1 + 312) = 0;
  *(_QWORD *)(a1 + 320) = 0;
  *(_QWORD *)(a1 + 328) = 0;
  *(_QWORD *)(a1 + 336) = 0;
  *(_DWORD *)(a1 + 344) = 0;
  *(_QWORD *)(a1 + 352) = 0;
  *(_QWORD *)(a1 + 360) = 0;
  *(_QWORD *)(a1 + 368) = 0;
  *(_BYTE *)(a1 + 376) = 0;
  *(_QWORD *)(a1 + 384) = 0;
  *(_QWORD *)(a1 + 392) = 0;
  *(_QWORD *)(a1 + 400) = 0;
  *(_QWORD *)(a1 + 408) = 0;
  *(_QWORD *)(a1 + 432) = 0x100000000LL;
  v9 = *(_QWORD *)(a1 + 256);
  *(_QWORD *)(a1 + 544) = a1 + 560;
  *(_QWORD *)(a1 + 552) = 0x400000000LL;
  *(_QWORD *)(a1 + 480) = 0;
  *(_QWORD *)(a1 + 488) = 0;
  *(_QWORD *)(a1 + 496) = 0;
  *(_QWORD *)(a1 + 504) = 0;
  *(_BYTE *)(a1 + 536) = 0;
  *(_DWORD *)(a1 + 720) = 0;
  *(_QWORD *)(a1 + 728) = 0;
  *(_QWORD *)(a1 + 736) = 0xFFFFFFFF00000000LL;
  v10 = *(__int64 (**)(void))(*(_QWORD *)v9 + 80LL);
  result = 0;
  if ( v10 != sub_168DB50 )
    result = v10();
  *(_BYTE *)(a1 + 416) = result;
  return result;
}
