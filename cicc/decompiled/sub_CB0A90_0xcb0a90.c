// Function: sub_CB0A90
// Address: 0xcb0a90
//
unsigned __int64 *__fastcall sub_CB0A90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rsi
  __int64 v11; // rdi
  __int64 *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  unsigned __int64 *v16; // r12
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // r8
  unsigned __int64 *result; // rax

  v7 = a4;
  sub_CB09D0((_QWORD *)a1, a4);
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  v11 = 16;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)a1 = &unk_49DCE78;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  v12 = (__int64 *)sub_22077B0(16);
  v16 = (unsigned __int64 *)v12;
  if ( v12 )
  {
    v7 = a2;
    v11 = (__int64)v12;
    sub_CA9F50(v12, a2, a3, a1 + 16, 0, a1 + 96);
  }
  *(_QWORD *)(a1 + 80) = v16;
  *(_QWORD *)(a1 + 88) = 0;
  *(_DWORD *)(a1 + 96) = 0;
  v17 = sub_2241E40(v11, v7, v13, v14, v15);
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 176) = a1 + 192;
  *(_QWORD *)(a1 + 224) = a1 + 240;
  *(_QWORD *)(a1 + 272) = a1 + 288;
  *(_QWORD *)(a1 + 320) = a1 + 336;
  *(_QWORD *)(a1 + 104) = v17;
  *(_QWORD *)(a1 + 368) = a1 + 384;
  *(_QWORD *)(a1 + 128) = a1 + 144;
  *(_QWORD *)(a1 + 416) = a1 + 432;
  *(_QWORD *)(a1 + 136) = 0x400000000LL;
  *(_QWORD *)(a1 + 232) = 0x400000000LL;
  *(_QWORD *)(a1 + 328) = 0x400000000LL;
  *(_QWORD *)(a1 + 424) = 0x400000000LL;
  *(_QWORD *)(a1 + 464) = a1 + 480;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_QWORD *)(a1 + 200) = 1;
  *(_QWORD *)(a1 + 208) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_QWORD *)(a1 + 280) = 0;
  *(_QWORD *)(a1 + 288) = 0;
  *(_QWORD *)(a1 + 296) = 0;
  *(_QWORD *)(a1 + 304) = 0;
  *(_QWORD *)(a1 + 312) = 0;
  *(_QWORD *)(a1 + 376) = 0;
  *(_QWORD *)(a1 + 384) = 0;
  *(_QWORD *)(a1 + 392) = 0;
  *(_QWORD *)(a1 + 400) = 0;
  *(_QWORD *)(a1 + 408) = 0;
  *(_QWORD *)(a1 + 472) = 0;
  *(_QWORD *)(a1 + 480) = 0;
  *(_QWORD *)(a1 + 520) = 0x400000000LL;
  *(_QWORD *)(a1 + 560) = a1 + 576;
  *(_QWORD *)(a1 + 600) = a1 + 616;
  *(_QWORD *)(a1 + 608) = 0x600000000LL;
  *(_QWORD *)(a1 + 488) = 0;
  *(_QWORD *)(a1 + 496) = 0;
  *(_QWORD *)(a1 + 504) = 0;
  *(_QWORD *)(a1 + 512) = a1 + 528;
  *(_QWORD *)(a1 + 568) = 0;
  *(_QWORD *)(a1 + 576) = 0;
  *(_QWORD *)(a1 + 584) = 0;
  *(_QWORD *)(a1 + 592) = 0;
  *(_DWORD *)(a1 + 664) = 0;
  *(_QWORD *)(a1 + 672) = 0;
  *(_WORD *)(a1 + 680) = 0;
  if ( a5 )
  {
    *(_QWORD *)(a1 + 64) = a5;
    *(_QWORD *)(a1 + 72) = a6;
  }
  result = sub_CAFE70(v16, v7, a1 + 528, v18, v19);
  *(_QWORD *)(a1 + 592) = result;
  return result;
}
