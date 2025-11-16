// Function: sub_2AB2780
// Address: 0x2ab2780
//
char __fastcall sub_2AB2780(
        __int64 a1,
        int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int64 a11,
        __int64 a12,
        __int64 a13,
        __int64 a14,
        __int64 a15,
        __int64 *a16)
{
  __int64 v17; // r14
  __int64 v18; // rax
  char result; // al
  int v20; // r14d
  __int64 v21; // [rsp+0h] [rbp-40h] BYREF
  __int64 v22; // [rsp+8h] [rbp-38h]

  *(_QWORD *)(a1 + 48) = a1 + 64;
  *(_DWORD *)a1 = 0;
  *(_BYTE *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_DWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_DWORD *)(a1 + 88) = 0;
  *(_DWORD *)(a1 + 96) = a2;
  *(_BYTE *)(a1 + 108) = 0;
  *(_BYTE *)(a1 + 113) = 0;
  *(_BYTE *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_DWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_DWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 264) = a1 + 288;
  *(_QWORD *)(a1 + 192) = 0;
  *(_QWORD *)(a1 + 200) = 0;
  *(_QWORD *)(a1 + 208) = 0;
  *(_DWORD *)(a1 + 216) = 0;
  *(_QWORD *)(a1 + 224) = 0;
  *(_QWORD *)(a1 + 232) = 0;
  *(_QWORD *)(a1 + 240) = 0;
  *(_DWORD *)(a1 + 248) = 0;
  *(_QWORD *)(a1 + 256) = 0;
  *(_QWORD *)(a1 + 272) = 4;
  *(_DWORD *)(a1 + 280) = 0;
  *(_BYTE *)(a1 + 284) = 1;
  *(_QWORD *)(a1 + 320) = 0;
  *(_QWORD *)(a1 + 328) = 0;
  *(_QWORD *)(a1 + 336) = 0;
  *(_DWORD *)(a1 + 344) = 0;
  *(_QWORD *)(a1 + 352) = 0;
  *(_QWORD *)(a1 + 360) = 0;
  *(_QWORD *)(a1 + 368) = 0;
  *(_DWORD *)(a1 + 376) = 0;
  *(_QWORD *)(a1 + 384) = 0;
  *(_QWORD *)(a1 + 392) = 0;
  *(_QWORD *)(a1 + 400) = 0;
  *(_DWORD *)(a1 + 408) = 0;
  *(_QWORD *)(a1 + 416) = a3;
  *(_QWORD *)(a1 + 424) = a4;
  *(_QWORD *)(a1 + 432) = a5;
  *(_QWORD *)(a1 + 440) = a6;
  *(_QWORD *)(a1 + 448) = a7;
  *(_QWORD *)(a1 + 456) = a8;
  *(_QWORD *)(a1 + 488) = a12;
  *(_QWORD *)(a1 + 464) = a9;
  *(_QWORD *)(a1 + 512) = 0;
  *(_QWORD *)(a1 + 472) = a10;
  *(_QWORD *)(a1 + 528) = 16;
  *(_QWORD *)(a1 + 480) = a11;
  *(_DWORD *)(a1 + 536) = 0;
  *(_QWORD *)(a1 + 496) = a13;
  *(_BYTE *)(a1 + 540) = 1;
  *(_QWORD *)(a1 + 504) = a14;
  *(_QWORD *)(a1 + 520) = a1 + 544;
  *(_QWORD *)(a1 + 680) = a1 + 704;
  *(_QWORD *)(a1 + 672) = 0;
  *(_DWORD *)(a1 + 688) = 16;
  *(_QWORD *)(a1 + 692) = 0;
  *(_BYTE *)(a1 + 700) = 1;
  *(_QWORD *)(a1 + 832) = 0;
  *(_QWORD *)(a1 + 840) = a1 + 864;
  *(_QWORD *)(a1 + 848) = 16;
  *(_DWORD *)(a1 + 856) = 0;
  *(_BYTE *)(a1 + 860) = 1;
  if ( (unsigned __int8)sub_DFE610(a7) || byte_500DD68 )
  {
    v17 = *(_QWORD *)(**(_QWORD **)(*(_QWORD *)(a1 + 416) + 32LL) + 72LL);
    if ( !(unsigned __int8)sub_B2D610(v17, 96)
      || (v21 = sub_B2D7D0(v17, 96), v20 = sub_A71EB0(&v21), v18 = sub_A71ED0(&v21), v22 = v18, !BYTE4(v18))
      || v20 != (_DWORD)v22 )
    {
      v18 = sub_DFB260(*(_QWORD *)(a1 + 448));
    }
    *(_QWORD *)(a1 + 4) = v18;
  }
  *(_DWORD *)(a1 + 992) = 2 * ((unsigned __int8)sub_B2D610(a12, 18) != 0);
  result = sub_11F3070(**(_QWORD **)(a3 + 32), a15, a16);
  *(_BYTE *)(a1 + 996) = result;
  return result;
}
