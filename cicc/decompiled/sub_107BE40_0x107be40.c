// Function: sub_107BE40
// Address: 0x107be40
//
__int64 *__fastcall sub_107BE40(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v5; // rax

  v3 = *a2;
  *a2 = 0;
  v5 = sub_22077B0(1112);
  if ( v5 )
  {
    *(_BYTE *)(v5 + 40) = 0;
    *(_QWORD *)(v5 + 8) = v5 + 24;
    *(_QWORD *)(v5 + 24) = v5 + 40;
    *(_WORD *)(v5 + 80) = 0;
    *(_QWORD *)(v5 + 88) = v5 + 104;
    *(_QWORD *)v5 = off_49E6150;
    *(_QWORD *)(v5 + 16) = 0;
    *(_QWORD *)(v5 + 32) = 0;
    *(_QWORD *)(v5 + 56) = 0;
    *(_QWORD *)(v5 + 64) = 0;
    *(_QWORD *)(v5 + 72) = 0;
    *(_QWORD *)(v5 + 96) = 0;
    *(_QWORD *)(v5 + 104) = 0;
    *(_QWORD *)(v5 + 112) = v3;
    *(_QWORD *)(v5 + 120) = 0;
    *(_QWORD *)(v5 + 128) = 0;
    *(_QWORD *)(v5 + 136) = 0;
    *(_QWORD *)(v5 + 144) = 0;
    *(_QWORD *)(v5 + 152) = 0;
    *(_QWORD *)(v5 + 160) = 0;
    *(_QWORD *)(v5 + 168) = 0;
    *(_QWORD *)(v5 + 176) = 0;
    *(_QWORD *)(v5 + 184) = 0;
    *(_DWORD *)(v5 + 192) = 0;
    *(_QWORD *)(v5 + 200) = 0;
    *(_QWORD *)(v5 + 208) = 0;
    *(_QWORD *)(v5 + 216) = 0;
    *(_DWORD *)(v5 + 224) = 0;
    *(_QWORD *)(v5 + 232) = 0;
    *(_QWORD *)(v5 + 240) = 0;
    *(_QWORD *)(v5 + 248) = 0;
    *(_DWORD *)(v5 + 256) = 0;
    *(_QWORD *)(v5 + 264) = 0;
    *(_QWORD *)(v5 + 464) = v5 + 480;
    *(_QWORD *)(v5 + 272) = 0;
    *(_QWORD *)(v5 + 280) = 0;
    *(_DWORD *)(v5 + 288) = 0;
    *(_QWORD *)(v5 + 296) = 0;
    *(_QWORD *)(v5 + 304) = 0;
    *(_QWORD *)(v5 + 312) = 0;
    *(_DWORD *)(v5 + 320) = 0;
    *(_QWORD *)(v5 + 328) = 0;
    *(_QWORD *)(v5 + 336) = 0;
    *(_QWORD *)(v5 + 344) = 0;
    *(_QWORD *)(v5 + 352) = 0;
    *(_QWORD *)(v5 + 360) = 0;
    *(_QWORD *)(v5 + 368) = 0;
    *(_QWORD *)(v5 + 376) = 0;
    *(_QWORD *)(v5 + 384) = 0;
    *(_DWORD *)(v5 + 392) = 0;
    *(_QWORD *)(v5 + 400) = 0;
    *(_QWORD *)(v5 + 408) = 0;
    *(_QWORD *)(v5 + 416) = 0;
    *(_DWORD *)(v5 + 424) = 0;
    *(_QWORD *)(v5 + 432) = 0;
    *(_QWORD *)(v5 + 440) = 0;
    *(_QWORD *)(v5 + 448) = 0;
    *(_DWORD *)(v5 + 456) = 0;
    *(_QWORD *)(v5 + 472) = 0x400000000LL;
    *(_QWORD *)(v5 + 736) = v5 + 752;
    *(_QWORD *)(v5 + 744) = 0x400000000LL;
    *(_QWORD *)(v5 + 1072) = 0;
    *(_QWORD *)(v5 + 1080) = 0;
    *(_DWORD *)(v5 + 1088) = 0;
    *(_BYTE *)(v5 + 1092) = 0;
    *(_QWORD *)(v5 + 1096) = a3;
    *(_QWORD *)(v5 + 1104) = 0;
LABEL_3:
    *a1 = v5;
    return a1;
  }
  if ( !v3 )
    goto LABEL_3;
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v3 + 8LL))(v3);
  *a1 = 0;
  return a1;
}
