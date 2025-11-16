// Function: sub_107C150
// Address: 0x107c150
//
__int64 *__fastcall sub_107C150(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // rbx

  v6 = *a2;
  *a2 = 0;
  v7 = sub_22077B0(1112);
  v8 = v7;
  if ( v7 )
  {
    *(_BYTE *)(v7 + 40) = 0;
    *(_QWORD *)(v7 + 8) = v7 + 24;
    *(_QWORD *)(v7 + 24) = v7 + 40;
    *(_WORD *)(v7 + 80) = 0;
    *(_QWORD *)(v7 + 88) = v7 + 104;
    *(_QWORD *)v7 = off_49E6150;
    *(_QWORD *)(v7 + 16) = 0;
    *(_QWORD *)(v7 + 32) = 0;
    *(_QWORD *)(v7 + 56) = 0;
    *(_QWORD *)(v7 + 64) = 0;
    *(_QWORD *)(v7 + 72) = 0;
    *(_QWORD *)(v7 + 96) = 0;
    *(_QWORD *)(v7 + 104) = 0;
    *(_QWORD *)(v7 + 112) = v6;
    *(_QWORD *)(v7 + 120) = 0;
    *(_QWORD *)(v7 + 128) = 0;
    *(_QWORD *)(v7 + 136) = 0;
    *(_QWORD *)(v7 + 144) = 0;
    *(_QWORD *)(v7 + 152) = 0;
    *(_QWORD *)(v7 + 160) = 0;
    *(_QWORD *)(v7 + 168) = 0;
    *(_QWORD *)(v7 + 176) = 0;
    *(_QWORD *)(v7 + 184) = 0;
    *(_DWORD *)(v7 + 192) = 0;
    *(_QWORD *)(v7 + 200) = 0;
    *(_QWORD *)(v7 + 208) = 0;
    *(_QWORD *)(v7 + 216) = 0;
    *(_DWORD *)(v7 + 224) = 0;
    *(_QWORD *)(v7 + 232) = 0;
    *(_QWORD *)(v7 + 240) = 0;
    *(_QWORD *)(v7 + 248) = 0;
    *(_DWORD *)(v7 + 256) = 0;
    *(_QWORD *)(v7 + 264) = 0;
    *(_QWORD *)(v7 + 464) = v7 + 480;
    *(_QWORD *)(v7 + 272) = 0;
    *(_QWORD *)(v7 + 280) = 0;
    *(_DWORD *)(v7 + 288) = 0;
    *(_QWORD *)(v7 + 296) = 0;
    *(_QWORD *)(v7 + 304) = 0;
    *(_QWORD *)(v7 + 312) = 0;
    *(_DWORD *)(v7 + 320) = 0;
    *(_QWORD *)(v7 + 328) = 0;
    *(_QWORD *)(v7 + 336) = 0;
    *(_QWORD *)(v7 + 344) = 0;
    *(_QWORD *)(v7 + 352) = 0;
    *(_QWORD *)(v7 + 360) = 0;
    *(_QWORD *)(v7 + 368) = 0;
    *(_QWORD *)(v7 + 376) = 0;
    *(_QWORD *)(v7 + 384) = 0;
    *(_DWORD *)(v7 + 392) = 0;
    *(_QWORD *)(v7 + 400) = 0;
    *(_QWORD *)(v7 + 408) = 0;
    *(_QWORD *)(v7 + 416) = 0;
    *(_DWORD *)(v7 + 424) = 0;
    *(_QWORD *)(v7 + 432) = 0;
    *(_QWORD *)(v7 + 440) = 0;
    *(_QWORD *)(v7 + 448) = 0;
    *(_DWORD *)(v7 + 456) = 0;
    *(_QWORD *)(v7 + 472) = 0x400000000LL;
    *(_QWORD *)(v7 + 736) = v7 + 752;
    *(_QWORD *)(v7 + 744) = 0x400000000LL;
    *(_QWORD *)(v7 + 1072) = 0;
    *(_QWORD *)(v7 + 1080) = 0;
    *(_DWORD *)(v7 + 1088) = 0;
    *(_BYTE *)(v7 + 1092) = 1;
    *(_QWORD *)(v7 + 1096) = a3;
    *(_QWORD *)(v7 + 1104) = a4;
  }
  else if ( v6 )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v6 + 8LL))(v6);
  }
  *a1 = v8;
  return a1;
}
