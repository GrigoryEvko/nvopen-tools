// Function: sub_CA00D0
// Address: 0xca00d0
//
__int64 __fastcall sub_CA00D0(__int64 a1)
{
  __int64 v1; // r12
  int v2; // edx
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 *v5; // r13
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  char v11; // al
  bool v12; // zf
  __int64 v13; // rax
  int v14; // edx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 *v17; // r12
  __int64 v18; // rax
  int v19; // edx
  __int64 *v20; // rbx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rax
  char v24; // al
  __int64 result; // rax
  const char *v26; // [rsp+0h] [rbp-60h] BYREF
  char v27; // [rsp+20h] [rbp-40h]
  char v28; // [rsp+21h] [rbp-3Fh]

  v1 = a1 + 32;
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 32) = &unk_49DC150;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)(a1 + 44) &= 0x8000u;
  *(_QWORD *)(a1 + 112) = 0x100000000LL;
  *(_DWORD *)(a1 + 40) = v2;
  *(_WORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = a1 + 120;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = a1 + 160;
  *(_QWORD *)(a1 + 144) = 1;
  *(_DWORD *)(a1 + 152) = 0;
  *(_BYTE *)(a1 + 156) = 1;
  v5 = sub_C57470();
  v6 = *(unsigned int *)(a1 + 112);
  if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 116) )
  {
    sub_C8D5F0(a1 + 104, (const void *)(a1 + 120), v6 + 1, 8u, v3, v4);
    v6 = *(unsigned int *)(a1 + 112);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 104) + 8 * v6) = v5;
  *(_QWORD *)(a1 + 184) = a1 + 200;
  ++*(_DWORD *)(a1 + 112);
  *(_BYTE *)(a1 + 200) = 0;
  *(_QWORD *)(a1 + 176) = &unk_49DC130;
  *(_QWORD *)(a1 + 168) = 0;
  *(_BYTE *)(a1 + 216) = 0;
  *(_QWORD *)(a1 + 32) = &unk_49DCA98;
  *(_QWORD *)(a1 + 192) = 0;
  *(_QWORD *)(a1 + 224) = &unk_49DC350;
  *(_QWORD *)(a1 + 256) = nullsub_164;
  *(_QWORD *)(a1 + 248) = sub_C8B390;
  sub_C53080(v1, (__int64)"info-output-file", 16);
  *(_QWORD *)(a1 + 96) = 8;
  *(_QWORD *)(a1 + 88) = "filename";
  *(_QWORD *)(a1 + 72) = "File to append -stats and -timer output to";
  v11 = *(_BYTE *)(a1 + 44);
  *(_QWORD *)(a1 + 80) = 42;
  v12 = *(_QWORD *)(a1 + 168) == 0;
  *(_BYTE *)(a1 + 44) = v11 & 0x9F | 0x20;
  if ( v12 )
  {
    *(_QWORD *)(a1 + 168) = a1;
    *(_BYTE *)(a1 + 216) = 1;
    sub_2240AE0(a1 + 184, a1);
  }
  else
  {
    v13 = sub_CEADF0(v1, "info-output-file", v7, v8, v9, v10);
    v28 = 1;
    v26 = "cl::location(x) specified more than once!";
    v27 = 3;
    sub_C53280(v1, (__int64)&v26, 0, 0, v13);
  }
  sub_C53130(v1);
  *(_QWORD *)(a1 + 264) = &unk_49DC150;
  v14 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)(a1 + 276) &= 0x8000u;
  *(_QWORD *)(a1 + 344) = 0x100000000LL;
  *(_DWORD *)(a1 + 272) = v14;
  *(_WORD *)(a1 + 280) = 0;
  *(_QWORD *)(a1 + 288) = 0;
  *(_QWORD *)(a1 + 296) = 0;
  *(_QWORD *)(a1 + 304) = 0;
  *(_QWORD *)(a1 + 312) = 0;
  *(_QWORD *)(a1 + 320) = 0;
  *(_QWORD *)(a1 + 328) = 0;
  *(_QWORD *)(a1 + 336) = a1 + 352;
  *(_QWORD *)(a1 + 360) = 0;
  *(_QWORD *)(a1 + 368) = a1 + 392;
  *(_QWORD *)(a1 + 376) = 1;
  *(_DWORD *)(a1 + 384) = 0;
  *(_BYTE *)(a1 + 388) = 1;
  v17 = sub_C57470();
  v18 = *(unsigned int *)(a1 + 344);
  if ( v18 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 348) )
  {
    sub_C8D5F0(a1 + 336, (const void *)(a1 + 352), v18 + 1, 8u, v15, v16);
    v18 = *(unsigned int *)(a1 + 344);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 336) + 8 * v18) = v17;
  ++*(_DWORD *)(a1 + 344);
  *(_WORD *)(a1 + 416) = 0;
  *(_QWORD *)(a1 + 408) = &unk_49D9748;
  *(_BYTE *)(a1 + 400) = 0;
  *(_QWORD *)(a1 + 264) = &unk_49DC090;
  *(_QWORD *)(a1 + 424) = &unk_49DC1D0;
  *(_QWORD *)(a1 + 456) = nullsub_23;
  *(_QWORD *)(a1 + 448) = sub_984030;
  sub_C53080(a1 + 264, (__int64)"track-memory", 12);
  *(_QWORD *)(a1 + 312) = 54;
  *(_QWORD *)(a1 + 304) = "Enable -time-passes memory tracking (this may be slow)";
  *(_BYTE *)(a1 + 276) = *(_BYTE *)(a1 + 276) & 0x9F | 0x20;
  sub_C53130(a1 + 264);
  *(_QWORD *)(a1 + 464) = &unk_49DC150;
  v19 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)(a1 + 476) &= 0x8000u;
  *(_QWORD *)(a1 + 544) = 0x100000000LL;
  *(_DWORD *)(a1 + 472) = v19;
  *(_WORD *)(a1 + 480) = 0;
  *(_QWORD *)(a1 + 488) = 0;
  *(_QWORD *)(a1 + 496) = 0;
  *(_QWORD *)(a1 + 504) = 0;
  *(_QWORD *)(a1 + 512) = 0;
  *(_QWORD *)(a1 + 520) = 0;
  *(_QWORD *)(a1 + 528) = 0;
  *(_QWORD *)(a1 + 536) = a1 + 552;
  *(_QWORD *)(a1 + 560) = 0;
  *(_QWORD *)(a1 + 568) = a1 + 592;
  *(_QWORD *)(a1 + 576) = 1;
  *(_DWORD *)(a1 + 584) = 0;
  *(_BYTE *)(a1 + 588) = 1;
  v20 = sub_C57470();
  v23 = *(unsigned int *)(a1 + 544);
  if ( v23 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 548) )
  {
    sub_C8D5F0(a1 + 536, (const void *)(a1 + 552), v23 + 1, 8u, v21, v22);
    v23 = *(unsigned int *)(a1 + 544);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 536) + 8 * v23) = v20;
  *(_WORD *)(a1 + 616) = 0;
  ++*(_DWORD *)(a1 + 544);
  *(_BYTE *)(a1 + 600) = 0;
  *(_QWORD *)(a1 + 608) = &unk_49D9748;
  *(_QWORD *)(a1 + 464) = &unk_49DC090;
  *(_QWORD *)(a1 + 624) = &unk_49DC1D0;
  *(_QWORD *)(a1 + 656) = nullsub_23;
  *(_QWORD *)(a1 + 648) = sub_984030;
  sub_C53080(a1 + 464, (__int64)"sort-timers", 11);
  *(_QWORD *)(a1 + 504) = "In the report, sort the timers in each group in wall clock time order";
  v24 = *(_BYTE *)(a1 + 476);
  *(_WORD *)(a1 + 616) = 257;
  *(_BYTE *)(a1 + 600) = 1;
  *(_QWORD *)(a1 + 512) = 69;
  *(_BYTE *)(a1 + 476) = v24 & 0x9F | 0x20;
  sub_C53130(a1 + 464);
  *(_QWORD *)(a1 + 696) = 0;
  *(_OWORD *)(a1 + 680) = 0;
  *(_DWORD *)(a1 + 680) = 1;
  *(_DWORD *)(a1 + 704) = 0;
  *(_OWORD *)(a1 + 664) = 0;
  sub_C9E810((__int64 *)(a1 + 712), "misc", 4, "Miscellaneous Ungrouped Timers", 30, (pthread_mutex_t *)(a1 + 664));
  result = sub_F04DE0(a1 + 824);
  *(_BYTE *)(a1 + 864) = 0;
  *(_DWORD *)(a1 + 832) = 0;
  return result;
}
