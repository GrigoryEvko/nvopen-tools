// Function: sub_16FFE70
// Address: 0x16ffe70
//
void __fastcall sub_16FFE70(
        __int64 a1,
        __int64 a2,
        __int8 *a3,
        size_t a4,
        __int64 a5,
        __int64 a6,
        _BYTE *a7,
        __int64 a8,
        _BYTE *a9,
        __int64 a10)
{
  __int8 *v13; // rsi
  __int64 v17; // rcx
  __int64 v18; // rax
  __int64 v19; // rdi

  v13 = a3;
  *(_QWORD *)(a1 + 64) = a1 + 80;
  *(_QWORD *)(a1 + 8) = a2;
  *(_QWORD *)(a1 + 72) = 0x1000000000LL;
  *(_QWORD *)(a1 + 208) = a1 + 224;
  *(_QWORD *)(a1 + 240) = a1 + 256;
  v17 = a1 + 440;
  *(_QWORD *)a1 = &unk_49EFE98;
  v18 = a1 + 56;
  v19 = a1 + 16;
  *(_QWORD *)(v19 + 24) = v18;
  *(_QWORD *)(v19 + 32) = 0x800000000LL;
  *(_QWORD *)(v19 + 232) = 0x800000000LL;
  *(_QWORD *)(v19 + 408) = v17;
  *(_QWORD *)(v19 + 416) = 0x800000000LL;
  *(_QWORD *)(v19 + 200) = 0;
  *(_BYTE *)(v19 + 208) = 0;
  *(_QWORD *)(v19 + 400) = 0;
  sub_15A9300(v19, v13, a4);
  *(_QWORD *)(a1 + 472) = a1 + 488;
  sub_16FF8E0((__int64 *)(a1 + 472), *(_BYTE **)a5, *(_QWORD *)a5 + *(_QWORD *)(a5 + 8));
  *(_QWORD *)(a1 + 504) = *(_QWORD *)(a5 + 32);
  *(_QWORD *)(a1 + 512) = *(_QWORD *)(a5 + 40);
  *(_QWORD *)(a1 + 520) = *(_QWORD *)(a5 + 48);
  *(_QWORD *)(a1 + 528) = a1 + 544;
  if ( a7 )
  {
    sub_16FF830((__int64 *)(a1 + 528), a7, (__int64)&a7[a8]);
  }
  else
  {
    *(_QWORD *)(a1 + 536) = 0;
    *(_BYTE *)(a1 + 544) = 0;
  }
  *(_QWORD *)(a1 + 560) = a1 + 576;
  if ( a9 )
  {
    sub_16FF830((__int64 *)(a1 + 560), a9, (__int64)&a9[a10]);
  }
  else
  {
    *(_QWORD *)(a1 + 568) = 0;
    *(_BYTE *)(a1 + 576) = 0;
  }
  *(_BYTE *)(a1 + 640) &= ~1u;
  *(_QWORD *)(a1 + 592) = 0;
  *(_DWORD *)(a1 + 600) = 2;
  *(_QWORD *)(a1 + 608) = 0;
  *(_QWORD *)(a1 + 616) = 0;
  *(_QWORD *)(a1 + 624) = 0;
  *(_QWORD *)(a1 + 632) = 0;
  sub_16FFCD0((_QWORD *)(a1 + 648), a6);
  sub_16FFCD0((_QWORD *)(a1 + 792), a6);
}
