// Function: sub_38BB200
// Address: 0x38bb200
//
__int64 __fastcall sub_38BB200(__int64 a1)
{
  __m128i v1; // xmm0
  int v2; // eax

  *(_WORD *)(a1 + 20) = 0;
  *(_BYTE *)(a1 + 22) = 0;
  *(_QWORD *)a1 = &unk_4A3DFA8;
  *(_QWORD *)(a1 + 8) = 0x400000004LL;
  *(_QWORD *)(a1 + 24) = 0x100000004LL;
  *(_QWORD *)(a1 + 300) = 0x1010100000000LL;
  *(_QWORD *)(a1 + 336) = 0x1200000009LL;
  *(_QWORD *)(a1 + 352) = 0x100000000LL;
  *(_DWORD *)(a1 + 16) = 1;
  *(_BYTE *)(a1 + 32) = 0;
  *(_WORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_DWORD *)(a1 + 168) = 0;
  *(_WORD *)(a1 + 172) = 256;
  *(_BYTE *)(a1 + 174) = 0;
  *(_QWORD *)(a1 + 232) = 0;
  *(_QWORD *)(a1 + 240) = 0;
  *(_QWORD *)(a1 + 248) = 0;
  *(_QWORD *)(a1 + 256) = 0;
  *(_QWORD *)(a1 + 264) = 0;
  *(_QWORD *)(a1 + 272) = 0;
  *(_QWORD *)(a1 + 280) = &loc_1000000;
  *(_WORD *)(a1 + 296) = 256;
  *(_BYTE *)(a1 + 298) = 1;
  *(_WORD *)(a1 + 308) = 0;
  *(_QWORD *)(a1 + 320) = 0;
  *(_WORD *)(a1 + 328) = 0;
  *(_BYTE *)(a1 + 330) = 0;
  *(_DWORD *)(a1 + 332) = 9;
  *(_BYTE *)(a1 + 344) = 0;
  *(_DWORD *)(a1 + 348) = 0;
  *(_BYTE *)(a1 + 360) = 1;
  *(_QWORD *)(a1 + 40) = ";";
  *(_QWORD *)(a1 + 48) = "#";
  *(_QWORD *)(a1 + 64) = ":";
  *(_QWORD *)(a1 + 80) = "L";
  *(_QWORD *)(a1 + 112) = byte_3F871B3;
  *(_QWORD *)(a1 + 128) = "APP";
  *(_QWORD *)(a1 + 136) = "NO_APP";
  *(_QWORD *)(a1 + 144) = ".code16";
  *(_QWORD *)(a1 + 152) = ".code32";
  *(_QWORD *)(a1 + 160) = ".code64";
  *(_QWORD *)(a1 + 176) = "\t.zero\t";
  *(_QWORD *)(a1 + 184) = "\t.ascii\t";
  *(_QWORD *)(a1 + 192) = "\t.asciz\t";
  *(_QWORD *)(a1 + 200) = "\t.byte\t";
  *(_QWORD *)(a1 + 208) = "\t.short\t";
  *(_QWORD *)(a1 + 216) = "\t.long\t";
  *(_QWORD *)(a1 + 224) = "\t.quad\t";
  *(_QWORD *)(a1 + 288) = "\t.globl\t";
  *(_QWORD *)(a1 + 88) = 1;
  v1 = _mm_loadu_si128((const __m128i *)(a1 + 80));
  *(_QWORD *)(a1 + 312) = "\t.weak\t";
  v2 = dword_5052740;
  *(_QWORD *)(a1 + 368) = 0;
  *(_QWORD *)(a1 + 376) = 0;
  *(_QWORD *)(a1 + 384) = 0;
  *(_DWORD *)(a1 + 396) = 0;
  *(_WORD *)(a1 + 400) = 257;
  *(_BYTE *)(a1 + 402) = 0;
  *(_QWORD *)(a1 + 56) = 1;
  *(__m128i *)(a1 + 96) = v1;
  if ( v2 )
    *(_BYTE *)(a1 + 360) = v2 == 1;
  *(_WORD *)(a1 + 392) = 256;
  return 256;
}
