// Function: sub_106EBF0
// Address: 0x106ebf0
//
__int64 __fastcall sub_106EBF0(__int64 a1)
{
  __m128i v1; // xmm0

  *(_DWORD *)(a1 + 16) = 1;
  *(_WORD *)(a1 + 20) = 0;
  *(_QWORD *)a1 = &unk_49E6008;
  *(_QWORD *)(a1 + 8) = 0x400000004LL;
  *(_QWORD *)(a1 + 24) = 0x100000004LL;
  *(_QWORD *)(a1 + 284) = 0x1010100000000LL;
  *(_QWORD *)(a1 + 316) = 0xD0000000CLL;
  *(_QWORD *)(a1 + 324) = 0x160000000CLL;
  *(_QWORD *)(a1 + 344) = 0x1000100000000LL;
  *(_BYTE *)(a1 + 22) = 0;
  *(_BYTE *)(a1 + 32) = 0;
  *(_BYTE *)(a1 + 64) = 1;
  *(_WORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_DWORD *)(a1 + 184) = 65537;
  *(_DWORD *)(a1 + 216) = 0;
  *(_DWORD *)(a1 + 256) = 1;
  *(_BYTE *)(a1 + 260) = 1;
  *(_DWORD *)(a1 + 264) = 0;
  *(_WORD *)(a1 + 280) = 256;
  *(_BYTE *)(a1 + 292) = 0;
  *(_QWORD *)(a1 + 304) = 0;
  *(_WORD *)(a1 + 312) = 0;
  *(_BYTE *)(a1 + 332) = 0;
  *(_DWORD *)(a1 + 336) = 0;
  *(_BYTE *)(a1 + 340) = 0;
  *(_WORD *)(a1 + 352) = 256;
  *(_BYTE *)(a1 + 354) = 1;
  *(_QWORD *)(a1 + 360) = 0;
  *(_QWORD *)(a1 + 368) = 0;
  *(_QWORD *)(a1 + 376) = 0;
  *(_QWORD *)(a1 + 384) = 0x1A00000002LL;
  *(_QWORD *)(a1 + 456) = 0x1000000000LL;
  *(_QWORD *)(a1 + 40) = ";";
  *(_QWORD *)(a1 + 48) = "#";
  *(_QWORD *)(a1 + 72) = ":";
  *(_QWORD *)(a1 + 88) = "L";
  *(_QWORD *)(a1 + 120) = byte_3F871B3;
  *(_QWORD *)(a1 + 136) = "APP";
  *(_QWORD *)(a1 + 144) = "NO_APP";
  *(_QWORD *)(a1 + 152) = ".code16";
  *(_QWORD *)(a1 + 160) = ".code32";
  *(_QWORD *)(a1 + 168) = ".code64";
  *(_QWORD *)(a1 + 192) = "\t.zero\t";
  *(_QWORD *)(a1 + 200) = "\t.ascii\t";
  *(_QWORD *)(a1 + 208) = "\t.asciz\t";
  *(_QWORD *)(a1 + 224) = "\t.byte\t";
  *(_QWORD *)(a1 + 232) = "\t.short\t";
  *(_QWORD *)(a1 + 240) = "\t.long\t";
  *(_QWORD *)(a1 + 96) = 1;
  v1 = _mm_loadu_si128((const __m128i *)(a1 + 88));
  *(_QWORD *)(a1 + 248) = "\t.quad\t";
  *(_QWORD *)(a1 + 272) = "\t.globl\t";
  *(_DWORD *)(a1 + 396) = 40;
  *(_WORD *)(a1 + 400) = 1;
  *(_QWORD *)(a1 + 408) = 0;
  *(_QWORD *)(a1 + 416) = 0;
  *(_QWORD *)(a1 + 424) = 0;
  *(_DWORD *)(a1 + 432) = 0;
  *(_QWORD *)(a1 + 440) = 0;
  *(_QWORD *)(a1 + 448) = 0;
  *(_QWORD *)(a1 + 56) = 1;
  *(__m128i *)(a1 + 104) = v1;
  *(_QWORD *)(a1 + 296) = "\t.weak\t";
  if ( (_DWORD)qword_4F8FF48 )
    *(_BYTE *)(a1 + 354) = (_DWORD)qword_4F8FF48 == 1;
  if ( LODWORD(qword_4F8FE28[8]) )
    *(_BYTE *)(a1 + 186) = LODWORD(qword_4F8FE28[8]) == 1;
  *(_BYTE *)(a1 + 394) = 1;
  *(_WORD *)(a1 + 392) = 1;
  return 1;
}
