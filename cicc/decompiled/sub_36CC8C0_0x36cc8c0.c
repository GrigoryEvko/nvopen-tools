// Function: sub_36CC8C0
// Address: 0x36cc8c0
//
const char *__fastcall sub_36CC8C0(__int64 a1, __int64 a2)
{
  bool v2; // zf
  __m128i v3; // xmm0

  sub_106EBF0(a1);
  v2 = *(_DWORD *)(a2 + 32) == 43;
  *(_QWORD *)a1 = &unk_4A3AEA0;
  if ( v2 )
    *(_QWORD *)(a1 + 8) = 0x800000008LL;
  *(_QWORD *)(a1 + 96) = 4;
  *(_QWORD *)(a1 + 48) = "//";
  *(_QWORD *)(a1 + 136) = " begin inline asm";
  *(_QWORD *)(a1 + 144) = " end inline asm";
  *(_WORD *)(a1 + 288) = 0;
  *(_QWORD *)(a1 + 224) = ".b8 ";
  *(_QWORD *)(a1 + 240) = ".b32 ";
  *(_QWORD *)(a1 + 248) = ".b64 ";
  *(_QWORD *)(a1 + 192) = ".b8";
  *(_QWORD *)(a1 + 88) = "$L__";
  v3 = _mm_loadu_si128((const __m128i *)(a1 + 88));
  *(_QWORD *)(a1 + 296) = "\t// .weak\t";
  *(_QWORD *)(a1 + 56) = 2;
  *(_BYTE *)(a1 + 290) = 0;
  *(_DWORD *)(a1 + 316) = 0;
  *(_QWORD *)(a1 + 324) = 0;
  *(_BYTE *)(a1 + 332) = 1;
  *(_QWORD *)(a1 + 232) = 0;
  *(_QWORD *)(a1 + 200) = 0;
  *(_QWORD *)(a1 + 208) = 0;
  *(_BYTE *)(a1 + 184) = 0;
  *(_BYTE *)(a1 + 354) = 0;
  *(_BYTE *)(a1 + 256) = 0;
  *(_BYTE *)(a1 + 291) = 1;
  *(_QWORD *)(a1 + 272) = "\t// .globl\t";
  *(_BYTE *)(a1 + 392) = 0;
  *(_BYTE *)(a1 + 353) = 0;
  *(_BYTE *)(a1 + 350) = 0;
  *(__m128i *)(a1 + 104) = v3;
  return "\t// .globl\t";
}
