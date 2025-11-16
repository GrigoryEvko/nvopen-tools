// Function: sub_891F00
// Address: 0x891f00
//
void *__fastcall sub_891F00(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  int v4; // eax
  __int64 v5; // rax
  __m128i v7; // xmm0

  v3 = a1 + 248;
  *(_QWORD *)(v3 - 248) = a2;
  *(_QWORD *)(v3 - 240) = 0;
  *(_QWORD *)(a2 + 120) = *(_QWORD *)(a2 + 120) & 0xFFFFBFFFF7FFFFFFLL
                        | ((unsigned __int64)(word_4D04430 & 1) << 27)
                        | 0x400000000000LL;
  *(_QWORD *)(v3 - 232) = 0;
  v4 = unk_4D043FC;
  *(_QWORD *)(v3 - 224) = 0;
  *(_QWORD *)(v3 - 216) = 0;
  *(_DWORD *)(v3 - 164) = v4;
  *(_QWORD *)(v3 - 208) = 0;
  v5 = *(_QWORD *)&dword_4F077C8;
  *(_QWORD *)(v3 - 200) = 0;
  *(_QWORD *)(v3 - 192) = 0;
  *(_QWORD *)(v3 - 108) = v5;
  *(_QWORD *)(v3 - 100) = v5;
  *(_QWORD *)(v3 - 184) = 0;
  *(_QWORD *)(v3 - 176) = 0;
  *(_DWORD *)(v3 - 168) = 0;
  *(_QWORD *)(v3 - 160) = 0;
  *(_QWORD *)(v3 - 152) = 0;
  *(_QWORD *)(v3 - 144) = 0;
  *(_QWORD *)(v3 - 136) = 0;
  *(_QWORD *)(v3 - 128) = 0;
  *(_QWORD *)(v3 - 120) = 0;
  *(_DWORD *)(v3 - 112) = 0;
  *(_QWORD *)(v3 - 92) = 0;
  *(_BYTE *)(v3 - 84) = 0;
  *(_QWORD *)(v3 - 80) = 0;
  *(_DWORD *)(v3 - 72) = 0;
  *(_QWORD *)(v3 - 64) = 0;
  *(_QWORD *)(v3 - 56) = 0;
  *(_QWORD *)(v3 - 48) = -1;
  *(_DWORD *)(v3 - 40) = -1;
  *(_QWORD *)(v3 - 32) = 0;
  *(_QWORD *)(v3 - 24) = 0;
  *(_QWORD *)(v3 - 16) = 0;
  *(_QWORD *)(v3 - 8) = 0;
  sub_7ADF70(v3, 1);
  *(_DWORD *)(a1 + 280) = 0;
  sub_7ADF70(a1 + 288, 1);
  *(_QWORD *)(a1 + 344) = 0;
  *(_QWORD *)(a1 + 424) = 0;
  *(_DWORD *)(a1 + 320) = 0;
  *(_QWORD *)(a1 + 328) = 0;
  *(_QWORD *)(a1 + 336) = 0;
  memset(
    (void *)((a1 + 352) & 0xFFFFFFFFFFFFFFF8LL),
    0,
    8LL * (((unsigned int)a1 - (((_DWORD)a1 + 352) & 0xFFFFFFF8) + 432) >> 3));
  *(_QWORD *)(a1 + 432) = 0;
  *(_QWORD *)(a1 + 440) = 0;
  *(_QWORD *)(a1 + 448) = 0;
  v7 = _mm_loadu_si128((const __m128i *)&unk_4F07370);
  *(_QWORD *)(a1 + 456) = 0;
  *(_QWORD *)(a1 + 464) = 0;
  *(_QWORD *)(a1 + 488) = 0;
  *(_QWORD *)(a1 + 496) = 0;
  *(__m128i *)(a1 + 472) = v7;
  return &unk_4F07370;
}
