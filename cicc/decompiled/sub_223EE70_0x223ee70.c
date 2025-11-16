// Function: sub_223EE70
// Address: 0x223ee70
//
void __fastcall __noreturn sub_223EE70(void *src, __int64 a2)
{
  void *v2; // rsp
  _OWORD v3[6]; // [rsp+0h] [rbp-80h] BYREF
  char v4[32]; // [rsp+60h] [rbp-20h] BYREF

  v2 = alloca(a2 - (_QWORD)src + 113);
  strcpy(v4, "/):\n    ");
  v3[0] = _mm_load_si128((const __m128i *)&xmmword_4362810);
  v3[1] = _mm_load_si128((const __m128i *)&xmmword_4362820);
  v3[2] = _mm_load_si128((const __m128i *)&xmmword_4362830);
  v3[3] = _mm_load_si128((const __m128i *)&xmmword_4362840);
  v3[4] = _mm_load_si128((const __m128i *)&xmmword_4362850);
  v3[5] = _mm_load_si128((const __m128i *)&xmmword_4362860);
  memcpy(&v4[8], src, a2 - (_QWORD)src);
  v4[a2 - (_QWORD)src + 8] = 0;
  sub_426248((__int64)v3);
}
