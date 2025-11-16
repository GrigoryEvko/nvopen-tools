// Function: sub_21BC460
// Address: 0x21bc460
//
const char *__fastcall sub_21BC460(__m128i *a1, __int64 a2)
{
  bool v2; // zf
  __m128i v3; // xmm0

  sub_38BB200();
  v2 = *(_DWORD *)(a2 + 32) == 35;
  a1->m128i_i64[0] = (__int64)&unk_4A032C8;
  if ( v2 )
    a1->m128i_i64[1] = 0x800000008LL;
  a1[5].m128i_i64[1] = 4;
  a1[3].m128i_i64[0] = (__int64)"//";
  a1[8].m128i_i64[0] = (__int64)" begin inline asm";
  a1[8].m128i_i64[1] = (__int64)" end inline asm";
  a1[19].m128i_i16[0] = 0;
  a1[12].m128i_i64[1] = (__int64)".b8 ";
  a1[13].m128i_i64[1] = (__int64)".b32 ";
  a1[14].m128i_i64[0] = (__int64)".b64 ";
  a1[11].m128i_i64[0] = (__int64)".b8";
  a1[5].m128i_i64[0] = (__int64)"$L__";
  v3 = _mm_loadu_si128(a1 + 5);
  a1[19].m128i_i64[1] = (__int64)"\t// .weak\t";
  a1[3].m128i_i64[1] = 2;
  a1[19].m128i_i8[2] = 0;
  a1[20].m128i_i32[3] = 0;
  a1[21].m128i_i64[0] = 0;
  a1[21].m128i_i8[8] = 1;
  a1[13].m128i_i64[0] = 0;
  a1[11].m128i_i64[1] = 0;
  a1[12].m128i_i64[0] = 0;
  a1[10].m128i_i8[13] = 0;
  a1[22].m128i_i8[8] = 0;
  a1[19].m128i_i8[3] = 1;
  a1[18].m128i_i64[0] = (__int64)"\t// .globl\t";
  a1[6] = v3;
  return "\t// .globl\t";
}
