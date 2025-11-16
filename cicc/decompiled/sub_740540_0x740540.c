// Function: sub_740540
// Address: 0x740540
//
__m128i *__fastcall sub_740540(const __m128i *a1, int a2, int a3)
{
  __m128i *v3; // r12
  const __m128i *v5; // rdi
  unsigned int v6; // edx

  if ( !a3 && dword_4F07270[0] == unk_4F073B8 && (unsigned int)sub_72AA80((__int64)a1) )
  {
    v6 = 0;
    if ( !a2 )
      v6 = 1024;
    v3 = sub_740190((__int64)a1, 0, v6);
  }
  else
  {
    v3 = (__m128i *)sub_724D50(a1[10].m128i_i8[13]);
    sub_72A510(a1, v3);
    if ( a2 )
      v3[-1].m128i_i8[8] = a1[-1].m128i_i8[8] & 8 | v3[-1].m128i_i8[8] & 0xF7;
  }
  sub_72A1A0((__int64)v3);
  if ( a1[10].m128i_i8[13] == 6 )
  {
    v5 = (const __m128i *)a1[12].m128i_i64[1];
    if ( v5 )
      v3[12].m128i_i64[1] = sub_72A820(v5);
  }
  sub_73B910((__int64)v3);
  return v3;
}
