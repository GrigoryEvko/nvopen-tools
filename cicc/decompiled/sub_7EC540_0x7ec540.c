// Function: sub_7EC540
// Address: 0x7ec540
//
void __fastcall sub_7EC540(const __m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  const __m128i *v6; // rax

  if ( a1[1].m128i_i8[8] == 1 )
    sub_7E88E0((__int64)a1, a2, a3, a4, a5, a6);
  if ( (a1[1].m128i_i8[9] & 1) == 0
    && (unsigned int)sub_7E1E50(a1->m128i_i64[0])
    && ((a1[1].m128i_i8[8] - 2) & 0xFD) != 0
    && (a1[1].m128i_i8[9] & 4) == 0
    && !unk_4F189C4 )
  {
    v6 = (const __m128i *)sub_7EC460(a1->m128i_i64[0], a1);
    sub_730620((__int64)a1, v6);
  }
}
