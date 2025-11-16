// Function: sub_770090
// Address: 0x770090
//
__int64 __fastcall sub_770090(__int64 a1, __int64 a2, __int64 a3, char **a4, __m128i *a5)
{
  char v5; // al
  char v7; // al

  v5 = **a4;
  if ( v5 == 28 || v5 == 23 && ((v7 = *(_BYTE *)(*((_QWORD *)*a4 + 1) + 28LL), (unsigned __int8)(v7 - 3) <= 1u) || !v7) )
  {
    *a5 = _mm_load_si128((const __m128i *)&xmmword_4F08280);
    return 1;
  }
  else
  {
    *a5 = _mm_load_si128((const __m128i *)&xmmword_4F08290);
    return 1;
  }
}
