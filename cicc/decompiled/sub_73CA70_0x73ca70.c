// Function: sub_73CA70
// Address: 0x73ca70
//
__m128i *__fastcall sub_73CA70(const __m128i *a1, __int64 a2)
{
  int v2; // esi

  if ( (*(_BYTE *)(a2 + 140) & 0xFB) == 8 )
    v2 = sub_8D4C10(a2, dword_4F077C4 != 2);
  else
    v2 = 0;
  return sub_73C570(a1, v2);
}
