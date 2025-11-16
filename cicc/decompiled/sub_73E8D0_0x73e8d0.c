// Function: sub_73E8D0
// Address: 0x73e8d0
//
__m128i *__fastcall sub_73E8D0(__m128i **a1, const __m128i **a2)
{
  int v2; // r13d
  const __m128i *v3; // r15
  const __m128i *v4; // r14
  __m128i *result; // rax
  __m128i *v6; // r15
  int v7; // ecx
  const __m128i *v8; // r14
  int v9; // r13d
  int v10; // [rsp+Ch] [rbp-34h]

  v2 = 0;
  v3 = *a1;
  v4 = *a2;
  if ( ((*a1)[8].m128i_i8[12] & 0xFB) == 8 )
  {
    v2 = sub_8D4C10(*a1, dword_4F077C4 != 2);
    result = (__m128i *)(v4[8].m128i_i8[12] & 0xFB);
    if ( (v4[8].m128i_i8[12] & 0xFB) != 8 )
      return result;
  }
  else
  {
    result = (__m128i *)(v4[8].m128i_i8[12] & 0xFB);
    if ( (v4[8].m128i_i8[12] & 0xFB) != 8 )
      return result;
  }
  result = (__m128i *)sub_8D4C10(v4, dword_4F077C4 != 2);
  if ( v2 )
  {
    v10 = (int)result;
    if ( (_DWORD)result )
    {
      v6 = sub_73D4C0(v3, dword_4F077C4 == 2);
      result = sub_73D4C0(v4, dword_4F077C4 == 2);
      v7 = v10;
      v8 = result;
      if ( v2 != v10 )
      {
        if ( (v2 & ~v10) != 0 )
        {
          result = sub_73C570(v6, v2 & (unsigned int)~v10);
          v7 = v10;
          v6 = result;
        }
        v9 = v7 & ~v2;
        if ( v9 )
        {
          result = sub_73C570(v8, v9);
          v8 = result;
        }
      }
      *a1 = v6;
      *a2 = v8;
    }
  }
  return result;
}
