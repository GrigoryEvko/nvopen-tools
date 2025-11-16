// Function: sub_73C7D0
// Address: 0x73c7d0
//
__m128i *__fastcall sub_73C7D0(__m128i **a1)
{
  __m128i *v1; // rbx
  __m128i *result; // rax
  int i; // r13d
  __int64 v4; // rax
  const __m128i *v5; // rax

  v1 = *a1;
  result = (__m128i *)sub_8D2B80(*a1);
  if ( (_DWORD)result )
  {
    result = (__m128i *)v1[8].m128i_u8[12];
    if ( (_BYTE)result == 12 )
    {
      result = v1;
      do
        result = (__m128i *)result[10].m128i_i64[0];
      while ( result[8].m128i_i8[12] == 12 );
      if ( (result[11].m128i_i8[0] & 1) == 0 )
        return result;
    }
    else
    {
      if ( (v1[11].m128i_i8[0] & 1) == 0 )
        return result;
      if ( ((unsigned __int8)result & 0xFB) != 8 )
      {
        i = 0;
LABEL_9:
        v4 = sub_8D4620(v1);
        v5 = (const __m128i *)sub_72B5A0(v1[10].m128i_i64[0], v4, 0);
        result = sub_73C570(v5, i);
        *a1 = result;
        return result;
      }
    }
    for ( i = sub_8D4C10(v1, dword_4F077C4 != 2); v1[8].m128i_i8[12] == 12; v1 = (__m128i *)v1[10].m128i_i64[0] )
      ;
    goto LABEL_9;
  }
  return result;
}
