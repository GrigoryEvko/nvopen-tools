// Function: sub_770010
// Address: 0x770010
//
__int64 __fastcall sub_770010(__int64 a1, __m128i *a2)
{
  char v2; // al
  const __m128i *v3; // rdx
  __int64 result; // rax

  v2 = *(_BYTE *)(a1 + 8);
  if ( (v2 & 1) == 0 )
  {
    if ( (v2 & 0x20) != 0 )
    {
      LOBYTE(result) = *(_QWORD *)(a1 + 16) == 0;
      if ( !*(_QWORD *)(a1 + 16) )
        goto LABEL_6;
    }
    else
    {
      LOBYTE(result) = *(_QWORD *)a1 == 0;
      if ( !*(_QWORD *)a1 )
      {
LABEL_6:
        if ( a2 )
        {
          *a2 = _mm_load_si128((const __m128i *)&xmmword_4F08290);
          return 1;
        }
      }
    }
    return (unsigned __int8)result;
  }
  v3 = *(const __m128i **)(a1 + 16);
  result = 0;
  if ( v3[10].m128i_i8[13] == 1 )
  {
    result = 1;
    if ( a2 )
      *a2 = _mm_loadu_si128(v3 + 11);
  }
  return result;
}
