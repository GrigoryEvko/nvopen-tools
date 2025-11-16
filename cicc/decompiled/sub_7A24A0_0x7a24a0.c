// Function: sub_7A24A0
// Address: 0x7a24a0
//
__int64 __fastcall sub_7A24A0(__int64 a1, unsigned int a2, __m128i *a3)
{
  __int64 v3; // rbx
  char v5; // al
  __int64 result; // rax
  __m128i *v7; // rax
  unsigned __int64 v8; // rdx
  const __m128i *v9; // rax
  __int64 v10; // rdx
  const __m128i *v11; // roff
  __m128i v12; // xmm4
  const __m128i *v13[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = *(_QWORD *)(a1 + 176);
  if ( v3 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v5 = *(_BYTE *)(v3 + 173);
        if ( v5 == 9 )
          break;
        if ( v5 == 10 && (*(_BYTE *)(v3 + 192) & 1) != 0 )
        {
          result = sub_7A24A0(v3, a2, a3);
          if ( !(_DWORD)result )
            return result;
        }
        v3 = *(_QWORD *)(v3 + 120);
        if ( !v3 )
          goto LABEL_11;
      }
      v7 = (__m128i *)sub_724DC0();
      v8 = *(_QWORD *)(v3 + 128);
      v13[0] = v7;
      if ( !(unsigned int)sub_7A1C60(*(_QWORD *)(v3 + 176), (FILE *)(v3 + 64), v8, 1, v7, a3, a2) )
        break;
      v9 = v13[0];
      v10 = *(_QWORD *)(v3 + 120);
      v11 = v13[0];
      *(__m128i *)v3 = _mm_loadu_si128(v13[0]);
      *(__m128i *)(v3 + 16) = _mm_loadu_si128(v11 + 1);
      *(__m128i *)(v3 + 32) = _mm_loadu_si128(v9 + 2);
      *(__m128i *)(v3 + 48) = _mm_loadu_si128(v9 + 3);
      *(__m128i *)(v3 + 64) = _mm_loadu_si128(v9 + 4);
      *(__m128i *)(v3 + 80) = _mm_loadu_si128(v9 + 5);
      *(__m128i *)(v3 + 96) = _mm_loadu_si128(v9 + 6);
      *(__m128i *)(v3 + 112) = _mm_loadu_si128(v9 + 7);
      *(__m128i *)(v3 + 128) = _mm_loadu_si128(v9 + 8);
      *(__m128i *)(v3 + 144) = _mm_loadu_si128(v9 + 9);
      *(__m128i *)(v3 + 160) = _mm_loadu_si128(v9 + 10);
      *(__m128i *)(v3 + 176) = _mm_loadu_si128(v9 + 11);
      v12 = _mm_loadu_si128(v9 + 12);
      *(_QWORD *)(v3 + 120) = v10;
      *(__m128i *)(v3 + 192) = v12;
      sub_724E30((__int64)v13);
      v3 = *(_QWORD *)(v3 + 120);
      if ( !v3 )
        goto LABEL_11;
    }
    sub_724E30((__int64)v13);
    return 0;
  }
  else
  {
LABEL_11:
    *(_BYTE *)(a1 + 192) &= ~1u;
    return 1;
  }
}
