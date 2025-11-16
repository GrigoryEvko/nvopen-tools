// Function: sub_C1FF60
// Address: 0xc1ff60
//
__int64 __fastcall sub_C1FF60(__int64 a1, _DWORD *a2)
{
  __int64 v4; // rax
  __m128i *v5; // rdx
  __int64 v6; // rdi
  __m128i si128; // xmm0
  __int64 v8; // rdi
  _BYTE *v9; // rax

  if ( (unsigned __int64)(*(_QWORD *)(a1 + 24) + 4LL) > *(_QWORD *)(a1 + 8) )
  {
    *a2 = 0;
    v4 = sub_CB72A0(a1, a2);
    v5 = *(__m128i **)(v4 + 32);
    v6 = v4;
    if ( *(_QWORD *)(v4 + 24) - (_QWORD)v5 <= 0x20u )
    {
      v6 = sub_CB6200(v4, "unexpected end of memory buffer: ", 33);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F64EF0);
      v5[2].m128i_i8[0] = 32;
      *v5 = si128;
      v5[1] = _mm_load_si128((const __m128i *)&xmmword_3F64F00);
      *(_QWORD *)(v4 + 32) += 33LL;
    }
    v8 = sub_CB59D0(v6, *(_QWORD *)(a1 + 24));
    v9 = *(_BYTE **)(v8 + 32);
    if ( *(_BYTE **)(v8 + 24) == v9 )
    {
      sub_CB6200(v8, "\n", 1);
      return 0;
    }
    else
    {
      *v9 = 10;
      ++*(_QWORD *)(v8 + 32);
      return 0;
    }
  }
  else
  {
    *a2 = sub_C5F610(a1, a1 + 24, a1 + 32);
    return 1;
  }
}
