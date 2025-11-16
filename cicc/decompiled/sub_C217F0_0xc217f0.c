// Function: sub_C217F0
// Address: 0xc217f0
//
__int64 __fastcall sub_C217F0(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __m128i *v5; // rdx
  __int64 v6; // rdi
  __m128i si128; // xmm0
  __int64 v8; // rdi
  _BYTE *v9; // rax

  if ( (unsigned __int64)(*(_QWORD *)(a1 + 232) + 4LL) > *(_QWORD *)(a1 + 216) )
  {
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
    v8 = sub_CB59D0(v6, *(_QWORD *)(a1 + 232));
    v9 = *(_BYTE **)(v8 + 32);
    if ( *(_BYTE **)(v8 + 24) == v9 )
    {
      sub_CB6200(v8, "\n", 1);
    }
    else
    {
      *v9 = 10;
      ++*(_QWORD *)(v8 + 32);
    }
    sub_C1AFD0();
    return 4;
  }
  else
  {
    sub_C5F610(a1 + 208, a1 + 232, a1 + 240);
    sub_C1AFD0();
    return 0;
  }
}
