// Function: sub_C20040
// Address: 0xc20040
//
__int64 __fastcall sub_C20040(__int64 a1, unsigned __int64 *a2)
{
  __int64 v3; // r13
  __int64 result; // rax
  __int64 v5; // rax
  __m128i *v6; // rdx
  __int64 v7; // rdi
  __m128i si128; // xmm0
  __int64 v9; // rdi
  _BYTE *v10; // rax
  _DWORD v11[9]; // [rsp+Ch] [rbp-24h] BYREF

  if ( (unsigned __int64)(*(_QWORD *)(a1 + 24) + 4LL) > *(_QWORD *)(a1 + 8) )
  {
    v5 = sub_CB72A0(a1, a2);
    v6 = *(__m128i **)(v5 + 32);
    v7 = v5;
    if ( *(_QWORD *)(v5 + 24) - (_QWORD)v6 <= 0x20u )
    {
      v7 = sub_CB6200(v5, "unexpected end of memory buffer: ", 33);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F64EF0);
      v6[2].m128i_i8[0] = 32;
      *v6 = si128;
      v6[1] = _mm_load_si128((const __m128i *)&xmmword_3F64F00);
      *(_QWORD *)(v5 + 32) += 33LL;
    }
    v9 = sub_CB59D0(v7, *(_QWORD *)(a1 + 24));
    v10 = *(_BYTE **)(v9 + 32);
    if ( *(_BYTE **)(v9 + 24) == v10 )
    {
      sub_CB6200(v9, "\n", 1);
      return 0;
    }
    else
    {
      *v10 = 10;
      ++*(_QWORD *)(v9 + 32);
      return 0;
    }
  }
  else
  {
    v3 = (unsigned int)sub_C5F610(a1, a1 + 24, a1 + 32);
    result = sub_C1FF60(a1, v11);
    if ( (_BYTE)result )
      *a2 = ((unsigned __int64)v11[0] << 32) | v3;
  }
  return result;
}
