// Function: sub_39E1330
// Address: 0x39e1330
//
_BYTE *__fastcall sub_39E1330(__int64 a1)
{
  __int64 v2; // rdi
  __m128i *v3; // rdx
  __m128i si128; // xmm0
  unsigned __int64 v5; // r13
  _BYTE *result; // rax
  __int64 v7; // rdi
  __int64 v8; // r14
  char *v9; // rsi
  size_t v10; // rdx
  void *v11; // rdi

  v2 = *(_QWORD *)(a1 + 272);
  v3 = *(__m128i **)(v2 + 24);
  if ( *(_QWORD *)(v2 + 16) - (_QWORD)v3 <= 0x11u )
  {
    sub_16E7EE0(v2, "\t.cv_filechecksums", 0x12u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F7F870);
    v3[1].m128i_i16[0] = 29549;
    *v3 = si128;
    *(_QWORD *)(v2 + 24) += 18LL;
  }
  v5 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
  {
    v8 = *(_QWORD *)(a1 + 272);
    v9 = *(char **)(a1 + 304);
    v10 = *(unsigned int *)(a1 + 312);
    v11 = *(void **)(v8 + 24);
    if ( v5 > *(_QWORD *)(v8 + 16) - (_QWORD)v11 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 272), v9, v10);
    }
    else
    {
      memcpy(v11, v9, v10);
      *(_QWORD *)(v8 + 24) += v5;
    }
  }
  *(_DWORD *)(a1 + 312) = 0;
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
    return sub_39E0440(a1);
  v7 = *(_QWORD *)(a1 + 272);
  result = *(_BYTE **)(v7 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(v7 + 16) )
    return (_BYTE *)sub_16E7DE0(v7, 10);
  *(_QWORD *)(v7 + 24) = result + 1;
  *result = 10;
  return result;
}
