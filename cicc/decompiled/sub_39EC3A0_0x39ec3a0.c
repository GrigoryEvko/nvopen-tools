// Function: sub_39EC3A0
// Address: 0x39ec3a0
//
_BYTE *__fastcall sub_39EC3A0(__int64 a1, _BYTE *a2, unsigned int a3)
{
  __int64 v5; // rdi
  __m128i *v6; // rdx
  __m128i si128; // xmm0
  __int64 v8; // rax
  _WORD *v9; // rdx
  unsigned __int64 v10; // r13
  _BYTE *result; // rax
  __int64 v12; // rdi
  __int64 v13; // r14
  char *v14; // rsi
  size_t v15; // rdx
  void *v16; // rdi

  sub_38DD190(a1, (__int64)a2, a3);
  v5 = *(_QWORD *)(a1 + 272);
  v6 = *(__m128i **)(v5 + 24);
  if ( *(_QWORD *)(v5 + 16) - (_QWORD)v6 <= 0x11u )
  {
    v5 = sub_16E7EE0(v5, "\t.cfi_personality ", 0x12u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F7F9F0);
    v6[1].m128i_i16[0] = 8313;
    *v6 = si128;
    *(_QWORD *)(v5 + 24) += 18LL;
  }
  v8 = sub_16E7A90(v5, a3);
  v9 = *(_WORD **)(v8 + 24);
  if ( *(_QWORD *)(v8 + 16) - (_QWORD)v9 <= 1u )
  {
    sub_16E7EE0(v8, ", ", 2u);
  }
  else
  {
    *v9 = 8236;
    *(_QWORD *)(v8 + 24) += 2LL;
  }
  sub_38E2490(a2, *(_QWORD *)(a1 + 272), *(_BYTE **)(a1 + 280));
  v10 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
  {
    v13 = *(_QWORD *)(a1 + 272);
    v14 = *(char **)(a1 + 304);
    v15 = *(unsigned int *)(a1 + 312);
    v16 = *(void **)(v13 + 24);
    if ( v10 > *(_QWORD *)(v13 + 16) - (_QWORD)v16 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 272), v14, v15);
    }
    else
    {
      memcpy(v16, v14, v15);
      *(_QWORD *)(v13 + 24) += v10;
    }
  }
  *(_DWORD *)(a1 + 312) = 0;
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
    return sub_39E0440(a1);
  v12 = *(_QWORD *)(a1 + 272);
  result = *(_BYTE **)(v12 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(v12 + 16) )
    return (_BYTE *)sub_16E7DE0(v12, 10);
  *(_QWORD *)(v12 + 24) = result + 1;
  *result = 10;
  return result;
}
