// Function: sub_39EA810
// Address: 0x39ea810
//
_BYTE *__fastcall sub_39EA810(__int64 a1, unsigned int a2, unsigned __int64 a3)
{
  __int64 v4; // rdi
  __m128i *v5; // rdx
  __m128i si128; // xmm0
  unsigned __int64 v7; // r13
  _BYTE *result; // rax
  __int64 v9; // rdi
  __int64 v10; // r14
  char *v11; // rsi
  size_t v12; // rdx
  void *v13; // rdi

  sub_38E0FE0((_QWORD *)a1, a2, a3);
  v4 = *(_QWORD *)(a1 + 272);
  v5 = *(__m128i **)(v4 + 24);
  if ( *(_QWORD *)(v4 + 16) - (_QWORD)v5 <= 0x10u )
  {
    v4 = sub_16E7EE0(v4, "\t.seh_stackalloc ", 0x11u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F7F900);
    v5[1].m128i_i8[0] = 32;
    *v5 = si128;
    *(_QWORD *)(v4 + 24) += 17LL;
  }
  sub_16E7A90(v4, a2);
  v7 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
  {
    v10 = *(_QWORD *)(a1 + 272);
    v11 = *(char **)(a1 + 304);
    v12 = *(unsigned int *)(a1 + 312);
    v13 = *(void **)(v10 + 24);
    if ( v7 > *(_QWORD *)(v10 + 16) - (_QWORD)v13 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 272), v11, v12);
    }
    else
    {
      memcpy(v13, v11, v12);
      *(_QWORD *)(v10 + 24) += v7;
    }
  }
  *(_DWORD *)(a1 + 312) = 0;
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
    return sub_39E0440(a1);
  v9 = *(_QWORD *)(a1 + 272);
  result = *(_BYTE **)(v9 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(v9 + 16) )
    return (_BYTE *)sub_16E7DE0(v9, 10);
  *(_QWORD *)(v9 + 24) = result + 1;
  *result = 10;
  return result;
}
