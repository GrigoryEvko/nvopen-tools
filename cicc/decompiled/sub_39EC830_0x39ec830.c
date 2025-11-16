// Function: sub_39EC830
// Address: 0x39ec830
//
_BYTE *__fastcall sub_39EC830(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __m128i *v4; // rdx
  __m128i si128; // xmm0
  unsigned __int64 v6; // r13
  _BYTE *result; // rax
  __int64 v8; // rdi
  __int64 v9; // r14
  char *v10; // rsi
  size_t v11; // rdx
  void *v12; // rdi

  sub_38DE980(a1, a2);
  v3 = *(_QWORD *)(a1 + 272);
  v4 = *(__m128i **)(v3 + 24);
  if ( *(_QWORD *)(v3 + 16) - (_QWORD)v4 <= 0x14u )
  {
    v3 = sub_16E7EE0(v3, "\t.cfi_def_cfa_offset ", 0x15u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F7FA20);
    v4[1].m128i_i32[0] = 1952805734;
    v4[1].m128i_i8[4] = 32;
    *v4 = si128;
    *(_QWORD *)(v3 + 24) += 21LL;
  }
  sub_16E7AB0(v3, a2);
  v6 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
  {
    v9 = *(_QWORD *)(a1 + 272);
    v10 = *(char **)(a1 + 304);
    v11 = *(unsigned int *)(a1 + 312);
    v12 = *(void **)(v9 + 24);
    if ( v6 > *(_QWORD *)(v9 + 16) - (_QWORD)v12 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 272), v10, v11);
    }
    else
    {
      memcpy(v12, v10, v11);
      *(_QWORD *)(v9 + 24) += v6;
    }
  }
  *(_DWORD *)(a1 + 312) = 0;
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
    return sub_39E0440(a1);
  v8 = *(_QWORD *)(a1 + 272);
  result = *(_BYTE **)(v8 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(v8 + 16) )
    return (_BYTE *)sub_16E7DE0(v8, 10);
  *(_QWORD *)(v8 + 24) = result + 1;
  *result = 10;
  return result;
}
