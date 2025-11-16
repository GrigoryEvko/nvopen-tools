// Function: sub_39E63A0
// Address: 0x39e63a0
//
_BYTE *__fastcall sub_39E63A0(__int64 a1)
{
  _BYTE *result; // rax
  __int64 v3; // rdi
  __m128i *v4; // rdx
  __m128i si128; // xmm0
  unsigned __int64 v6; // r13
  __int64 v7; // rdi
  __int64 v8; // r14
  char *v9; // rsi
  size_t v10; // rdx
  void *v11; // rdi

  result = *(_BYTE **)(a1 + 280);
  if ( *((_DWORD *)result + 42) == 1 )
  {
    v3 = *(_QWORD *)(a1 + 272);
    v4 = *(__m128i **)(v3 + 24);
    if ( *(_QWORD *)(v3 + 16) - (_QWORD)v4 <= 0x16u )
    {
      sub_16E7EE0(v3, "\t.intel_syntax noprefix", 0x17u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F7F830);
      v4[1].m128i_i32[0] = 1701998703;
      v4[1].m128i_i16[2] = 26982;
      v4[1].m128i_i8[6] = 120;
      *v4 = si128;
      *(_QWORD *)(v3 + 24) += 23LL;
    }
    v6 = *(unsigned int *)(a1 + 312);
    if ( *(_DWORD *)(a1 + 312) )
    {
      v8 = *(_QWORD *)(a1 + 272);
      v9 = *(char **)(a1 + 304);
      v10 = *(unsigned int *)(a1 + 312);
      v11 = *(void **)(v8 + 24);
      if ( v6 > *(_QWORD *)(v8 + 16) - (_QWORD)v11 )
      {
        sub_16E7EE0(*(_QWORD *)(a1 + 272), v9, v10);
      }
      else
      {
        memcpy(v11, v9, v10);
        *(_QWORD *)(v8 + 24) += v6;
      }
    }
    *(_DWORD *)(a1 + 312) = 0;
    if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
    {
      return sub_39E0440(a1);
    }
    else
    {
      v7 = *(_QWORD *)(a1 + 272);
      result = *(_BYTE **)(v7 + 24);
      if ( (unsigned __int64)result >= *(_QWORD *)(v7 + 16) )
      {
        return (_BYTE *)sub_16E7DE0(v7, 10);
      }
      else
      {
        *(_QWORD *)(v7 + 24) = result + 1;
        *result = 10;
      }
    }
  }
  return result;
}
