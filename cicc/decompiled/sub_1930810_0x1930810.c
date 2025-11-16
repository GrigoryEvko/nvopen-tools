// Function: sub_1930810
// Address: 0x1930810
//
_BYTE *__fastcall sub_1930810(__int64 a1, __int64 a2, char a3)
{
  __int64 v5; // rdx
  __int64 v6; // rdi
  __int64 v7; // rax
  _WORD *v8; // rdx
  void *v9; // rdx
  _BYTE *result; // rax
  __int64 v11; // r14
  __int64 i; // rbx
  __int64 v13; // rdi
  __int64 v14; // rax
  _DWORD *v15; // rdx
  _WORD *v16; // rdx
  __m128i si128; // xmm0

  v5 = *(_QWORD *)(a2 + 24);
  if ( a3 )
  {
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v5) <= 0x14 )
    {
      sub_16E7EE0(a2, "ExpressionTypeBasic, ", 0x15u);
      v5 = *(_QWORD *)(a2 + 24);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_42BE550);
      *(_DWORD *)(v5 + 16) = 744712563;
      *(_BYTE *)(v5 + 20) = 32;
      *(__m128i *)v5 = si128;
      v5 = *(_QWORD *)(a2 + 24) + 21LL;
      *(_QWORD *)(a2 + 24) = v5;
    }
  }
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v5) <= 8 )
  {
    v6 = sub_16E7EE0(a2, "opcode = ", 9u);
  }
  else
  {
    *(_BYTE *)(v5 + 8) = 32;
    v6 = a2;
    *(_QWORD *)v5 = 0x3D2065646F63706FLL;
    *(_QWORD *)(a2 + 24) += 9LL;
  }
  v7 = sub_16E7A90(v6, *(unsigned int *)(a1 + 12));
  v8 = *(_WORD **)(v7 + 24);
  if ( *(_QWORD *)(v7 + 16) - (_QWORD)v8 <= 1u )
  {
    sub_16E7EE0(v7, ", ", 2u);
  }
  else
  {
    *v8 = 8236;
    *(_QWORD *)(v7 + 24) += 2LL;
  }
  v9 = *(void **)(a2 + 24);
  if ( *(_QWORD *)(a2 + 16) - (_QWORD)v9 <= 0xBu )
  {
    sub_16E7EE0(a2, "operands = {", 0xCu);
    result = *(_BYTE **)(a2 + 24);
  }
  else
  {
    qmemcpy(v9, "operands = {", 12);
    result = (_BYTE *)(*(_QWORD *)(a2 + 24) + 12LL);
    *(_QWORD *)(a2 + 24) = result;
  }
  v11 = *(unsigned int *)(a1 + 36);
  if ( (_DWORD)v11 )
  {
    for ( i = 0; i != v11; ++i )
    {
      while ( 1 )
      {
        if ( result == *(_BYTE **)(a2 + 16) )
        {
          v13 = sub_16E7EE0(a2, "[", 1u);
        }
        else
        {
          *result = 91;
          v13 = a2;
          ++*(_QWORD *)(a2 + 24);
        }
        v14 = sub_16E7A90(v13, i);
        v15 = *(_DWORD **)(v14 + 24);
        if ( *(_QWORD *)(v14 + 16) - (_QWORD)v15 <= 3u )
        {
          sub_16E7EE0(v14, "] = ", 4u);
        }
        else
        {
          *v15 = 540876893;
          *(_QWORD *)(v14 + 24) += 4LL;
        }
        sub_15537D0(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * i), a2, 1, 0);
        v16 = *(_WORD **)(a2 + 24);
        if ( *(_QWORD *)(a2 + 16) - (_QWORD)v16 <= 1u )
          break;
        ++i;
        *v16 = 8224;
        result = (_BYTE *)(*(_QWORD *)(a2 + 24) + 2LL);
        *(_QWORD *)(a2 + 24) = result;
        if ( v11 == i )
          goto LABEL_18;
      }
      sub_16E7EE0(a2, "  ", 2u);
      result = *(_BYTE **)(a2 + 24);
    }
  }
LABEL_18:
  if ( *(_QWORD *)(a2 + 16) - (_QWORD)result <= 1u )
    return (_BYTE *)sub_16E7EE0(a2, "} ", 2u);
  *(_WORD *)result = 8317;
  *(_QWORD *)(a2 + 24) += 2LL;
  return result;
}
