// Function: sub_39E6F60
// Address: 0x39e6f60
//
_BYTE *__fastcall sub_39E6F60(__int64 a1, int a2)
{
  __int64 v3; // rdi
  _BYTE *v4; // rax
  unsigned __int64 v5; // r13
  _BYTE *result; // rax
  __int64 v7; // rdi
  __int64 v8; // r14
  char *v9; // rsi
  size_t v10; // rdx
  void *v11; // rdi
  _BYTE *v12; // rax
  __m128i *v13; // rdx
  __m128i si128; // xmm0
  __m128i *v15; // rdx
  _BYTE *v16; // rax

  v3 = *(_QWORD *)(a1 + 272);
  switch ( a2 )
  {
    case 0:
      v15 = *(__m128i **)(v3 + 24);
      if ( *(_QWORD *)(v3 + 16) - (_QWORD)v15 <= 0xFu )
      {
        sub_16E7EE0(v3, "\t.syntax unified", 0x10u);
      }
      else
      {
        *v15 = _mm_load_si128((const __m128i *)&xmmword_4534D40);
        *(_QWORD *)(v3 + 24) += 16LL;
      }
      break;
    case 1:
      v13 = *(__m128i **)(v3 + 24);
      if ( *(_QWORD *)(v3 + 16) - (_QWORD)v13 <= 0x17u )
      {
        sub_16E7EE0(v3, ".subsections_via_symbols", 0x18u);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_4534D50);
        v13[1].m128i_i64[0] = 0x736C6F626D79735FLL;
        *v13 = si128;
        *(_QWORD *)(v3 + 24) += 24LL;
      }
      break;
    case 2:
      v16 = *(_BYTE **)(v3 + 24);
      if ( (unsigned __int64)v16 >= *(_QWORD *)(v3 + 16) )
      {
        v3 = sub_16E7DE0(v3, 9);
      }
      else
      {
        *(_QWORD *)(v3 + 24) = v16 + 1;
        *v16 = 9;
      }
      sub_1263B40(v3, *(const char **)(*(_QWORD *)(a1 + 280) + 144LL));
      break;
    case 3:
      v12 = *(_BYTE **)(v3 + 24);
      if ( (unsigned __int64)v12 >= *(_QWORD *)(v3 + 16) )
      {
        v3 = sub_16E7DE0(v3, 9);
      }
      else
      {
        *(_QWORD *)(v3 + 24) = v12 + 1;
        *v12 = 9;
      }
      sub_1263B40(v3, *(const char **)(*(_QWORD *)(a1 + 280) + 152LL));
      break;
    case 4:
      v4 = *(_BYTE **)(v3 + 24);
      if ( (unsigned __int64)v4 >= *(_QWORD *)(v3 + 16) )
      {
        v3 = sub_16E7DE0(v3, 9);
      }
      else
      {
        *(_QWORD *)(v3 + 24) = v4 + 1;
        *v4 = 9;
      }
      sub_1263B40(v3, *(const char **)(*(_QWORD *)(a1 + 280) + 160LL));
      break;
    default:
      break;
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
