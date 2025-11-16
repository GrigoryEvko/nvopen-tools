// Function: sub_39E5380
// Address: 0x39e5380
//
_BYTE *__fastcall sub_39E5380(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdi
  __m128i *v6; // rdx
  __m128i si128; // xmm0
  __int64 v8; // rdi
  _BYTE *v9; // rax
  __int64 v10; // r14
  __int64 v11; // rbx
  _BYTE *v12; // rax
  __int64 v13; // rdi
  _BYTE *v14; // rax
  __int64 v15; // rdi
  _WORD *v16; // rdx
  unsigned __int64 v17; // r13
  _BYTE *result; // rax
  __int64 v19; // rdi
  __int64 v20; // r14
  char *v21; // rsi
  size_t v22; // rdx
  void *v23; // rdi

  v4 = *(_QWORD *)(a1 + 272);
  v6 = *(__m128i **)(v4 + 24);
  if ( *(_QWORD *)(v4 + 16) - (_QWORD)v6 <= 0x10u )
  {
    v4 = sub_16E7EE0(v4, "\t.linker_option \"", 0x11u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F7F860);
    v6[1].m128i_i8[0] = 34;
    *v6 = si128;
    *(_QWORD *)(v4 + 24) += 17LL;
  }
  v8 = sub_16E7EE0(v4, *(char **)a2, *(_QWORD *)(a2 + 8));
  v9 = *(_BYTE **)(v8 + 24);
  if ( (unsigned __int64)v9 >= *(_QWORD *)(v8 + 16) )
  {
    sub_16E7DE0(v8, 34);
  }
  else
  {
    *(_QWORD *)(v8 + 24) = v9 + 1;
    *v9 = 34;
  }
  v10 = a2 + 32;
  v11 = a2 + 32 * a3;
  if ( a2 + 32 != v11 )
  {
    while ( 1 )
    {
      v15 = *(_QWORD *)(a1 + 272);
      v16 = *(_WORD **)(v15 + 24);
      if ( *(_QWORD *)(v15 + 16) - (_QWORD)v16 > 1u )
      {
        *v16 = 8236;
        v12 = (_BYTE *)(*(_QWORD *)(v15 + 24) + 2LL);
        *(_QWORD *)(v15 + 24) = v12;
        if ( *(_QWORD *)(v15 + 16) <= (unsigned __int64)v12 )
          goto LABEL_13;
      }
      else
      {
        v15 = sub_16E7EE0(v15, ", ", 2u);
        v12 = *(_BYTE **)(v15 + 24);
        if ( *(_QWORD *)(v15 + 16) <= (unsigned __int64)v12 )
        {
LABEL_13:
          v15 = sub_16E7DE0(v15, 34);
          goto LABEL_9;
        }
      }
      *(_QWORD *)(v15 + 24) = v12 + 1;
      *v12 = 34;
LABEL_9:
      v13 = sub_16E7EE0(v15, *(char **)v10, *(_QWORD *)(v10 + 8));
      v14 = *(_BYTE **)(v13 + 24);
      if ( (unsigned __int64)v14 >= *(_QWORD *)(v13 + 16) )
      {
        v10 += 32;
        sub_16E7DE0(v13, 34);
        if ( v10 == v11 )
          break;
      }
      else
      {
        v10 += 32;
        *(_QWORD *)(v13 + 24) = v14 + 1;
        *v14 = 34;
        if ( v10 == v11 )
          break;
      }
    }
  }
  v17 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
  {
    v20 = *(_QWORD *)(a1 + 272);
    v21 = *(char **)(a1 + 304);
    v22 = *(unsigned int *)(a1 + 312);
    v23 = *(void **)(v20 + 24);
    if ( v17 > *(_QWORD *)(v20 + 16) - (_QWORD)v23 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 272), v21, v22);
    }
    else
    {
      memcpy(v23, v21, v22);
      *(_QWORD *)(v20 + 24) += v17;
    }
  }
  *(_DWORD *)(a1 + 312) = 0;
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
    return sub_39E0440(a1);
  v19 = *(_QWORD *)(a1 + 272);
  result = *(_BYTE **)(v19 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(v19 + 16) )
    return (_BYTE *)sub_16E7DE0(v19, 10);
  *(_QWORD *)(v19 + 24) = result + 1;
  *result = 10;
  return result;
}
