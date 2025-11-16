// Function: sub_39E9E50
// Address: 0x39e9e50
//
_BYTE *__fastcall sub_39E9E50(__int64 a1, unsigned __int64 a2)
{
  __int64 v3; // rbx
  unsigned __int64 v4; // rax
  __int64 v5; // rsi
  __int64 v6; // rdx
  __int64 *v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rsi
  __int64 *v10; // rax
  __int64 v11; // rdi
  __m128i *v12; // rdx
  __m128i si128; // xmm0
  unsigned __int64 v14; // r13
  _BYTE *result; // rax
  __int64 v16; // rdi
  __int64 v17; // r14
  char *v18; // rsi
  size_t v19; // rdx
  void *v20; // rdi

  sub_38DD4D0(a1, a2);
  v3 = *(_QWORD *)(*(_QWORD *)(a1 + 72) + 24LL);
  v4 = *(_QWORD *)v3 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v4
    || (*(_BYTE *)(v3 + 9) & 0xC) == 8
    && (*(_BYTE *)(v3 + 8) |= 4u,
        v4 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v3 + 24)),
        *(_QWORD *)v3 = v4 | *(_QWORD *)v3 & 7LL,
        v4) )
  {
    v5 = *(_QWORD *)(v4 + 24);
  }
  else
  {
    v5 = 0;
  }
  v6 = sub_38DD570(a1, v5);
  v7 = (__int64 *)(*(_QWORD *)(a1 + 112) + 32LL * *(unsigned int *)(a1 + 120) - 32);
  v8 = *v7;
  v9 = v7[1];
  v7[2] = *v7;
  v7[3] = v9;
  if ( v6 != v8 || v9 )
  {
    v10 = (__int64 *)(*(_QWORD *)(a1 + 112) + 32LL * *(unsigned int *)(a1 + 120) - 32);
    *v10 = v6;
    v10[1] = 0;
  }
  v11 = *(_QWORD *)(a1 + 272);
  v12 = *(__m128i **)(v11 + 24);
  if ( *(_QWORD *)(v11 + 16) - (_QWORD)v12 <= 0x10u )
  {
    sub_16E7EE0(v11, "\t.seh_handlerdata", 0x11u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F7F8C0);
    v12[1].m128i_i8[0] = 97;
    *v12 = si128;
    *(_QWORD *)(v11 + 24) += 17LL;
  }
  v14 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
  {
    v17 = *(_QWORD *)(a1 + 272);
    v18 = *(char **)(a1 + 304);
    v19 = *(unsigned int *)(a1 + 312);
    v20 = *(void **)(v17 + 24);
    if ( v14 > *(_QWORD *)(v17 + 16) - (_QWORD)v20 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 272), v18, v19);
    }
    else
    {
      memcpy(v20, v18, v19);
      *(_QWORD *)(v17 + 24) += v14;
    }
  }
  *(_DWORD *)(a1 + 312) = 0;
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
    return sub_39E0440(a1);
  v16 = *(_QWORD *)(a1 + 272);
  result = *(_BYTE **)(v16 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(v16 + 16) )
    return (_BYTE *)sub_16E7DE0(v16, 10);
  *(_QWORD *)(v16 + 24) = result + 1;
  *result = 10;
  return result;
}
