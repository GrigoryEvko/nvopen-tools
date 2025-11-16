// Function: sub_13B8450
// Address: 0x13b8450
//
_BYTE *__fastcall sub_13B8450(__int64 *a1, __int64 **a2)
{
  __int64 **v2; // r12
  __m128i *v4; // rax
  __int64 *v5; // rdx
  __int64 v6; // r14
  __int64 v7; // rdx
  __int64 v8; // rax
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  _DWORD *v11; // rdx
  __int64 v12; // r8
  __int64 v13; // rdi
  __int64 v14; // rdi
  _BYTE *result; // rax
  __m128i si128; // xmm0
  _QWORD *v17; // rdx
  __int64 **v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rdx
  _QWORD *v21; // rdx
  __int64 v22; // [rsp+0h] [rbp-80h]
  __int64 *v23; // [rsp+10h] [rbp-70h] BYREF
  __int64 v24; // [rsp+18h] [rbp-68h]
  _QWORD v25[2]; // [rsp+20h] [rbp-60h] BYREF
  __int64 v26; // [rsp+30h] [rbp-50h] BYREF
  __int64 v27; // [rsp+38h] [rbp-48h]
  char v28[64]; // [rsp+40h] [rbp-40h] BYREF

  v2 = a2;
  v23 = v25;
  v26 = 19;
  v4 = (__m128i *)sub_22409D0(&v23, &v26, 0);
  v23 = (__int64 *)v4;
  v25[0] = v26;
  *v4 = _mm_load_si128((const __m128i *)&xmmword_4289C10);
  v5 = v23;
  v4[1].m128i_i16[0] = 25970;
  v4[1].m128i_i8[2] = 101;
  v24 = v26;
  *((_BYTE *)v5 + v26) = 0;
  v6 = *a1;
  v7 = *(_QWORD *)(*a1 + 24);
  v8 = *(_QWORD *)(*a1 + 16);
  if ( a2[1] )
  {
    if ( (unsigned __int64)(v8 - v7) <= 8 )
    {
      v6 = sub_16E7EE0(*a1, "digraph \"", 9);
    }
    else
    {
      *(_BYTE *)(v7 + 8) = 34;
      *(_QWORD *)v7 = 0x2068706172676964LL;
      *(_QWORD *)(v6 + 24) += 9LL;
    }
  }
  else
  {
    v9 = v8 - v7;
    if ( !v24 )
    {
      if ( v9 <= 0x11 )
      {
        sub_16E7EE0(*a1, "digraph unnamed {\n", 18);
      }
      else
      {
        si128 = _mm_load_si128(xmmword_3F8CB00);
        *(_WORD *)(v7 + 16) = 2683;
        *(__m128i *)v7 = si128;
        *(_QWORD *)(v6 + 24) += 18LL;
      }
      goto LABEL_10;
    }
    if ( v9 <= 8 )
    {
      v6 = sub_16E7EE0(*a1, "digraph \"", 9);
    }
    else
    {
      *(_BYTE *)(v7 + 8) = 34;
      *(_QWORD *)v7 = 0x2068706172676964LL;
      *(_QWORD *)(v6 + 24) += 9LL;
    }
    a2 = &v23;
  }
  sub_16BE9B0(&v26, a2);
  v10 = sub_16E7EE0(v6, (const char *)v26, v27);
  v11 = *(_DWORD **)(v10 + 24);
  if ( *(_QWORD *)(v10 + 16) - (_QWORD)v11 <= 3u )
  {
    sub_16E7EE0(v10, "\" {\n", 4);
  }
  else
  {
    *v11 = 175841314;
    *(_QWORD *)(v10 + 24) += 4LL;
  }
  if ( (char *)v26 != v28 )
    j_j___libc_free_0(v26, *(_QWORD *)v28 + 1LL);
LABEL_10:
  v12 = *a1;
  v13 = *a1;
  if ( v2[1] )
  {
    v17 = *(_QWORD **)(v12 + 24);
    if ( *(_QWORD *)(v12 + 16) - (_QWORD)v17 <= 7u )
    {
      v12 = sub_16E7EE0(v13, "\tlabel=\"", 8);
    }
    else
    {
      *v17 = 0x223D6C6562616C09LL;
      *(_QWORD *)(v12 + 24) += 8LL;
    }
    v22 = v12;
    v18 = v2;
  }
  else
  {
    if ( !v24 )
      goto LABEL_12;
    v21 = *(_QWORD **)(v12 + 24);
    if ( *(_QWORD *)(v12 + 16) - (_QWORD)v21 <= 7u )
    {
      v12 = sub_16E7EE0(v13, "\tlabel=\"", 8);
    }
    else
    {
      *v21 = 0x223D6C6562616C09LL;
      *(_QWORD *)(v12 + 24) += 8LL;
    }
    v22 = v12;
    v18 = &v23;
  }
  sub_16BE9B0(&v26, v18);
  v19 = sub_16E7EE0(v22, (const char *)v26, v27);
  v20 = *(_QWORD *)(v19 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(v19 + 16) - v20) <= 2 )
  {
    sub_16E7EE0(v19, "\";\n", 3);
  }
  else
  {
    *(_BYTE *)(v20 + 2) = 10;
    *(_WORD *)v20 = 15138;
    *(_QWORD *)(v19 + 24) += 3LL;
  }
  if ( (char *)v26 != v28 )
    j_j___libc_free_0(v26, *(_QWORD *)v28 + 1LL);
  v13 = *a1;
LABEL_12:
  v26 = (__int64)v28;
  v27 = 0;
  v28[0] = 0;
  sub_16E7EE0(v13, v28, 0);
  if ( (char *)v26 != v28 )
    j_j___libc_free_0(v26, *(_QWORD *)v28 + 1LL);
  v14 = *a1;
  result = *(_BYTE **)(*a1 + 24);
  if ( *(_BYTE **)(*a1 + 16) == result )
  {
    result = (_BYTE *)sub_16E7EE0(v14, "\n", 1);
  }
  else
  {
    *result = 10;
    ++*(_QWORD *)(v14 + 24);
  }
  if ( v23 != v25 )
    return (_BYTE *)j_j___libc_free_0(v23, v25[0] + 1LL);
  return result;
}
