// Function: sub_137F580
// Address: 0x137f580
//
_BYTE *__fastcall sub_137F580(__int64 *a1, __m128i **a2)
{
  __m128i **v2; // r13
  _BYTE *v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rcx
  __m128i *v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r15
  __int64 v11; // rdx
  __int64 v12; // rax
  unsigned __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rcx
  _DWORD *v16; // rdx
  __int64 v17; // r15
  __int64 v18; // r8
  __int64 v19; // rdi
  _BYTE *result; // rax
  __m128i si128; // xmm0
  _QWORD *v22; // rdx
  __m128i **v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rdx
  _QWORD *v26; // rdx
  __int64 v27; // [rsp+0h] [rbp-A0h]
  __m128i *v28; // [rsp+10h] [rbp-90h] BYREF
  __int64 v29; // [rsp+18h] [rbp-88h]
  __m128i v30; // [rsp+20h] [rbp-80h] BYREF
  __int64 v31[2]; // [rsp+30h] [rbp-70h] BYREF
  _QWORD v32[2]; // [rsp+40h] [rbp-60h] BYREF
  const char *v33; // [rsp+50h] [rbp-50h] BYREF
  __int64 v34; // [rsp+58h] [rbp-48h]
  _OWORD v35[4]; // [rsp+60h] [rbp-40h] BYREF

  v2 = a2;
  v4 = (_BYTE *)sub_1649960(*(_QWORD *)a1[1]);
  if ( v4 )
  {
    v31[0] = (__int64)v32;
    sub_137E9E0(v31, v4, (__int64)&v4[v5]);
  }
  else
  {
    v31[1] = 0;
    v31[0] = (__int64)v32;
    LOBYTE(v32[0]) = 0;
  }
  v6 = sub_2241130(v31, 0, 0, "CFG for '", 9);
  v33 = (const char *)v35;
  if ( *(_QWORD *)v6 == v6 + 16 )
  {
    v35[0] = _mm_loadu_si128((const __m128i *)(v6 + 16));
  }
  else
  {
    v33 = *(const char **)v6;
    *(_QWORD *)&v35[0] = *(_QWORD *)(v6 + 16);
  }
  v7 = *(_QWORD *)(v6 + 8);
  v34 = v7;
  *(_QWORD *)v6 = v6 + 16;
  *(_QWORD *)(v6 + 8) = 0;
  *(_BYTE *)(v6 + 16) = 0;
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v34) <= 9 )
    sub_4262D8((__int64)"basic_string::append");
  v8 = (__m128i *)sub_2241490(&v33, "' function", 10, v7);
  v28 = &v30;
  if ( (__m128i *)v8->m128i_i64[0] == &v8[1] )
  {
    v30 = _mm_loadu_si128(v8 + 1);
  }
  else
  {
    v28 = (__m128i *)v8->m128i_i64[0];
    v30.m128i_i64[0] = v8[1].m128i_i64[0];
  }
  v9 = v8->m128i_i64[1];
  v8[1].m128i_i8[0] = 0;
  v29 = v9;
  v8->m128i_i64[0] = (__int64)v8[1].m128i_i64;
  v8->m128i_i64[1] = 0;
  if ( v33 != (const char *)v35 )
    j_j___libc_free_0(v33, *(_QWORD *)&v35[0] + 1LL);
  if ( (_QWORD *)v31[0] != v32 )
    j_j___libc_free_0(v31[0], v32[0] + 1LL);
  v10 = *a1;
  v11 = *(_QWORD *)(*a1 + 24);
  v12 = *(_QWORD *)(*a1 + 16);
  if ( a2[1] )
  {
    if ( (unsigned __int64)(v12 - v11) <= 8 )
    {
      v10 = sub_16E7EE0(*a1, "digraph \"", 9);
    }
    else
    {
      *(_BYTE *)(v11 + 8) = 34;
      *(_QWORD *)v11 = 0x2068706172676964LL;
      *(_QWORD *)(v10 + 24) += 9LL;
    }
  }
  else
  {
    v13 = v12 - v11;
    if ( !v29 )
    {
      if ( v13 <= 0x11 )
      {
        sub_16E7EE0(*a1, "digraph unnamed {\n", 18);
      }
      else
      {
        si128 = _mm_load_si128(xmmword_3F8CB00);
        v15 = 2683;
        *(_WORD *)(v11 + 16) = 2683;
        *(__m128i *)v11 = si128;
        *(_QWORD *)(v10 + 24) += 18LL;
      }
      goto LABEL_21;
    }
    if ( v13 <= 8 )
    {
      v10 = sub_16E7EE0(*a1, "digraph \"", 9);
    }
    else
    {
      *(_BYTE *)(v11 + 8) = 34;
      *(_QWORD *)v11 = 0x2068706172676964LL;
      *(_QWORD *)(v10 + 24) += 9LL;
    }
    a2 = &v28;
  }
  sub_16BE9B0(&v33, a2);
  v14 = sub_16E7EE0(v10, v33, v34);
  v16 = *(_DWORD **)(v14 + 24);
  if ( *(_QWORD *)(v14 + 16) - (_QWORD)v16 <= 3u )
  {
    sub_16E7EE0(v14, "\" {\n", 4);
  }
  else
  {
    *v16 = 175841314;
    *(_QWORD *)(v14 + 24) += 4LL;
  }
  if ( v33 != (const char *)v35 )
    j_j___libc_free_0(v33, *(_QWORD *)&v35[0] + 1LL);
LABEL_21:
  v17 = *a1;
  v18 = *a1;
  if ( v2[1] )
  {
    v22 = *(_QWORD **)(v17 + 24);
    if ( *(_QWORD *)(v17 + 16) - (_QWORD)v22 <= 7u )
    {
      v17 = sub_16E7EE0(*a1, "\tlabel=\"", 8, v15, v18);
    }
    else
    {
      *v22 = 0x223D6C6562616C09LL;
      *(_QWORD *)(v17 + 24) += 8LL;
    }
    v23 = v2;
  }
  else
  {
    if ( !v29 )
      goto LABEL_23;
    v26 = *(_QWORD **)(v17 + 24);
    if ( *(_QWORD *)(v17 + 16) - (_QWORD)v26 <= 7u )
    {
      v17 = sub_16E7EE0(*a1, "\tlabel=\"", 8, v15, v18);
    }
    else
    {
      *v26 = 0x223D6C6562616C09LL;
      *(_QWORD *)(v17 + 24) += 8LL;
    }
    v23 = &v28;
  }
  sub_16BE9B0(&v33, v23);
  v24 = sub_16E7EE0(v17, v33, v34);
  v25 = *(_QWORD *)(v24 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(v24 + 16) - v25) <= 2 )
  {
    sub_16E7EE0(v24, "\";\n", 3);
  }
  else
  {
    *(_BYTE *)(v25 + 2) = 10;
    *(_WORD *)v25 = 15138;
    *(_QWORD *)(v24 + 24) += 3LL;
  }
  if ( v33 != (const char *)v35 )
    j_j___libc_free_0(v33, *(_QWORD *)&v35[0] + 1LL);
  v18 = *a1;
LABEL_23:
  v27 = v18;
  v33 = (const char *)v35;
  sub_137E9E0((__int64 *)&v33, byte_3F871B3, (__int64)byte_3F871B3);
  sub_16E7EE0(v27, v33, v34);
  if ( v33 != (const char *)v35 )
    j_j___libc_free_0(v33, *(_QWORD *)&v35[0] + 1LL);
  v19 = *a1;
  result = *(_BYTE **)(*a1 + 24);
  if ( *(_BYTE **)(*a1 + 16) == result )
  {
    result = (_BYTE *)sub_16E7EE0(v19, "\n", 1);
  }
  else
  {
    *result = 10;
    ++*(_QWORD *)(v19 + 24);
  }
  if ( v28 != &v30 )
    return (_BYTE *)j_j___libc_free_0(v28, v30.m128i_i64[0] + 1);
  return result;
}
