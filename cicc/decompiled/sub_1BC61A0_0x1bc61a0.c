// Function: sub_1BC61A0
// Address: 0x1bc61a0
//
_BYTE *__fastcall sub_1BC61A0(__int64 *a1, __int64 *a2)
{
  __int64 v2; // r12
  __int64 v4; // r13
  __int64 v5; // rdx
  __int64 v6; // rax
  unsigned __int64 v7; // rax
  __int64 v8; // rax
  _DWORD *v9; // rdx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rdi
  _BYTE *result; // rax
  __m128i si128; // xmm0
  _QWORD *v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r12
  __int64 v19; // rax
  __int64 v20; // [rsp+0h] [rbp-80h]
  __int64 v21; // [rsp+0h] [rbp-80h]
  _QWORD *v22; // [rsp+10h] [rbp-70h] BYREF
  __int64 v23; // [rsp+18h] [rbp-68h]
  _QWORD v24[2]; // [rsp+20h] [rbp-60h] BYREF
  char *v25; // [rsp+30h] [rbp-50h] BYREF
  size_t v26; // [rsp+38h] [rbp-48h]
  _QWORD v27[8]; // [rsp+40h] [rbp-40h] BYREF

  v2 = (__int64)a2;
  v22 = v24;
  sub_1BB98B0((__int64 *)&v22, byte_3F871B3, (__int64)byte_3F871B3);
  v4 = *a1;
  v5 = *(_QWORD *)(*a1 + 24);
  v6 = *(_QWORD *)(*a1 + 16);
  if ( a2[1] )
  {
    if ( (unsigned __int64)(v6 - v5) <= 8 )
    {
      v4 = sub_16E7EE0(*a1, "digraph \"", 9u);
    }
    else
    {
      *(_BYTE *)(v5 + 8) = 34;
      *(_QWORD *)v5 = 0x2068706172676964LL;
      *(_QWORD *)(v4 + 24) += 9LL;
    }
  }
  else
  {
    v7 = v6 - v5;
    if ( !v23 )
    {
      if ( v7 <= 0x11 )
      {
        sub_16E7EE0(*a1, "digraph unnamed {\n", 0x12u);
      }
      else
      {
        si128 = _mm_load_si128(xmmword_3F8CB00);
        *(_WORD *)(v5 + 16) = 2683;
        *(__m128i *)v5 = si128;
        *(_QWORD *)(v4 + 24) += 18LL;
      }
      goto LABEL_10;
    }
    if ( v7 <= 8 )
    {
      v4 = sub_16E7EE0(*a1, "digraph \"", 9u);
    }
    else
    {
      *(_BYTE *)(v5 + 8) = 34;
      *(_QWORD *)v5 = 0x2068706172676964LL;
      *(_QWORD *)(v4 + 24) += 9LL;
    }
    a2 = (__int64 *)&v22;
  }
  sub_16BE9B0((__int64 *)&v25, (__int64)a2);
  v8 = sub_16E7EE0(v4, v25, v26);
  v9 = *(_DWORD **)(v8 + 24);
  if ( *(_QWORD *)(v8 + 16) - (_QWORD)v9 <= 3u )
  {
    sub_16E7EE0(v8, "\" {\n", 4u);
  }
  else
  {
    *v9 = 175841314;
    *(_QWORD *)(v8 + 24) += 4LL;
  }
  if ( v25 != (char *)v27 )
    j_j___libc_free_0(v25, v27[0] + 1LL);
LABEL_10:
  v10 = *a1;
  v11 = *a1;
  if ( *(_QWORD *)(v2 + 8) )
  {
    v15 = *(_QWORD **)(v10 + 24);
    if ( *(_QWORD *)(v10 + 16) - (_QWORD)v15 <= 7u )
    {
      v10 = sub_16E7EE0(*a1, "\tlabel=\"", 8u);
    }
    else
    {
      *v15 = 0x223D6C6562616C09LL;
      *(_QWORD *)(v10 + 24) += 8LL;
    }
    v21 = v10;
    sub_16BE9B0((__int64 *)&v25, v2);
    v16 = sub_16E7EE0(v21, v25, v26);
    v17 = *(_QWORD *)(v16 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(v16 + 16) - v17) <= 2 )
    {
      sub_16E7EE0(v16, "\";\n", 3u);
    }
    else
    {
      *(_BYTE *)(v17 + 2) = 10;
      *(_WORD *)v17 = 15138;
      *(_QWORD *)(v16 + 24) += 3LL;
    }
  }
  else
  {
    if ( !v23 )
      goto LABEL_12;
    v18 = sub_1263B40(*a1, "\tlabel=\"");
    sub_16BE9B0((__int64 *)&v25, (__int64)&v22);
    v19 = sub_16E7EE0(v18, v25, v26);
    sub_1263B40(v19, "\";\n");
  }
  if ( v25 != (char *)v27 )
    j_j___libc_free_0(v25, v27[0] + 1LL);
  v11 = *a1;
LABEL_12:
  v20 = v11;
  v25 = (char *)v27;
  sub_1BB98B0((__int64 *)&v25, byte_3F871B3, (__int64)byte_3F871B3);
  sub_16E7EE0(v20, v25, v26);
  if ( v25 != (char *)v27 )
    j_j___libc_free_0(v25, v27[0] + 1LL);
  v12 = *a1;
  result = *(_BYTE **)(*a1 + 24);
  if ( *(_BYTE **)(*a1 + 16) == result )
  {
    result = (_BYTE *)sub_16E7EE0(v12, "\n", 1u);
  }
  else
  {
    *result = 10;
    ++*(_QWORD *)(v12 + 24);
  }
  if ( v22 != v24 )
    return (_BYTE *)j_j___libc_free_0(v22, v24[0] + 1LL);
  return result;
}
