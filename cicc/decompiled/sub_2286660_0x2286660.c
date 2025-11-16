// Function: sub_2286660
// Address: 0x2286660
//
void __fastcall sub_2286660(__int64 *a1, __m128i **a2)
{
  __m128i **v2; // r13
  __int64 v4; // rax
  __m128i *v5; // rax
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // rdx
  __int64 v9; // rax
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  _DWORD *v12; // rdx
  __int64 v13; // r8
  __int64 v14; // rdi
  __int64 v15; // rdi
  _BYTE *v16; // rax
  __m128i si128; // xmm0
  _QWORD *v18; // rdx
  __m128i **v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rdx
  _QWORD *v22; // rdx
  __int64 v23; // [rsp+8h] [rbp-78h]
  __int64 v24; // [rsp+8h] [rbp-78h]
  __m128i *v25; // [rsp+10h] [rbp-70h] BYREF
  __int64 v26; // [rsp+18h] [rbp-68h]
  __m128i v27; // [rsp+20h] [rbp-60h] BYREF
  unsigned __int8 *v28; // [rsp+30h] [rbp-50h] BYREF
  size_t v29; // [rsp+38h] [rbp-48h]
  unsigned __int8 v30[64]; // [rsp+40h] [rbp-40h] BYREF

  v2 = a2;
  v4 = **(_QWORD **)a1[1];
  v28 = v30;
  sub_11F4570((__int64 *)&v28, *(_BYTE **)(v4 + 168), *(_QWORD *)(v4 + 168) + *(_QWORD *)(v4 + 176));
  v5 = (__m128i *)sub_2241130((unsigned __int64 *)&v28, 0, 0, "Call graph: ", 0xCu);
  v25 = &v27;
  if ( (__m128i *)v5->m128i_i64[0] == &v5[1] )
  {
    v27 = _mm_loadu_si128(v5 + 1);
  }
  else
  {
    v25 = (__m128i *)v5->m128i_i64[0];
    v27.m128i_i64[0] = v5[1].m128i_i64[0];
  }
  v6 = v5->m128i_i64[1];
  v5[1].m128i_i8[0] = 0;
  v26 = v6;
  v5->m128i_i64[0] = (__int64)v5[1].m128i_i64;
  v5->m128i_i64[1] = 0;
  if ( v28 != v30 )
    j_j___libc_free_0((unsigned __int64)v28);
  v7 = *a1;
  v8 = *(_QWORD *)(*a1 + 32);
  v9 = *(_QWORD *)(*a1 + 24);
  if ( a2[1] )
  {
    if ( (unsigned __int64)(v9 - v8) <= 8 )
    {
      v23 = sub_CB6200(*a1, "digraph \"", 9u);
    }
    else
    {
      *(_BYTE *)(v8 + 8) = 34;
      *(_QWORD *)v8 = 0x2068706172676964LL;
      v23 = v7;
      *(_QWORD *)(v7 + 32) += 9LL;
    }
  }
  else
  {
    v10 = v9 - v8;
    if ( !v26 )
    {
      if ( v10 <= 0x11 )
      {
        sub_CB6200(*a1, "digraph unnamed {\n", 0x12u);
      }
      else
      {
        si128 = _mm_load_si128(xmmword_3F8CB00);
        *(_WORD *)(v8 + 16) = 2683;
        *(__m128i *)v8 = si128;
        *(_QWORD *)(v7 + 32) += 18LL;
      }
      goto LABEL_14;
    }
    if ( v10 <= 8 )
    {
      v7 = sub_CB6200(*a1, "digraph \"", 9u);
    }
    else
    {
      *(_BYTE *)(v8 + 8) = 34;
      *(_QWORD *)v8 = 0x2068706172676964LL;
      *(_QWORD *)(v7 + 32) += 9LL;
    }
    v23 = v7;
    a2 = &v25;
  }
  sub_C67200((__int64 *)&v28, (__int64)a2);
  v11 = sub_CB6200(v23, v28, v29);
  v12 = *(_DWORD **)(v11 + 32);
  if ( *(_QWORD *)(v11 + 24) - (_QWORD)v12 <= 3u )
  {
    sub_CB6200(v11, "\" {\n", 4u);
  }
  else
  {
    *v12 = 175841314;
    *(_QWORD *)(v11 + 32) += 4LL;
  }
  if ( v28 != v30 )
    j_j___libc_free_0((unsigned __int64)v28);
LABEL_14:
  v13 = *a1;
  v14 = *a1;
  if ( v2[1] )
  {
    v18 = *(_QWORD **)(v13 + 32);
    if ( *(_QWORD *)(v13 + 24) - (_QWORD)v18 <= 7u )
    {
      v13 = sub_CB6200(v14, "\tlabel=\"", 8u);
    }
    else
    {
      *v18 = 0x223D6C6562616C09LL;
      *(_QWORD *)(v13 + 32) += 8LL;
    }
    v24 = v13;
    v19 = v2;
  }
  else
  {
    if ( !v26 )
      goto LABEL_16;
    v22 = *(_QWORD **)(v13 + 32);
    if ( *(_QWORD *)(v13 + 24) - (_QWORD)v22 <= 7u )
    {
      v13 = sub_CB6200(v14, "\tlabel=\"", 8u);
    }
    else
    {
      *v22 = 0x223D6C6562616C09LL;
      *(_QWORD *)(v13 + 32) += 8LL;
    }
    v24 = v13;
    v19 = &v25;
  }
  sub_C67200((__int64 *)&v28, (__int64)v19);
  v20 = sub_CB6200(v24, v28, v29);
  v21 = *(_QWORD *)(v20 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v20 + 24) - v21) <= 2 )
  {
    sub_CB6200(v20, (unsigned __int8 *)"\";\n", 3u);
  }
  else
  {
    *(_BYTE *)(v21 + 2) = 10;
    *(_WORD *)v21 = 15138;
    *(_QWORD *)(v20 + 32) += 3LL;
  }
  if ( v28 != v30 )
    j_j___libc_free_0((unsigned __int64)v28);
  v14 = *a1;
LABEL_16:
  v28 = v30;
  v29 = 0;
  v30[0] = 0;
  sub_CB6200(v14, v30, 0);
  if ( v28 != v30 )
    j_j___libc_free_0((unsigned __int64)v28);
  v15 = *a1;
  v16 = *(_BYTE **)(*a1 + 32);
  if ( *(_BYTE **)(*a1 + 24) == v16 )
  {
    sub_CB6200(v15, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v16 = 10;
    ++*(_QWORD *)(v15 + 32);
  }
  if ( v25 != &v27 )
    j_j___libc_free_0((unsigned __int64)v25);
}
