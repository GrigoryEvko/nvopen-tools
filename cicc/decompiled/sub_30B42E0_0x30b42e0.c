// Function: sub_30B42E0
// Address: 0x30b42e0
//
void __fastcall sub_30B42E0(__int64 *a1, __int64 a2)
{
  __int64 v4; // rax
  _BYTE *v5; // rsi
  __int64 v6; // rdx
  __m128i *v7; // rax
  __int64 v8; // rsi
  __m128i *v9; // rax
  __int64 v10; // rsi
  __m128i *v11; // rdi
  __int64 v12; // r8
  __int64 v13; // rdx
  __int64 v14; // rax
  unsigned __int64 v15; // rax
  __m128i **v16; // rsi
  __int64 v17; // rax
  _DWORD *v18; // rdx
  __int64 v19; // r8
  __int64 v20; // rdi
  __int64 v21; // rdi
  _BYTE *v22; // rax
  __m128i si128; // xmm0
  _QWORD *v24; // rdx
  __m128i **v25; // rsi
  __int64 v26; // rax
  __int64 v27; // rdx
  _QWORD *v28; // rdx
  __int64 v29; // [rsp+8h] [rbp-98h]
  __int64 v30; // [rsp+8h] [rbp-98h]
  __m128i *v31; // [rsp+10h] [rbp-90h] BYREF
  __int64 v32; // [rsp+18h] [rbp-88h]
  __m128i v33; // [rsp+20h] [rbp-80h] BYREF
  __m128i *v34; // [rsp+30h] [rbp-70h] BYREF
  __int64 v35; // [rsp+38h] [rbp-68h]
  __m128i v36; // [rsp+40h] [rbp-60h] BYREF
  unsigned __int8 *v37; // [rsp+50h] [rbp-50h] BYREF
  size_t v38; // [rsp+58h] [rbp-48h]
  unsigned __int8 v39[64]; // [rsp+60h] [rbp-40h] BYREF

  v4 = *(_QWORD *)a1[1];
  v5 = *(_BYTE **)(v4 + 8);
  v6 = *(_QWORD *)(v4 + 16);
  v37 = v39;
  sub_30B30D0((__int64 *)&v37, v5, (__int64)&v5[v6]);
  v7 = (__m128i *)sub_2241130((unsigned __int64 *)&v37, 0, 0, "DDG for '", 9u);
  v34 = &v36;
  if ( (__m128i *)v7->m128i_i64[0] == &v7[1] )
  {
    v36 = _mm_loadu_si128(v7 + 1);
  }
  else
  {
    v34 = (__m128i *)v7->m128i_i64[0];
    v36.m128i_i64[0] = v7[1].m128i_i64[0];
  }
  v8 = v7->m128i_i64[1];
  v7[1].m128i_i8[0] = 0;
  v35 = v8;
  v7->m128i_i64[0] = (__int64)v7[1].m128i_i64;
  v7->m128i_i64[1] = 0;
  if ( v35 == 0x3FFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"basic_string::append");
  v9 = (__m128i *)sub_2241490((unsigned __int64 *)&v34, "'", 1u);
  v31 = &v33;
  if ( (__m128i *)v9->m128i_i64[0] == &v9[1] )
  {
    v33 = _mm_loadu_si128(v9 + 1);
  }
  else
  {
    v31 = (__m128i *)v9->m128i_i64[0];
    v33.m128i_i64[0] = v9[1].m128i_i64[0];
  }
  v10 = v9->m128i_i64[1];
  v9[1].m128i_i8[0] = 0;
  v32 = v10;
  v9->m128i_i64[0] = (__int64)v9[1].m128i_i64;
  v11 = v34;
  v9->m128i_i64[1] = 0;
  if ( v11 != &v36 )
    j_j___libc_free_0((unsigned __int64)v11);
  if ( v37 != v39 )
    j_j___libc_free_0((unsigned __int64)v37);
  v12 = *a1;
  v13 = *(_QWORD *)(*a1 + 32);
  v14 = *(_QWORD *)(*a1 + 24);
  if ( *(_QWORD *)(a2 + 8) )
  {
    if ( (unsigned __int64)(v14 - v13) <= 8 )
    {
      v16 = (__m128i **)a2;
      v29 = sub_CB6200(*a1, "digraph \"", 9u);
    }
    else
    {
      *(_BYTE *)(v13 + 8) = 34;
      v16 = (__m128i **)a2;
      *(_QWORD *)v13 = 0x2068706172676964LL;
      v29 = v12;
      *(_QWORD *)(v12 + 32) += 9LL;
    }
  }
  else
  {
    v15 = v14 - v13;
    if ( !v32 )
    {
      if ( v15 <= 0x11 )
      {
        sub_CB6200(*a1, "digraph unnamed {\n", 0x12u);
      }
      else
      {
        si128 = _mm_load_si128(xmmword_3F8CB00);
        *(_WORD *)(v13 + 16) = 2683;
        *(__m128i *)v13 = si128;
        *(_QWORD *)(v12 + 32) += 18LL;
      }
      goto LABEL_19;
    }
    if ( v15 <= 8 )
    {
      v12 = sub_CB6200(*a1, "digraph \"", 9u);
    }
    else
    {
      *(_BYTE *)(v13 + 8) = 34;
      *(_QWORD *)v13 = 0x2068706172676964LL;
      *(_QWORD *)(v12 + 32) += 9LL;
    }
    v29 = v12;
    v16 = &v31;
  }
  sub_C67200((__int64 *)&v37, (__int64)v16);
  v17 = sub_CB6200(v29, v37, v38);
  v18 = *(_DWORD **)(v17 + 32);
  if ( *(_QWORD *)(v17 + 24) - (_QWORD)v18 <= 3u )
  {
    sub_CB6200(v17, "\" {\n", 4u);
  }
  else
  {
    *v18 = 175841314;
    *(_QWORD *)(v17 + 32) += 4LL;
  }
  if ( v37 != v39 )
    j_j___libc_free_0((unsigned __int64)v37);
LABEL_19:
  v19 = *a1;
  v20 = *a1;
  if ( *(_QWORD *)(a2 + 8) )
  {
    v24 = *(_QWORD **)(v19 + 32);
    if ( *(_QWORD *)(v19 + 24) - (_QWORD)v24 <= 7u )
    {
      v19 = sub_CB6200(v20, "\tlabel=\"", 8u);
    }
    else
    {
      *v24 = 0x223D6C6562616C09LL;
      *(_QWORD *)(v19 + 32) += 8LL;
    }
    v30 = v19;
    v25 = (__m128i **)a2;
  }
  else
  {
    if ( !v32 )
      goto LABEL_21;
    v28 = *(_QWORD **)(v19 + 32);
    if ( *(_QWORD *)(v19 + 24) - (_QWORD)v28 <= 7u )
    {
      v19 = sub_CB6200(v20, "\tlabel=\"", 8u);
    }
    else
    {
      *v28 = 0x223D6C6562616C09LL;
      *(_QWORD *)(v19 + 32) += 8LL;
    }
    v30 = v19;
    v25 = &v31;
  }
  sub_C67200((__int64 *)&v37, (__int64)v25);
  v26 = sub_CB6200(v30, v37, v38);
  v27 = *(_QWORD *)(v26 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v26 + 24) - v27) <= 2 )
  {
    sub_CB6200(v26, (unsigned __int8 *)"\";\n", 3u);
  }
  else
  {
    *(_BYTE *)(v27 + 2) = 10;
    *(_WORD *)v27 = 15138;
    *(_QWORD *)(v26 + 32) += 3LL;
  }
  if ( v37 != v39 )
    j_j___libc_free_0((unsigned __int64)v37);
  v20 = *a1;
LABEL_21:
  v37 = v39;
  v38 = 0;
  v39[0] = 0;
  sub_CB6200(v20, v39, 0);
  if ( v37 != v39 )
    j_j___libc_free_0((unsigned __int64)v37);
  v21 = *a1;
  v22 = *(_BYTE **)(*a1 + 32);
  if ( *(_BYTE **)(*a1 + 24) == v22 )
  {
    sub_CB6200(v21, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v22 = 10;
    ++*(_QWORD *)(v21 + 32);
  }
  if ( v31 != &v33 )
    j_j___libc_free_0((unsigned __int64)v31);
}
