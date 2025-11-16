// Function: sub_31584C0
// Address: 0x31584c0
//
void __fastcall sub_31584C0(__int64 ****a1, __m128i **a2)
{
  __m128i **v2; // r13
  char *v4; // rax
  __int64 v5; // rdx
  __m128i *v6; // rax
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 **v9; // rdx
  __int64 **v10; // rax
  unsigned __int64 v11; // rax
  __int64 v12; // rax
  _DWORD *v13; // rdx
  __int64 v14; // r8
  __int64 v15; // rdi
  __int64 v16; // rdi
  _BYTE *v17; // rax
  __m128i si128; // xmm0
  _QWORD *v19; // rdx
  __m128i **v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rdx
  _QWORD *v23; // rdx
  __int64 v24; // [rsp+8h] [rbp-78h]
  __int64 v25; // [rsp+8h] [rbp-78h]
  __m128i *v26; // [rsp+10h] [rbp-70h] BYREF
  __int64 v27; // [rsp+18h] [rbp-68h]
  __m128i v28; // [rsp+20h] [rbp-60h] BYREF
  unsigned __int8 *v29; // [rsp+30h] [rbp-50h] BYREF
  size_t v30; // [rsp+38h] [rbp-48h]
  unsigned __int8 v31[64]; // [rsp+40h] [rbp-40h] BYREF

  v2 = a2;
  v4 = (char *)sub_BD5D20(***a1[1]);
  if ( v4 )
  {
    v29 = v31;
    sub_3157F50((__int64 *)&v29, v4, (__int64)&v4[v5]);
  }
  else
  {
    v30 = 0;
    v29 = v31;
    v31[0] = 0;
  }
  v6 = (__m128i *)sub_2241130((unsigned __int64 *)&v29, 0, 0, "BCI CFG for ", 0xCu);
  v26 = &v28;
  if ( (__m128i *)v6->m128i_i64[0] == &v6[1] )
  {
    v28 = _mm_loadu_si128(v6 + 1);
  }
  else
  {
    v26 = (__m128i *)v6->m128i_i64[0];
    v28.m128i_i64[0] = v6[1].m128i_i64[0];
  }
  v7 = v6->m128i_i64[1];
  v6[1].m128i_i8[0] = 0;
  v27 = v7;
  v6->m128i_i64[0] = (__int64)v6[1].m128i_i64;
  v6->m128i_i64[1] = 0;
  if ( v29 != v31 )
    j_j___libc_free_0((unsigned __int64)v29);
  v8 = (__int64)*a1;
  v9 = (*a1)[4];
  v10 = (*a1)[3];
  if ( a2[1] )
  {
    if ( (unsigned __int64)((char *)v10 - (char *)v9) <= 8 )
    {
      v24 = sub_CB6200((__int64)*a1, "digraph \"", 9u);
    }
    else
    {
      *((_BYTE *)v9 + 8) = 34;
      *v9 = (__int64 *)0x2068706172676964LL;
      v24 = v8;
      *(_QWORD *)(v8 + 32) += 9LL;
    }
  }
  else
  {
    v11 = (char *)v10 - (char *)v9;
    if ( !v27 )
    {
      if ( v11 <= 0x11 )
      {
        sub_CB6200((__int64)*a1, "digraph unnamed {\n", 0x12u);
      }
      else
      {
        si128 = _mm_load_si128(xmmword_3F8CB00);
        *((_WORD *)v9 + 8) = 2683;
        *(__m128i *)v9 = si128;
        *(_QWORD *)(v8 + 32) += 18LL;
      }
      goto LABEL_16;
    }
    if ( v11 <= 8 )
    {
      v8 = sub_CB6200((__int64)*a1, "digraph \"", 9u);
    }
    else
    {
      *((_BYTE *)v9 + 8) = 34;
      *v9 = (__int64 *)0x2068706172676964LL;
      *(_QWORD *)(v8 + 32) += 9LL;
    }
    v24 = v8;
    a2 = &v26;
  }
  sub_C67200((__int64 *)&v29, (__int64)a2);
  v12 = sub_CB6200(v24, v29, v30);
  v13 = *(_DWORD **)(v12 + 32);
  if ( *(_QWORD *)(v12 + 24) - (_QWORD)v13 <= 3u )
  {
    sub_CB6200(v12, "\" {\n", 4u);
  }
  else
  {
    *v13 = 175841314;
    *(_QWORD *)(v12 + 32) += 4LL;
  }
  if ( v29 != v31 )
    j_j___libc_free_0((unsigned __int64)v29);
LABEL_16:
  v14 = (__int64)*a1;
  v15 = (__int64)*a1;
  if ( v2[1] )
  {
    v19 = *(_QWORD **)(v14 + 32);
    if ( *(_QWORD *)(v14 + 24) - (_QWORD)v19 <= 7u )
    {
      v14 = sub_CB6200(v15, "\tlabel=\"", 8u);
    }
    else
    {
      *v19 = 0x223D6C6562616C09LL;
      *(_QWORD *)(v14 + 32) += 8LL;
    }
    v25 = v14;
    v20 = v2;
  }
  else
  {
    if ( !v27 )
      goto LABEL_18;
    v23 = *(_QWORD **)(v14 + 32);
    if ( *(_QWORD *)(v14 + 24) - (_QWORD)v23 <= 7u )
    {
      v14 = sub_CB6200(v15, "\tlabel=\"", 8u);
    }
    else
    {
      *v23 = 0x223D6C6562616C09LL;
      *(_QWORD *)(v14 + 32) += 8LL;
    }
    v25 = v14;
    v20 = &v26;
  }
  sub_C67200((__int64 *)&v29, (__int64)v20);
  v21 = sub_CB6200(v25, v29, v30);
  v22 = *(_QWORD *)(v21 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v21 + 24) - v22) <= 2 )
  {
    sub_CB6200(v21, (unsigned __int8 *)"\";\n", 3u);
  }
  else
  {
    *(_BYTE *)(v22 + 2) = 10;
    *(_WORD *)v22 = 15138;
    *(_QWORD *)(v21 + 32) += 3LL;
  }
  if ( v29 != v31 )
    j_j___libc_free_0((unsigned __int64)v29);
  v15 = (__int64)*a1;
LABEL_18:
  v29 = v31;
  v30 = 0;
  v31[0] = 0;
  sub_CB6200(v15, v31, 0);
  if ( v29 != v31 )
    j_j___libc_free_0((unsigned __int64)v29);
  v16 = (__int64)*a1;
  v17 = (*a1)[4];
  if ( (*a1)[3] == (__int64 **)v17 )
  {
    sub_CB6200(v16, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v17 = 10;
    ++*(_QWORD *)(v16 + 32);
  }
  if ( v26 != &v28 )
    j_j___libc_free_0((unsigned __int64)v26);
}
