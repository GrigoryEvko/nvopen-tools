// Function: sub_35258A0
// Address: 0x35258a0
//
void __fastcall sub_35258A0(__int64 ****a1, __int64 a2)
{
  char *v4; // rax
  __int64 v5; // rdx
  __m128i *v6; // rax
  __int64 v7; // rsi
  __m128i *v8; // rax
  __int64 v9; // rsi
  __m128i *v10; // rdi
  __int64 v11; // r8
  __int64 **v12; // rdx
  __int64 **v13; // rax
  unsigned __int64 v14; // rax
  __m128i **v15; // rsi
  __int64 v16; // rax
  _DWORD *v17; // rdx
  __int64 v18; // r8
  __int64 v19; // rdi
  __int64 v20; // rdi
  _BYTE *v21; // rax
  __m128i si128; // xmm0
  _QWORD *v23; // rdx
  __m128i **v24; // rsi
  __int64 v25; // rax
  __int64 v26; // rdx
  _QWORD *v27; // rdx
  __int64 v28; // [rsp+8h] [rbp-98h]
  __int64 v29; // [rsp+8h] [rbp-98h]
  __m128i *v30; // [rsp+10h] [rbp-90h] BYREF
  __int64 v31; // [rsp+18h] [rbp-88h]
  __m128i v32; // [rsp+20h] [rbp-80h] BYREF
  __m128i *v33; // [rsp+30h] [rbp-70h] BYREF
  __int64 v34; // [rsp+38h] [rbp-68h]
  __m128i v35; // [rsp+40h] [rbp-60h] BYREF
  unsigned __int8 *v36; // [rsp+50h] [rbp-50h] BYREF
  size_t v37; // [rsp+58h] [rbp-48h]
  unsigned __int8 v38[64]; // [rsp+60h] [rbp-40h] BYREF

  v4 = (char *)sub_2E791E0(**a1[1]);
  if ( v4 )
  {
    v36 = v38;
    sub_3525230((__int64 *)&v36, v4, (__int64)&v4[v5]);
  }
  else
  {
    v37 = 0;
    v36 = v38;
    v38[0] = 0;
  }
  v6 = (__m128i *)sub_2241130((unsigned __int64 *)&v36, 0, 0, "Machine CFG for '", 0x11u);
  v33 = &v35;
  if ( (__m128i *)v6->m128i_i64[0] == &v6[1] )
  {
    v35 = _mm_loadu_si128(v6 + 1);
  }
  else
  {
    v33 = (__m128i *)v6->m128i_i64[0];
    v35.m128i_i64[0] = v6[1].m128i_i64[0];
  }
  v7 = v6->m128i_i64[1];
  v6[1].m128i_i8[0] = 0;
  v34 = v7;
  v6->m128i_i64[0] = (__int64)v6[1].m128i_i64;
  v6->m128i_i64[1] = 0;
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v34) <= 9 )
    sub_4262D8((__int64)"basic_string::append");
  v8 = (__m128i *)sub_2241490((unsigned __int64 *)&v33, "' function", 0xAu);
  v30 = &v32;
  if ( (__m128i *)v8->m128i_i64[0] == &v8[1] )
  {
    v32 = _mm_loadu_si128(v8 + 1);
  }
  else
  {
    v30 = (__m128i *)v8->m128i_i64[0];
    v32.m128i_i64[0] = v8[1].m128i_i64[0];
  }
  v9 = v8->m128i_i64[1];
  v8[1].m128i_i8[0] = 0;
  v31 = v9;
  v8->m128i_i64[0] = (__int64)v8[1].m128i_i64;
  v10 = v33;
  v8->m128i_i64[1] = 0;
  if ( v10 != &v35 )
    j_j___libc_free_0((unsigned __int64)v10);
  if ( v36 != v38 )
    j_j___libc_free_0((unsigned __int64)v36);
  v11 = (__int64)*a1;
  v12 = (*a1)[4];
  v13 = (*a1)[3];
  if ( *(_QWORD *)(a2 + 8) )
  {
    if ( (unsigned __int64)((char *)v13 - (char *)v12) <= 8 )
    {
      v15 = (__m128i **)a2;
      v28 = sub_CB6200((__int64)*a1, "digraph \"", 9u);
    }
    else
    {
      *((_BYTE *)v12 + 8) = 34;
      v15 = (__m128i **)a2;
      *v12 = (__int64 *)0x2068706172676964LL;
      v28 = v11;
      *(_QWORD *)(v11 + 32) += 9LL;
    }
  }
  else
  {
    v14 = (char *)v13 - (char *)v12;
    if ( !v31 )
    {
      if ( v14 <= 0x11 )
      {
        sub_CB6200((__int64)*a1, "digraph unnamed {\n", 0x12u);
      }
      else
      {
        si128 = _mm_load_si128(xmmword_3F8CB00);
        *((_WORD *)v12 + 8) = 2683;
        *(__m128i *)v12 = si128;
        *(_QWORD *)(v11 + 32) += 18LL;
      }
      goto LABEL_21;
    }
    if ( v14 <= 8 )
    {
      v11 = sub_CB6200((__int64)*a1, "digraph \"", 9u);
    }
    else
    {
      *((_BYTE *)v12 + 8) = 34;
      *v12 = (__int64 *)0x2068706172676964LL;
      *(_QWORD *)(v11 + 32) += 9LL;
    }
    v28 = v11;
    v15 = &v30;
  }
  sub_C67200((__int64 *)&v36, (__int64)v15);
  v16 = sub_CB6200(v28, v36, v37);
  v17 = *(_DWORD **)(v16 + 32);
  if ( *(_QWORD *)(v16 + 24) - (_QWORD)v17 <= 3u )
  {
    sub_CB6200(v16, "\" {\n", 4u);
  }
  else
  {
    *v17 = 175841314;
    *(_QWORD *)(v16 + 32) += 4LL;
  }
  if ( v36 != v38 )
    j_j___libc_free_0((unsigned __int64)v36);
LABEL_21:
  v18 = (__int64)*a1;
  v19 = (__int64)*a1;
  if ( *(_QWORD *)(a2 + 8) )
  {
    v23 = *(_QWORD **)(v18 + 32);
    if ( *(_QWORD *)(v18 + 24) - (_QWORD)v23 <= 7u )
    {
      v18 = sub_CB6200(v19, "\tlabel=\"", 8u);
    }
    else
    {
      *v23 = 0x223D6C6562616C09LL;
      *(_QWORD *)(v18 + 32) += 8LL;
    }
    v29 = v18;
    v24 = (__m128i **)a2;
  }
  else
  {
    if ( !v31 )
      goto LABEL_23;
    v27 = *(_QWORD **)(v18 + 32);
    if ( *(_QWORD *)(v18 + 24) - (_QWORD)v27 <= 7u )
    {
      v18 = sub_CB6200(v19, "\tlabel=\"", 8u);
    }
    else
    {
      *v27 = 0x223D6C6562616C09LL;
      *(_QWORD *)(v18 + 32) += 8LL;
    }
    v29 = v18;
    v24 = &v30;
  }
  sub_C67200((__int64 *)&v36, (__int64)v24);
  v25 = sub_CB6200(v29, v36, v37);
  v26 = *(_QWORD *)(v25 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v25 + 24) - v26) <= 2 )
  {
    sub_CB6200(v25, (unsigned __int8 *)"\";\n", 3u);
  }
  else
  {
    *(_BYTE *)(v26 + 2) = 10;
    *(_WORD *)v26 = 15138;
    *(_QWORD *)(v25 + 32) += 3LL;
  }
  if ( v36 != v38 )
    j_j___libc_free_0((unsigned __int64)v36);
  v19 = (__int64)*a1;
LABEL_23:
  v36 = v38;
  v37 = 0;
  v38[0] = 0;
  sub_CB6200(v19, v38, 0);
  if ( v36 != v38 )
    j_j___libc_free_0((unsigned __int64)v36);
  v20 = (__int64)*a1;
  v21 = (*a1)[4];
  if ( (*a1)[3] == (__int64 **)v21 )
  {
    sub_CB6200(v20, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v21 = 10;
    ++*(_QWORD *)(v20 + 32);
  }
  if ( v30 != &v32 )
    j_j___libc_free_0((unsigned __int64)v30);
}
