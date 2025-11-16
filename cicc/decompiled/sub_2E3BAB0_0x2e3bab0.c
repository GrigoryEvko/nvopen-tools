// Function: sub_2E3BAB0
// Address: 0x2e3bab0
//
void __fastcall sub_2E3BAB0(__int64 ***a1, _QWORD **a2)
{
  _QWORD **v2; // r12
  __int64 v4; // rax
  _BYTE *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r13
  __int64 *v8; // rdx
  __int64 *v9; // rax
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  _DWORD *v12; // rdx
  __int64 v13; // r8
  __int64 v14; // rdi
  __int64 v15; // rdi
  __int64 *v16; // rax
  __m128i si128; // xmm0
  _QWORD *v18; // rdx
  _QWORD **v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rdx
  _QWORD *v22; // rdx
  __int64 v23; // [rsp+8h] [rbp-78h]
  _QWORD *v24; // [rsp+10h] [rbp-70h] BYREF
  __int64 v25; // [rsp+18h] [rbp-68h]
  _QWORD v26[2]; // [rsp+20h] [rbp-60h] BYREF
  unsigned __int8 *v27; // [rsp+30h] [rbp-50h] BYREF
  size_t v28; // [rsp+38h] [rbp-48h]
  unsigned __int8 v29[64]; // [rsp+40h] [rbp-40h] BYREF

  v2 = a2;
  v4 = sub_2E3A060(*a1[1]);
  v5 = (_BYTE *)sub_2E791E0(v4);
  v24 = v26;
  sub_2E396D0((__int64 *)&v24, v5, (__int64)&v5[v6]);
  v7 = (__int64)*a1;
  v8 = (*a1)[4];
  v9 = (*a1)[3];
  if ( a2[1] )
  {
    if ( (unsigned __int64)((char *)v9 - (char *)v8) <= 8 )
    {
      v7 = sub_CB6200((__int64)*a1, "digraph \"", 9u);
    }
    else
    {
      *((_BYTE *)v8 + 8) = 34;
      *v8 = 0x2068706172676964LL;
      *(_QWORD *)(v7 + 32) += 9LL;
    }
  }
  else
  {
    v10 = (char *)v9 - (char *)v8;
    if ( !v25 )
    {
      if ( v10 <= 0x11 )
      {
        sub_CB6200((__int64)*a1, "digraph unnamed {\n", 0x12u);
      }
      else
      {
        si128 = _mm_load_si128(xmmword_3F8CB00);
        *((_WORD *)v8 + 8) = 2683;
        *(__m128i *)v8 = si128;
        *(_QWORD *)(v7 + 32) += 18LL;
      }
      goto LABEL_10;
    }
    if ( v10 <= 8 )
    {
      v7 = sub_CB6200((__int64)*a1, "digraph \"", 9u);
    }
    else
    {
      *((_BYTE *)v8 + 8) = 34;
      *v8 = 0x2068706172676964LL;
      *(_QWORD *)(v7 + 32) += 9LL;
    }
    a2 = &v24;
  }
  sub_C67200((__int64 *)&v27, (__int64)a2);
  v11 = sub_CB6200(v7, v27, v28);
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
  if ( v27 != v29 )
    j_j___libc_free_0((unsigned __int64)v27);
LABEL_10:
  v13 = (__int64)*a1;
  v14 = (__int64)*a1;
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
    v23 = v13;
    v19 = v2;
  }
  else
  {
    if ( !v25 )
      goto LABEL_12;
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
    v23 = v13;
    v19 = &v24;
  }
  sub_C67200((__int64 *)&v27, (__int64)v19);
  v20 = sub_CB6200(v23, v27, v28);
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
  if ( v27 != v29 )
    j_j___libc_free_0((unsigned __int64)v27);
  v14 = (__int64)*a1;
LABEL_12:
  v27 = v29;
  v28 = 0;
  v29[0] = 0;
  sub_CB6200(v14, v29, 0);
  if ( v27 != v29 )
    j_j___libc_free_0((unsigned __int64)v27);
  v15 = (__int64)*a1;
  v16 = (*a1)[4];
  if ( (*a1)[3] == v16 )
  {
    sub_CB6200(v15, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *(_BYTE *)v16 = 10;
    ++*(_QWORD *)(v15 + 32);
  }
  if ( v24 != v26 )
    j_j___libc_free_0((unsigned __int64)v24);
}
