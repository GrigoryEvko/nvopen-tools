// Function: sub_22E6E00
// Address: 0x22e6e00
//
void __fastcall sub_22E6E00(__int64 *a1, _QWORD **a2)
{
  _QWORD **v2; // r12
  __int64 v4; // r13
  __int64 v5; // rdx
  __int64 v6; // rax
  unsigned __int64 v7; // rax
  __int64 v8; // rax
  _DWORD *v9; // rdx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rdi
  _BYTE *v13; // rax
  __m128i si128; // xmm0
  _QWORD *v15; // rdx
  _QWORD **v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rdx
  _QWORD *v19; // rdx
  __int64 v20; // [rsp+0h] [rbp-80h]
  __int64 v21; // [rsp+0h] [rbp-80h]
  _QWORD *v22; // [rsp+10h] [rbp-70h] BYREF
  __int64 v23; // [rsp+18h] [rbp-68h]
  _QWORD v24[2]; // [rsp+20h] [rbp-60h] BYREF
  unsigned __int8 *v25; // [rsp+30h] [rbp-50h] BYREF
  size_t v26; // [rsp+38h] [rbp-48h]
  _QWORD v27[8]; // [rsp+40h] [rbp-40h] BYREF

  v2 = a2;
  v22 = v24;
  sub_22E4AB0((__int64 *)&v22, "Region Graph", (__int64)"");
  v4 = *a1;
  v5 = *(_QWORD *)(*a1 + 32);
  v6 = *(_QWORD *)(*a1 + 24);
  if ( a2[1] )
  {
    if ( (unsigned __int64)(v6 - v5) <= 8 )
    {
      v4 = sub_CB6200(*a1, "digraph \"", 9u);
    }
    else
    {
      *(_BYTE *)(v5 + 8) = 34;
      *(_QWORD *)v5 = 0x2068706172676964LL;
      *(_QWORD *)(v4 + 32) += 9LL;
    }
  }
  else
  {
    v7 = v6 - v5;
    if ( !v23 )
    {
      if ( v7 <= 0x11 )
      {
        sub_CB6200(*a1, "digraph unnamed {\n", 0x12u);
      }
      else
      {
        si128 = _mm_load_si128(xmmword_3F8CB00);
        *(_WORD *)(v5 + 16) = 2683;
        *(__m128i *)v5 = si128;
        *(_QWORD *)(v4 + 32) += 18LL;
      }
      goto LABEL_10;
    }
    if ( v7 <= 8 )
    {
      v4 = sub_CB6200(*a1, "digraph \"", 9u);
    }
    else
    {
      *(_BYTE *)(v5 + 8) = 34;
      *(_QWORD *)v5 = 0x2068706172676964LL;
      *(_QWORD *)(v4 + 32) += 9LL;
    }
    a2 = &v22;
  }
  sub_C67200((__int64 *)&v25, (__int64)a2);
  v8 = sub_CB6200(v4, v25, v26);
  v9 = *(_DWORD **)(v8 + 32);
  if ( *(_QWORD *)(v8 + 24) - (_QWORD)v9 <= 3u )
  {
    sub_CB6200(v8, "\" {\n", 4u);
  }
  else
  {
    *v9 = 175841314;
    *(_QWORD *)(v8 + 32) += 4LL;
  }
  if ( v25 != (unsigned __int8 *)v27 )
    j_j___libc_free_0((unsigned __int64)v25);
LABEL_10:
  v10 = *a1;
  v11 = *a1;
  if ( v2[1] )
  {
    v15 = *(_QWORD **)(v10 + 32);
    if ( *(_QWORD *)(v10 + 24) - (_QWORD)v15 <= 7u )
    {
      v10 = sub_CB6200(*a1, "\tlabel=\"", 8u);
    }
    else
    {
      *v15 = 0x223D6C6562616C09LL;
      *(_QWORD *)(v10 + 32) += 8LL;
    }
    v21 = v10;
    v16 = v2;
  }
  else
  {
    if ( !v23 )
      goto LABEL_12;
    v19 = *(_QWORD **)(v10 + 32);
    if ( *(_QWORD *)(v10 + 24) - (_QWORD)v19 <= 7u )
    {
      v10 = sub_CB6200(*a1, "\tlabel=\"", 8u);
    }
    else
    {
      *v19 = 0x223D6C6562616C09LL;
      *(_QWORD *)(v10 + 32) += 8LL;
    }
    v21 = v10;
    v16 = &v22;
  }
  sub_C67200((__int64 *)&v25, (__int64)v16);
  v17 = sub_CB6200(v21, v25, v26);
  v18 = *(_QWORD *)(v17 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v17 + 24) - v18) <= 2 )
  {
    sub_CB6200(v17, (unsigned __int8 *)"\";\n", 3u);
  }
  else
  {
    *(_BYTE *)(v18 + 2) = 10;
    *(_WORD *)v18 = 15138;
    *(_QWORD *)(v17 + 32) += 3LL;
  }
  if ( v25 != (unsigned __int8 *)v27 )
    j_j___libc_free_0((unsigned __int64)v25);
  v11 = *a1;
LABEL_12:
  v20 = v11;
  v25 = (unsigned __int8 *)v27;
  sub_22E4AB0((__int64 *)&v25, byte_3F871B3, (__int64)byte_3F871B3);
  sub_CB6200(v20, v25, v26);
  if ( v25 != (unsigned __int8 *)v27 )
    j_j___libc_free_0((unsigned __int64)v25);
  v12 = *a1;
  v13 = *(_BYTE **)(*a1 + 32);
  if ( *(_BYTE **)(*a1 + 24) == v13 )
  {
    sub_CB6200(v12, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v13 = 10;
    ++*(_QWORD *)(v12 + 32);
  }
  if ( v22 != v24 )
    j_j___libc_free_0((unsigned __int64)v22);
}
