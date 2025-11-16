// Function: sub_136A2B0
// Address: 0x136a2b0
//
__int64 *__fastcall sub_136A2B0(__int64 ***a1, __int64 *a2)
{
  __int64 *v2; // r12
  __int64 v4; // rax
  _BYTE *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r13
  __int64 *v8; // rdx
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  _DWORD *v11; // rdx
  __int64 v12; // r8
  __int64 **v13; // r9
  __int64 **v14; // rdi
  __int64 *result; // rax
  __m128i si128; // xmm0
  _QWORD *v17; // rdx
  __int64 *v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rdx
  _QWORD *v21; // rdx
  __int64 **v22; // [rsp+8h] [rbp-78h]
  __int64 v23; // [rsp+8h] [rbp-78h]
  _QWORD *v24; // [rsp+10h] [rbp-70h] BYREF
  __int64 v25; // [rsp+18h] [rbp-68h]
  _QWORD v26[2]; // [rsp+20h] [rbp-60h] BYREF
  const char *v27; // [rsp+30h] [rbp-50h] BYREF
  __int64 v28; // [rsp+38h] [rbp-48h]
  _QWORD v29[8]; // [rsp+40h] [rbp-40h] BYREF

  v2 = a2;
  v4 = sub_1368BD0(*a1[1]);
  v5 = (_BYTE *)sub_1649960(v4);
  if ( v5 )
  {
    v24 = v26;
    sub_1367D20((__int64 *)&v24, v5, (__int64)&v5[v6]);
  }
  else
  {
    v25 = 0;
    v24 = v26;
    LOBYTE(v26[0]) = 0;
  }
  v7 = (__int64)*a1;
  v8 = (*a1)[3];
  v9 = (char *)(*a1)[2] - (char *)v8;
  if ( a2[1] )
  {
    if ( v9 <= 8 )
    {
      v7 = sub_16E7EE0(*a1, "digraph \"", 9);
    }
    else
    {
      *((_BYTE *)v8 + 8) = 34;
      *v8 = 0x2068706172676964LL;
      *(_QWORD *)(v7 + 24) += 9LL;
    }
  }
  else
  {
    if ( !v25 )
    {
      if ( v9 <= 0x11 )
      {
        sub_16E7EE0(*a1, "digraph unnamed {\n", 18);
      }
      else
      {
        si128 = _mm_load_si128(xmmword_3F8CB00);
        *((_WORD *)v8 + 8) = 2683;
        *(__m128i *)v8 = si128;
        *(_QWORD *)(v7 + 24) += 18LL;
      }
      goto LABEL_12;
    }
    if ( v9 <= 8 )
    {
      v7 = sub_16E7EE0(*a1, "digraph \"", 9);
    }
    else
    {
      *((_BYTE *)v8 + 8) = 34;
      *v8 = 0x2068706172676964LL;
      *(_QWORD *)(v7 + 24) += 9LL;
    }
    a2 = (__int64 *)&v24;
  }
  sub_16BE9B0(&v27, a2);
  v10 = sub_16E7EE0(v7, v27, v28);
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
  if ( v27 != (const char *)v29 )
    j_j___libc_free_0(v27, v29[0] + 1LL);
LABEL_12:
  v12 = (__int64)*a1;
  v13 = *a1;
  if ( v2[1] )
  {
    v17 = *(_QWORD **)(v12 + 24);
    if ( *(_QWORD *)(v12 + 16) - (_QWORD)v17 <= 7u )
    {
      v12 = sub_16E7EE0(*a1, "\tlabel=\"", 8);
    }
    else
    {
      *v17 = 0x223D6C6562616C09LL;
      *(_QWORD *)(v12 + 24) += 8LL;
    }
    v23 = v12;
    v18 = v2;
  }
  else
  {
    if ( !v25 )
      goto LABEL_14;
    v21 = *(_QWORD **)(v12 + 24);
    if ( *(_QWORD *)(v12 + 16) - (_QWORD)v21 <= 7u )
    {
      v12 = sub_16E7EE0(*a1, "\tlabel=\"", 8);
    }
    else
    {
      *v21 = 0x223D6C6562616C09LL;
      *(_QWORD *)(v12 + 24) += 8LL;
    }
    v23 = v12;
    v18 = (__int64 *)&v24;
  }
  sub_16BE9B0(&v27, v18);
  v19 = sub_16E7EE0(v23, v27, v28);
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
  if ( v27 != (const char *)v29 )
    j_j___libc_free_0(v27, v29[0] + 1LL);
  v13 = *a1;
LABEL_14:
  v22 = v13;
  v27 = (const char *)v29;
  sub_1367D20((__int64 *)&v27, byte_3F871B3, (__int64)byte_3F871B3);
  sub_16E7EE0(v22, v27, v28);
  if ( v27 != (const char *)v29 )
    j_j___libc_free_0(v27, v29[0] + 1LL);
  v14 = *a1;
  result = (*a1)[3];
  if ( (*a1)[2] == result )
  {
    result = (__int64 *)sub_16E7EE0(v14, "\n", 1);
  }
  else
  {
    *(_BYTE *)result = 10;
    v14[3] = (__int64 *)((char *)v14[3] + 1);
  }
  if ( v24 != v26 )
    return (__int64 *)j_j___libc_free_0(v24, v26[0] + 1LL);
  return result;
}
