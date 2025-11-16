// Function: sub_13B8800
// Address: 0x13b8800
//
_BYTE *__fastcall sub_13B8800(__int64 *a1, _QWORD **a2)
{
  _QWORD **v2; // r12
  __int64 v4; // r13
  bool v5; // zf
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rax
  _DWORD *v9; // rdx
  __int64 v10; // r8
  __int64 v11; // rdi
  __int64 v12; // rdi
  _BYTE *result; // rax
  _QWORD *v14; // rdx
  _QWORD *v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rdx
  _QWORD *v18; // rdx
  __int64 v19; // [rsp+8h] [rbp-78h]
  _QWORD *v20; // [rsp+10h] [rbp-70h] BYREF
  __int64 v21; // [rsp+18h] [rbp-68h]
  _QWORD v22[2]; // [rsp+20h] [rbp-60h] BYREF
  const char *v23; // [rsp+30h] [rbp-50h] BYREF
  __int64 v24; // [rsp+38h] [rbp-48h]
  char v25[64]; // [rsp+40h] [rbp-40h] BYREF

  v2 = a2;
  v4 = *a1;
  v5 = a2[1] == 0;
  strcpy((char *)v22, "Dominator tree");
  v6 = *(_QWORD *)(v4 + 24);
  v20 = v22;
  v7 = *(_QWORD *)(v4 + 16);
  v21 = 14;
  if ( v5 )
  {
    if ( (unsigned __int64)(v7 - v6) <= 8 )
    {
      v4 = sub_16E7EE0(v4, "digraph \"", 9);
    }
    else
    {
      *(_BYTE *)(v6 + 8) = 34;
      *(_QWORD *)v6 = 0x2068706172676964LL;
      *(_QWORD *)(v4 + 24) += 9LL;
    }
    a2 = &v20;
  }
  else if ( (unsigned __int64)(v7 - v6) <= 8 )
  {
    v4 = sub_16E7EE0(v4, "digraph \"", 9);
  }
  else
  {
    *(_BYTE *)(v6 + 8) = 34;
    *(_QWORD *)v6 = 0x2068706172676964LL;
    *(_QWORD *)(v4 + 24) += 9LL;
  }
  sub_16BE9B0(&v23, a2);
  v8 = sub_16E7EE0(v4, v23, v24);
  v9 = *(_DWORD **)(v8 + 24);
  if ( *(_QWORD *)(v8 + 16) - (_QWORD)v9 <= 3u )
  {
    sub_16E7EE0(v8, "\" {\n", 4);
  }
  else
  {
    *v9 = 175841314;
    *(_QWORD *)(v8 + 24) += 4LL;
  }
  if ( v23 != v25 )
    j_j___libc_free_0(v23, *(_QWORD *)v25 + 1LL);
  v10 = *a1;
  v11 = *a1;
  if ( v2[1] )
  {
    v14 = *(_QWORD **)(v10 + 24);
    if ( *(_QWORD *)(v10 + 16) - (_QWORD)v14 <= 7u )
    {
      v10 = sub_16E7EE0(v11, "\tlabel=\"", 8);
    }
    else
    {
      *v14 = 0x223D6C6562616C09LL;
      *(_QWORD *)(v10 + 24) += 8LL;
    }
    v19 = v10;
    v15 = v2;
  }
  else
  {
    if ( !v21 )
      goto LABEL_13;
    v18 = *(_QWORD **)(v10 + 24);
    if ( *(_QWORD *)(v10 + 16) - (_QWORD)v18 <= 7u )
    {
      v10 = sub_16E7EE0(v11, "\tlabel=\"", 8);
    }
    else
    {
      *v18 = 0x223D6C6562616C09LL;
      *(_QWORD *)(v10 + 24) += 8LL;
    }
    v19 = v10;
    v15 = &v20;
  }
  sub_16BE9B0(&v23, v15);
  v16 = sub_16E7EE0(v19, v23, v24);
  v17 = *(_QWORD *)(v16 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(v16 + 16) - v17) <= 2 )
  {
    sub_16E7EE0(v16, "\";\n", 3);
  }
  else
  {
    *(_BYTE *)(v17 + 2) = 10;
    *(_WORD *)v17 = 15138;
    *(_QWORD *)(v16 + 24) += 3LL;
  }
  if ( v23 != v25 )
    j_j___libc_free_0(v23, *(_QWORD *)v25 + 1LL);
  v11 = *a1;
LABEL_13:
  v23 = v25;
  v24 = 0;
  v25[0] = 0;
  sub_16E7EE0(v11, v25, 0);
  if ( v23 != v25 )
    j_j___libc_free_0(v23, *(_QWORD *)v25 + 1LL);
  v12 = *a1;
  result = *(_BYTE **)(*a1 + 24);
  if ( *(_BYTE **)(*a1 + 16) == result )
  {
    result = (_BYTE *)sub_16E7EE0(v12, "\n", 1);
  }
  else
  {
    *result = 10;
    ++*(_QWORD *)(v12 + 24);
  }
  if ( v20 != v22 )
    return (_BYTE *)j_j___libc_free_0(v20, v22[0] + 1LL);
  return result;
}
