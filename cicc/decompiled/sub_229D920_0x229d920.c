// Function: sub_229D920
// Address: 0x229d920
//
void __fastcall sub_229D920(__int64 *a1, char **a2)
{
  char **v2; // r12
  __int64 v4; // r13
  bool v5; // zf
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  _DWORD *v9; // rdx
  __int64 v10; // r8
  __int64 v11; // rdi
  __int64 v12; // rdi
  _BYTE *v13; // rax
  _QWORD *v14; // rdx
  char **v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rdx
  _QWORD *v18; // rdx
  __int64 v19; // [rsp+8h] [rbp-78h]
  char *v20; // [rsp+10h] [rbp-70h] BYREF
  __int64 v21; // [rsp+18h] [rbp-68h]
  char v22[16]; // [rsp+20h] [rbp-60h] BYREF
  unsigned __int8 *v23; // [rsp+30h] [rbp-50h] BYREF
  size_t v24; // [rsp+38h] [rbp-48h]
  unsigned __int8 v25[64]; // [rsp+40h] [rbp-40h] BYREF

  v2 = a2;
  v4 = *a1;
  v5 = a2[1] == 0;
  strcpy(v22, "Dominator tree");
  v6 = *(_QWORD *)(v4 + 32);
  v7 = *(_QWORD *)(v4 + 24);
  v20 = v22;
  v21 = 14;
  if ( v5 )
  {
    if ( (unsigned __int64)(v7 - v6) <= 8 )
    {
      v4 = sub_CB6200(v4, "digraph \"", 9u);
    }
    else
    {
      *(_BYTE *)(v6 + 8) = 34;
      *(_QWORD *)v6 = 0x2068706172676964LL;
      *(_QWORD *)(v4 + 32) += 9LL;
    }
    a2 = &v20;
  }
  else if ( (unsigned __int64)(v7 - v6) <= 8 )
  {
    v4 = sub_CB6200(v4, "digraph \"", 9u);
  }
  else
  {
    *(_BYTE *)(v6 + 8) = 34;
    *(_QWORD *)v6 = 0x2068706172676964LL;
    *(_QWORD *)(v4 + 32) += 9LL;
  }
  sub_C67200((__int64 *)&v23, (__int64)a2);
  v8 = sub_CB6200(v4, v23, v24);
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
  if ( v23 != v25 )
    j_j___libc_free_0((unsigned __int64)v23);
  v10 = *a1;
  v11 = *a1;
  if ( v2[1] )
  {
    v14 = *(_QWORD **)(v10 + 32);
    if ( *(_QWORD *)(v10 + 24) - (_QWORD)v14 <= 7u )
    {
      v10 = sub_CB6200(v11, "\tlabel=\"", 8u);
    }
    else
    {
      *v14 = 0x223D6C6562616C09LL;
      *(_QWORD *)(v10 + 32) += 8LL;
    }
    v19 = v10;
    v15 = v2;
  }
  else
  {
    if ( !v21 )
      goto LABEL_13;
    v18 = *(_QWORD **)(v10 + 32);
    if ( *(_QWORD *)(v10 + 24) - (_QWORD)v18 <= 7u )
    {
      v10 = sub_CB6200(v11, "\tlabel=\"", 8u);
    }
    else
    {
      *v18 = 0x223D6C6562616C09LL;
      *(_QWORD *)(v10 + 32) += 8LL;
    }
    v19 = v10;
    v15 = &v20;
  }
  sub_C67200((__int64 *)&v23, (__int64)v15);
  v16 = sub_CB6200(v19, v23, v24);
  v17 = *(_QWORD *)(v16 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v16 + 24) - v17) <= 2 )
  {
    sub_CB6200(v16, (unsigned __int8 *)"\";\n", 3u);
  }
  else
  {
    *(_BYTE *)(v17 + 2) = 10;
    *(_WORD *)v17 = 15138;
    *(_QWORD *)(v16 + 32) += 3LL;
  }
  if ( v23 != v25 )
    j_j___libc_free_0((unsigned __int64)v23);
  v11 = *a1;
LABEL_13:
  v23 = v25;
  v24 = 0;
  v25[0] = 0;
  sub_CB6200(v11, v25, 0);
  if ( v23 != v25 )
    j_j___libc_free_0((unsigned __int64)v23);
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
  if ( v20 != v22 )
    j_j___libc_free_0((unsigned __int64)v20);
}
