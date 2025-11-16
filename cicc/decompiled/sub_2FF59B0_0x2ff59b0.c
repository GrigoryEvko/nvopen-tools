// Function: sub_2FF59B0
// Address: 0x2ff59b0
//
__int64 __fastcall sub_2FF59B0(unsigned int *a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 v5; // rdx
  _QWORD *v6; // rdx
  unsigned __int16 *v8; // rdx
  unsigned __int16 v9; // r13
  char *v10; // rsi
  size_t v11; // rax
  void *v12; // rdi
  size_t v13; // rbx
  _BYTE *v14; // rax
  char *v15; // rsi
  size_t v16; // rax
  void *v17; // rdi
  size_t v18; // r14
  __int64 v19; // rdx
  __int64 v20; // [rsp+0h] [rbp-30h]

  v3 = a2;
  v4 = *((_QWORD *)a1 + 1);
  if ( !v4 )
  {
    v19 = *(_QWORD *)(a2 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v19) <= 4 )
    {
      v3 = sub_CB6200(a2, (unsigned __int8 *)"Unit~", 5u);
    }
    else
    {
      *(_DWORD *)v19 = 1953066581;
      *(_BYTE *)(v19 + 4) = 126;
      *(_QWORD *)(a2 + 32) += 5LL;
    }
    return sub_CB59D0(v3, *a1);
  }
  v5 = *a1;
  if ( *(_DWORD *)(v4 + 44) <= (unsigned int)v5 )
  {
    v6 = *(_QWORD **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v6 <= 7u )
    {
      v3 = sub_CB6200(a2, "BadUnit~", 8u);
    }
    else
    {
      *v6 = 0x7E74696E55646142LL;
      *(_QWORD *)(a2 + 32) += 8LL;
    }
    return sub_CB59D0(v3, *a1);
  }
  v8 = (unsigned __int16 *)(*(_QWORD *)(v4 + 48) + 4 * v5);
  v9 = v8[1];
  v10 = (char *)(*(_QWORD *)(v4 + 72) + *(unsigned int *)(*(_QWORD *)(v4 + 8) + 24LL * *v8));
  if ( v10 )
  {
    v11 = strlen(v10);
    v12 = *(void **)(v3 + 32);
    v13 = v11;
    if ( v11 > *(_QWORD *)(v3 + 24) - (_QWORD)v12 )
    {
      sub_CB6200(v3, (unsigned __int8 *)v10, v11);
    }
    else if ( v11 )
    {
      memcpy(v12, v10, v11);
      *(_QWORD *)(v3 + 32) += v13;
    }
  }
  if ( !v9 )
    return v20;
  v14 = *(_BYTE **)(v3 + 32);
  if ( (unsigned __int64)v14 >= *(_QWORD *)(v3 + 24) )
  {
    v3 = sub_CB5D20(v3, 126);
  }
  else
  {
    *(_QWORD *)(v3 + 32) = v14 + 1;
    *v14 = 126;
  }
  v15 = (char *)(*(_QWORD *)(*((_QWORD *)a1 + 1) + 72LL)
               + *(unsigned int *)(*(_QWORD *)(*((_QWORD *)a1 + 1) + 8LL) + 24LL * v9));
  if ( !v15 )
    return v20;
  v16 = strlen(v15);
  v17 = *(void **)(v3 + 32);
  v18 = v16;
  if ( v16 <= *(_QWORD *)(v3 + 24) - (_QWORD)v17 )
  {
    if ( v16 )
    {
      memcpy(v17, v15, v16);
      *(_QWORD *)(v3 + 32) += v18;
    }
    return v20;
  }
  return sub_CB6200(v3, (unsigned __int8 *)v15, v16);
}
