// Function: sub_B7FAA0
// Address: 0xb7faa0
//
unsigned __int64 __fastcall sub_B7FAA0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  void *v4; // rdx
  __int64 v5; // rax
  size_t v6; // rdx
  _BYTE *v7; // rdi
  const void *v8; // rsi
  _BYTE *v9; // rax
  size_t v10; // r13
  __int64 v11; // r14
  unsigned __int64 v12; // rax
  _BYTE *v13; // rdx
  char v14; // al
  __int64 v15; // rdx
  _WORD *v16; // rdx
  _WORD *v17; // rdx
  unsigned __int64 result; // rax
  _BYTE *v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rdi
  _QWORD *v23; // rdx

  v2 = a2;
  if ( *(_QWORD *)(a1 + 24) || *(_QWORD *)(a1 + 32) )
  {
    v4 = *(void **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v4 <= 0xDu )
    {
      sub_CB6200(a2, "Running pass '", 14);
    }
    else
    {
      qmemcpy(v4, "Running pass '", 14);
      *(_QWORD *)(a2 + 32) += 14LL;
    }
  }
  else
  {
    sub_904010(a2, "Releasing pass '");
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 16) + 16LL))(*(_QWORD *)(a1 + 16));
  v7 = *(_BYTE **)(a2 + 32);
  v8 = (const void *)v5;
  v9 = *(_BYTE **)(v2 + 24);
  v10 = v6;
  if ( v9 - v7 < v6 )
  {
    v11 = sub_CB6200(v2, v8, v6);
    v9 = *(_BYTE **)(v11 + 24);
    v7 = *(_BYTE **)(v11 + 32);
  }
  else
  {
    v11 = v2;
    if ( v6 )
    {
      memcpy(v7, v8, v6);
      v19 = *(_BYTE **)(v2 + 24);
      v7 = (_BYTE *)(v10 + *(_QWORD *)(v2 + 32));
      *(_QWORD *)(v2 + 32) = v7;
      if ( v19 != v7 )
        goto LABEL_7;
      goto LABEL_20;
    }
  }
  if ( v9 != v7 )
  {
LABEL_7:
    *v7 = 39;
    ++*(_QWORD *)(v11 + 32);
    goto LABEL_8;
  }
LABEL_20:
  sub_CB6200(v11, "'", 1);
LABEL_8:
  v12 = *(_QWORD *)(v2 + 24);
  v13 = *(_BYTE **)(v2 + 32);
  if ( !*(_QWORD *)(a1 + 32) )
  {
    if ( !*(_QWORD *)(a1 + 24) )
    {
      if ( (unsigned __int64)v13 >= v12 )
        return sub_CB5D20(v2, 10);
      *(_QWORD *)(v2 + 32) = v13 + 1;
      *v13 = 10;
      return (unsigned __int64)(v13 + 1);
    }
    if ( v12 - (unsigned __int64)v13 <= 3 )
    {
      sub_CB6200(v2, " on ", 4);
      v14 = **(_BYTE **)(a1 + 24);
      if ( v14 )
      {
LABEL_12:
        if ( v14 == 23 )
        {
          sub_904010(v2, "basic block");
          v16 = *(_WORD **)(v2 + 32);
        }
        else
        {
          v15 = *(_QWORD *)(v2 + 32);
          if ( (unsigned __int64)(*(_QWORD *)(v2 + 24) - v15) <= 4 )
          {
            sub_CB6200(v2, "value", 5);
            v16 = *(_WORD **)(v2 + 32);
          }
          else
          {
            *(_DWORD *)v15 = 1970037110;
            *(_BYTE *)(v15 + 4) = 101;
            v16 = (_WORD *)(*(_QWORD *)(v2 + 32) + 5LL);
            *(_QWORD *)(v2 + 32) = v16;
          }
        }
        goto LABEL_15;
      }
    }
    else
    {
      *(_DWORD *)v13 = 544108320;
      *(_QWORD *)(v2 + 32) += 4LL;
      v14 = **(_BYTE **)(a1 + 24);
      if ( v14 )
        goto LABEL_12;
    }
    v23 = *(_QWORD **)(v2 + 32);
    if ( *(_QWORD *)(v2 + 24) - (_QWORD)v23 <= 7u )
    {
      sub_CB6200(v2, "function", 8);
      v16 = *(_WORD **)(v2 + 32);
    }
    else
    {
      *v23 = 0x6E6F6974636E7566LL;
      v16 = (_WORD *)(*(_QWORD *)(v2 + 32) + 8LL);
      *(_QWORD *)(v2 + 32) = v16;
    }
LABEL_15:
    if ( *(_QWORD *)(v2 + 24) - (_QWORD)v16 <= 1u )
    {
      sub_CB6200(v2, " '", 2);
    }
    else
    {
      *v16 = 10016;
      *(_QWORD *)(v2 + 32) += 2LL;
    }
    sub_A5BF40(*(unsigned __int8 **)(a1 + 24), v2, 0, *(_QWORD *)(a1 + 32));
    v17 = *(_WORD **)(v2 + 32);
    if ( *(_QWORD *)(v2 + 24) - (_QWORD)v17 <= 1u )
      return sub_CB6200(v2, "'\n", 2);
    *v17 = 2599;
    *(_QWORD *)(v2 + 32) += 2LL;
    return 2599;
  }
  if ( v12 - (unsigned __int64)v13 <= 0xB )
  {
    v2 = sub_CB6200(v2, " on module '", 12);
  }
  else
  {
    qmemcpy(v13, " on module '", 12);
    *(_QWORD *)(v2 + 32) += 12LL;
  }
  v20 = sub_CB6200(v2, *(_QWORD *)(*(_QWORD *)(a1 + 32) + 168LL), *(_QWORD *)(*(_QWORD *)(a1 + 32) + 176LL));
  v21 = *(_QWORD *)(v20 + 32);
  v22 = v20;
  result = *(_QWORD *)(v20 + 24) - v21;
  if ( result <= 2 )
    return sub_CB6200(v22, "'.\n", 3);
  *(_BYTE *)(v21 + 2) = 10;
  *(_WORD *)v21 = 11815;
  *(_QWORD *)(v22 + 32) += 3LL;
  return result;
}
