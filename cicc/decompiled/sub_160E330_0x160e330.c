// Function: sub_160E330
// Address: 0x160e330
//
__int64 __fastcall sub_160E330(__int64 a1, __int64 a2)
{
  void *v4; // rdx
  __int64 v5; // rax
  size_t v6; // rdx
  _BYTE *v7; // rdi
  const char *v8; // rsi
  _BYTE *v9; // rax
  size_t v10; // r13
  __int64 v11; // r14
  unsigned __int64 v12; // rax
  _BYTE *v13; // rdx
  char v14; // al
  _WORD *v15; // rdx
  _WORD *v16; // rdx
  _BYTE *v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax

  if ( *(_QWORD *)(a1 + 24) || *(_QWORD *)(a1 + 32) )
  {
    v4 = *(void **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v4 <= 0xDu )
    {
      sub_16E7EE0(a2, "Running pass '", 14);
    }
    else
    {
      qmemcpy(v4, "Running pass '", 14);
      *(_QWORD *)(a2 + 24) += 14LL;
    }
  }
  else
  {
    sub_1263B40(a2, "Releasing pass '");
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 16) + 16LL))(*(_QWORD *)(a1 + 16));
  v7 = *(_BYTE **)(a2 + 24);
  v8 = (const char *)v5;
  v9 = *(_BYTE **)(a2 + 16);
  v10 = v6;
  if ( v9 - v7 < v6 )
  {
    v11 = sub_16E7EE0(a2, v8, v6);
    v9 = *(_BYTE **)(v11 + 16);
    v7 = *(_BYTE **)(v11 + 24);
LABEL_6:
    if ( v9 != v7 )
      goto LABEL_7;
    goto LABEL_18;
  }
  v11 = a2;
  if ( !v6 )
    goto LABEL_6;
  memcpy(v7, v8, v6);
  v18 = *(_BYTE **)(a2 + 16);
  v7 = (_BYTE *)(v10 + *(_QWORD *)(a2 + 24));
  *(_QWORD *)(a2 + 24) = v7;
  if ( v18 != v7 )
  {
LABEL_7:
    *v7 = 39;
    ++*(_QWORD *)(v11 + 24);
    if ( !*(_QWORD *)(a1 + 32) )
      goto LABEL_8;
LABEL_19:
    v19 = sub_1263B40(a2, " on module '");
    v20 = sub_16E7EE0(v19, *(const char **)(*(_QWORD *)(a1 + 32) + 176LL), *(_QWORD *)(*(_QWORD *)(a1 + 32) + 184LL));
    return sub_1263B40(v20, "'.\n");
  }
LABEL_18:
  sub_16E7EE0(v11, "'", 1);
  if ( *(_QWORD *)(a1 + 32) )
    goto LABEL_19;
LABEL_8:
  v12 = *(_QWORD *)(a2 + 16);
  v13 = *(_BYTE **)(a2 + 24);
  if ( !*(_QWORD *)(a1 + 24) )
  {
    if ( (unsigned __int64)v13 >= v12 )
      return sub_16E7DE0(a2, 10);
    *(_QWORD *)(a2 + 24) = v13 + 1;
    *v13 = 10;
    return (__int64)(v13 + 1);
  }
  if ( v12 - (unsigned __int64)v13 > 3 )
  {
    *(_DWORD *)v13 = 544108320;
    *(_QWORD *)(a2 + 24) += 4LL;
    v14 = *(_BYTE *)(*(_QWORD *)(a1 + 24) + 16LL);
    if ( v14 )
      goto LABEL_11;
LABEL_25:
    sub_1263B40(a2, "function");
    goto LABEL_13;
  }
  sub_16E7EE0(a2, " on ", 4);
  v14 = *(_BYTE *)(*(_QWORD *)(a1 + 24) + 16LL);
  if ( !v14 )
    goto LABEL_25;
LABEL_11:
  if ( v14 == 18 )
    sub_1263B40(a2, "basic block");
  else
    sub_1263B40(a2, "value");
LABEL_13:
  v15 = *(_WORD **)(a2 + 24);
  if ( *(_QWORD *)(a2 + 16) - (_QWORD)v15 <= 1u )
  {
    sub_16E7EE0(a2, " '", 2);
  }
  else
  {
    *v15 = 10016;
    *(_QWORD *)(a2 + 24) += 2LL;
  }
  sub_15537D0(*(_QWORD *)(a1 + 24), a2, 0, *(_QWORD **)(a1 + 32));
  v16 = *(_WORD **)(a2 + 24);
  if ( *(_QWORD *)(a2 + 16) - (_QWORD)v16 <= 1u )
    return sub_16E7EE0(a2, "'\n", 2);
  *v16 = 2599;
  *(_QWORD *)(a2 + 24) += 2LL;
  return 2599;
}
