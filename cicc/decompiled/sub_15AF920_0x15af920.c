// Function: sub_15AF920
// Address: 0x15af920
//
bool __fastcall sub_15AF920(__int64 a1, __int64 a2)
{
  bool result; // al
  char *v5; // rdx
  __int64 v6; // rax
  char v7; // di
  int v8; // esi
  char *v9; // rax
  char v10; // cl
  char *v11; // rsi
  char *v12; // r8
  const char *v13; // r13
  __int64 v14; // rdx
  __int64 v15; // r14
  const char *v16; // rdi
  size_t v17; // rdx
  __int64 v18; // rdi
  const char *v19; // r12
  __int64 v20; // rdx
  __int64 v21; // r13
  __int64 v22; // rdi
  size_t v23; // rdx
  const char *v24; // rdi

  if ( *(_DWORD *)(a2 + 4) != *(_DWORD *)(a1 + 4) )
    return 1;
  if ( *(_WORD *)(a2 + 2) != *(_WORD *)(a1 + 2) )
    return 1;
  v5 = *(char **)(a1 - 8LL * *(unsigned int *)(a1 + 8));
  v6 = *(unsigned int *)(a2 + 8);
  v7 = *v5;
  if ( *v5 == 19 )
  {
    v8 = *((_DWORD *)v5 + 6);
    v9 = *(char **)(a2 - 8 * v6);
    v10 = *v9;
    if ( *v9 != 19 )
    {
      if ( !v8 )
        goto LABEL_9;
      return 1;
    }
  }
  else
  {
    v9 = *(char **)(a2 - 8 * v6);
    v8 = 0;
    v10 = *v9;
    if ( *v9 != 19 )
    {
LABEL_9:
      v11 = v9;
      v12 = v9;
      if ( v10 == 15 )
        goto LABEL_10;
      goto LABEL_30;
    }
  }
  if ( *((_DWORD *)v9 + 6) != v8 )
    return 1;
LABEL_30:
  v11 = *(char **)&v9[-8 * *((unsigned int *)v9 + 2)];
  if ( !v11 )
  {
    v15 = 0;
    v13 = byte_3F871B3;
    goto LABEL_12;
  }
  v12 = *(char **)&v9[-8 * *((unsigned int *)v9 + 2)];
LABEL_10:
  v13 = *(const char **)&v12[-8 * *((unsigned int *)v11 + 2)];
  if ( v13 )
  {
    v13 = (const char *)sub_161E970(*(_QWORD *)&v12[-8 * *((unsigned int *)v11 + 2)]);
    v15 = v14;
    v5 = *(char **)(a1 - 8LL * *(unsigned int *)(a1 + 8));
    v7 = *v5;
  }
  else
  {
    v15 = 0;
  }
LABEL_12:
  if ( v7 == 15 || (v5 = *(char **)&v5[-8 * *((unsigned int *)v5 + 2)]) != 0 )
  {
    v16 = *(const char **)&v5[-8 * *((unsigned int *)v5 + 2)];
    if ( v16 )
      v16 = (const char *)sub_161E970(v16);
    else
      v17 = 0;
  }
  else
  {
    v17 = 0;
    v16 = byte_3F871B3;
  }
  if ( v17 != v15 || v17 && memcmp(v16, v13, v17) )
    return 1;
  v18 = *(_QWORD *)(a2 - 8LL * *(unsigned int *)(a2 + 8));
  if ( *(_BYTE *)v18 == 15 || (v18 = *(_QWORD *)(v18 - 8LL * *(unsigned int *)(v18 + 8))) != 0 )
  {
    v19 = (const char *)sub_15AF8F0(v18, 1u);
    v21 = v20;
  }
  else
  {
    v21 = 0;
    v19 = byte_3F871B3;
  }
  v22 = *(_QWORD *)(a1 - 8LL * *(unsigned int *)(a1 + 8));
  if ( *(_BYTE *)v22 == 15 || (v22 = *(_QWORD *)(v22 - 8LL * *(unsigned int *)(v22 + 8))) != 0 )
  {
    v24 = (const char *)sub_15AF8F0(v22, 1u);
  }
  else
  {
    v23 = 0;
    v24 = byte_3F871B3;
  }
  if ( v23 != v21 )
    return 1;
  result = 0;
  if ( v23 )
    return memcmp(v24, v19, v23) != 0;
  return result;
}
