// Function: sub_2EA4AB0
// Address: 0x2ea4ab0
//
unsigned __int64 __fastcall sub_2EA4AB0(__int64 a1, __int64 a2, char a3, char a4, int a5)
{
  void *v7; // rdx
  __int64 v8; // rdi
  _QWORD *v9; // rax
  unsigned int v10; // edx
  unsigned __int64 v11; // rsi
  __int64 v12; // rax
  void *v13; // rdx
  unsigned __int64 result; // rax
  __int64 v15; // rax
  _QWORD *v16; // rdi
  _QWORD *v17; // rsi
  __int64 *v18; // rcx
  __int64 *v19; // r12
  __int64 *v20; // r13
  __int64 v21; // rsi
  _QWORD *v22; // rax
  _QWORD *v23; // rdx
  __int64 v24; // rdx
  _BYTE *v25; // rax
  _QWORD *v26; // rdx
  __int64 v27; // rdx
  __int64 v28; // rdx
  _BYTE *v29; // rax
  __int64 *v30; // r12
  __int64 *i; // r13
  __int64 v32; // rdi
  __int64 v35; // [rsp+18h] [rbp-58h]
  unsigned int v37; // [rsp+24h] [rbp-4Ch]
  __int64 v38; // [rsp+28h] [rbp-48h]
  __int64 v39[7]; // [rsp+38h] [rbp-38h] BYREF

  sub_CB69B0(a2, 2 * a5);
  v7 = *(void **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v7 <= 0xDu )
  {
    v8 = sub_CB6200(a2, "Loop at depth ", 0xEu);
    v9 = *(_QWORD **)a1;
    if ( *(_QWORD *)a1 )
    {
LABEL_3:
      v10 = 1;
      do
      {
        v9 = (_QWORD *)*v9;
        ++v10;
      }
      while ( v9 );
      v11 = v10;
      goto LABEL_6;
    }
  }
  else
  {
    v8 = a2;
    qmemcpy(v7, "Loop at depth ", 14);
    *(_QWORD *)(a2 + 32) += 14LL;
    v9 = *(_QWORD **)a1;
    if ( *(_QWORD *)a1 )
      goto LABEL_3;
  }
  v11 = 1;
LABEL_6:
  v12 = sub_CB59D0(v8, v11);
  v13 = *(void **)(v12 + 32);
  if ( *(_QWORD *)(v12 + 24) - (_QWORD)v13 <= 0xCu )
  {
    sub_CB6200(v12, " containing: ", 0xDu);
  }
  else
  {
    qmemcpy(v13, " containing: ", 13);
    *(_QWORD *)(v12 + 32) += 13LL;
  }
  result = *(_QWORD *)(a1 + 32);
  v35 = *(_QWORD *)result;
  if ( result == *(_QWORD *)(a1 + 40) )
    goto LABEL_36;
  v38 = *(_QWORD *)result;
  v37 = 0;
  if ( a3 )
    goto LABEL_24;
  while ( 2 )
  {
    if ( v37 )
    {
      v29 = *(_BYTE **)(a2 + 32);
      if ( *(_BYTE **)(a2 + 24) == v29 )
      {
        sub_CB6200(a2, (unsigned __int8 *)",", 1u);
      }
      else
      {
        *v29 = 44;
        ++*(_QWORD *)(a2 + 32);
      }
    }
    sub_2E39560(v38, a2);
LABEL_12:
    if ( v35 == v38 )
    {
LABEL_26:
      v26 = *(_QWORD **)(a2 + 32);
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v26 <= 7u )
      {
        sub_CB6200(a2, "<header>", 8u);
      }
      else
      {
        *v26 = 0x3E7265646165683CLL;
        *(_QWORD *)(a2 + 32) += 8LL;
      }
    }
LABEL_13:
    v39[0] = v38;
    v15 = **(_QWORD **)(a1 + 32);
    v16 = *(_QWORD **)(v15 + 64);
    v17 = &v16[*(unsigned int *)(v15 + 72)];
    if ( v17 != sub_2EA40A0(v16, (__int64)v17, v39) )
    {
      v28 = *(_QWORD *)(a2 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v28) <= 6 )
      {
        sub_CB6200(a2, "<latch>", 7u);
      }
      else
      {
        *(_DWORD *)v28 = 1952541756;
        *(_WORD *)(v28 + 4) = 26723;
        *(_BYTE *)(v28 + 6) = 62;
        *(_QWORD *)(a2 + 32) += 7LL;
      }
    }
    v18 = *(__int64 **)(v38 + 112);
    v19 = &v18[*(unsigned int *)(v38 + 120)];
    v20 = v18;
    if ( v18 == v19 )
    {
LABEL_21:
      if ( a3 )
        goto LABEL_31;
      goto LABEL_22;
    }
    while ( 1 )
    {
      v21 = *v20;
      if ( !*(_BYTE *)(a1 + 84) )
        break;
      v22 = *(_QWORD **)(a1 + 64);
      v23 = &v22[*(unsigned int *)(a1 + 76)];
      if ( v22 == v23 )
        goto LABEL_29;
      while ( v21 != *v22 )
      {
        if ( v23 == ++v22 )
          goto LABEL_29;
      }
LABEL_20:
      if ( v19 == ++v20 )
        goto LABEL_21;
    }
    if ( sub_C8CA60(a1 + 56, v21) )
      goto LABEL_20;
LABEL_29:
    v27 = *(_QWORD *)(a2 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v27) <= 8 )
    {
      sub_CB6200(a2, "<exiting>", 9u);
      goto LABEL_21;
    }
    *(_BYTE *)(v27 + 8) = 62;
    *(_QWORD *)v27 = 0x676E69746978653CLL;
    *(_QWORD *)(a2 + 32) += 9LL;
    if ( a3 )
LABEL_31:
      sub_2E393D0(v38, a2, 0, 1u);
LABEL_22:
    v24 = *(_QWORD *)(a1 + 32);
    ++v37;
    result = (*(_QWORD *)(a1 + 40) - v24) >> 3;
    if ( v37 < result )
    {
      v38 = *(_QWORD *)(v24 + 8LL * v37);
      if ( !a3 )
        continue;
LABEL_24:
      v25 = *(_BYTE **)(a2 + 32);
      if ( *(_BYTE **)(a2 + 24) != v25 )
      {
        *v25 = 10;
        ++*(_QWORD *)(a2 + 32);
        if ( v35 == v38 )
          goto LABEL_26;
        goto LABEL_13;
      }
      sub_CB6200(a2, (unsigned __int8 *)"\n", 1u);
      goto LABEL_12;
    }
    break;
  }
LABEL_36:
  if ( a4 )
  {
    result = *(_QWORD *)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) == result )
    {
      result = sub_CB6200(a2, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *(_BYTE *)result = 10;
      ++*(_QWORD *)(a2 + 32);
    }
    v30 = *(__int64 **)(a1 + 8);
    for ( i = *(__int64 **)(a1 + 16); i != v30; result = sub_2EA4AB0(v32, a2, 0, 1, (unsigned int)(a5 + 2)) )
      v32 = *v30++;
  }
  return result;
}
