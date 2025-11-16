// Function: sub_2820A80
// Address: 0x2820a80
//
__int64 __fastcall sub_2820A80(__int64 *a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rbx
  __int64 v10; // rdi
  __int64 v11; // rsi
  _QWORD *v12; // rax
  _QWORD *v13; // rdx
  __int64 v14; // r15
  char v15; // bl
  __int64 v16; // r15
  __int64 *v17; // rax
  __int64 v18; // rsi
  _QWORD *v19; // rax
  _QWORD *v20; // rdx
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  __int64 v23; // rax
  char v24; // cl
  __int64 v26; // rax
  unsigned __int8 v27; // [rsp+4h] [rbp-4Ch]
  unsigned __int8 v29; // [rsp+13h] [rbp-3Dh]

  v9 = *(_QWORD *)(a5 + 16);
  v10 = *a1;
  if ( !v9 )
  {
LABEL_8:
    v14 = *(_QWORD *)(a6 + 16);
    v15 = 0;
    if ( v14 )
      goto LABEL_12;
    goto LABEL_9;
  }
  while ( 1 )
  {
    v11 = *(_QWORD *)(*(_QWORD *)(v9 + 24) + 40LL);
    if ( !*(_BYTE *)(v10 + 84) )
      break;
    v12 = *(_QWORD **)(v10 + 64);
    v13 = &v12[*(unsigned int *)(v10 + 76)];
    if ( v12 == v13 )
      goto LABEL_11;
    while ( v11 != *v12 )
    {
      if ( v13 == ++v12 )
        goto LABEL_11;
    }
LABEL_7:
    v9 = *(_QWORD *)(v9 + 8);
    if ( !v9 )
      goto LABEL_8;
  }
  v17 = sub_C8CA60(v10 + 56, v11);
  v10 = *a1;
  if ( v17 )
    goto LABEL_7;
LABEL_11:
  v14 = *(_QWORD *)(a6 + 16);
  v15 = 1;
  if ( !v14 )
  {
    v26 = sub_D4B130(v10);
    v24 = 0;
    v16 = v26;
    goto LABEL_25;
  }
  while ( 1 )
  {
LABEL_12:
    while ( 1 )
    {
      v18 = *(_QWORD *)(*(_QWORD *)(v14 + 24) + 40LL);
      if ( !*(_BYTE *)(v10 + 84) )
        break;
      v19 = *(_QWORD **)(v10 + 64);
      v20 = &v19[*(unsigned int *)(v10 + 76)];
      if ( v19 == v20 )
        goto LABEL_27;
      while ( v18 != *v19 )
      {
        if ( v20 == ++v19 )
          goto LABEL_27;
      }
      v14 = *(_QWORD *)(v14 + 8);
      if ( !v14 )
      {
LABEL_18:
        v16 = sub_D4B130(v10);
        if ( !v15 )
          goto LABEL_19;
        v24 = 0;
        goto LABEL_25;
      }
    }
    if ( !sub_C8CA60(v10 + 56, v18) )
      break;
    v14 = *(_QWORD *)(v14 + 8);
    v10 = *a1;
    if ( !v14 )
      goto LABEL_18;
  }
LABEL_27:
  if ( v15 )
    return 0;
  v10 = *a1;
LABEL_9:
  v16 = sub_D4B130(v10);
LABEL_19:
  v21 = sub_AA54C0(v16);
  if ( v21 )
  {
    v22 = *(_QWORD *)(v21 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v22 == v21 + 48 )
      goto LABEL_35;
    if ( !v22 )
      BUG();
    if ( (unsigned int)*(unsigned __int8 *)(v22 - 24) - 30 > 0xA )
LABEL_35:
      BUG();
    if ( *(_BYTE *)(v22 - 24) == 31 )
    {
      v15 = 0;
      v23 = sub_281E580(v22 - 24, v16);
      v24 = 1;
      if ( a3 == v23 )
      {
LABEL_25:
        v27 = v24;
        v29 = sub_281E640(a1, a2, a3, v24);
        if ( v29 )
        {
          sub_281F920(a1, a2, v16, a6, a5, a3, (char *)a4, (const char **)(a4 + 48), v27, v15, 0);
          return v29;
        }
      }
    }
  }
  return 0;
}
