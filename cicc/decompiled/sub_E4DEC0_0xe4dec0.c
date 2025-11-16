// Function: sub_E4DEC0
// Address: 0xe4dec0
//
_BYTE *__fastcall sub_E4DEC0(__int64 a1, __int64 a2, _QWORD *a3, unsigned __int64 a4, unsigned __int8 a5)
{
  __int64 v9; // rdi
  void *v10; // rdx
  __int64 v11; // r12
  void *v12; // rsi
  _BYTE *v13; // rdi
  size_t v14; // rdx
  __int64 v15; // rax
  void *v16; // rdi
  size_t v17; // rax
  _BYTE *v18; // rax
  size_t v19; // rdx
  unsigned __int8 *v20; // rsi
  __int64 v21; // rdi
  _BYTE *v22; // rax
  __int64 v23; // rdi
  _BYTE *v24; // rax
  __int64 v25; // rdi
  _BYTE *v26; // rax
  _BYTE *v28; // rdx
  size_t v29; // [rsp+0h] [rbp-40h]
  __int64 v30; // [rsp+0h] [rbp-40h]

  if ( a3 )
    *a3 = a2 + 56;
  v9 = *(_QWORD *)(a1 + 304);
  v10 = *(void **)(v9 + 32);
  if ( *(_QWORD *)(v9 + 24) - (_QWORD)v10 <= 9u )
  {
    sub_CB6200(v9, ".zerofill ", 0xAu);
  }
  else
  {
    qmemcpy(v10, ".zerofill ", 10);
    *(_QWORD *)(v9 + 32) += 10LL;
  }
  v11 = *(_QWORD *)(a1 + 304);
  v12 = (void *)(a2 + 148);
  if ( *(_BYTE *)(a2 + 163) )
  {
    v13 = *(_BYTE **)(v11 + 32);
    v14 = 16;
    if ( *(_QWORD *)(v11 + 24) - (_QWORD)v13 <= 0xFu )
      goto LABEL_7;
LABEL_25:
    v29 = v14;
    memcpy(v13, v12, v14);
    v28 = (_BYTE *)(*(_QWORD *)(v11 + 32) + v29);
    *(_QWORD *)(v11 + 32) = v28;
    v18 = *(_BYTE **)(v11 + 24);
    v13 = v28;
LABEL_11:
    if ( v18 == v13 )
      goto LABEL_8;
    goto LABEL_12;
  }
  v17 = strlen((const char *)(a2 + 148));
  v13 = *(_BYTE **)(v11 + 32);
  v12 = (void *)(a2 + 148);
  v14 = v17;
  v18 = *(_BYTE **)(v11 + 24);
  if ( v18 - v13 >= v14 )
  {
    if ( !v14 )
      goto LABEL_11;
    goto LABEL_25;
  }
LABEL_7:
  v11 = sub_CB6200(v11, (unsigned __int8 *)v12, v14);
  v13 = *(_BYTE **)(v11 + 32);
  if ( *(_BYTE **)(v11 + 24) == v13 )
  {
LABEL_8:
    v15 = sub_CB6200(v11, (unsigned __int8 *)",", 1u);
    v16 = *(void **)(v15 + 32);
    v11 = v15;
    goto LABEL_13;
  }
LABEL_12:
  *v13 = 44;
  v16 = (void *)(*(_QWORD *)(v11 + 32) + 1LL);
  *(_QWORD *)(v11 + 32) = v16;
LABEL_13:
  v19 = *(_QWORD *)(a2 + 136);
  v20 = *(unsigned __int8 **)(a2 + 128);
  if ( v19 > *(_QWORD *)(v11 + 24) - (_QWORD)v16 )
  {
    sub_CB6200(v11, v20, v19);
  }
  else if ( v19 )
  {
    v30 = *(_QWORD *)(a2 + 136);
    memcpy(v16, v20, v19);
    *(_QWORD *)(v11 + 32) += v30;
  }
  if ( a3 )
  {
    v21 = *(_QWORD *)(a1 + 304);
    v22 = *(_BYTE **)(v21 + 32);
    if ( (unsigned __int64)v22 >= *(_QWORD *)(v21 + 24) )
    {
      sub_CB5D20(v21, 44);
    }
    else
    {
      *(_QWORD *)(v21 + 32) = v22 + 1;
      *v22 = 44;
    }
    sub_EA12C0(a3, *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 312));
    v23 = *(_QWORD *)(a1 + 304);
    v24 = *(_BYTE **)(v23 + 32);
    if ( (unsigned __int64)v24 >= *(_QWORD *)(v23 + 24) )
    {
      v23 = sub_CB5D20(v23, 44);
    }
    else
    {
      *(_QWORD *)(v23 + 32) = v24 + 1;
      *v24 = 44;
    }
    sub_CB59D0(v23, a4);
    v25 = *(_QWORD *)(a1 + 304);
    v26 = *(_BYTE **)(v25 + 32);
    if ( (unsigned __int64)v26 >= *(_QWORD *)(v25 + 24) )
    {
      v25 = sub_CB5D20(v25, 44);
    }
    else
    {
      *(_QWORD *)(v25 + 32) = v26 + 1;
      *v26 = 44;
    }
    sub_CB59D0(v25, a5);
  }
  return sub_E4D880(a1);
}
