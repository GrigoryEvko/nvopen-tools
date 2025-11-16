// Function: sub_2EF03A0
// Address: 0x2ef03a0
//
_BYTE *__fastcall sub_2EF03A0(__int64 a1, char *a2, unsigned __int64 a3)
{
  __int64 v5; // r12
  void *v6; // rdx
  __int64 v7; // rdx
  _BYTE *v8; // rax
  const char *v9; // rax
  size_t v10; // rdx
  _WORD *v11; // rdi
  unsigned __int8 *v12; // rsi
  unsigned __int64 v13; // rax
  __int64 v14; // rdi
  _BYTE *v15; // rax
  __int64 v16; // r12
  _WORD *v17; // rdx
  _BYTE *v18; // rax
  _BYTE *v19; // rax
  __int64 v20; // rdi
  _BYTE *result; // rax
  unsigned __int64 v22; // rax
  __int64 v23; // rax
  size_t v24; // [rsp+8h] [rbp-48h]
  _QWORD v25[2]; // [rsp+10h] [rbp-40h] BYREF
  void (__fastcall *v26)(_QWORD *, _QWORD *, __int64); // [rsp+20h] [rbp-30h]
  void (__fastcall *v27)(_QWORD *, __int64); // [rsp+28h] [rbp-28h]

  sub_2EEFF60(a1, a2, *(__int64 **)(a3 + 32));
  v5 = *(_QWORD *)(a1 + 16);
  v6 = *(void **)(v5 + 32);
  if ( *(_QWORD *)(v5 + 24) - (_QWORD)v6 <= 0xEu )
  {
    v5 = sub_CB6200(*(_QWORD *)(a1 + 16), "- basic block: ", 0xFu);
  }
  else
  {
    qmemcpy(v6, "- basic block: ", 15);
    *(_QWORD *)(v5 + 32) += 15LL;
  }
  sub_2E31000(v25, a3);
  if ( !v26 )
    sub_4263D6(v25, a3, v7);
  v27(v25, v5);
  v8 = *(_BYTE **)(v5 + 32);
  if ( (unsigned __int64)v8 >= *(_QWORD *)(v5 + 24) )
  {
    v5 = sub_CB5D20(v5, 32);
  }
  else
  {
    *(_QWORD *)(v5 + 32) = v8 + 1;
    *v8 = 32;
  }
  v9 = sub_2E31BC0(a3);
  v11 = *(_WORD **)(v5 + 32);
  v12 = (unsigned __int8 *)v9;
  v13 = *(_QWORD *)(v5 + 24) - (_QWORD)v11;
  if ( v13 < v10 )
  {
    v23 = sub_CB6200(v5, v12, v10);
    v11 = *(_WORD **)(v23 + 32);
    v5 = v23;
    v13 = *(_QWORD *)(v23 + 24) - (_QWORD)v11;
  }
  else if ( v10 )
  {
    v24 = v10;
    memcpy(v11, v12, v10);
    v11 = (_WORD *)(v24 + *(_QWORD *)(v5 + 32));
    v22 = *(_QWORD *)(v5 + 24) - (_QWORD)v11;
    *(_QWORD *)(v5 + 32) = v11;
    if ( v22 > 1 )
      goto LABEL_9;
    goto LABEL_24;
  }
  if ( v13 > 1 )
  {
LABEL_9:
    *v11 = 10272;
    *(_QWORD *)(v5 + 32) += 2LL;
    goto LABEL_10;
  }
LABEL_24:
  v5 = sub_CB6200(v5, (unsigned __int8 *)" (", 2u);
LABEL_10:
  v14 = sub_CB5A80(v5, a3);
  v15 = *(_BYTE **)(v14 + 32);
  if ( (unsigned __int64)v15 >= *(_QWORD *)(v14 + 24) )
  {
    sub_CB5D20(v14, 41);
  }
  else
  {
    *(_QWORD *)(v14 + 32) = v15 + 1;
    *v15 = 41;
  }
  if ( v26 )
    v26(v25, v25, 3);
  if ( *(_QWORD *)(a1 + 656) )
  {
    v16 = *(_QWORD *)(a1 + 16);
    v17 = *(_WORD **)(v16 + 32);
    if ( *(_QWORD *)(v16 + 24) - (_QWORD)v17 <= 1u )
    {
      v16 = sub_CB6200(*(_QWORD *)(a1 + 16), (unsigned __int8 *)" [", 2u);
    }
    else
    {
      *v17 = 23328;
      *(_QWORD *)(v16 + 32) += 2LL;
    }
    v25[0] = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 656) + 152LL) + 16LL * *(unsigned int *)(a3 + 24));
    sub_2FAD600(v25, v16);
    v18 = *(_BYTE **)(v16 + 32);
    if ( (unsigned __int64)v18 >= *(_QWORD *)(v16 + 24) )
    {
      v16 = sub_CB5D20(v16, 59);
    }
    else
    {
      *(_QWORD *)(v16 + 32) = v18 + 1;
      *v18 = 59;
    }
    v25[0] = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 656) + 152LL) + 16LL * *(unsigned int *)(a3 + 24) + 8);
    sub_2FAD600(v25, v16);
    v19 = *(_BYTE **)(v16 + 32);
    if ( (unsigned __int64)v19 >= *(_QWORD *)(v16 + 24) )
    {
      sub_CB5D20(v16, 41);
    }
    else
    {
      *(_QWORD *)(v16 + 32) = v19 + 1;
      *v19 = 41;
    }
  }
  v20 = *(_QWORD *)(a1 + 16);
  result = *(_BYTE **)(v20 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(v20 + 24) )
    return (_BYTE *)sub_CB5D20(v20, 10);
  *(_QWORD *)(v20 + 32) = result + 1;
  *result = 10;
  return result;
}
