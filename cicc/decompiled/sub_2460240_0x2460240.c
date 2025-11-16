// Function: sub_2460240
// Address: 0x2460240
//
_BYTE *__fastcall sub_2460240(
        unsigned int **a1,
        __int64 a2,
        __int64 (__fastcall *a3)(__int64, char *, __int64),
        __int64 a4)
{
  __int64 v6; // rax
  size_t v7; // rdx
  _BYTE *v8; // rdi
  unsigned __int8 *v9; // rsi
  _BYTE *v10; // rax
  size_t v11; // r12
  _BYTE *result; // rax
  unsigned int *v13; // rbx
  unsigned int *v14; // r12
  signed __int64 v15; // r13
  __int64 v16; // rdi
  __int64 v17; // rax
  _WORD *v18; // rdx
  __int64 v19; // rdi
  unsigned int v20; // r14d
  unsigned __int64 v21; // rdx
  _BYTE *v22; // rax

  v6 = a3(a4, "LowerAllowCheckPass]", 19);
  v8 = *(_BYTE **)(a2 + 32);
  v9 = (unsigned __int8 *)v6;
  v10 = *(_BYTE **)(a2 + 24);
  v11 = v7;
  if ( v10 - v8 < v7 )
  {
    sub_CB6200(a2, v9, v7);
    v10 = *(_BYTE **)(a2 + 24);
    v8 = *(_BYTE **)(a2 + 32);
  }
  else if ( v7 )
  {
    memcpy(v8, v9, v7);
    v22 = *(_BYTE **)(a2 + 24);
    v8 = (_BYTE *)(v11 + *(_QWORD *)(a2 + 32));
    *(_QWORD *)(a2 + 32) = v8;
    if ( v22 != v8 )
      goto LABEL_4;
    goto LABEL_23;
  }
  if ( v10 != v8 )
  {
LABEL_4:
    *v8 = 60;
    result = (_BYTE *)(*(_QWORD *)(a2 + 32) + 1LL);
    *(_QWORD *)(a2 + 32) = result;
    goto LABEL_5;
  }
LABEL_23:
  sub_CB6200(a2, "<", 1u);
  result = *(_BYTE **)(a2 + 32);
LABEL_5:
  v13 = *a1;
  v14 = a1[1];
  if ( *a1 != v14 )
  {
    v15 = 0;
    while ( 1 )
    {
      v20 = *v13;
      if ( *v13 )
        break;
LABEL_12:
      ++v13;
      ++v15;
      if ( v14 == v13 )
        goto LABEL_19;
    }
    if ( v15 )
    {
      if ( *(_BYTE **)(a2 + 24) != result )
      {
        *result = 59;
        result = (_BYTE *)(*(_QWORD *)(a2 + 32) + 1LL);
        v21 = *(_QWORD *)(a2 + 24) - (_QWORD)result;
        *(_QWORD *)(a2 + 32) = result;
        if ( v21 <= 7 )
          goto LABEL_17;
        goto LABEL_8;
      }
      sub_CB6200(a2, (unsigned __int8 *)";", 1u);
      result = *(_BYTE **)(a2 + 32);
    }
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)result <= 7u )
    {
LABEL_17:
      v16 = sub_CB6200(a2, "cutoffs[", 8u);
      goto LABEL_9;
    }
LABEL_8:
    v16 = a2;
    *(_QWORD *)result = 0x5B7366666F747563LL;
    *(_QWORD *)(a2 + 32) += 8LL;
LABEL_9:
    v17 = sub_CB59F0(v16, v15);
    v18 = *(_WORD **)(v17 + 32);
    v19 = v17;
    if ( *(_QWORD *)(v17 + 24) - (_QWORD)v18 <= 1u )
    {
      v19 = sub_CB6200(v17, "]=", 2u);
    }
    else
    {
      *v18 = 15709;
      *(_QWORD *)(v17 + 32) += 2LL;
    }
    sub_CB59D0(v19, v20);
    result = *(_BYTE **)(a2 + 32);
    goto LABEL_12;
  }
LABEL_19:
  if ( *(_QWORD *)(a2 + 24) <= (unsigned __int64)result )
    return (_BYTE *)sub_CB5D20(a2, 62);
  *(_QWORD *)(a2 + 32) = result + 1;
  *result = 62;
  return result;
}
