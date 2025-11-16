// Function: sub_A6AAF0
// Address: 0xa6aaf0
//
_BYTE *__fastcall sub_A6AAF0(__int64 *a1, unsigned __int64 **a2, const char *a3)
{
  __int64 v5; // r13
  size_t v6; // r14
  _BYTE *v7; // rdi
  unsigned __int64 v8; // rax
  unsigned __int64 *v9; // r14
  unsigned __int64 *v10; // rbx
  char v11; // r13
  __int64 v12; // rdi
  _WORD *v13; // rdx
  unsigned __int64 v14; // rsi
  unsigned __int64 v15; // rdx
  __int64 v16; // rdi
  _BYTE *result; // rax
  unsigned __int64 v18; // rax
  __int64 v19; // rax

  v5 = *a1;
  v6 = strlen(a3);
  v7 = *(_BYTE **)(v5 + 32);
  v8 = *(_QWORD *)(v5 + 24) - (_QWORD)v7;
  if ( v6 > v8 )
  {
    v19 = sub_CB6200(v5, a3, v6);
    v7 = *(_BYTE **)(v19 + 32);
    v5 = v19;
    v8 = *(_QWORD *)(v19 + 24) - (_QWORD)v7;
  }
  else if ( v6 )
  {
    memcpy(v7, a3, v6);
    v7 = (_BYTE *)(v6 + *(_QWORD *)(v5 + 32));
    v18 = *(_QWORD *)(v5 + 24) - (_QWORD)v7;
    *(_QWORD *)(v5 + 32) = v7;
    if ( v18 > 2 )
      goto LABEL_4;
    goto LABEL_16;
  }
  if ( v8 > 2 )
  {
LABEL_4:
    v7[2] = 40;
    *(_WORD *)v7 = 8250;
    *(_QWORD *)(v5 + 32) += 3LL;
    goto LABEL_5;
  }
LABEL_16:
  sub_CB6200(v5, ": (", 3);
LABEL_5:
  v9 = a2[1];
  v10 = *a2;
  v11 = 1;
  while ( v9 != v10 )
  {
    if ( v11 )
    {
      v11 = 0;
    }
    else
    {
      v12 = *a1;
      v13 = *(_WORD **)(*a1 + 32);
      if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v13 <= 1u )
      {
        sub_CB6200(v12, ", ", 2);
      }
      else
      {
        *v13 = 8236;
        *(_QWORD *)(v12 + 32) += 2LL;
      }
    }
    v14 = *v10;
    v15 = v10[1];
    v10 += 2;
    sub_A6A880(a1, v14, v15);
  }
  v16 = *a1;
  result = *(_BYTE **)(*a1 + 32);
  if ( *(_BYTE **)(*a1 + 24) == result )
    return (_BYTE *)sub_CB6200(v16, ")", 1);
  *result = 41;
  ++*(_QWORD *)(v16 + 32);
  return result;
}
