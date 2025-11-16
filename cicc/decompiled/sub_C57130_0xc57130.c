// Function: sub_C57130
// Address: 0xc57130
//
__int64 __fastcall sub_C57130(__int64 a1, __int64 a2, const void *a3, size_t a4, __int64 a5, int a6)
{
  __int64 v9; // rax
  _WORD *v10; // rdx
  _QWORD *v11; // r12
  void *v12; // rdi
  unsigned int v13; // r12d
  __int64 v14; // rax
  __int64 v15; // rax
  void *v16; // rdx
  __int64 v17; // rdi
  __int64 v18; // rdi
  __int64 v19; // rax
  _WORD *v20; // rdx
  __int64 v22; // rax

  sub_C54F20(a1, a2, a6);
  v9 = sub_CB7210(a1);
  v10 = *(_WORD **)(v9 + 32);
  v11 = (_QWORD *)v9;
  if ( *(_QWORD *)(v9 + 24) - (_QWORD)v10 <= 1u )
  {
    v22 = sub_CB6200(v9, "= ", 2);
    v12 = *(void **)(v22 + 32);
    v11 = (_QWORD *)v22;
  }
  else
  {
    *v10 = 8253;
    v12 = (void *)(*(_QWORD *)(v9 + 32) + 2LL);
    *(_QWORD *)(v9 + 32) = v12;
  }
  if ( v11[3] - (_QWORD)v12 < a4 )
  {
    v12 = v11;
    sub_CB6200(v11, a3, a4);
  }
  else
  {
    if ( !a4 )
    {
LABEL_5:
      v13 = 8 - a4;
      goto LABEL_6;
    }
    memcpy(v12, a3, a4);
    v11[4] += a4;
  }
  v13 = 0;
  if ( a4 <= 7 )
    goto LABEL_5;
LABEL_6:
  v14 = sub_CB7210(v12);
  v15 = sub_CB69B0(v14, v13);
  v16 = *(void **)(v15 + 32);
  v17 = v15;
  if ( *(_QWORD *)(v15 + 24) - (_QWORD)v16 <= 0xAu )
  {
    sub_CB6200(v15, " (default: ", 11);
  }
  else
  {
    qmemcpy(v16, " (default: ", 11);
    *(_QWORD *)(v15 + 32) += 11LL;
  }
  if ( *(_BYTE *)(a5 + 40) )
  {
    v18 = sub_CB7210(v17);
    sub_CB6200(v18, *(_QWORD *)(a5 + 8), *(_QWORD *)(a5 + 16));
  }
  else
  {
    v18 = sub_CB7210(v17);
    sub_904010(v18, "*no default*");
  }
  v19 = sub_CB7210(v18);
  v20 = *(_WORD **)(v19 + 32);
  if ( *(_QWORD *)(v19 + 24) - (_QWORD)v20 <= 1u )
    return sub_CB6200(v19, ")\n", 2);
  *v20 = 2601;
  *(_QWORD *)(v19 + 32) += 2LL;
  return 2601;
}
