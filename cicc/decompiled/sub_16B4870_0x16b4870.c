// Function: sub_16B4870
// Address: 0x16b4870
//
__int64 __fastcall sub_16B4870(__int64 a1, char *a2, const char *a3, size_t a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rdx
  __int64 v10; // rax
  _WORD *v11; // rdx
  _QWORD *v12; // r12
  void *v13; // rdi
  unsigned int v14; // r12d
  __int64 v15; // rax
  unsigned __int64 v16; // rsi
  __int64 v17; // rax
  void *v18; // rdx
  __int64 v19; // rdi
  __int64 v20; // rax
  const char *v21; // rsi
  __int64 v22; // rdi
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v26; // rax
  __int64 v27; // rax

  sub_16B2F80(a1, (__int64)a2, a6, a4);
  v10 = sub_16E8C20(a1, a2, v9);
  v11 = *(_WORD **)(v10 + 24);
  v12 = (_QWORD *)v10;
  if ( *(_QWORD *)(v10 + 16) - (_QWORD)v11 <= 1u )
  {
    a2 = "= ";
    v27 = sub_16E7EE0(v10, "= ", 2);
    v13 = *(void **)(v27 + 24);
    v12 = (_QWORD *)v27;
  }
  else
  {
    *v11 = 8253;
    v13 = (void *)(*(_QWORD *)(v10 + 24) + 2LL);
    *(_QWORD *)(v10 + 24) = v13;
  }
  if ( v12[2] - (_QWORD)v13 < a4 )
  {
    a2 = (char *)a3;
    v13 = v12;
    sub_16E7EE0(v12, a3, a4);
  }
  else
  {
    if ( !a4 )
    {
LABEL_5:
      v14 = 8 - a4;
      goto LABEL_6;
    }
    a2 = (char *)a3;
    memcpy(v13, a3, a4);
    v12[3] += a4;
  }
  v14 = 0;
  if ( a4 <= 7 )
    goto LABEL_5;
LABEL_6:
  v15 = sub_16E8C20(v13, a2, v11);
  v16 = v14;
  v17 = sub_16E8750(v15, v14);
  v18 = *(void **)(v17 + 24);
  v19 = v17;
  if ( *(_QWORD *)(v17 + 16) - (_QWORD)v18 <= 0xAu )
  {
    v16 = (unsigned __int64)" (default: ";
    sub_16E7EE0(v17, " (default: ", 11);
  }
  else
  {
    qmemcpy(v18, " (default: ", 11);
    *(_QWORD *)(v17 + 24) += 11LL;
  }
  if ( *(_BYTE *)(a5 + 40) )
  {
    v20 = sub_16E8C20(v19, v16, v18);
    v21 = *(const char **)(a5 + 8);
    v22 = v20;
    sub_16E7EE0(v20, v21, *(_QWORD *)(a5 + 16));
  }
  else
  {
    v26 = sub_16E8C20(v19, v16, v18);
    v21 = "*no default*";
    v22 = v26;
    sub_1263B40(v26, "*no default*");
  }
  v24 = sub_16E8C20(v22, v21, v23);
  return sub_1263B40(v24, ")\n");
}
