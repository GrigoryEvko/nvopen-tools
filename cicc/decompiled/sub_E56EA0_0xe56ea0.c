// Function: sub_E56EA0
// Address: 0xe56ea0
//
_BYTE *__fastcall sub_E56EA0(_QWORD *a1, __int64 a2, unsigned __int8 a3, unsigned __int8 a4)
{
  __int64 v7; // rdi
  void *v8; // rdx
  int v9; // eax
  char v10; // r14
  __int64 v12; // rdi
  char *v13; // rax
  __int64 v14; // rdi
  char *v15; // rax

  sub_E99720(a1, a2, a3, a4);
  v7 = a1[38];
  v8 = *(void **)(v7 + 32);
  if ( *(_QWORD *)(v7 + 24) - (_QWORD)v8 <= 0xDu )
  {
    sub_CB6200(v7, "\t.seh_handler ", 0xEu);
  }
  else
  {
    qmemcpy(v8, "\t.seh_handler ", 14);
    *(_QWORD *)(v7 + 32) += 14LL;
  }
  sub_EA12C0(a2, a1[38], a1[39]);
  v9 = *(_DWORD *)(a1[1] + 56LL);
  if ( v9 == 36 || (v10 = 64, v9 == 1) )
    v10 = 37;
  if ( a3 )
  {
    v14 = sub_904010(a1[38], ", ");
    v15 = *(char **)(v14 + 32);
    if ( (unsigned __int64)v15 >= *(_QWORD *)(v14 + 24) )
    {
      v14 = sub_CB5D20(v14, v10);
    }
    else
    {
      *(_QWORD *)(v14 + 32) = v15 + 1;
      *v15 = v10;
    }
    sub_904010(v14, "unwind");
  }
  if ( !a4 )
    return sub_E4D880((__int64)a1);
  v12 = sub_904010(a1[38], ", ");
  v13 = *(char **)(v12 + 32);
  if ( (unsigned __int64)v13 >= *(_QWORD *)(v12 + 24) )
  {
    v12 = sub_CB5D20(v12, v10);
  }
  else
  {
    *(_QWORD *)(v12 + 32) = v13 + 1;
    *v13 = v10;
  }
  sub_904010(v12, "except");
  return sub_E4D880((__int64)a1);
}
