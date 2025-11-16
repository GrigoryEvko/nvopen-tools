// Function: sub_C51E50
// Address: 0xc51e50
//
__int64 __fastcall sub_C51E50(__int64 **a1, char a2, const void *a3, size_t a4)
{
  __int64 v8; // rax
  void *v9; // rdx
  __int64 v10; // rdi
  const char *v11; // rsi
  __int64 v12; // rax
  _WORD *v13; // rdx
  __int64 v14; // rdi
  __int64 v15; // rax
  void *v16; // rdx
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 result; // rax
  __int64 v20; // rax
  __int64 *v21; // rax
  __int64 v22; // rdi
  _QWORD v23[8]; // [rsp+0h] [rbp-40h] BYREF

  v8 = sub_CB6200(**a1, *a1[1], a1[1][1]);
  v9 = *(void **)(v8 + 32);
  v10 = v8;
  if ( *(_QWORD *)(v8 + 24) - (_QWORD)v9 <= 9u )
  {
    v10 = sub_CB6200(v8, ": Unknown ", 10);
  }
  else
  {
    qmemcpy(v9, ": Unknown ", 10);
    *(_QWORD *)(v8 + 32) += 10LL;
  }
  v11 = "command line argument";
  if ( !a2 )
    v11 = "subcommand";
  v12 = sub_904010(v10, v11);
  v13 = *(_WORD **)(v12 + 32);
  v14 = v12;
  if ( *(_QWORD *)(v12 + 24) - (_QWORD)v13 <= 1u )
  {
    v14 = sub_CB6200(v12, " '", 2);
  }
  else
  {
    *v13 = 10016;
    *(_QWORD *)(v12 + 32) += 2LL;
  }
  v15 = sub_904010(v14, *(const char **)(*a1[2] + 8LL * *(int *)a1[3]));
  v16 = *(void **)(v15 + 32);
  v17 = v15;
  if ( *(_QWORD *)(v15 + 24) - (_QWORD)v16 <= 9u )
  {
    v17 = sub_CB6200(v15, "'.  Try: '", 10);
  }
  else
  {
    qmemcpy(v16, "'.  Try: '", 10);
    *(_QWORD *)(v15 + 32) += 10LL;
  }
  v18 = sub_904010(v17, *(const char **)*a1[2]);
  result = sub_904010(v18, " --help'\n");
  if ( a4 )
  {
    v20 = sub_CB6200(**a1, *a1[1], a1[1][1]);
    sub_904010(v20, ": Did you mean '");
    v21 = *a1;
    if ( a2 )
    {
      v22 = *v21;
      v23[0] = a3;
      v23[1] = a4;
      v23[2] = 0;
      sub_C51AE0(v22, (__int64)v23);
    }
    else
    {
      sub_A51340(*v21, a3, a4);
    }
    return sub_904010(**a1, "'?\n");
  }
  return result;
}
