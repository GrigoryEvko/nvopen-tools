// Function: sub_E82420
// Address: 0xe82420
//
_BYTE *__fastcall sub_E82420(unsigned int *a1, __int64 a2, void *a3, size_t a4, void *a5, size_t a6, __int64 a7)
{
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rax
  char *v15; // r15
  char *v16; // rdi
  void *v17; // rdi
  _BYTE *result; // rax
  __int64 v19; // rax
  _BYTE *v20; // rax
  __int64 v21; // r8
  void *v22; // rdi
  __int64 v23; // [rsp+8h] [rbp-48h]
  char *v26; // [rsp+18h] [rbp-38h]

  v10 = *(_QWORD *)(a2 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v10) <= 8 )
  {
    v19 = sub_CB6200(a2, "<MCInst #", 9u);
    sub_CB59D0(v19, *a1);
    if ( !a4 )
      goto LABEL_3;
  }
  else
  {
    *(_BYTE *)(v10 + 8) = 35;
    *(_QWORD *)v10 = 0x2074736E49434D3CLL;
    *(_QWORD *)(a2 + 32) += 9LL;
    sub_CB59D0(a2, *a1);
    if ( !a4 )
      goto LABEL_3;
  }
  v20 = *(_BYTE **)(a2 + 32);
  if ( (unsigned __int64)v20 >= *(_QWORD *)(a2 + 24) )
  {
    v21 = sub_CB5D20(a2, 32);
    v22 = *(void **)(v21 + 32);
    if ( a4 <= *(_QWORD *)(v21 + 24) - (_QWORD)v22 )
      goto LABEL_15;
  }
  else
  {
    v21 = a2;
    *(_QWORD *)(a2 + 32) = v20 + 1;
    *v20 = 32;
    v22 = *(void **)(a2 + 32);
    if ( a4 <= *(_QWORD *)(a2 + 24) - (_QWORD)v22 )
    {
LABEL_15:
      v23 = v21;
      memcpy(v22, a3, a4);
      v12 = v23;
      *(_QWORD *)(v23 + 32) += a4;
      goto LABEL_3;
    }
  }
  sub_CB6200(v21, (unsigned __int8 *)a3, a4);
LABEL_3:
  v14 = a1[6];
  v26 = (char *)(16 * v14);
  v15 = 0;
  if ( (_DWORD)v14 )
  {
    do
    {
      v17 = *(void **)(a2 + 32);
      if ( a6 <= *(_QWORD *)(a2 + 24) - (_QWORD)v17 )
      {
        if ( a6 )
        {
          memcpy(v17, a5, a6);
          *(_QWORD *)(a2 + 32) += a6;
        }
      }
      else
      {
        sub_CB6200(a2, (unsigned __int8 *)a5, a6);
      }
      v16 = &v15[*((_QWORD *)a1 + 2)];
      v15 += 16;
      sub_E81F00(v16, a2, a7, v11, v12, v13);
    }
    while ( v26 != v15 );
  }
  result = *(_BYTE **)(a2 + 32);
  if ( *(_BYTE **)(a2 + 24) == result )
    return (_BYTE *)sub_CB6200(a2, (unsigned __int8 *)">", 1u);
  *result = 62;
  ++*(_QWORD *)(a2 + 32);
  return result;
}
