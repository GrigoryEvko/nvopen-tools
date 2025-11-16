// Function: sub_2BDE5C0
// Address: 0x2bde5c0
//
_QWORD *__fastcall sub_2BDE5C0(__int64 *a1, const void *a2, size_t a3, __int64 a4, __int64 a5)
{
  bool v5; // zf
  char *v7; // r13
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // r12
  __int64 v11; // rbx
  __int64 v12; // rax
  unsigned __int64 v13; // rcx
  unsigned __int64 v14; // rsi
  int v15; // edx
  _QWORD *result; // rax
  __int64 v17; // rdi
  char *v18; // r13
  void *v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  _QWORD *v22; // r8
  char *v23; // rax
  __int64 v24; // [rsp+8h] [rbp-58h] BYREF
  _QWORD v25[2]; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v26[8]; // [rsp+20h] [rbp-40h] BYREF

  if ( !a3 )
  {
    v22 = sub_CB72A0();
    v23 = (char *)v22[4];
    if ( v22[3] - (_QWORD)v23 <= 0x16u )
    {
      sub_CB6200((__int64)v22, "Found empty pass name.\n", 0x17u);
    }
    else
    {
      qmemcpy(v23, "Found empty pass name.\n", 0x17u);
      v22[4] += 23LL;
    }
LABEL_17:
    exit(1);
  }
  v5 = a1[3] == 0;
  v25[0] = a2;
  v25[1] = a3;
  v26[0] = a4;
  v26[1] = a5;
  if ( v5 )
    sub_4263D6(a1, a2, a3);
  v7 = (char *)v26;
  ((void (__fastcall *)(__int64 *, __int64 *, _QWORD *, _QWORD *))a1[4])(&v24, a1 + 1, v25, v26);
  v10 = v24;
  if ( !v24 )
  {
    v19 = sub_CB72A0();
    v20 = sub_904010((__int64)v19, "Pass '");
    v21 = sub_A51340(v20, a2, a3);
    sub_904010(v21, "' not registered!\n");
    goto LABEL_17;
  }
  v11 = *a1;
  v26[0] = v24;
  v24 = 0;
  v12 = *(unsigned int *)(v11 + 48);
  v13 = *(_QWORD *)(v11 + 40);
  v14 = v12 + 1;
  v15 = v12;
  if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(v11 + 52) )
  {
    v17 = v11 + 40;
    if ( v13 > (unsigned __int64)v26 || (unsigned __int64)v26 >= v13 + 8 * v12 )
    {
      sub_2BDE4F0(v17, v14, v12, v13, v8, v9);
      v12 = *(unsigned int *)(v11 + 48);
      v13 = *(_QWORD *)(v11 + 40);
      v15 = *(_DWORD *)(v11 + 48);
    }
    else
    {
      v18 = (char *)v26 - v13;
      sub_2BDE4F0(v17, v14, v12, v13, v8, v9);
      v13 = *(_QWORD *)(v11 + 40);
      v12 = *(unsigned int *)(v11 + 48);
      v7 = &v18[v13];
      v15 = *(_DWORD *)(v11 + 48);
    }
  }
  result = (_QWORD *)(v13 + 8 * v12);
  if ( result )
  {
    *result = *(_QWORD *)v7;
    *(_QWORD *)v7 = 0;
    v10 = v26[0];
    ++*(_DWORD *)(v11 + 48);
    if ( !v10 )
      goto LABEL_8;
  }
  else
  {
    *(_DWORD *)(v11 + 48) = v15 + 1;
  }
  result = (_QWORD *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10 + 8LL))(v10);
LABEL_8:
  if ( v24 )
    return (_QWORD *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v24 + 8LL))(v24);
  return result;
}
