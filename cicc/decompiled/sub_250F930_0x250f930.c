// Function: sub_250F930
// Address: 0x250f930
//
void __fastcall sub_250F930(unsigned int *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  void *v4; // rdx
  void *v5; // rdx
  __int64 v6; // rax
  void *v7; // rdx
  __int64 v8; // r15
  __int64 (__fastcall *v9)(__int64, __int64); // rax
  __int64 v10; // rdi
  _BYTE *v11; // rax
  unsigned __int64 v12; // [rsp+8h] [rbp-58h]
  unsigned __int8 *v13; // [rsp+10h] [rbp-50h] BYREF
  size_t v14; // [rsp+18h] [rbp-48h]
  _QWORD v15[8]; // [rsp+20h] [rbp-40h] BYREF

  v3 = a3;
  sub_904010(a3, "[");
  (*(void (__fastcall **)(unsigned __int8 **, unsigned int *))(*(_QWORD *)a1 + 72LL))(&v13, a1);
  sub_CB6200(v3, v13, v14);
  if ( v13 != (unsigned __int8 *)v15 )
    j_j___libc_free_0((unsigned __int64)v13);
  v4 = *(void **)(v3 + 32);
  if ( *(_QWORD *)(v3 + 24) - (_QWORD)v4 <= 0xAu )
  {
    sub_CB6200(v3, "] for CtxI ", 0xBu);
  }
  else
  {
    qmemcpy(v4, "] for CtxI ", 11);
    *(_QWORD *)(v3 + 32) += 11LL;
  }
  v12 = sub_2509740((_QWORD *)a1 + 9);
  if ( v12 )
  {
    sub_904010(v3, "'");
    sub_A69870(v12, (_BYTE *)v3, 0);
    sub_904010(v3, "'");
  }
  else
  {
    sub_904010(v3, "<<null inst>>");
  }
  v5 = *(void **)(v3 + 32);
  if ( *(_QWORD *)(v3 + 24) - (_QWORD)v5 <= 0xCu )
  {
    v3 = sub_CB6200(v3, " at position ", 0xDu);
  }
  else
  {
    qmemcpy(v5, " at position ", 13);
    *(_QWORD *)(v3 + 32) += 13LL;
  }
  v6 = sub_250F6E0(v3, (_QWORD *)a1 + 9);
  v7 = *(void **)(v6 + 32);
  v8 = v6;
  if ( *(_QWORD *)(v6 + 24) - (_QWORD)v7 <= 0xBu )
  {
    v8 = sub_CB6200(v6, " with state ", 0xCu);
  }
  else
  {
    qmemcpy(v7, " with state ", 12);
    *(_QWORD *)(v6 + 32) += 12LL;
  }
  v9 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 64LL);
  if ( v9 == sub_25060C0 )
    sub_2554360(&v13, a1[25]);
  else
    ((void (__fastcall *)(unsigned __int8 **, unsigned int *, __int64))v9)(&v13, a1, a2);
  v10 = sub_CB6200(v8, v13, v14);
  v11 = *(_BYTE **)(v10 + 32);
  if ( (unsigned __int64)v11 >= *(_QWORD *)(v10 + 24) )
  {
    sub_CB5D20(v10, 10);
  }
  else
  {
    *(_QWORD *)(v10 + 32) = v11 + 1;
    *v11 = 10;
  }
  if ( v13 != (unsigned __int8 *)v15 )
    j_j___libc_free_0((unsigned __int64)v13);
}
