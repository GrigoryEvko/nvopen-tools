// Function: sub_23201C0
// Address: 0x23201c0
//
__int64 __fastcall sub_23201C0(__int64 a1, __int64 a2, __int64 (__fastcall *a3)(__int64, char *, __int64), __int64 a4)
{
  __int64 v6; // r12
  unsigned __int8 *v7; // r14
  size_t v8; // rdx
  size_t v9; // r13
  _QWORD *v10; // rdx
  _BYTE *v11; // rdi
  unsigned __int64 v12; // rax
  unsigned __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  char *v17; // [rsp+0h] [rbp-30h] BYREF
  __int64 v18; // [rsp+8h] [rbp-28h]

  v6 = a2;
  v17 = "llvm::FunctionAnalysisManagerCGSCCProxy]";
  v18 = 39;
  sub_95CB50((const void **)&v17, "llvm::", 6u);
  v7 = (unsigned __int8 *)a3(a4, v17, v18);
  v9 = v8;
  v10 = *(_QWORD **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v10 <= 7u )
  {
    v16 = sub_CB6200(a2, "require<", 8u);
    v11 = *(_BYTE **)(v16 + 32);
    v6 = v16;
  }
  else
  {
    *v10 = 0x3C65726975716572LL;
    v11 = (_BYTE *)(*(_QWORD *)(a2 + 32) + 8LL);
    *(_QWORD *)(a2 + 32) = v11;
  }
  v12 = *(_QWORD *)(v6 + 24);
  if ( v12 - (unsigned __int64)v11 < v9 )
  {
    v15 = sub_CB6200(v6, v7, v9);
    v11 = *(_BYTE **)(v15 + 32);
    v6 = v15;
    v12 = *(_QWORD *)(v15 + 24);
  }
  else if ( v9 )
  {
    memcpy(v11, v7, v9);
    v14 = *(_QWORD *)(v6 + 24);
    v11 = (_BYTE *)(v9 + *(_QWORD *)(v6 + 32));
    *(_QWORD *)(v6 + 32) = v11;
    if ( (unsigned __int64)v11 < v14 )
      goto LABEL_6;
    return sub_CB5D20(v6, 62);
  }
  if ( (unsigned __int64)v11 < v12 )
  {
LABEL_6:
    *(_QWORD *)(v6 + 32) = v11 + 1;
    *v11 = 62;
    return (__int64)(v11 + 1);
  }
  return sub_CB5D20(v6, 62);
}
