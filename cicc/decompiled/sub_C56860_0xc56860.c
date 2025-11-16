// Function: sub_C56860
// Address: 0xc56860
//
__int64 __fastcall sub_C56860(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, int a5)
{
  __int64 v7; // rax
  _WORD *v8; // rdx
  __int64 v9; // rdi
  unsigned int v10; // r13d
  __int64 v11; // rax
  __int64 v12; // rax
  void *v13; // rdx
  __int64 v14; // rdi
  __int64 v15; // rdi
  __int64 v16; // rax
  _WORD *v17; // rdx
  __int64 v18; // rdi
  __int64 result; // rax
  _QWORD *v20; // [rsp+0h] [rbp-90h] BYREF
  unsigned __int64 v21; // [rsp+8h] [rbp-88h]
  _QWORD v22[2]; // [rsp+10h] [rbp-80h] BYREF
  _QWORD v23[14]; // [rsp+20h] [rbp-70h] BYREF

  sub_C54F20(a1, a2, a5);
  LOBYTE(v22[0]) = 0;
  v20 = v22;
  v23[5] = 0x100000000LL;
  v23[6] = &v20;
  v21 = 0;
  memset(&v23[1], 0, 32);
  v23[0] = &unk_49DD210;
  sub_CB5980(v23, 0, 0, 0);
  sub_CB59D0(v23, a3);
  v23[0] = &unk_49DD210;
  sub_CB5840(v23);
  v7 = sub_CB7210(v23);
  v8 = *(_WORD **)(v7 + 32);
  v9 = v7;
  if ( *(_QWORD *)(v7 + 24) - (_QWORD)v8 <= 1u )
  {
    v9 = sub_CB6200(v7, "= ", 2);
  }
  else
  {
    *v8 = 8253;
    *(_QWORD *)(v7 + 32) += 2LL;
  }
  v10 = 0;
  sub_CB6200(v9, v20, v21);
  if ( v21 < 8 )
    v10 = 8 - v21;
  v11 = sub_CB7210(v9);
  v12 = sub_CB69B0(v11, v10);
  v13 = *(void **)(v12 + 32);
  v14 = v12;
  if ( *(_QWORD *)(v12 + 24) - (_QWORD)v13 <= 0xAu )
  {
    sub_CB6200(v12, " (default: ", 11);
  }
  else
  {
    qmemcpy(v13, " (default: ", 11);
    *(_QWORD *)(v12 + 32) += 11LL;
  }
  if ( *(_BYTE *)(a4 + 12) )
  {
    v15 = sub_CB7210(v14);
    sub_CB59D0(v15, *(unsigned int *)(a4 + 8));
  }
  else
  {
    v15 = sub_CB7210(v14);
    sub_904010(v15, "*no default*");
  }
  v16 = sub_CB7210(v15);
  v17 = *(_WORD **)(v16 + 32);
  v18 = v16;
  if ( *(_QWORD *)(v16 + 24) - (_QWORD)v17 <= 1u )
  {
    result = sub_CB6200(v16, ")\n", 2);
  }
  else
  {
    result = 2601;
    *v17 = 2601;
    *(_QWORD *)(v18 + 32) += 2LL;
  }
  if ( v20 != v22 )
    return j_j___libc_free_0(v20, v22[0] + 1LL);
  return result;
}
