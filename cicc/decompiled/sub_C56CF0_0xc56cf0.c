// Function: sub_C56CF0
// Address: 0xc56cf0
//
__int64 __fastcall sub_C56CF0(__int64 a1, __int64 a2, __int64 a3, int a4, double a5)
{
  __int64 v6; // rax
  _WORD *v7; // rdx
  __int64 v8; // rdi
  unsigned int v9; // r13d
  __int64 v10; // rax
  __int64 v11; // rax
  void *v12; // rdx
  __int64 v13; // rdi
  __int64 v14; // rdi
  __int64 v15; // rax
  _WORD *v16; // rdx
  __int64 v17; // rdi
  __int64 result; // rax
  _QWORD *v19; // [rsp+10h] [rbp-80h] BYREF
  unsigned __int64 v20; // [rsp+18h] [rbp-78h]
  _QWORD v21[2]; // [rsp+20h] [rbp-70h] BYREF
  _QWORD v22[12]; // [rsp+30h] [rbp-60h] BYREF

  sub_C54F20(a1, a2, a4);
  v22[6] = &v19;
  v22[5] = 0x100000000LL;
  v19 = v21;
  v20 = 0;
  LOBYTE(v21[0]) = 0;
  memset(&v22[1], 0, 32);
  v22[0] = &unk_49DD210;
  sub_CB5980(v22, 0, 0, 0);
  sub_CB5AB0(v22, a5);
  v22[0] = &unk_49DD210;
  sub_CB5840(v22);
  v6 = sub_CB7210(v22);
  v7 = *(_WORD **)(v6 + 32);
  v8 = v6;
  if ( *(_QWORD *)(v6 + 24) - (_QWORD)v7 <= 1u )
  {
    v8 = sub_CB6200(v6, "= ", 2);
  }
  else
  {
    *v7 = 8253;
    *(_QWORD *)(v6 + 32) += 2LL;
  }
  v9 = 0;
  sub_CB6200(v8, v19, v20);
  if ( v20 < 8 )
    v9 = 8 - v20;
  v10 = sub_CB7210(v8);
  v11 = sub_CB69B0(v10, v9);
  v12 = *(void **)(v11 + 32);
  v13 = v11;
  if ( *(_QWORD *)(v11 + 24) - (_QWORD)v12 <= 0xAu )
  {
    sub_CB6200(v11, " (default: ", 11);
  }
  else
  {
    qmemcpy(v12, " (default: ", 11);
    *(_QWORD *)(v11 + 32) += 11LL;
  }
  if ( *(_BYTE *)(a3 + 16) )
  {
    v14 = sub_CB7210(v13);
    sub_CB5AB0(v14, *(double *)(a3 + 8));
  }
  else
  {
    v14 = sub_CB7210(v13);
    sub_904010(v14, "*no default*");
  }
  v15 = sub_CB7210(v14);
  v16 = *(_WORD **)(v15 + 32);
  v17 = v15;
  if ( *(_QWORD *)(v15 + 24) - (_QWORD)v16 <= 1u )
  {
    result = sub_CB6200(v15, ")\n", 2);
  }
  else
  {
    result = 2601;
    *v16 = 2601;
    *(_QWORD *)(v17 + 32) += 2LL;
  }
  if ( v19 != v21 )
    return j_j___libc_free_0(v19, v21[0] + 1LL);
  return result;
}
