// Function: sub_13AC760
// Address: 0x13ac760
//
void __fastcall sub_13AC760(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __int64 v6; // rbx
  __int64 v7; // r8
  __int64 *v8; // rbx
  bool v9; // zf
  __int64 v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // rsi
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rdi
  __int64 v18; // r12
  __int64 v19; // r13
  __int64 v20; // r14
  unsigned __int64 v21[2]; // [rsp+0h] [rbp-40h] BYREF
  _QWORD v22[6]; // [rsp+10h] [rbp-30h] BYREF

  v6 = 9LL * a5;
  v7 = 32LL * a5;
  v8 = (__int64 *)(a4 + 16 * v6);
  v9 = *v8 == 0;
  v8[11] = 0;
  v8[3] = 0;
  v10 = *(_QWORD *)(a1 + 8);
  v11 = *(_QWORD *)(v7 + a3);
  v12 = *(_QWORD *)(v7 + a2);
  if ( !v9 )
  {
    v13 = sub_14806B0(v10, v12, v11, 0, 0);
    v14 = sub_13AC720(a1, v13);
    v8[11] = sub_13A5B60(*(_QWORD *)(a1 + 8), v14, *v8, 0, 0);
    v15 = sub_13AC6E0(a1, v13);
    v16 = *v8;
    v17 = *(_QWORD *)(a1 + 8);
    v22[0] = v15;
    v22[1] = v16;
    v21[0] = (unsigned __int64)v22;
    v21[1] = 0x200000002LL;
    v18 = sub_147EE30(v17, v21, 0, 0);
    if ( (_QWORD *)v21[0] != v22 )
      _libc_free(v21[0]);
LABEL_4:
    v8[3] = v18;
    return;
  }
  v19 = sub_14806B0(v10, v12, v11, 0, 0);
  v20 = sub_13AC720(a1, v19);
  if ( (unsigned __int8)sub_14560B0(v20) )
    v8[11] = v20;
  v18 = sub_13AC6E0(a1, v19);
  if ( (unsigned __int8)sub_14560B0(v18) )
    goto LABEL_4;
}
