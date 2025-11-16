// Function: sub_13AC560
// Address: 0x13ac560
//
void __fastcall sub_13AC560(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __int64 v5; // rbx
  __int64 v6; // r8
  __int64 v7; // r14
  _QWORD *v8; // r12
  __int64 *v9; // rbx
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r9
  __int64 v14; // r15
  __int64 v15; // r12
  __int64 v16; // r13
  __int64 v17; // rax
  __int64 v18; // r15
  __int64 v19; // rax
  __int64 v20; // [rsp+8h] [rbp-58h]
  __int64 v21; // [rsp+8h] [rbp-58h]
  unsigned __int64 v22[2]; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v23[8]; // [rsp+20h] [rbp-40h] BYREF

  v5 = 9LL * a5;
  v6 = 32LL * a5;
  v7 = a3 + v6;
  v8 = (_QWORD *)(a2 + v6);
  v9 = (__int64 *)(a4 + 16 * v5);
  v10 = *v9;
  v9[8] = 0;
  v9[16] = 0;
  if ( v10 )
  {
    v20 = *(_QWORD *)(a1 + 8);
    v11 = sub_14806B0(v20, v8[2], *(_QWORD *)(v7 + 8), 0, 0);
    v12 = sub_13A5B60(v20, v11, v10, 0, 0);
    v13 = *v9;
    v9[16] = v12;
    v14 = *(_QWORD *)(a1 + 8);
    v21 = v13;
    v23[0] = sub_14806B0(v14, v8[1], *(_QWORD *)(v7 + 16), 0, 0);
    v22[0] = (unsigned __int64)v23;
    v23[1] = v21;
    v22[1] = 0x200000002LL;
    v15 = sub_147EE30(v14, v22, 0, 0);
    if ( (_QWORD *)v22[0] != v23 )
      _libc_free(v22[0]);
    v9[8] = v15;
  }
  else
  {
    if ( (unsigned __int8)sub_13A7760(a1, 32, v8[2], *(_QWORD *)(v7 + 8)) )
    {
      v18 = *(_QWORD *)(a1 + 8);
      v19 = sub_1456040(*v8);
      v9[16] = sub_145CF80(v18, v19, 0, 0);
    }
    if ( (unsigned __int8)sub_13A7760(a1, 32, v8[1], *(_QWORD *)(v7 + 16)) )
    {
      v16 = *(_QWORD *)(a1 + 8);
      v17 = sub_1456040(*v8);
      v9[8] = sub_145CF80(v16, v17, 0, 0);
    }
  }
}
