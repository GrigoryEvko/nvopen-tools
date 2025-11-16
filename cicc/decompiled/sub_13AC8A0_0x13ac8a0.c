// Function: sub_13AC8A0
// Address: 0x13ac8a0
//
__int64 __fastcall sub_13AC8A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __int64 v5; // r12
  __int64 v6; // r8
  _QWORD *v8; // r13
  _QWORD *v9; // r12
  __int64 v10; // r14
  __int64 v11; // rdi
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // r14
  __int64 v22; // r13
  __int64 v23; // rax
  __int64 result; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // [rsp+0h] [rbp-40h]
  __int64 v30; // [rsp+8h] [rbp-38h]

  v5 = 9LL * a5;
  v6 = 32LL * a5;
  v8 = (_QWORD *)(a3 + v6);
  v9 = (_QWORD *)(a4 + 16 * v5);
  v10 = a2 + v6;
  v11 = *v9;
  v9[10] = 0;
  v9[2] = 0;
  if ( v11 )
  {
    v12 = *(_QWORD *)(a1 + 8);
    v13 = sub_1456040(v11);
    v14 = sub_145CF80(v12, v13, 1, 0);
    v15 = sub_14806B0(v12, *v9, v14, 0, 0);
    v16 = sub_14806B0(*(_QWORD *)(a1 + 8), *(_QWORD *)(v10 + 16), *v8, 0, 0);
    v17 = sub_13AC720(a1, v16);
    v29 = *v8;
    v30 = *(_QWORD *)(a1 + 8);
    v18 = sub_13A5B60(v30, v17, v15, 0, 0);
    v9[10] = sub_14806B0(v30, v18, v29, 0, 0);
    v19 = sub_14806B0(*(_QWORD *)(a1 + 8), *(_QWORD *)(v10 + 8), *v8, 0, 0);
    v20 = sub_13AC6E0(a1, v19);
    v21 = *(_QWORD *)(a1 + 8);
    v22 = *v8;
    v23 = sub_13A5B60(v21, v20, v15, 0, 0);
    result = sub_14806B0(v21, v23, v22, 0, 0);
    v9[2] = result;
  }
  else
  {
    v25 = sub_14806B0(*(_QWORD *)(a1 + 8), *(_QWORD *)(v10 + 16), *v8, 0, 0);
    v26 = sub_13AC720(a1, v25);
    if ( (unsigned __int8)sub_14560B0(v26) )
      v9[10] = sub_1480620(*(_QWORD *)(a1 + 8), *v8, 0);
    v27 = sub_14806B0(*(_QWORD *)(a1 + 8), *(_QWORD *)(v10 + 8), *v8, 0, 0);
    v28 = sub_13AC6E0(a1, v27);
    result = sub_14560B0(v28);
    if ( (_BYTE)result )
    {
      result = sub_1480620(*(_QWORD *)(a1 + 8), *v8, 0);
      v9[2] = result;
    }
  }
  return result;
}
