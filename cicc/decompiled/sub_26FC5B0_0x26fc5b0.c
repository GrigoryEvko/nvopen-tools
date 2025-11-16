// Function: sub_26FC5B0
// Address: 0x26fc5b0
//
char __fastcall sub_26FC5B0(__int64 a1, __int64 a2, unsigned __int8 *a3, unsigned __int8 **a4)
{
  unsigned __int8 *v4; // rax
  const char *v5; // rax
  __int64 *v6; // rdx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rcx
  __int64 v10; // r12
  char *v11; // rbx
  unsigned __int64 v12; // r13
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 *v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // r12
  unsigned __int8 *v22; // [rsp+18h] [rbp-58h] BYREF
  unsigned __int8 **v23[10]; // [rsp+20h] [rbp-50h] BYREF

  v22 = a3;
  v4 = sub_BD3990(a3, a2);
  v5 = sub_BD5D20((__int64)v4);
  v9 = *(_QWORD *)(a1 + 400);
  v10 = *(_QWORD *)(a1 + 408);
  if ( v9 == v10 )
  {
LABEL_6:
    v23[0] = (unsigned __int8 **)a1;
    v23[1] = &v22;
    v23[2] = a4;
    LOBYTE(v14) = sub_26FBDD0(v23, a2, v6, v9, v7, v8);
    v19 = *(_QWORD *)(a2 + 104);
    if ( v19 != a2 + 88 )
    {
      do
      {
        sub_26FBDD0(v23, v19 + 56, v15, v16, v17, v18);
        v14 = sub_220EEE0(v19);
        v19 = v14;
      }
      while ( a2 + 88 != v14 );
    }
  }
  else
  {
    v11 = (char *)v5;
    v12 = (unsigned __int64)v6;
    v13 = *(_QWORD *)(a1 + 400);
    while ( 1 )
    {
      LOBYTE(v14) = sub_1099960(v13, v11, v12);
      if ( (_BYTE)v14 )
        break;
      v13 += 72;
      if ( v10 == v13 )
        goto LABEL_6;
    }
  }
  return v14;
}
