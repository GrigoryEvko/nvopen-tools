// Function: sub_3180CD0
// Address: 0x3180cd0
//
__int64 __fastcall sub_3180CD0(__int64 a1, __int64 a2, __int64 a3)
{
  int *v4; // rax
  size_t v5; // rdx
  _QWORD *v6; // rax
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // rax
  unsigned __int64 v11; // r13
  _QWORD *v12; // rbx
  unsigned __int64 v13; // r14
  _QWORD *v14; // rdi
  int *v16; // rax
  size_t v17; // rdx
  _QWORD *v18; // [rsp+8h] [rbp-58h]
  __int64 v19; // [rsp+10h] [rbp-50h]
  __int64 i; // [rsp+18h] [rbp-48h]
  __int64 v21; // [rsp+20h] [rbp-40h] BYREF
  __int64 v22[7]; // [rsp+28h] [rbp-38h] BYREF

  v21 = 0;
  v22[0] = sub_317E640(a2);
  v18 = (_QWORD *)sub_317E650(a2);
  v19 = a1 + 120;
  if ( a1 + 120 != a3 )
    v21 = v22[0];
  v4 = (int *)sub_317E460(a2);
  v6 = sub_317E540(a3, &v21, v4, v5);
  v7 = (__int64)v6;
  if ( !v6 )
  {
    v7 = sub_31807D0(a1, a3, &v21, a2);
    if ( v19 != a3 )
      return v7;
LABEL_11:
    v16 = (int *)sub_317E460(v7);
    sub_317E210(v18, v22, v16, v17);
    return v7;
  }
  sub_317F330(a1, a2, (__int64)v6);
  v8 = sub_317E450(a2);
  v9 = *(_QWORD *)(v8 + 24);
  for ( i = v8 + 8; i != v9; v9 = sub_220EEE0(v9) )
    sub_3180CD0(a1, v9 + 40, v7);
  v10 = sub_317E450(a2);
  v11 = *(_QWORD *)(v10 + 16);
  v12 = (_QWORD *)v10;
  while ( v11 )
  {
    v13 = v11;
    sub_317D930(*(_QWORD **)(v11 + 24));
    v14 = *(_QWORD **)(v11 + 56);
    v11 = *(_QWORD *)(v11 + 16);
    sub_317D930(v14);
    j_j___libc_free_0(v13);
  }
  v12[2] = 0;
  v12[3] = v12 + 1;
  v12[4] = v12 + 1;
  v12[5] = 0;
  if ( v19 == a3 )
    goto LABEL_11;
  return v7;
}
