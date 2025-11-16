// Function: sub_DC0560
// Address: 0xdc0560
//
char __fastcall sub_DC0560(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // r14
  __int64 *v7; // rax
  __int64 v8; // r15
  _QWORD *v9; // rax
  __int64 v10; // r13
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // rax
  __int64 v15; // r12
  __int64 v16; // r14
  __int64 v17; // r15
  _QWORD *v18; // rax
  __int64 v19; // r14
  __int64 v20; // r15
  _QWORD *v21; // rax
  char result; // al
  _QWORD *v23; // rax
  __int64 v24; // [rsp-48h] [rbp-48h]
  __int64 v25; // [rsp-48h] [rbp-48h]
  __int64 v26; // [rsp-40h] [rbp-40h]
  __int64 v27; // [rsp-40h] [rbp-40h]

  if ( a2 == (_QWORD *)a3 )
    return 1;
  v6 = **(_QWORD **)(a3 + 32);
  v7 = (__int64 *)a2[4];
  v8 = *v7;
  if ( *v7 == v6
    || (v24 = *(_QWORD *)(a1 + 128),
        v26 = *(_QWORD *)(a1 + 112),
        v9 = sub_DA4260(v26, *v7, **(_QWORD **)(a3 + 32)),
        sub_D92140(v24, (__int64)v9, v26))
    || (v25 = *(_QWORD *)(a1 + 128),
        v27 = *(_QWORD *)(a1 + 112),
        v23 = sub_DA4260(v27, v6, v8),
        (result = sub_D92140(v25, (__int64)v23, v27)) != 0) )
  {
    v10 = sub_D33D80((_QWORD *)a3, *(_QWORD *)(a1 + 112), a3, a4, a5);
    v14 = sub_D33D80(a2, *(_QWORD *)(a1 + 112), v11, v12, v13);
    v15 = v14;
    if ( v10 == v14 )
      return 1;
    v16 = *(_QWORD *)(a1 + 112);
    v17 = *(_QWORD *)(a1 + 128);
    v18 = sub_DA4260(v16, v14, v10);
    if ( sub_D92140(v17, (__int64)v18, v16) )
    {
      return 1;
    }
    else
    {
      v19 = *(_QWORD *)(a1 + 112);
      v20 = *(_QWORD *)(a1 + 128);
      v21 = sub_DA4260(v19, v10, v15);
      return sub_D92140(v20, (__int64)v21, v19);
    }
  }
  return result;
}
