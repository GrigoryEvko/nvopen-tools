// Function: sub_D8AE10
// Address: 0xd8ae10
//
__int64 __fastcall sub_D8AE10(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // rdi
  __int64 v9; // r12
  __int64 *v10; // rdx
  __int64 *v11; // rax
  unsigned __int64 v12; // r12
  char v13; // al
  __int64 v14; // r12
  unsigned __int64 v15; // rbx
  _QWORD *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r14
  _BOOL4 v20; // r8d
  __int64 v21; // rax
  __int64 v22; // r13
  __int64 v23; // r14
  unsigned __int64 v24; // rbx
  _QWORD *v25; // rax
  __int64 v26; // rdx
  __int64 v27; // r12
  _BOOL4 v28; // r8d
  __int64 v29; // rax
  _BOOL4 v30; // [rsp+Ch] [rbp-64h]
  _BOOL4 v31; // [rsp+Ch] [rbp-64h]
  unsigned __int64 v32; // [rsp+18h] [rbp-58h] BYREF
  __int64 v33[2]; // [rsp+20h] [rbp-50h] BYREF
  __int64 v34[8]; // [rsp+30h] [rbp-40h] BYREF

  if ( **(_QWORD **)a1 == a2
    || (v4 = **(_QWORD **)(a1 + 32)) != 0
    && !(unsigned __int8)sub_104D360(*(_QWORD *)(a1 + 40), v4, **(_QWORD **)(a1 + 16)) )
  {
    v22 = *(_QWORD *)(a1 + 8);
    v23 = *(_QWORD *)(a1 + 24) + 32LL;
    v24 = **(_QWORD **)(a1 + 16);
    v33[0] = v24;
    v25 = sub_D8ACC0(v22 + 32, (unsigned __int64 *)v33);
    v27 = v26;
    if ( v26 )
    {
      v28 = 1;
      if ( !v25 && v26 != v22 + 40 )
        v28 = v24 < *(_QWORD *)(v26 + 32);
      v31 = v28;
      v29 = sub_22077B0(40);
      *(_QWORD *)(v29 + 32) = v33[0];
      sub_220F040(v31, v29, v27, v22 + 40);
      ++*(_QWORD *)(v22 + 72);
    }
    return sub_D87370(v22, v23);
  }
  else
  {
    v5 = sub_9208B0(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL), *(_QWORD *)(a2 + 8));
    v6 = *(_QWORD *)(a1 + 24);
    v33[0] = v5;
    v8 = v7;
    v9 = v5;
    v10 = *(__int64 **)(a1 + 56);
    v11 = *(__int64 **)(a1 + 48);
    v33[1] = v8;
    v12 = (unsigned __int64)(v9 + 7) >> 3;
    sub_D89430((__int64)v33, v6, *v11, *v10, v12, v8);
    v13 = sub_D88E20(*(_QWORD *)(a1 + 24), *(__int64 **)(a1 + 48), **(_QWORD **)(a1 + 32), v12, v8);
    v14 = *(_QWORD *)(a1 + 8);
    v15 = **(_QWORD **)(a1 + 16);
    v32 = v15;
    if ( !v13 )
    {
      v17 = sub_D8ACC0(v14 + 32, &v32);
      v19 = v18;
      if ( v18 )
      {
        v20 = 1;
        if ( !v17 && v18 != v14 + 40 )
          v20 = v15 < *(_QWORD *)(v18 + 32);
        v30 = v20;
        v21 = sub_22077B0(40);
        *(_QWORD *)(v21 + 32) = v32;
        sub_220F040(v30, v21, v19, v14 + 40);
        ++*(_QWORD *)(v14 + 72);
      }
    }
    sub_D87370(v14, (__int64)v33);
    sub_969240(v34);
    return sub_969240(v33);
  }
}
