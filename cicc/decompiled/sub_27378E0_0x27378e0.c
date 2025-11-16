// Function: sub_27378E0
// Address: 0x27378e0
//
__int64 __fastcall sub_27378E0(_QWORD *a1, __int64 a2)
{
  __int64 *v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // r15
  __int64 v8; // r13
  __int64 *v9; // rbx
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 *v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 *v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // [rsp+8h] [rbp-38h]

  if ( (unsigned __int8)sub_BB98D0(a1, a2) )
    return 0;
  v3 = (__int64 *)a1[1];
  v4 = *v3;
  v5 = v3[1];
  if ( v4 == v5 )
LABEL_30:
    BUG();
  while ( *(_UNKNOWN **)v4 != &unk_4F87C64 )
  {
    v4 += 16;
    if ( v5 == v4 )
      goto LABEL_30;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v4 + 8) + 104LL))(*(_QWORD *)(v4 + 8), &unk_4F87C64);
  v7 = *(_QWORD *)(a2 + 80);
  v8 = *(_QWORD *)(v6 + 176);
  if ( v7 )
    v7 -= 24;
  v9 = 0;
  if ( (_BYTE)qword_4FFA008 )
  {
    v20 = (__int64 *)a1[1];
    v21 = *v20;
    v22 = v20[1];
    if ( v22 == v21 )
LABEL_28:
      BUG();
    while ( *(_UNKNOWN **)v21 != &unk_4F8D9B0 )
    {
      v21 += 16;
      if ( v22 == v21 )
        goto LABEL_28;
    }
    v9 = (__int64 *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v21 + 8) + 104LL))(
                       *(_QWORD *)(v21 + 8),
                       &unk_4F8D9B0)
                   + 176);
  }
  v10 = (__int64 *)a1[1];
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
LABEL_29:
    BUG();
  while ( *(_UNKNOWN **)v11 != &unk_4F8144C )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_29;
  }
  v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(*(_QWORD *)(v11 + 8), &unk_4F8144C);
  v14 = (__int64 *)a1[1];
  v15 = v13 + 176;
  v16 = *v14;
  v17 = v14[1];
  if ( v16 == v17 )
LABEL_31:
    BUG();
  while ( *(_UNKNOWN **)v16 != &unk_4F89C28 )
  {
    v16 += 16;
    if ( v17 == v16 )
      goto LABEL_31;
  }
  v23 = v15;
  v18 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v16 + 8) + 104LL))(*(_QWORD *)(v16 + 8), &unk_4F89C28);
  v19 = sub_DFED00(v18, a2);
  return sub_2737780((__int64)(a1 + 22), a2, v19, v23, v9, v7, v8);
}
