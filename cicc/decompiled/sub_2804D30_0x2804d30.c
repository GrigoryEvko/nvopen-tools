// Function: sub_2804D30
// Address: 0x2804d30
//
__int64 __fastcall sub_2804D30(_QWORD *a1, __int64 a2)
{
  __int64 *v3; // rcx
  __int64 v4; // rax
  __int64 v5; // rcx
  __int64 v6; // rax
  __int64 *v7; // rcx
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rax
  __int64 *v11; // rcx
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 *v15; // rcx
  __int64 v16; // r15
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 *v21; // rcx
  __int64 v22; // r14
  __int64 v23; // rax
  __int64 v24; // rcx
  __int64 v25; // rax
  __int64 *v26; // rdx
  __int64 v27; // r13
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // [rsp+0h] [rbp-70h]
  __int64 v32; // [rsp+8h] [rbp-68h]
  _QWORD v33[12]; // [rsp+10h] [rbp-60h] BYREF

  if ( (unsigned __int8)sub_BB98D0(a1, a2) )
    return 0;
  v3 = (__int64 *)a1[1];
  v4 = *v3;
  v5 = v3[1];
  if ( v4 == v5 )
LABEL_39:
    BUG();
  while ( *(_UNKNOWN **)v4 != &unk_4F8144C )
  {
    v4 += 16;
    if ( v5 == v4 )
      goto LABEL_39;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v4 + 8) + 104LL))(*(_QWORD *)(v4 + 8), &unk_4F8144C);
  v7 = (__int64 *)a1[1];
  v31 = v6 + 176;
  v8 = *v7;
  v9 = v7[1];
  if ( v8 == v9 )
LABEL_40:
    BUG();
  while ( *(_UNKNOWN **)v8 != &unk_4F875EC )
  {
    v8 += 16;
    if ( v9 == v8 )
      goto LABEL_40;
  }
  v10 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(*(_QWORD *)(v8 + 8), &unk_4F875EC);
  v11 = (__int64 *)a1[1];
  v32 = v10 + 176;
  v12 = *v11;
  v13 = v11[1];
  if ( v12 == v13 )
LABEL_41:
    BUG();
  while ( *(_UNKNOWN **)v12 != &unk_4F881C8 )
  {
    v12 += 16;
    if ( v13 == v12 )
      goto LABEL_41;
  }
  v14 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v12 + 8) + 104LL))(*(_QWORD *)(v12 + 8), &unk_4F881C8);
  v15 = (__int64 *)a1[1];
  v16 = *(_QWORD *)(v14 + 176);
  v17 = *v15;
  v18 = v15[1];
  if ( v17 == v18 )
LABEL_42:
    BUG();
  while ( *(_UNKNOWN **)v17 != &unk_4F8662C )
  {
    v17 += 16;
    if ( v18 == v17 )
      goto LABEL_42;
  }
  v19 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v17 + 8) + 104LL))(*(_QWORD *)(v17 + 8), &unk_4F8662C);
  v20 = sub_CFFAC0(v19, a2);
  v21 = (__int64 *)a1[1];
  v22 = v20;
  v23 = *v21;
  v24 = v21[1];
  if ( v23 == v24 )
LABEL_43:
    BUG();
  while ( *(_UNKNOWN **)v23 != &unk_4F8FAE4 )
  {
    v23 += 16;
    if ( v24 == v23 )
      goto LABEL_43;
  }
  v25 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v23 + 8) + 104LL))(*(_QWORD *)(v23 + 8), &unk_4F8FAE4);
  v26 = (__int64 *)a1[1];
  v27 = *(_QWORD *)(v25 + 176);
  v28 = *v26;
  v29 = v26[1];
  if ( v28 == v29 )
LABEL_44:
    BUG();
  while ( *(_UNKNOWN **)v28 != &unk_4F89C28 )
  {
    v28 += 16;
    if ( v29 == v28 )
      goto LABEL_44;
  }
  v30 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v28 + 8) + 104LL))(*(_QWORD *)(v28 + 8), &unk_4F89C28);
  v33[0] = v22;
  v33[3] = v16;
  v33[1] = v31;
  v33[5] = v27;
  v33[2] = v32;
  v33[4] = sub_DFED00(v30, a2);
  return sub_2804C20(v33);
}
