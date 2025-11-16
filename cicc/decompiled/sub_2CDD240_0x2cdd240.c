// Function: sub_2CDD240
// Address: 0x2cdd240
//
__int64 __fastcall sub_2CDD240(__int64 a1, __int64 a2, unsigned __int8 a3)
{
  __int64 *v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 *v8; // rdx
  __int64 **v9; // r13
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 *v13; // rdx
  __int64 v14; // r14
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 *v18; // rdx
  __int64 v19; // r9
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v24; // [rsp+8h] [rbp-38h]

  v4 = *(__int64 **)(a1 + 8);
  v5 = *v4;
  v6 = v4[1];
  if ( v5 == v6 )
LABEL_28:
    BUG();
  while ( *(_UNKNOWN **)v5 != &unk_4F8144C )
  {
    v5 += 16;
    if ( v6 == v5 )
      goto LABEL_28;
  }
  v7 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v5 + 8) + 104LL))(*(_QWORD *)(v5 + 8), &unk_4F8144C);
  v8 = *(__int64 **)(a1 + 8);
  v9 = (__int64 **)(v7 + 176);
  v10 = *v8;
  v11 = v8[1];
  if ( v10 == v11 )
LABEL_25:
    BUG();
  while ( *(_UNKNOWN **)v10 != &unk_4F86530 )
  {
    v10 += 16;
    if ( v11 == v10 )
      goto LABEL_25;
  }
  v12 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v10 + 8) + 104LL))(*(_QWORD *)(v10 + 8), &unk_4F86530);
  v13 = *(__int64 **)(a1 + 8);
  v14 = *(_QWORD *)(v12 + 176);
  v15 = *v13;
  v16 = v13[1];
  if ( v15 == v16 )
LABEL_26:
    BUG();
  while ( *(_UNKNOWN **)v15 != &unk_5035D54 )
  {
    v15 += 16;
    if ( v16 == v15 )
      goto LABEL_26;
  }
  v17 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v15 + 8) + 104LL))(*(_QWORD *)(v15 + 8), &unk_5035D54);
  v18 = *(__int64 **)(a1 + 8);
  v19 = v17 + 172;
  v20 = *v18;
  v21 = v18[1];
  if ( v20 == v21 )
LABEL_27:
    BUG();
  while ( *(_UNKNOWN **)v20 != &unk_5010CD4 )
  {
    v20 += 16;
    if ( v21 == v20 )
      goto LABEL_27;
  }
  v24 = v19;
  v22 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v20 + 8) + 104LL))(*(_QWORD *)(v20 + 8), &unk_5010CD4);
  return sub_2CDCF50((__int64 *)(a1 + 176), a2, a3, v9, v14, v24, *(_QWORD *)(v22 + 176));
}
