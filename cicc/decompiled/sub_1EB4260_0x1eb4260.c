// Function: sub_1EB4260
// Address: 0x1eb4260
//
__int64 __fastcall sub_1EB4260(_QWORD *a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v6; // rax
  __int64 *v7; // rdx
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 *v12; // rdx
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 *v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 *v21; // rdx
  __int64 v22; // r13
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rdi
  __int64 v28; // rdi

  v2 = (__int64 *)a1[1];
  a1[85] = a2;
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_39:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4FCF954 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_39;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4FCF954);
  v7 = (__int64 *)a1[1];
  v8 = v6;
  v9 = *v7;
  v10 = v7[1];
  if ( v9 == v10 )
LABEL_35:
    BUG();
  while ( *(_UNKNOWN **)v9 != &unk_4FC450C )
  {
    v9 += 16;
    if ( v10 == v9 )
      goto LABEL_35;
  }
  v11 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v9 + 8) + 104LL))(*(_QWORD *)(v9 + 8), &unk_4FC450C);
  v12 = (__int64 *)a1[1];
  v13 = v11;
  v14 = *v12;
  v15 = v12[1];
  if ( v14 == v15 )
LABEL_36:
    BUG();
  while ( *(_UNKNOWN **)v14 != &unk_4FCE424 )
  {
    v14 += 16;
    if ( v15 == v14 )
      goto LABEL_36;
  }
  v16 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v14 + 8) + 104LL))(*(_QWORD *)(v14 + 8), &unk_4FCE424);
  sub_210BBD0(a1 + 29, v16, v13, v8);
  v17 = (__int64 *)a1[1];
  v18 = *v17;
  v19 = v17[1];
  if ( v18 == v19 )
LABEL_37:
    BUG();
  while ( *(_UNKNOWN **)v18 != &unk_4FC453D )
  {
    v18 += 16;
    if ( v19 == v18 )
      goto LABEL_37;
  }
  v20 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v18 + 8) + 104LL))(*(_QWORD *)(v18 + 8), &unk_4FC453D);
  v21 = (__int64 *)a1[1];
  v22 = v20;
  v23 = *v21;
  v24 = v21[1];
  if ( v23 == v24 )
LABEL_38:
    BUG();
  while ( *(_UNKNOWN **)v23 != &unk_4FC6A0C )
  {
    v23 += 16;
    if ( v24 == v23 )
      goto LABEL_38;
  }
  v25 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v23 + 8) + 104LL))(*(_QWORD *)(v23 + 8), &unk_4FC6A0C);
  sub_20E2E20(a1[33], a1[85], a1[32], v25, v22, sub_1EB3B70);
  v26 = sub_20EAFC0(a1, a1[85], a1[32]);
  v27 = a1[86];
  a1[86] = v26;
  if ( v27 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v27 + 16LL))(v27);
  sub_210BE60(a1 + 29);
  sub_210B880(a1 + 29);
  v28 = a1[86];
  a1[86] = 0;
  if ( v28 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v28 + 16LL))(v28);
  return 1;
}
