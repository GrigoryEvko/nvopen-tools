// Function: sub_1968CE0
// Address: 0x1968ce0
//
bool __fastcall sub_1968CE0(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 *v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 *v9; // rdx
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 *v14; // rdx
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  int v19; // ebx

  if ( (unsigned __int8)sub_1404700(a1, (__int64)a2) )
    return 0;
  v5 = *(__int64 **)(a1 + 8);
  v6 = *v5;
  v7 = v5[1];
  if ( v6 == v7 )
LABEL_23:
    BUG();
  while ( *(_UNKNOWN **)v6 != &unk_4F9E06C )
  {
    v6 += 16;
    if ( v7 == v6 )
      goto LABEL_23;
  }
  v8 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v6 + 8) + 104LL))(*(_QWORD *)(v6 + 8), &unk_4F9E06C);
  v9 = *(__int64 **)(a1 + 8);
  v10 = v8 + 160;
  v11 = *v9;
  v12 = v9[1];
  if ( v11 == v12 )
LABEL_25:
    BUG();
  while ( *(_UNKNOWN **)v11 != &unk_4F9A488 )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_25;
  }
  v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(*(_QWORD *)(v11 + 8), &unk_4F9A488);
  v14 = *(__int64 **)(a1 + 8);
  v15 = *(_QWORD *)(v13 + 160);
  v16 = *v14;
  v17 = v14[1];
  if ( v16 == v17 )
LABEL_24:
    BUG();
  while ( *(_UNKNOWN **)v16 != &unk_4F9920C )
  {
    v16 += 16;
    if ( v17 == v16 )
      goto LABEL_24;
  }
  v18 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v16 + 8) + 104LL))(*(_QWORD *)(v16 + 8), &unk_4F9920C);
  v19 = sub_1968470(a2, v10, v15, v18 + 160);
  if ( v19 == 2 )
    sub_1407870(a3, (__int64)a2);
  return v19 != 0;
}
