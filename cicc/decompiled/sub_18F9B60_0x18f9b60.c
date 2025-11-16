// Function: sub_18F9B60
// Address: 0x18f9b60
//
__int64 __fastcall sub_18F9B60(__int64 a1, __int64 a2)
{
  __int64 *v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rdx
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
  __int64 v18; // r15
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax

  if ( (unsigned __int8)sub_1636880(a1, a2) )
    return 0;
  v3 = *(__int64 **)(a1 + 8);
  v4 = *v3;
  v5 = v3[1];
  if ( v4 == v5 )
    goto LABEL_20;
  while ( *(_UNKNOWN **)v4 != &unk_4F9E06C )
  {
    v4 += 16;
    if ( v5 == v4 )
      goto LABEL_20;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v4 + 8) + 104LL))(*(_QWORD *)(v4 + 8), &unk_4F9E06C);
  v7 = *(__int64 **)(a1 + 8);
  v8 = v6 + 160;
  v9 = *v7;
  v10 = v7[1];
  if ( v9 == v10 )
    goto LABEL_20;
  while ( *(_UNKNOWN **)v9 != &unk_4F96DB4 )
  {
    v9 += 16;
    if ( v10 == v9 )
      goto LABEL_20;
  }
  v11 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v9 + 8) + 104LL))(*(_QWORD *)(v9 + 8), &unk_4F96DB4);
  v12 = *(__int64 **)(a1 + 8);
  v13 = *(_QWORD *)(v11 + 160);
  v14 = *v12;
  v15 = v12[1];
  if ( v14 == v15 )
    goto LABEL_20;
  while ( *(_UNKNOWN **)v14 != &unk_4F99308 )
  {
    v14 += 16;
    if ( v15 == v14 )
      goto LABEL_20;
  }
  v16 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v14 + 8) + 104LL))(*(_QWORD *)(v14 + 8), &unk_4F99308);
  v17 = *(__int64 **)(a1 + 8);
  v18 = v16 + 160;
  v19 = *v17;
  v20 = v17[1];
  if ( v19 == v20 )
LABEL_20:
    BUG();
  while ( *(_UNKNOWN **)v19 != &unk_4F9B6E8 )
  {
    v19 += 16;
    if ( v20 == v19 )
      goto LABEL_20;
  }
  v21 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v19 + 8) + 104LL))(*(_QWORD *)(v19 + 8), &unk_4F9B6E8);
  return sub_18F9A50(a2, v13, v18, v8, (__int64 *)(v21 + 360));
}
