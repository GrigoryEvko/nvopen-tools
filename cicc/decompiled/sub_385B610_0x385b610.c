// Function: sub_385B610
// Address: 0x385b610
//
__int64 __fastcall sub_385B610(_QWORD *a1)
{
  __int64 *v1; // rdx
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 *v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx

  v1 = (__int64 *)a1[1];
  v2 = *v1;
  v3 = v1[1];
  if ( v2 == v3 )
LABEL_31:
    BUG();
  while ( *(_UNKNOWN **)v2 != &unk_4F9A488 )
  {
    v2 += 16;
    if ( v3 == v2 )
      goto LABEL_31;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v2 + 8) + 104LL))(*(_QWORD *)(v2 + 8), &unk_4F9A488);
  v6 = a1[1];
  a1[24] = *(_QWORD *)(v5 + 160);
  v7 = sub_160F9A0(v6, (__int64)&unk_4F9B6E8, 1u);
  if ( v7 && (v8 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v7 + 104LL))(v7, &unk_4F9B6E8)) != 0 )
    v9 = v8 + 360;
  else
    v9 = 0;
  v10 = (__int64 *)a1[1];
  a1[25] = v9;
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
LABEL_28:
    BUG();
  while ( *(_UNKNOWN **)v11 != &unk_4F96DB4 )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_28;
  }
  v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(*(_QWORD *)(v11 + 8), &unk_4F96DB4);
  v14 = (__int64 *)a1[1];
  a1[26] = *(_QWORD *)(v13 + 160);
  v15 = *v14;
  v16 = v14[1];
  if ( v15 == v16 )
LABEL_29:
    BUG();
  while ( *(_UNKNOWN **)v15 != &unk_4F9E06C )
  {
    v15 += 16;
    if ( v16 == v15 )
      goto LABEL_29;
  }
  v17 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v15 + 8) + 104LL))(*(_QWORD *)(v15 + 8), &unk_4F9E06C);
  v18 = (__int64 *)a1[1];
  a1[27] = v17 + 160;
  v19 = *v18;
  v20 = v18[1];
  if ( v19 == v20 )
LABEL_30:
    BUG();
  while ( *(_UNKNOWN **)v19 != &unk_4F9920C )
  {
    v19 += 16;
    if ( v20 == v19 )
      goto LABEL_30;
  }
  a1[28] = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v19 + 8) + 104LL))(
             *(_QWORD *)(v19 + 8),
             &unk_4F9920C)
         + 160;
  return 0;
}
