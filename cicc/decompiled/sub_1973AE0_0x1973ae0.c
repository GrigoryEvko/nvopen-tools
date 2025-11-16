// Function: sub_1973AE0
// Address: 0x1973ae0
//
__int64 __fastcall sub_1973AE0(__int64 a1, __int64 a2)
{
  __int64 *v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 *v7; // rdx
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 *v12; // rdx
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 *v18; // rdx
  __int64 v19; // r15
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax

  if ( (unsigned __int8)sub_1404700(a1, a2) )
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
  while ( *(_UNKNOWN **)v9 != &unk_4F9920C )
  {
    v9 += 16;
    if ( v10 == v9 )
      goto LABEL_20;
  }
  v11 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v9 + 8) + 104LL))(*(_QWORD *)(v9 + 8), &unk_4F9920C);
  v12 = *(__int64 **)(a1 + 8);
  v13 = v11 + 160;
  v14 = *v12;
  v15 = v12[1];
  if ( v14 == v15 )
    goto LABEL_20;
  while ( *(_UNKNOWN **)v14 != &unk_4F9D764 )
  {
    v14 += 16;
    if ( v15 == v14 )
      goto LABEL_20;
  }
  v16 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v14 + 8) + 104LL))(*(_QWORD *)(v14 + 8), &unk_4F9D764);
  v17 = sub_14CF090(v16, *(_QWORD *)(**(_QWORD **)(a2 + 32) + 56LL));
  v18 = *(__int64 **)(a1 + 8);
  v19 = v17;
  v20 = *v18;
  v21 = v18[1];
  if ( v20 == v21 )
LABEL_20:
    BUG();
  while ( *(_UNKNOWN **)v20 != &unk_4F9B6E8 )
  {
    v20 += 16;
    if ( v21 == v20 )
      goto LABEL_20;
  }
  v22 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v20 + 8) + 104LL))(*(_QWORD *)(v20 + 8), &unk_4F9B6E8);
  return sub_1972D40(a2, v8, v13, v19, v22 + 360);
}
