// Function: sub_1413E40
// Address: 0x1413e40
//
__int64 __fastcall sub_1413E40(__int64 a1, __int64 a2)
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
  __int64 v12; // rax
  __int64 *v13; // rdx
  __int64 v14; // r12
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 *v18; // rdx
  __int64 v19; // r14
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 *v23; // rdx
  __int64 v24; // r15
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rdi
  __int64 v29; // r9
  __int64 v31; // [rsp+0h] [rbp-40h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_37:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F96DB4 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_37;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F96DB4);
  v7 = *(__int64 **)(a1 + 8);
  v8 = *(_QWORD *)(v6 + 160);
  v9 = *v7;
  v10 = v7[1];
  if ( v9 == v10 )
LABEL_33:
    BUG();
  while ( *(_UNKNOWN **)v9 != &unk_4F9D764 )
  {
    v9 += 16;
    if ( v10 == v9 )
      goto LABEL_33;
  }
  v11 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v9 + 8) + 104LL))(*(_QWORD *)(v9 + 8), &unk_4F9D764);
  v12 = sub_14CF090(v11, a2);
  v13 = *(__int64 **)(a1 + 8);
  v14 = v12;
  v15 = *v13;
  v16 = v13[1];
  if ( v15 == v16 )
LABEL_34:
    BUG();
  while ( *(_UNKNOWN **)v15 != &unk_4F9B6E8 )
  {
    v15 += 16;
    if ( v16 == v15 )
      goto LABEL_34;
  }
  v17 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v15 + 8) + 104LL))(*(_QWORD *)(v15 + 8), &unk_4F9B6E8);
  v18 = *(__int64 **)(a1 + 8);
  v19 = v17 + 360;
  v20 = *v18;
  v21 = v18[1];
  if ( v20 == v21 )
LABEL_35:
    BUG();
  while ( *(_UNKNOWN **)v20 != &unk_4F9E06C )
  {
    v20 += 16;
    if ( v21 == v20 )
      goto LABEL_35;
  }
  v22 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v20 + 8) + 104LL))(*(_QWORD *)(v20 + 8), &unk_4F9E06C);
  v23 = *(__int64 **)(a1 + 8);
  v24 = v22 + 160;
  v25 = *v23;
  v26 = v23[1];
  if ( v25 == v26 )
LABEL_36:
    BUG();
  while ( *(_UNKNOWN **)v25 != &unk_4F99CC4 )
  {
    v25 += 16;
    if ( v26 == v25 )
      goto LABEL_36;
  }
  v27 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v25 + 8) + 104LL))(*(_QWORD *)(v25 + 8), &unk_4F99CC4);
  v28 = a1 + 160;
  v29 = *(_QWORD *)(v27 + 160);
  if ( *(_BYTE *)(a1 + 1104) )
  {
    v31 = *(_QWORD *)(v27 + 160);
    sub_14139D0(v28);
    v29 = v31;
    v28 = a1 + 160;
  }
  *(_BYTE *)(a1 + 1104) = 1;
  sub_1412230(v28, v8, v14, v19, v24, v29);
  return 0;
}
