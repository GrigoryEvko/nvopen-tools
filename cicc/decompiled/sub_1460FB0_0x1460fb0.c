// Function: sub_1460FB0
// Address: 0x1460fb0
//
__int64 __fastcall sub_1460FB0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 *v12; // rdx
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 *v17; // rdx
  __int64 v18; // r8
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // r15
  __int64 v23; // r12
  __int64 v25; // [rsp+0h] [rbp-40h]
  __int64 v26; // [rsp+8h] [rbp-38h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_32:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F9B6E8 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_32;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F9B6E8);
  v6 = *(__int64 **)(a1 + 8);
  v7 = v5 + 360;
  v8 = *v6;
  v9 = v6[1];
  if ( v8 == v9 )
LABEL_29:
    BUG();
  while ( *(_UNKNOWN **)v8 != &unk_4F9D764 )
  {
    v8 += 16;
    if ( v9 == v8 )
      goto LABEL_29;
  }
  v10 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(*(_QWORD *)(v8 + 8), &unk_4F9D764);
  v11 = sub_14CF090(v10, a2);
  v12 = *(__int64 **)(a1 + 8);
  v13 = v11;
  v14 = *v12;
  v15 = v12[1];
  if ( v14 == v15 )
LABEL_30:
    BUG();
  while ( *(_UNKNOWN **)v14 != &unk_4F9E06C )
  {
    v14 += 16;
    if ( v15 == v14 )
      goto LABEL_30;
  }
  v16 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v14 + 8) + 104LL))(*(_QWORD *)(v14 + 8), &unk_4F9E06C);
  v17 = *(__int64 **)(a1 + 8);
  v18 = v16 + 160;
  v19 = *v17;
  v20 = v17[1];
  if ( v19 == v20 )
LABEL_31:
    BUG();
  while ( *(_UNKNOWN **)v19 != &unk_4F9920C )
  {
    v19 += 16;
    if ( v20 == v19 )
      goto LABEL_31;
  }
  v25 = v18;
  v26 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v19 + 8) + 104LL))(*(_QWORD *)(v19 + 8), &unk_4F9920C)
      + 160;
  v21 = sub_22077B0(1040);
  v22 = v21;
  if ( v21 )
    sub_1457DF0(v21, a2, v7, v13, v25, v26);
  v23 = *(_QWORD *)(a1 + 160);
  *(_QWORD *)(a1 + 160) = v22;
  if ( v23 )
  {
    sub_14602B0(v23);
    j_j___libc_free_0(v23, 1040);
  }
  return 0;
}
