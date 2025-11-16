// Function: sub_1C124D0
// Address: 0x1c124d0
//
__int64 __fastcall sub_1C124D0(__int64 a1, __int64 a2)
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
  __int64 v18; // r8
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // r15
  __int64 v23; // rdi
  __int64 v25; // [rsp+0h] [rbp-40h]
  __int64 v26; // [rsp+8h] [rbp-38h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_32:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4FBA0D0 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_32;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4FBA0D0);
  v7 = *(__int64 **)(a1 + 8);
  v8 = v6 + 160;
  v9 = *v7;
  v10 = v7[1];
  if ( v9 == v10 )
LABEL_29:
    BUG();
  while ( *(_UNKNOWN **)v9 != &unk_4FBA36C )
  {
    v9 += 16;
    if ( v10 == v9 )
      goto LABEL_29;
  }
  v11 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v9 + 8) + 104LL))(*(_QWORD *)(v9 + 8), &unk_4FBA36C);
  v12 = *(__int64 **)(a1 + 8);
  v13 = *(_QWORD *)(v11 + 160);
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
  while ( *(_UNKNOWN **)v19 != &unk_4F99CCC )
  {
    v19 += 16;
    if ( v20 == v19 )
      goto LABEL_31;
  }
  v25 = v18;
  v26 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v19 + 8) + 104LL))(*(_QWORD *)(v19 + 8), &unk_4F99CCC)
      + 160;
  v21 = sub_22077B0(32);
  v22 = v21;
  if ( v21 )
    sub_1C121D0(v21, a2, *(_BYTE *)(a1 + 168), v8, v25, v13, v26);
  v23 = *(_QWORD *)(a1 + 160);
  *(_QWORD *)(a1 + 160) = v22;
  if ( v23 )
    j_j___libc_free_0(v23, 32);
  return 0;
}
