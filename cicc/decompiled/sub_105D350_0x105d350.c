// Function: sub_105D350
// Address: 0x105d350
//
__int64 __fastcall sub_105D350(_QWORD *a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 *v7; // r12
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 *v11; // rdx
  __int64 v12; // r13
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // r15
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rsi
  __int64 *v21; // rdx
  __int64 v23[7]; // [rsp+8h] [rbp-38h] BYREF

  v2 = (__int64 *)a1[1];
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_26:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F92384 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_26;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F92384);
  v6 = (__int64 *)a1[1];
  v7 = (__int64 *)(v5 + 184);
  v8 = *v6;
  v9 = v6[1];
  if ( v8 == v9 )
LABEL_24:
    BUG();
  while ( *(_UNKNOWN **)v8 != &unk_4F8144C )
  {
    v8 += 16;
    if ( v9 == v8 )
      goto LABEL_24;
  }
  v10 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(*(_QWORD *)(v8 + 8), &unk_4F8144C);
  v11 = (__int64 *)a1[1];
  v12 = v10;
  v13 = *v11;
  v14 = v11[1];
  if ( v13 == v14 )
LABEL_25:
    BUG();
  while ( *(_UNKNOWN **)v13 != &unk_4F89C28 )
  {
    v13 += 16;
    if ( v14 == v13 )
      goto LABEL_25;
  }
  v15 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v13 + 8) + 104LL))(*(_QWORD *)(v13 + 8), &unk_4F89C28);
  v16 = sub_DFED00(v15, a2);
  a1[22] = a2;
  v17 = v16;
  sub_105CFE0(v23, v12 + 176, v7, v16);
  v18 = v23[0];
  v19 = a1[23];
  v23[0] = 0;
  a1[23] = v18;
  if ( v19 )
  {
    sub_10568E0((__int64)(a1 + 23), v19);
    if ( v23[0] )
      sub_10568E0((__int64)v23, v23[0]);
  }
  v20 = a1[22];
  if ( (unsigned __int8)sub_DF9710(v17) )
  {
    sub_1058900(a1[23], v20, v21);
    sub_105BE30(a1[23]);
  }
  return 0;
}
