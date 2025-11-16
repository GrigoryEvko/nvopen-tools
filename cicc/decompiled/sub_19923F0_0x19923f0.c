// Function: sub_19923F0
// Address: 0x19923f0
//
__int64 __fastcall sub_19923F0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 *v10; // rdx
  __int64 *v11; // r14
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 *v15; // rdx
  __int64 v16; // r13
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 *v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // rax
  __int64 v25; // [rsp+8h] [rbp-38h]

  if ( (unsigned __int8)sub_1404700(a1, a2) )
    return 0;
  v3 = sub_160F9A0(*(_QWORD *)(a1 + 8), (__int64)&unk_4F9A488, 1u);
  if ( v3 && (v4 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v3 + 104LL))(v3, &unk_4F9A488)) != 0 )
    v5 = *(_QWORD *)(v4 + 160);
  else
    v5 = 0;
  v6 = *(__int64 **)(a1 + 8);
  v7 = *v6;
  v8 = v6[1];
  if ( v7 == v8 )
    goto LABEL_24;
  while ( *(_UNKNOWN **)v7 != &unk_4F97E48 )
  {
    v7 += 16;
    if ( v8 == v7 )
      goto LABEL_24;
  }
  v9 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v7 + 8) + 104LL))(*(_QWORD *)(v7 + 8), &unk_4F97E48);
  v10 = *(__int64 **)(a1 + 8);
  v11 = (__int64 *)(v9 + 160);
  v12 = *v10;
  v13 = v10[1];
  if ( v12 == v13 )
    goto LABEL_24;
  while ( *(_UNKNOWN **)v12 != &unk_4F9E06C )
  {
    v12 += 16;
    if ( v13 == v12 )
      goto LABEL_24;
  }
  v14 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v12 + 8) + 104LL))(*(_QWORD *)(v12 + 8), &unk_4F9E06C);
  v15 = *(__int64 **)(a1 + 8);
  v16 = v14 + 160;
  v17 = *v15;
  v18 = v15[1];
  if ( v17 == v18 )
    goto LABEL_24;
  while ( *(_UNKNOWN **)v17 != &unk_4F9920C )
  {
    v17 += 16;
    if ( v18 == v17 )
      goto LABEL_24;
  }
  v19 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v17 + 8) + 104LL))(*(_QWORD *)(v17 + 8), &unk_4F9920C);
  v20 = *(__int64 **)(a1 + 8);
  v21 = v19 + 160;
  v22 = *v20;
  v23 = v20[1];
  if ( v22 == v23 )
LABEL_24:
    BUG();
  while ( *(_UNKNOWN **)v22 != &unk_4F96DB4 )
  {
    v22 += 16;
    if ( v23 == v22 )
      goto LABEL_24;
  }
  v25 = v21;
  v24 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v22 + 8) + 104LL))(*(_QWORD *)(v22 + 8), &unk_4F96DB4);
  return sub_1990AC0(a2, *(_QWORD *)(v24 + 160), v25, v16, v11, v5);
}
