// Function: sub_198E630
// Address: 0x198e630
//
__int64 __fastcall sub_198E630(__int64 a1, __int64 a2)
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

  if ( (unsigned __int8)sub_1404700(a1, a2) )
    return 0;
  v3 = *(__int64 **)(a1 + 8);
  v4 = *v3;
  v5 = v3[1];
  if ( v4 == v5 )
    goto LABEL_16;
  while ( *(_UNKNOWN **)v4 != &unk_4F9E06C )
  {
    v4 += 16;
    if ( v5 == v4 )
      goto LABEL_16;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v4 + 8) + 104LL))(*(_QWORD *)(v4 + 8), &unk_4F9E06C);
  v7 = *(__int64 **)(a1 + 8);
  v8 = v6 + 160;
  v9 = *v7;
  v10 = v7[1];
  if ( v9 == v10 )
    goto LABEL_16;
  while ( *(_UNKNOWN **)v9 != &unk_4F9920C )
  {
    v9 += 16;
    if ( v10 == v9 )
      goto LABEL_16;
  }
  v11 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v9 + 8) + 104LL))(*(_QWORD *)(v9 + 8), &unk_4F9920C);
  v12 = *(__int64 **)(a1 + 8);
  v13 = v11 + 160;
  v14 = *v12;
  v15 = v12[1];
  if ( v14 == v15 )
LABEL_16:
    BUG();
  while ( *(_UNKNOWN **)v14 != &unk_4F9A488 )
  {
    v14 += 16;
    if ( v15 == v14 )
      goto LABEL_16;
  }
  v16 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v14 + 8) + 104LL))(*(_QWORD *)(v14 + 8), &unk_4F9A488);
  return sub_198E380(a2, v8, v13, *(_QWORD *)(v16 + 160));
}
