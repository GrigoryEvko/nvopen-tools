// Function: sub_35D3EE0
// Address: 0x35d3ee0
//
_BOOL8 __fastcall sub_35D3EE0(_QWORD *a1, __int64 *a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  bool v15; // [rsp+Fh] [rbp-31h]
  _BYTE v16[48]; // [rsp+10h] [rbp-30h] BYREF

  v2 = (__int64 *)a1[1];
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_26:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_501F1C8 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_26;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_501F1C8);
  v6 = (__int64 *)a1[1];
  a1[25] = v5 + 169;
  v7 = *v6;
  v8 = v6[1];
  if ( v7 == v8 )
LABEL_24:
    BUG();
  while ( *(_UNKNOWN **)v7 != &unk_501EC08 )
  {
    v7 += 16;
    if ( v8 == v7 )
      goto LABEL_24;
  }
  v9 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v7 + 8) + 104LL))(*(_QWORD *)(v7 + 8), &unk_501EC08);
  v10 = (__int64 *)a1[1];
  a1[26] = v9 + 200;
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
LABEL_25:
    BUG();
  while ( *(_UNKNOWN **)v11 != &unk_4F87C64 )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_25;
  }
  v13 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(
                      *(_QWORD *)(v11 + 8),
                      &unk_4F87C64)
                  + 176);
  a1[27] = v13;
  if ( v13 && *(_QWORD *)(v13 + 8) && a1[26] && (sub_B2EE70((__int64)v16, *a2, 0), v16[16]) )
  {
    v15 = sub_35D3D40((__int64)a1, (__int64)a2);
    sub_35D3EC0();
    return v15;
  }
  else
  {
    sub_35D3ED0();
    return 0;
  }
}
