// Function: sub_198DAF0
// Address: 0x198daf0
//
__int64 __fastcall sub_198DAF0(__int64 a1, __int64 a2)
{
  __int64 *v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r15
  int v7; // eax
  __int64 *v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  int v12; // eax
  __int64 *v13; // rdx
  int v14; // r14d
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  int v20; // r13d
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // r9
  int v24; // [rsp+0h] [rbp-80h]
  int v25; // [rsp+8h] [rbp-78h]
  int v26; // [rsp+10h] [rbp-70h]
  __int64 v27[12]; // [rsp+20h] [rbp-60h] BYREF

  if ( (unsigned __int8)sub_1404700(a1, a2) )
    return 0;
  v3 = *(__int64 **)(a1 + 8);
  v4 = *v3;
  v5 = v3[1];
  if ( v4 == v5 )
LABEL_29:
    BUG();
  while ( *(_UNKNOWN **)v4 != &unk_4F9920C )
  {
    v4 += 16;
    if ( v5 == v4 )
      goto LABEL_29;
  }
  v6 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 56LL);
  v7 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v4 + 8) + 104LL))(*(_QWORD *)(v4 + 8), &unk_4F9920C);
  v8 = *(__int64 **)(a1 + 8);
  v24 = v7 + 160;
  v9 = *v8;
  v10 = v8[1];
  if ( v9 == v10 )
LABEL_31:
    BUG();
  while ( *(_UNKNOWN **)v9 != &unk_4F9D3C0 )
  {
    v9 += 16;
    if ( v10 == v9 )
      goto LABEL_31;
  }
  v11 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v9 + 8) + 104LL))(*(_QWORD *)(v9 + 8), &unk_4F9D3C0);
  v12 = sub_14A4050(v11, v6);
  v13 = *(__int64 **)(a1 + 8);
  v14 = v12;
  v15 = *v13;
  v16 = v13[1];
  if ( v15 == v16 )
LABEL_30:
    BUG();
  while ( *(_UNKNOWN **)v15 != &unk_4F9D764 )
  {
    v15 += 16;
    if ( v16 == v15 )
      goto LABEL_30;
  }
  v17 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v15 + 8) + 104LL))(*(_QWORD *)(v15 + 8), &unk_4F9D764);
  v25 = sub_14CF090(v17, v6);
  v18 = sub_160F9A0(*(_QWORD *)(a1 + 8), (__int64)&unk_4F9E06C, 1u);
  if ( v18 && (v19 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v18 + 104LL))(v18, &unk_4F9E06C)) != 0 )
    v20 = v19 + 160;
  else
    v20 = 0;
  v21 = sub_160F9A0(*(_QWORD *)(a1 + 8), (__int64)&unk_4F9A488, 1u);
  if ( v21 && (v22 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v21 + 104LL))(v21, &unk_4F9A488)) != 0 )
    v23 = *(_QWORD *)(v22 + 160);
  else
    LODWORD(v23) = 0;
  v26 = v23;
  sub_13D1E20(v27, a1, v6);
  return sub_1AF8020(a2, v24, v14, v25, v20, v26, (__int64)v27, 0, *(_DWORD *)(a1 + 156), 0);
}
