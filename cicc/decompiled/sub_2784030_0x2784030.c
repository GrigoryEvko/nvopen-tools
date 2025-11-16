// Function: sub_2784030
// Address: 0x2784030
//
__int64 __fastcall sub_2784030(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 *v7; // rdx
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 *v13; // rdx
  __int64 v14; // r13
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 *v18; // rdx
  __int64 v19; // rbx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 *v24; // rdx
  __int64 v25; // r9
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rax
  unsigned int v29; // r12d
  __int64 v31; // [rsp+0h] [rbp-330h]
  __int64 v32; // [rsp+8h] [rbp-328h]
  _BYTE v33[800]; // [rsp+10h] [rbp-320h] BYREF

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_35:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F6D3F0 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_35;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F6D3F0);
  v6 = sub_277B500(v5, a2);
  v7 = *(__int64 **)(a1 + 8);
  v8 = v6;
  v9 = *v7;
  v10 = v7[1];
  if ( v9 == v10 )
LABEL_31:
    BUG();
  while ( *(_UNKNOWN **)v9 != &unk_4F89C28 )
  {
    v9 += 16;
    if ( v10 == v9 )
      goto LABEL_31;
  }
  v11 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v9 + 8) + 104LL))(*(_QWORD *)(v9 + 8), &unk_4F89C28);
  v12 = sub_DFED00(v11, a2);
  v13 = *(__int64 **)(a1 + 8);
  v14 = v12;
  v15 = *v13;
  v16 = v13[1];
  if ( v15 == v16 )
LABEL_32:
    BUG();
  while ( *(_UNKNOWN **)v15 != &unk_4F8144C )
  {
    v15 += 16;
    if ( v16 == v15 )
      goto LABEL_32;
  }
  v17 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v15 + 8) + 104LL))(*(_QWORD *)(v15 + 8), &unk_4F8144C);
  v18 = *(__int64 **)(a1 + 8);
  v19 = v17 + 176;
  v20 = *v18;
  v21 = v18[1];
  if ( v20 == v21 )
LABEL_33:
    BUG();
  while ( *(_UNKNOWN **)v20 != &unk_4F8662C )
  {
    v20 += 16;
    if ( v21 == v20 )
      goto LABEL_33;
  }
  v22 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v20 + 8) + 104LL))(*(_QWORD *)(v20 + 8), &unk_4F8662C);
  v23 = sub_CFFAC0(v22, a2);
  v24 = *(__int64 **)(a1 + 8);
  v25 = v23;
  v26 = *v24;
  v27 = v24[1];
  if ( v26 == v27 )
LABEL_34:
    BUG();
  while ( *(_UNKNOWN **)v26 != &unk_4F8F808 )
  {
    v26 += 16;
    if ( v27 == v26 )
      goto LABEL_34;
  }
  v31 = v25;
  v32 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v26 + 8) + 104LL))(
                      *(_QWORD *)(v26 + 8),
                      &unk_4F8F808)
                  + 176);
  v28 = sub_B2BEC0(a2);
  sub_2778920((__int64)v33, v28, v8, v14, v19, v31, v32);
  v29 = sub_2780B00((__int64)v33);
  sub_277A450((__int64)v33);
  return v29;
}
