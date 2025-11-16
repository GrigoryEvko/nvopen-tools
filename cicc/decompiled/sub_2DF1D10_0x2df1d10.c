// Function: sub_2DF1D10
// Address: 0x2df1d10
//
__int64 __fastcall sub_2DF1D10(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rcx
  __int64 *v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 *v11; // rdx
  __int64 v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 *v16; // rdx
  __int64 v17; // r15
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 (*v25)(); // rax
  __int64 v26; // rdi
  __int64 (*v27)(); // rdx
  __int64 v28; // rax
  __int64 v29; // [rsp-68h] [rbp-68h] BYREF
  __int64 v30; // [rsp-60h] [rbp-60h]
  __int64 v31; // [rsp-58h] [rbp-58h]
  __int64 v32; // [rsp-50h] [rbp-50h]
  __int64 v33; // [rsp-48h] [rbp-48h]

  if ( (_BYTE)qword_501E8C8 )
    return 0;
  v3 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_5027190);
  if ( !v3 )
    return 0;
  v4 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v3 + 104LL))(v3, &unk_5027190);
  if ( !v4 )
    return 0;
  v5 = *(__int64 **)(a1 + 8);
  v6 = *v5;
  v7 = v5[1];
  if ( v6 == v7 )
LABEL_28:
    BUG();
  while ( *(_UNKNOWN **)v6 != &unk_4F89C28 )
  {
    v6 += 16;
    if ( v7 == v6 )
      goto LABEL_28;
  }
  v8 = *(_QWORD *)(v4 + 256);
  v9 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v6 + 8) + 104LL))(*(_QWORD *)(v6 + 8), &unk_4F89C28);
  v10 = sub_DFED00(v9, a2);
  v11 = *(__int64 **)(a1 + 8);
  v12 = v10;
  v13 = *v11;
  v14 = v11[1];
  if ( v13 == v14 )
LABEL_29:
    BUG();
  while ( *(_UNKNOWN **)v13 != &unk_4F8F808 )
  {
    v13 += 16;
    if ( v14 == v13 )
      goto LABEL_29;
  }
  v15 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v13 + 8) + 104LL))(*(_QWORD *)(v13 + 8), &unk_4F8F808);
  v16 = *(__int64 **)(a1 + 8);
  v17 = *(_QWORD *)(v15 + 176);
  v18 = *v16;
  v19 = v16[1];
  if ( v18 == v19 )
LABEL_30:
    BUG();
  while ( *(_UNKNOWN **)v18 != &unk_4F8144C )
  {
    v18 += 16;
    if ( v19 == v18 )
      goto LABEL_30;
  }
  v20 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v18 + 8) + 104LL))(*(_QWORD *)(v18 + 8), &unk_4F8144C);
  v31 = v17;
  v29 = a2;
  v30 = v20 + 176;
  v25 = *(__int64 (**)())(*(_QWORD *)v8 + 16LL);
  if ( v25 == sub_23CE270 )
    BUG();
  v26 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64))v25)(
          v8,
          a2,
          v21,
          v22,
          v23,
          v24,
          v29,
          v30,
          v31);
  v27 = *(__int64 (**)())(*(_QWORD *)v26 + 144LL);
  v28 = 0;
  if ( v27 != sub_2C8F680 )
    v28 = ((__int64 (__fastcall *)(__int64))v27)(v26);
  v33 = v12;
  v32 = v28;
  return sub_2DF0A30((__int64)&v29);
}
