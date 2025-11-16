// Function: sub_299CB70
// Address: 0x299cb70
//
__int64 __fastcall sub_299CB70(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 *v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 *v13; // rdx
  __int64 *v14; // r13
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 *v18; // rdx
  __int64 v19; // r15
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  const char *v23; // rsi
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  _QWORD *v32; // rbx
  _QWORD *v33; // r15
  void (__fastcall *v34)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v35; // rax
  __int64 v37; // [rsp+8h] [rbp-308h]
  unsigned __int8 v38; // [rsp+8h] [rbp-308h]
  __int64 v39; // [rsp+18h] [rbp-2F8h] BYREF
  unsigned __int64 v40[2]; // [rsp+20h] [rbp-2F0h] BYREF
  _BYTE v41[512]; // [rsp+30h] [rbp-2E0h] BYREF
  __int64 v42; // [rsp+230h] [rbp-E0h]
  __int64 v43; // [rsp+238h] [rbp-D8h]
  __int64 v44; // [rsp+240h] [rbp-D0h]
  __int64 v45; // [rsp+248h] [rbp-C8h]
  char v46; // [rsp+250h] [rbp-C0h]
  __int64 v47; // [rsp+258h] [rbp-B8h]
  char *v48; // [rsp+260h] [rbp-B0h]
  __int64 v49; // [rsp+268h] [rbp-A8h]
  int v50; // [rsp+270h] [rbp-A0h]
  char v51; // [rsp+274h] [rbp-9Ch]
  char v52; // [rsp+278h] [rbp-98h] BYREF
  __int16 v53; // [rsp+2B8h] [rbp-58h]
  _QWORD *v54; // [rsp+2C0h] [rbp-50h]
  _QWORD *v55; // [rsp+2C8h] [rbp-48h]
  __int64 v56; // [rsp+2D0h] [rbp-40h]

  v3 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_4F8144C);
  if ( v3 && (v4 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v3 + 104LL))(v3, &unk_4F8144C)) != 0 )
    v5 = v4 + 176;
  else
    v5 = 0;
  v6 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_4F8FBD4);
  if ( v6 && (v7 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v6 + 104LL))(v6, &unk_4F8FBD4)) != 0 )
    v8 = v7 + 176;
  else
    v8 = 0;
  v45 = v8;
  v9 = *(__int64 **)(a1 + 8);
  v48 = &v52;
  v40[1] = 0x1000000000LL;
  v42 = 0;
  v43 = 0;
  v44 = v5;
  v46 = 0;
  v47 = 0;
  v49 = 8;
  v50 = 0;
  v51 = 1;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  v40[0] = (unsigned __int64)v41;
  v10 = *v9;
  v11 = v9[1];
  if ( v10 == v11 )
LABEL_46:
    BUG();
  while ( *(_UNKNOWN **)v10 != &unk_4F8FAE4 )
  {
    v10 += 16;
    if ( v11 == v10 )
      goto LABEL_46;
  }
  v12 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v10 + 8) + 104LL))(*(_QWORD *)(v10 + 8), &unk_4F8FAE4);
  v13 = *(__int64 **)(a1 + 8);
  v14 = *(__int64 **)(v12 + 176);
  v15 = *v13;
  v16 = v13[1];
  if ( v15 == v16 )
LABEL_44:
    BUG();
  while ( *(_UNKNOWN **)v15 != &unk_4F86530 )
  {
    v15 += 16;
    if ( v16 == v15 )
      goto LABEL_44;
  }
  v17 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v15 + 8) + 104LL))(*(_QWORD *)(v15 + 8), &unk_4F86530);
  v18 = *(__int64 **)(a1 + 8);
  v19 = *(_QWORD *)(v17 + 176);
  v20 = *v18;
  v21 = v18[1];
  if ( v20 == v21 )
LABEL_45:
    BUG();
  while ( *(_UNKNOWN **)v20 != &unk_4F89C28 )
  {
    v20 += 16;
    if ( v21 == v20 )
      goto LABEL_45;
  }
  v22 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v20 + 8) + 104LL))(*(_QWORD *)(v20 + 8), &unk_4F89C28);
  v23 = "disable-tail-calls";
  v37 = sub_DFED00(v22, a2);
  v39 = sub_B2D7E0(a2, "disable-tail-calls", 0x12u);
  if ( (unsigned __int8)sub_A72A30(&v39) )
  {
    v38 = 0;
  }
  else
  {
    v23 = (const char *)v37;
    v38 = sub_299AC30(a2, v37, v19, v14, (__int64)v40);
  }
  sub_FFCE90((__int64)v40, (__int64)v23, v24, v25, v26, v27);
  sub_FFD870((__int64)v40, (__int64)v23, v28, v29, v30, v31);
  sub_FFBC40((__int64)v40, (__int64)v23);
  v32 = v55;
  v33 = v54;
  if ( v55 != v54 )
  {
    do
    {
      v34 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v33[7];
      *v33 = &unk_49E5048;
      if ( v34 )
        v34(v33 + 5, v33 + 5, 3);
      *v33 = &unk_49DB368;
      v35 = v33[3];
      if ( v35 != -4096 && v35 != 0 && v35 != -8192 )
        sub_BD60C0(v33 + 1);
      v33 += 9;
    }
    while ( v32 != v33 );
    v33 = v54;
  }
  if ( v33 )
    j_j___libc_free_0((unsigned __int64)v33);
  if ( !v51 )
    _libc_free((unsigned __int64)v48);
  if ( (_BYTE *)v40[0] != v41 )
    _libc_free(v40[0]);
  return v38;
}
