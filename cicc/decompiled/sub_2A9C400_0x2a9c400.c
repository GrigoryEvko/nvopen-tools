// Function: sub_2A9C400
// Address: 0x2a9c400
//
__int64 __fastcall sub_2A9C400(__int64 a1, __int64 a2)
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
  __int64 *v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 *v15; // rdx
  __int64 *v16; // r14
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 *v21; // rdx
  __int64 v22; // rbx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  unsigned int v26; // r12d
  __int64 v28; // [rsp+8h] [rbp-578h]
  _QWORD v29[7]; // [rsp+10h] [rbp-570h] BYREF
  _BYTE *v30; // [rsp+48h] [rbp-538h]
  __int64 v31; // [rsp+50h] [rbp-530h]
  _BYTE v32[32]; // [rsp+58h] [rbp-528h] BYREF
  __int64 v33; // [rsp+78h] [rbp-508h]
  __int64 v34; // [rsp+80h] [rbp-500h]
  __int16 v35; // [rsp+88h] [rbp-4F8h]
  __int64 v36; // [rsp+90h] [rbp-4F0h]
  void **v37; // [rsp+98h] [rbp-4E8h]
  _QWORD *v38; // [rsp+A0h] [rbp-4E0h]
  __int64 v39; // [rsp+A8h] [rbp-4D8h]
  int v40; // [rsp+B0h] [rbp-4D0h]
  __int16 v41; // [rsp+B4h] [rbp-4CCh]
  char v42; // [rsp+B6h] [rbp-4CAh]
  __int64 v43; // [rsp+B8h] [rbp-4C8h]
  __int64 v44; // [rsp+C0h] [rbp-4C0h]
  void *v45; // [rsp+C8h] [rbp-4B8h] BYREF
  _QWORD v46[2]; // [rsp+D0h] [rbp-4B0h] BYREF
  __int64 v47; // [rsp+E0h] [rbp-4A0h]
  __int64 v48; // [rsp+E8h] [rbp-498h]
  unsigned int v49; // [rsp+F0h] [rbp-490h]
  _BYTE *v50; // [rsp+F8h] [rbp-488h]
  __int64 v51; // [rsp+100h] [rbp-480h]
  _BYTE v52[1024]; // [rsp+108h] [rbp-478h] BYREF
  __int64 v53; // [rsp+508h] [rbp-78h]
  __int64 v54; // [rsp+510h] [rbp-70h]
  __int64 v55; // [rsp+518h] [rbp-68h]
  __int64 v56; // [rsp+520h] [rbp-60h]
  __int64 v57; // [rsp+528h] [rbp-58h]
  __int64 v58; // [rsp+530h] [rbp-50h]
  __int64 v59; // [rsp+538h] [rbp-48h]
  unsigned int v60; // [rsp+540h] [rbp-40h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_39:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F86530 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_39;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F86530);
  v6 = *(__int64 **)(a1 + 8);
  v7 = *(_QWORD *)(v5 + 176);
  v8 = *v6;
  v9 = v6[1];
  if ( v8 == v9 )
LABEL_35:
    BUG();
  while ( *(_UNKNOWN **)v8 != &unk_4F8144C )
  {
    v8 += 16;
    if ( v9 == v8 )
      goto LABEL_35;
  }
  v10 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(*(_QWORD *)(v8 + 8), &unk_4F8144C);
  v11 = *(__int64 **)(a1 + 8);
  v28 = v10 + 176;
  v12 = *v11;
  v13 = v11[1];
  if ( v12 == v13 )
LABEL_36:
    BUG();
  while ( *(_UNKNOWN **)v12 != &unk_4F881C8 )
  {
    v12 += 16;
    if ( v13 == v12 )
      goto LABEL_36;
  }
  v14 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v12 + 8) + 104LL))(*(_QWORD *)(v12 + 8), &unk_4F881C8);
  v15 = *(__int64 **)(a1 + 8);
  v16 = *(__int64 **)(v14 + 176);
  v17 = *v15;
  v18 = v15[1];
  if ( v17 == v18 )
LABEL_37:
    BUG();
  while ( *(_UNKNOWN **)v17 != &unk_4F89C28 )
  {
    v17 += 16;
    if ( v18 == v17 )
      goto LABEL_37;
  }
  v19 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v17 + 8) + 104LL))(*(_QWORD *)(v17 + 8), &unk_4F89C28);
  v20 = sub_DFED00(v19, a2);
  v21 = *(__int64 **)(a1 + 8);
  v22 = v20;
  v23 = *v21;
  v24 = v21[1];
  if ( v23 == v24 )
LABEL_38:
    BUG();
  while ( *(_UNKNOWN **)v23 != &unk_4F8662C )
  {
    v23 += 16;
    if ( v24 == v23 )
      goto LABEL_38;
  }
  v25 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v23 + 8) + 104LL))(*(_QWORD *)(v23 + 8), &unk_4F8662C);
  v29[0] = a2;
  v29[2] = sub_CFFAC0(v25, a2);
  v29[1] = v7;
  v29[4] = v16;
  v29[5] = v22;
  v29[3] = v28;
  v29[6] = sub_B2BEC0(a2);
  v36 = sub_B2BE50(*v16);
  v41 = 512;
  v31 = 0x200000000LL;
  v35 = 0;
  v45 = &unk_49DA100;
  v30 = v32;
  v37 = &v45;
  v46[0] = &unk_49DA0B0;
  v38 = v46;
  v39 = 0;
  v40 = 0;
  v42 = 7;
  v43 = 0;
  v44 = 0;
  v33 = 0;
  v34 = 0;
  v46[1] = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v50 = v52;
  v51 = 0x8000000000LL;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v60 = 0;
  v26 = sub_2A99860((__int64)v29);
  sub_C7D6A0(v58, 24LL * v60, 8);
  sub_C7D6A0(v54, 8LL * (unsigned int)v56, 8);
  if ( v50 != v52 )
    _libc_free((unsigned __int64)v50);
  sub_C7D6A0(v47, 16LL * v49, 8);
  nullsub_61();
  v45 = &unk_49DA100;
  nullsub_63();
  if ( v30 != v32 )
    _libc_free((unsigned __int64)v30);
  return v26;
}
