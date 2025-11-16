// Function: sub_2CAEA60
// Address: 0x2caea60
//
_BOOL8 __fastcall sub_2CAEA60(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 *v11; // rdx
  __int64 v12; // r13
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r15
  __int64 v16; // rcx
  __int64 v17; // rdx
  _BOOL4 v18; // r12d
  __int64 *v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // [rsp+8h] [rbp-1A8h] BYREF
  _QWORD v24[9]; // [rsp+10h] [rbp-1A0h] BYREF
  _QWORD v25[8]; // [rsp+58h] [rbp-158h] BYREF
  int v26; // [rsp+98h] [rbp-118h]
  __int64 v27; // [rsp+A0h] [rbp-110h]
  __int64 v28; // [rsp+A8h] [rbp-108h]
  __int64 v29; // [rsp+B0h] [rbp-100h]
  int v30; // [rsp+B8h] [rbp-F8h]
  __int64 v31; // [rsp+C0h] [rbp-F0h]
  __int64 v32; // [rsp+C8h] [rbp-E8h]
  __int64 v33; // [rsp+D0h] [rbp-E0h]
  __int64 v34; // [rsp+D8h] [rbp-D8h]
  __int64 v35; // [rsp+E0h] [rbp-D0h]
  _QWORD *(__fastcall *v36)(__int64, unsigned __int64); // [rsp+E8h] [rbp-C8h]
  __int64 *v37; // [rsp+F0h] [rbp-C0h]
  __int64 v38; // [rsp+F8h] [rbp-B8h]
  __int64 v39; // [rsp+100h] [rbp-B0h]
  __int64 v40; // [rsp+108h] [rbp-A8h]
  __int64 v41; // [rsp+110h] [rbp-A0h]
  _QWORD v42[6]; // [rsp+120h] [rbp-90h] BYREF
  int v43; // [rsp+150h] [rbp-60h] BYREF
  __int64 v44; // [rsp+158h] [rbp-58h]
  int *v45; // [rsp+160h] [rbp-50h]
  int *v46; // [rsp+168h] [rbp-48h]
  __int64 v47; // [rsp+170h] [rbp-40h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_30:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F875EC )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_30;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F875EC);
  v6 = *(__int64 **)(a1 + 8);
  v7 = v5 + 176;
  v8 = *v6;
  v9 = v6[1];
  if ( v8 == v9 )
LABEL_27:
    BUG();
  while ( *(_UNKNOWN **)v8 != &unk_4F8144C )
  {
    v8 += 16;
    if ( v9 == v8 )
      goto LABEL_27;
  }
  v10 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(*(_QWORD *)(v8 + 8), &unk_4F8144C);
  v11 = *(__int64 **)(a1 + 8);
  v12 = v10 + 176;
  v13 = *v11;
  v14 = v11[1];
  if ( v13 == v14 )
LABEL_28:
    BUG();
  while ( *(_UNKNOWN **)v13 != &unk_5035D54 )
  {
    v13 += 16;
    if ( v14 == v13 )
      goto LABEL_28;
  }
  v15 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v13 + 8) + 104LL))(*(_QWORD *)(v13 + 8), &unk_5035D54)
      + 172;
  if ( (unsigned int)qword_5012E48 | (unsigned int)qword_5012D68 )
  {
    v20 = *(__int64 **)(a1 + 8);
    v21 = *v20;
    v22 = v20[1];
    if ( v21 == v22 )
LABEL_29:
      BUG();
    while ( *(_UNKNOWN **)v21 != &unk_4F881C8 )
    {
      v21 += 16;
      if ( v22 == v21 )
        goto LABEL_29;
    }
    v16 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v21 + 8) + 104LL))(
                        *(_QWORD *)(v21 + 8),
                        &unk_4F881C8)
                    + 176);
  }
  else
  {
    v16 = 0;
  }
  v17 = *(_QWORD *)(a1 + 176);
  v23 = a1;
  v25[2] = v25;
  v25[3] = v25;
  v36 = sub_2C90D80;
  v37 = &v23;
  v31 = v17;
  v32 = v16;
  v33 = v7;
  v34 = v12;
  v35 = v15;
  memset(v24, 0, 64);
  v25[0] = 0;
  v25[1] = 0;
  memset(&v25[4], 0, 32);
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v38 = 0;
  v39 = 0;
  v40 = 0;
  v42[2] = v42;
  v42[3] = v42;
  v41 = 0;
  v42[0] = 0;
  v42[1] = 0;
  v42[4] = 0;
  v43 = 0;
  v44 = 0;
  v45 = &v43;
  v46 = &v43;
  v47 = 0;
  v18 = sub_2CABB50((__int64)v24, a2);
  sub_2C91DD0((__int64)v24);
  return v18;
}
