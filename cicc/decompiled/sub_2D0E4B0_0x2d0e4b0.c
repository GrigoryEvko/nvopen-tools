// Function: sub_2D0E4B0
// Address: 0x2d0e4b0
//
__int64 __fastcall sub_2D0E4B0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 *v7; // rdx
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 *v12; // rdx
  __int64 v13; // r12
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 *v17; // rdx
  __int64 v18; // r14
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  unsigned int v23; // r12d
  _QWORD v25[7]; // [rsp+0h] [rbp-160h] BYREF
  int v26; // [rsp+38h] [rbp-128h]
  __int16 v27; // [rsp+3Ch] [rbp-124h]
  char v28; // [rsp+3Eh] [rbp-122h]
  int v29; // [rsp+40h] [rbp-120h]
  __int64 v30; // [rsp+48h] [rbp-118h]
  __int64 v31; // [rsp+50h] [rbp-110h]
  __int64 v32; // [rsp+58h] [rbp-108h]
  int v33; // [rsp+60h] [rbp-100h]
  _QWORD v34[3]; // [rsp+68h] [rbp-F8h] BYREF
  _QWORD v35[6]; // [rsp+80h] [rbp-E0h] BYREF
  int v36; // [rsp+B0h] [rbp-B0h]
  char v37; // [rsp+B4h] [rbp-ACh]
  char v38; // [rsp+B8h] [rbp-A8h] BYREF
  __int64 v39; // [rsp+F8h] [rbp-68h]
  __int64 v40; // [rsp+100h] [rbp-60h]
  __int64 v41; // [rsp+108h] [rbp-58h]
  int v42; // [rsp+110h] [rbp-50h]
  __int64 v43; // [rsp+118h] [rbp-48h]
  __int64 v44; // [rsp+120h] [rbp-40h]
  __int64 v45; // [rsp+128h] [rbp-38h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_28:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F89C28 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_28;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F89C28);
  v6 = sub_DFED00(v5, a2);
  v7 = *(__int64 **)(a1 + 8);
  v8 = v6;
  v9 = *v7;
  v10 = v7[1];
  if ( v9 == v10 )
LABEL_25:
    BUG();
  while ( *(_UNKNOWN **)v9 != &unk_4F8D9B0 )
  {
    v9 += 16;
    if ( v10 == v9 )
      goto LABEL_25;
  }
  v11 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v9 + 8) + 104LL))(*(_QWORD *)(v9 + 8), &unk_4F8D9B0);
  v12 = *(__int64 **)(a1 + 8);
  v13 = v11 + 176;
  v14 = *v12;
  v15 = v12[1];
  if ( v14 == v15 )
LABEL_26:
    BUG();
  while ( *(_UNKNOWN **)v14 != &unk_4F8144C )
  {
    v14 += 16;
    if ( v15 == v14 )
      goto LABEL_26;
  }
  v16 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v14 + 8) + 104LL))(*(_QWORD *)(v14 + 8), &unk_4F8144C);
  v17 = *(__int64 **)(a1 + 8);
  v18 = v16 + 176;
  v19 = *v17;
  v20 = v17[1];
  if ( v19 == v20 )
LABEL_27:
    BUG();
  while ( *(_UNKNOWN **)v19 != &unk_4F875EC )
  {
    v19 += 16;
    if ( v20 == v19 )
      goto LABEL_27;
  }
  v21 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v19 + 8) + 104LL))(*(_QWORD *)(v19 + 8), &unk_4F875EC);
  v22 = *(_QWORD *)(a2 + 40);
  v25[5] = v8;
  v25[0] = a2;
  v25[2] = v21 + 176;
  v27 = 256;
  v34[1] = v34;
  v34[0] = v34;
  v35[1] = v35;
  v35[0] = v35;
  v25[1] = v22 + 312;
  v25[3] = v18;
  v25[4] = v13;
  v35[4] = &v38;
  v25[6] = 0;
  v26 = 5;
  v28 = 0;
  v29 = 30;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v34[2] = 0;
  v35[2] = 0;
  v35[3] = 0;
  v35[5] = 8;
  v36 = 0;
  v37 = 1;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v23 = sub_2D0DB80((__int64)v25);
  sub_2D05B80((__int64)v25);
  return v23;
}
