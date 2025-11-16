// Function: sub_2E5CE30
// Address: 0x2e5ce30
//
__int64 __fastcall sub_2E5CE30(__int64 a1, __int64 *a2)
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
  __int64 v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 *v16; // rdx
  __int64 v17; // r15
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rax
  unsigned int v22; // r12d
  _QWORD v24[5]; // [rsp+0h] [rbp-370h] BYREF
  __int16 v25; // [rsp+28h] [rbp-348h]
  __int64 v26; // [rsp+30h] [rbp-340h]
  __int64 v27; // [rsp+38h] [rbp-338h]
  __int64 v28; // [rsp+40h] [rbp-330h]
  int v29; // [rsp+48h] [rbp-328h]
  __int64 v30; // [rsp+50h] [rbp-320h]
  __int64 v31; // [rsp+58h] [rbp-318h]
  __int64 v32; // [rsp+60h] [rbp-310h]
  int v33; // [rsp+68h] [rbp-308h]
  __int64 v34; // [rsp+70h] [rbp-300h]
  __int64 v35; // [rsp+78h] [rbp-2F8h]
  __int64 v36; // [rsp+80h] [rbp-2F0h]
  int v37; // [rsp+88h] [rbp-2E8h]
  __int64 v38; // [rsp+90h] [rbp-2E0h]
  __int64 v39; // [rsp+98h] [rbp-2D8h]
  __int64 v40; // [rsp+A0h] [rbp-2D0h]
  char *v41; // [rsp+A8h] [rbp-2C8h]
  __int64 v42; // [rsp+B0h] [rbp-2C0h]
  char v43; // [rsp+B8h] [rbp-2B8h] BYREF
  _QWORD *v44; // [rsp+D8h] [rbp-298h]
  __int64 v45; // [rsp+E0h] [rbp-290h]
  _QWORD v46[5]; // [rsp+E8h] [rbp-288h] BYREF
  int v47; // [rsp+110h] [rbp-260h]
  __int64 v48; // [rsp+118h] [rbp-258h]
  char *v49; // [rsp+120h] [rbp-250h]
  __int64 v50; // [rsp+128h] [rbp-248h]
  char v51; // [rsp+130h] [rbp-240h] BYREF
  int v52; // [rsp+330h] [rbp-40h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_28:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_501FE44 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_28;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_501FE44);
  v6 = *(__int64 **)(a1 + 8);
  v7 = v5 + 200;
  v8 = *v6;
  v9 = v6[1];
  if ( v8 == v9 )
LABEL_25:
    BUG();
  while ( *(_UNKNOWN **)v8 != &unk_501EC08 )
  {
    v8 += 16;
    if ( v9 == v8 )
      goto LABEL_25;
  }
  v10 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(*(_QWORD *)(v8 + 8), &unk_501EC08);
  v11 = *(__int64 **)(a1 + 8);
  v12 = v10 + 200;
  v13 = *v11;
  v14 = v11[1];
  if ( v13 == v14 )
LABEL_26:
    BUG();
  while ( *(_UNKNOWN **)v13 != &unk_502D274 )
  {
    v13 += 16;
    if ( v14 == v13 )
      goto LABEL_26;
  }
  v15 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v13 + 8) + 104LL))(*(_QWORD *)(v13 + 8), &unk_502D274);
  v16 = *(__int64 **)(a1 + 8);
  v17 = *(_QWORD *)(v15 + 200);
  v18 = *v16;
  v19 = v16[1];
  if ( v18 == v19 )
LABEL_27:
    BUG();
  while ( *(_UNKNOWN **)v18 != &unk_4F89C28 )
  {
    v18 += 16;
    if ( v19 == v18 )
      goto LABEL_27;
  }
  v20 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v18 + 8) + 104LL))(*(_QWORD *)(v18 + 8), &unk_4F89C28);
  v21 = sub_DFED00(v20, *a2);
  v24[2] = v7;
  v27 = v21;
  v41 = &v43;
  v42 = 0x400000000LL;
  v44 = v46;
  v25 = 257;
  v24[4] = v12;
  v26 = v17;
  v24[0] = 0;
  v24[1] = 0;
  v24[3] = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v34 = 0;
  v35 = 0;
  v36 = 0;
  v37 = 0;
  v38 = 0;
  v39 = 0;
  v40 = 0;
  v45 = 0;
  v46[0] = 0;
  v46[1] = 1;
  memset(&v46[2], 0, 24);
  v47 = 0;
  v48 = 0;
  v49 = &v51;
  v50 = 0x4000000000LL;
  v52 = 0;
  v22 = sub_2E5C7D0((__int64)v24, (__int64)a2);
  sub_2E50030((__int64)v24);
  return v22;
}
