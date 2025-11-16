// Function: sub_2987F60
// Address: 0x2987f60
//
__int64 __fastcall sub_2987F60(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 *v7; // rdx
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 *v12; // rdx
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  unsigned int v19; // r12d
  _QWORD v21[4]; // [rsp+0h] [rbp-170h] BYREF
  _QWORD v22[6]; // [rsp+20h] [rbp-150h] BYREF
  int v23; // [rsp+50h] [rbp-120h]
  __int64 v24; // [rsp+58h] [rbp-118h]
  __int64 v25; // [rsp+60h] [rbp-110h]
  __int64 v26; // [rsp+68h] [rbp-108h]
  int v27; // [rsp+70h] [rbp-100h]
  __int64 v28; // [rsp+78h] [rbp-F8h]
  __int64 v29; // [rsp+80h] [rbp-F0h]
  __int64 v30; // [rsp+88h] [rbp-E8h]
  int v31; // [rsp+90h] [rbp-E0h]
  __int64 v32; // [rsp+98h] [rbp-D8h]
  __int64 v33; // [rsp+A0h] [rbp-D0h]
  __int64 v34; // [rsp+A8h] [rbp-C8h]
  __int64 v35; // [rsp+B0h] [rbp-C0h]
  __int64 v36; // [rsp+B8h] [rbp-B8h]
  __int64 v37; // [rsp+C0h] [rbp-B0h]
  int v38; // [rsp+C8h] [rbp-A8h]
  __int64 v39; // [rsp+D0h] [rbp-A0h]
  __int64 v40; // [rsp+D8h] [rbp-98h]
  __int64 v41; // [rsp+E0h] [rbp-90h]
  int v42; // [rsp+E8h] [rbp-88h]
  _QWORD *v43; // [rsp+F0h] [rbp-80h]
  __int64 v44; // [rsp+F8h] [rbp-78h]
  _QWORD v45[3]; // [rsp+100h] [rbp-70h] BYREF
  int v46; // [rsp+118h] [rbp-58h]
  __int64 v47; // [rsp+120h] [rbp-50h]
  __int64 v48; // [rsp+128h] [rbp-48h]
  __int64 v49; // [rsp+130h] [rbp-40h]
  __int64 v50; // [rsp+138h] [rbp-38h]
  __int64 v51; // [rsp+140h] [rbp-30h]
  __int64 v52; // [rsp+148h] [rbp-28h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_21:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F89C28 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_21;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F89C28);
  v6 = sub_DFED00(v5, a2);
  v7 = *(__int64 **)(a1 + 8);
  v8 = v6;
  v9 = *v7;
  v10 = v7[1];
  if ( v9 == v10 )
LABEL_19:
    BUG();
  while ( *(_UNKNOWN **)v9 != &unk_4F8144C )
  {
    v9 += 16;
    if ( v10 == v9 )
      goto LABEL_19;
  }
  v11 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v9 + 8) + 104LL))(*(_QWORD *)(v9 + 8), &unk_4F8144C);
  v12 = *(__int64 **)(a1 + 8);
  v13 = v11 + 176;
  v14 = *v12;
  v15 = v12[1];
  if ( v14 == v15 )
LABEL_20:
    BUG();
  while ( *(_UNKNOWN **)v14 != &unk_4F881C8 )
  {
    v14 += 16;
    if ( v15 == v14 )
      goto LABEL_20;
  }
  v16 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v14 + 8) + 104LL))(*(_QWORD *)(v14 + 8), &unk_4F881C8);
  v17 = *(_QWORD *)(a1 + 176);
  v21[1] = v13;
  v18 = *(_QWORD *)(v16 + 176);
  v21[3] = v8;
  v21[0] = v17;
  v21[2] = v18;
  v22[1] = v22;
  v22[0] = v22;
  memset(&v22[2], 0, 32);
  v23 = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
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
  v41 = 0;
  v42 = 0;
  v43 = v45;
  v44 = 0;
  memset(v45, 0, sizeof(v45));
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v19 = sub_29851F0((__int64)v21);
  sub_297D250((__int64)v21);
  return v19;
}
