// Function: sub_35903C0
// Address: 0x35903c0
//
__int64 __fastcall sub_35903C0(_QWORD *a1, __int64 *a2)
{
  unsigned int v2; // r14d
  __int64 *v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 *v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 *v12; // rdx
  __int64 v13; // rbx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rcx
  unsigned __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 *v24; // rcx
  _QWORD *v25; // r14
  __int64 v26; // rax
  __int64 v27; // rcx
  __int64 v28; // rax
  __int64 *v29; // rcx
  __int64 v30; // r8
  __int64 v31; // rax
  __int64 v32; // rcx
  __int64 v33; // rax
  bool v34; // zf
  size_t v35; // r15
  const void *v36; // rbx
  const char *v37; // rax
  __int64 v38; // rdx
  __int64 v39; // r13
  const char *v40; // rax
  __int64 v41; // rdx
  size_t v42; // r14
  const void *v43; // rbx
  const char *v44; // rax
  __int64 v45; // rdx
  __int64 v46; // r14
  const char *v47; // rax
  __int64 v48; // rdx
  __int64 *v49; // rdx
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rbx
  __int64 v53; // rax
  __int64 v54; // [rsp+8h] [rbp-78h]
  __int64 v55; // [rsp+10h] [rbp-70h]
  __int64 v56; // [rsp+18h] [rbp-68h]
  __int64 v57; // [rsp+18h] [rbp-68h]
  void *v58[2]; // [rsp+20h] [rbp-60h] BYREF
  const char *v59; // [rsp+30h] [rbp-50h]
  __int64 v60; // [rsp+38h] [rbp-48h]
  __int16 v61; // [rsp+40h] [rbp-40h]

  v2 = *(unsigned __int8 *)(a1[32] + 1316LL);
  if ( (_BYTE)v2 )
  {
    v4 = (__int64 *)a1[1];
    v5 = *v4;
    v6 = v4[1];
    if ( v5 == v6 )
LABEL_51:
      BUG();
    while ( *(_UNKNOWN **)v5 != &unk_501EC08 )
    {
      v5 += 16;
      if ( v6 == v5 )
        goto LABEL_51;
    }
    v7 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v5 + 8) + 104LL))(*(_QWORD *)(v5 + 8), &unk_501EC08);
    v8 = (__int64 *)a1[1];
    a1[33] = v7 + 200;
    v9 = *v8;
    v10 = v8[1];
    if ( v9 == v10 )
LABEL_52:
      BUG();
    while ( *(_UNKNOWN **)v9 != &unk_501FE44 )
    {
      v9 += 16;
      if ( v10 == v9 )
        goto LABEL_52;
    }
    v11 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v9 + 8) + 104LL))(*(_QWORD *)(v9 + 8), &unk_501FE44);
    v12 = (__int64 *)a1[1];
    v13 = v11 + 200;
    v14 = *v12;
    v15 = v12[1];
    if ( v14 == v15 )
LABEL_53:
      BUG();
    while ( *(_UNKNOWN **)v14 != &unk_50209DC )
    {
      v14 += 16;
      if ( v15 == v14 )
        goto LABEL_53;
    }
    v56 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v14 + 8) + 104LL))(
            *(_QWORD *)(v14 + 8),
            &unk_50209DC)
        + 200;
    sub_2E7A760((__int64)a2, 0);
    sub_30052F0(v13, 0, v16, v17, v18, v19);
    sub_34BD420(v56, 0, v20, v21, v22, v23);
    v24 = (__int64 *)a1[1];
    v25 = (_QWORD *)a1[32];
    v26 = *v24;
    v27 = v24[1];
    if ( v26 == v27 )
LABEL_54:
      BUG();
    while ( *(_UNKNOWN **)v26 != &unk_50209AC )
    {
      v26 += 16;
      if ( v27 == v26 )
        goto LABEL_54;
    }
    v28 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v26 + 8) + 104LL))(
            *(_QWORD *)(v26 + 8),
            &unk_50209AC);
    v29 = (__int64 *)a1[1];
    v30 = *(_QWORD *)(v28 + 200);
    v31 = *v29;
    v32 = v29[1];
    if ( v31 == v32 )
LABEL_55:
      BUG();
    while ( *(_UNKNOWN **)v31 != &unk_50208AC )
    {
      v31 += 16;
      if ( v32 == v31 )
        goto LABEL_55;
    }
    v54 = a1[33];
    v55 = v30;
    v33 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v31 + 8) + 104LL))(
            *(_QWORD *)(v31 + 8),
            &unk_50208AC);
    v25[125] = v13;
    v34 = (_BYTE)qword_503F3C8 == 0;
    v25[127] = v33 + 200;
    v25[126] = v56;
    v25[162] = v54;
    v25[161] = v55;
    if ( !v34 )
    {
      if ( LODWORD(qword_501ED48[8]) )
      {
        v42 = qword_4F8DF28[9];
        if ( !qword_4F8DF28[9]
          || (v43 = (const void *)qword_4F8DF28[8], v44 = sub_BD5D20(*a2), v42 == v45) && !memcmp(v44, v43, v42) )
        {
          v46 = a1[33];
          v47 = sub_2E791E0(a2);
          v58[0] = "MIR_Prof_loader_b.";
          v61 = 1283;
          v60 = v48;
          v59 = v47;
          sub_2E43B30(v46, v58, 0);
        }
      }
    }
    v2 = sub_358FF90((_QWORD *)a1[32], a2);
    if ( (_BYTE)v2 )
    {
      v49 = (__int64 *)a1[1];
      v50 = *v49;
      v51 = v49[1];
      if ( v50 == v51 )
LABEL_56:
        BUG();
      while ( *(_UNKNOWN **)v50 != &unk_50208AC )
      {
        v50 += 16;
        if ( v51 == v50 )
          goto LABEL_56;
      }
      v57 = a1[33];
      v52 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v50 + 8) + 104LL))(
              *(_QWORD *)(v50 + 8),
              &unk_50208AC);
      v53 = sub_2E3A070((__int64 *)a1[33]);
      sub_2E43BF0(v57, (__int64)a2, v53, v52 + 200);
    }
    if ( (_BYTE)qword_503F2E8 )
    {
      if ( LODWORD(qword_501ED48[8]) )
      {
        v35 = qword_4F8DF28[9];
        if ( !qword_4F8DF28[9]
          || (v36 = (const void *)qword_4F8DF28[8], v37 = sub_BD5D20(*a2), v35 == v38) && !memcmp(v37, v36, v35) )
        {
          v39 = a1[33];
          v40 = sub_2E791E0(a2);
          v58[0] = "MIR_prof_loader_a.";
          v60 = v41;
          v61 = 1283;
          v59 = v40;
          sub_2E43B30(v39, v58, 0);
        }
      }
    }
  }
  return v2;
}
