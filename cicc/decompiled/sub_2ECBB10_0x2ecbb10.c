// Function: sub_2ECBB10
// Address: 0x2ecbb10
//
__int64 __fastcall sub_2ECBB10(_QWORD *a1, __int64 a2)
{
  _QWORD *v5; // r14
  _QWORD *v6; // r13
  unsigned __int64 v7; // rsi
  _QWORD *v8; // rax
  _QWORD *v9; // rdi
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rcx
  _QWORD *v14; // r8
  __int64 v15; // rax
  _QWORD *v16; // rdi
  __int64 *v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 *v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 *v25; // rdx
  __int64 v26; // r13
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 *v30; // rdx
  __int64 v31; // r15
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 *v35; // rdx
  __int64 v36; // r14
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rax
  __int64 v40; // [rsp+8h] [rbp-58h]
  _QWORD v41[10]; // [rsp+10h] [rbp-50h] BYREF

  if ( (unsigned __int8)sub_BB98D0(a1, *(_QWORD *)a2) )
    return 0;
  v5 = sub_C52410();
  v6 = v5 + 1;
  v7 = sub_C959E0();
  v8 = (_QWORD *)v5[2];
  if ( v8 )
  {
    v9 = v5 + 1;
    do
    {
      while ( 1 )
      {
        v10 = v8[2];
        v11 = v8[3];
        if ( v7 <= v8[4] )
          break;
        v8 = (_QWORD *)v8[3];
        if ( !v11 )
          goto LABEL_8;
      }
      v9 = v8;
      v8 = (_QWORD *)v8[2];
    }
    while ( v10 );
LABEL_8:
    if ( v6 != v9 && v7 >= v9[4] )
      v6 = v9;
  }
  if ( v6 == (_QWORD *)((char *)sub_C52410() + 8) )
    goto LABEL_60;
  v15 = v6[7];
  v14 = v6 + 6;
  if ( !v15 )
    goto LABEL_60;
  v7 = (unsigned int)dword_5020C88;
  v16 = v6 + 6;
  do
  {
    while ( 1 )
    {
      v13 = *(_QWORD *)(v15 + 16);
      v12 = *(_QWORD *)(v15 + 24);
      if ( *(_DWORD *)(v15 + 32) >= dword_5020C88 )
        break;
      v15 = *(_QWORD *)(v15 + 24);
      if ( !v12 )
        goto LABEL_17;
    }
    v16 = (_QWORD *)v15;
    v15 = *(_QWORD *)(v15 + 16);
  }
  while ( v13 );
LABEL_17:
  if ( v14 == v16 || dword_5020C88 < *((_DWORD *)v16 + 8) || !*((_DWORD *)v16 + 9) )
  {
LABEL_60:
    if ( (*(unsigned __int8 (__fastcall **)(_QWORD, unsigned __int64, __int64, __int64, _QWORD *))(**(_QWORD **)(a2 + 16)
                                                                                                 + 256LL))(
           *(_QWORD *)(a2 + 16),
           v7,
           v12,
           v13,
           v14) )
    {
      goto LABEL_21;
    }
    return 0;
  }
  if ( !(_BYTE)qword_5020D08 )
    return 0;
LABEL_21:
  v17 = (__int64 *)a1[1];
  v18 = *v17;
  v19 = v17[1];
  if ( v18 == v19 )
LABEL_54:
    BUG();
  while ( *(_UNKNOWN **)v18 != &unk_50208AC )
  {
    v18 += 16;
    if ( v19 == v18 )
      goto LABEL_54;
  }
  v20 = (*(__int64 (__fastcall **)(_QWORD, void *, __int64, __int64))(**(_QWORD **)(v18 + 8) + 104LL))(
          *(_QWORD *)(v18 + 8),
          &unk_50208AC,
          v19,
          v13);
  v21 = (__int64 *)a1[1];
  v40 = v20 + 200;
  v22 = *v21;
  v23 = v21[1];
  if ( v22 == v23 )
LABEL_53:
    BUG();
  while ( *(_UNKNOWN **)v22 != &unk_501FE44 )
  {
    v22 += 16;
    if ( v23 == v22 )
      goto LABEL_53;
  }
  v24 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v22 + 8) + 104LL))(*(_QWORD *)(v22 + 8), &unk_501FE44);
  v25 = (__int64 *)a1[1];
  v26 = v24 + 200;
  v27 = *v25;
  v28 = v25[1];
  if ( v27 == v28 )
LABEL_57:
    BUG();
  while ( *(_UNKNOWN **)v27 != &unk_5027190 )
  {
    v27 += 16;
    if ( v28 == v27 )
      goto LABEL_57;
  }
  v29 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v27 + 8) + 104LL))(*(_QWORD *)(v27 + 8), &unk_5027190);
  v30 = (__int64 *)a1[1];
  v31 = *(_QWORD *)(v29 + 256);
  v32 = *v30;
  v33 = v30[1];
  if ( v32 == v33 )
LABEL_56:
    BUG();
  while ( *(_UNKNOWN **)v32 != &unk_4F86530 )
  {
    v32 += 16;
    if ( v33 == v32 )
      goto LABEL_56;
  }
  v34 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v32 + 8) + 104LL))(*(_QWORD *)(v32 + 8), &unk_4F86530);
  v35 = (__int64 *)a1[1];
  v36 = *(_QWORD *)(v34 + 176);
  v37 = *v35;
  v38 = v35[1];
  if ( v37 == v38 )
LABEL_55:
    BUG();
  while ( *(_UNKNOWN **)v37 != &unk_501EACC )
  {
    v37 += 16;
    if ( v38 == v37 )
      goto LABEL_55;
  }
  v39 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v37 + 8) + 104LL))(*(_QWORD *)(v37 + 8), &unk_501EACC);
  a1[33] = a1;
  v41[0] = v40;
  v41[1] = v26;
  v41[2] = v36;
  v41[3] = v39 + 200;
  return sub_2ECB9C0(a1 + 25, a2, v31, v41);
}
