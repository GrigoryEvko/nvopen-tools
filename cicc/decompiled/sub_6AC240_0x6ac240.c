// Function: sub_6AC240
// Address: 0x6ac240
//
__int64 __fastcall sub_6AC240(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rax
  _BYTE *v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 result; // rax
  __int64 v16; // rax
  int v17; // r14d
  __int64 v18; // rbx
  int v19; // eax
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // r8
  _BYTE *v23; // r14
  __int64 v24; // rdi
  _BYTE *v25; // rdi
  _BYTE *v26; // rdx
  _BYTE *v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  _BYTE *i; // rax
  _BYTE *v31; // rax
  _BYTE *v32; // rdx
  __int64 v33; // rdx
  _BYTE *v34; // [rsp+0h] [rbp-70h]
  __int64 v35; // [rsp+8h] [rbp-68h]
  int v36; // [rsp+1Ch] [rbp-54h] BYREF
  _BYTE *v37; // [rsp+20h] [rbp-50h] BYREF
  __int64 v38; // [rsp+28h] [rbp-48h] BYREF
  __int64 v39; // [rsp+30h] [rbp-40h] BYREF
  _QWORD v40[7]; // [rsp+38h] [rbp-38h] BYREF

  v6 = a1;
  v7 = a2;
  v36 = 0;
  if ( a2 )
  {
    v39 = *(_QWORD *)(a2 + 68);
    v8 = qword_4D03C50;
  }
  else
  {
    v39 = *(_QWORD *)&dword_4F063F8;
    sub_7B8B50(a1, 0, a3, a4);
    a2 = 125;
    a1 = 27;
    sub_7BE280(27, 125, 0, 0);
    ++*(_BYTE *)(qword_4F061C8 + 36LL);
    v8 = qword_4D03C50;
    ++*(_QWORD *)(qword_4D03C50 + 40LL);
  }
  if ( (*(_BYTE *)(v8 + 19) & 0x40) != 0 )
  {
    if ( (unsigned int)sub_6E5430(a1, a2, a3, a4, a5, a6) )
      sub_6851C0(0x39u, &v39);
  }
  else if ( !(unsigned int)sub_6E9250(&v39) )
  {
    goto LABEL_7;
  }
  v36 = 1;
LABEL_7:
  ++*(_BYTE *)(qword_4F061C8 + 75LL);
  v35 = sub_6AC060(1u, 0x3A1u, &v36);
  ++*(_BYTE *)(qword_4F061C8 + 9LL);
  sub_7BE280(67, 253, 0, 0);
  v9 = qword_4F061C8;
  --*(_BYTE *)(qword_4F061C8 + 9LL);
  --*(_BYTE *)(v9 + 75);
  v40[0] = *(_QWORD *)&dword_4F063F8;
  sub_65CD60(&v37);
  v10 = v37;
  if ( (unsigned int)sub_8D2310(v37)
    || (v10 = v37, (unsigned int)sub_8D3410(v37))
    || (v10 = v37, (unsigned int)sub_8D32E0(v37)) )
  {
    if ( (unsigned int)sub_6E5430(v10, 253, v11, v12, v13, v14) )
      sub_6851C0(0x3A1u, v40);
    goto LABEL_11;
  }
  if ( dword_4F077C4 != 2 )
    goto LABEL_19;
  v25 = v37;
  if ( !(unsigned int)sub_8D3A70(v37) )
    goto LABEL_19;
  if ( dword_4F077C4 != 2 || unk_4F07778 <= 201102 && (v25 = (_BYTE *)dword_4F07774, !dword_4F07774) )
  {
    for ( i = v37; i[140] == 12; i = (_BYTE *)*((_QWORD *)i + 20) )
      ;
    if ( *(char *)(*(_QWORD *)(*(_QWORD *)i + 96LL) + 178LL) < 0 )
      goto LABEL_19;
    goto LABEL_37;
  }
  v31 = v37;
  v27 = v37;
  if ( v37[140] != 12 )
  {
    v33 = *(_QWORD *)(*(_QWORD *)v37 + 96LL);
    if ( !*(_QWORD *)(v33 + 8) )
      goto LABEL_56;
LABEL_47:
    v25 = *(_BYTE **)(*(_QWORD *)v27 + 96LL);
    if ( !(unsigned int)sub_879360(v25, 253, v33, v27) )
    {
      v31 = v37;
      if ( v37[140] == 12 )
        goto LABEL_49;
LABEL_56:
      v26 = *(_BYTE **)(*(_QWORD *)v31 + 96LL);
      if ( *((_QWORD *)v26 + 3) )
        goto LABEL_53;
      goto LABEL_19;
    }
LABEL_37:
    if ( (unsigned int)sub_6E5430(v25, 253, v26, v27, v28, v29) )
      sub_6851C0(0x50Bu, v40);
LABEL_11:
    v36 = 1;
    goto LABEL_12;
  }
  v32 = v37;
  do
    v32 = (_BYTE *)*((_QWORD *)v32 + 20);
  while ( v32[140] == 12 );
  v33 = *(_QWORD *)(*(_QWORD *)v32 + 96LL);
  if ( *(_QWORD *)(v33 + 8) )
  {
    do
      v27 = (_BYTE *)*((_QWORD *)v27 + 20);
    while ( v27[140] == 12 );
    goto LABEL_47;
  }
LABEL_49:
  v26 = v31;
  do
    v31 = (_BYTE *)*((_QWORD *)v31 + 20);
  while ( v31[140] == 12 );
  v27 = *(_BYTE **)(*(_QWORD *)v31 + 96LL);
  v31 = v26;
  if ( *((_QWORD *)v27 + 3) )
  {
    do
      v31 = (_BYTE *)*((_QWORD *)v31 + 20);
    while ( v31[140] == 12 );
LABEL_53:
    if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)v31 + 96LL) + 177LL) & 2) != 0 )
      goto LABEL_19;
    goto LABEL_37;
  }
LABEL_19:
  if ( unk_4F068F0 )
  {
    if ( !v36 )
    {
LABEL_21:
      v16 = sub_73DC30(112, v37, v35);
      sub_6E7150(v16, v6);
      goto LABEL_13;
    }
LABEL_12:
    sub_6E6260(v6);
    goto LABEL_13;
  }
  v20 = sub_8D6740(v37);
  if ( v37 == (_BYTE *)v20 || (v34 = (_BYTE *)v20, (unsigned int)sub_8D97D0(v37, v20, 0, v21, v22)) )
  {
    v23 = 0;
  }
  else
  {
    sub_6E5D70(5, 1145, v40, v37, v34);
    v23 = v37;
    v37 = v34;
  }
  if ( v36 )
    goto LABEL_12;
  if ( unk_4F068F0 )
    goto LABEL_21;
  v38 = sub_73DBF0(112, v37, v35);
  v24 = v38;
  if ( v23 )
  {
    sub_6E8160((unsigned int)&v38, (_DWORD)v23, 1, 1, 0, 0, 0, 1, (__int64)&v39);
    v24 = v38;
  }
  sub_6E70E0(v24, v6);
LABEL_13:
  result = sub_6E26D0(2, v6);
  if ( !v7 )
  {
    v17 = qword_4F063F0;
    v18 = WORD2(qword_4F063F0);
    sub_7BE280(28, 18, 0, 0);
    --*(_BYTE *)(qword_4F061C8 + 36LL);
    --*(_QWORD *)(qword_4D03C50 + 40LL);
    v19 = v39;
    *(_DWORD *)(v6 + 76) = v17;
    *(_DWORD *)(v6 + 68) = v19;
    LOWORD(v19) = WORD2(v39);
    *(_WORD *)(v6 + 80) = v18;
    *(_WORD *)(v6 + 72) = v19;
    *(_QWORD *)dword_4F07508 = *(_QWORD *)(v6 + 68);
    unk_4F061D8 = *(_QWORD *)(v6 + 76);
    return sub_6E3280(v6, &v39);
  }
  return result;
}
