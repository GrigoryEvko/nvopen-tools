// Function: sub_6DDE70
// Address: 0x6dde70
//
__int64 __fastcall sub_6DDE70(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 i; // rbx
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // r15
  __int64 v9; // rax
  __int64 v10; // rdx
  _BYTE *v11; // [rsp+8h] [rbp-478h]
  int v12; // [rsp+10h] [rbp-470h]
  __int16 v13; // [rsp+1Ah] [rbp-466h]
  int v14; // [rsp+1Ch] [rbp-464h]
  int v15; // [rsp+24h] [rbp-45Ch] BYREF
  __int64 v16; // [rsp+28h] [rbp-458h] BYREF
  _QWORD v17[44]; // [rsp+30h] [rbp-450h] BYREF
  _BYTE v18[352]; // [rsp+190h] [rbp-2F0h] BYREF
  _BYTE v19[68]; // [rsp+2F0h] [rbp-190h] BYREF
  __int64 v20; // [rsp+334h] [rbp-14Ch]
  __int64 v21; // [rsp+33Ch] [rbp-144h]

  v16 = *(_QWORD *)&dword_4F063F8;
  sub_7B8B50(a1, a2, a3, a4);
  sub_7BE280(27, 125, 0, 0);
  if ( (_BYTE)a1 == 52 )
  {
    ++*(_BYTE *)(qword_4F061C8 + 36LL);
    ++*(_QWORD *)(qword_4D03C50 + 40LL);
    sub_69ED20((__int64)v17, 0, 0, 1);
    v12 = 0;
    v11 = 0;
  }
  else
  {
    v9 = qword_4F061C8;
    v10 = qword_4D03C50;
    ++*(_BYTE *)(qword_4F061C8 + 36LL);
    ++*(_QWORD *)(v10 + 40);
    ++*(_BYTE *)(v9 + 75);
    sub_69ED20((__int64)v17, 0, 0, 1);
    sub_7BE280(67, 253, 0, 0);
    sub_69ED20((__int64)v18, 0, 0, 1);
    v11 = v18;
    v12 = 1;
    --*(_BYTE *)(qword_4F061C8 + 75LL);
  }
  v14 = qword_4F063F0;
  v13 = WORD2(qword_4F063F0);
  sub_7BE280(28, 18, 0, 0);
  --*(_BYTE *)(qword_4F061C8 + 36LL);
  --*(_QWORD *)(qword_4D03C50 + 40LL);
  if ( !(unsigned int)sub_68B5C0(v17, &v15) )
  {
    if ( !v15 )
    {
      sub_72C930(v17);
      sub_6E6260(a2);
      goto LABEL_11;
    }
LABEL_5:
    sub_6F40C0(v17);
    i = *(_QWORD *)&dword_4D03B80;
    v5 = sub_6F6F40(v17, 0);
    if ( !v12 )
      goto LABEL_6;
    goto LABEL_19;
  }
  if ( v15 )
    goto LABEL_5;
  sub_6F69D0(v17, 0);
  v5 = sub_6F6F40(v17, 0);
  if ( !v12 )
  {
LABEL_14:
    for ( i = v17[0]; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    goto LABEL_7;
  }
  i = 0;
LABEL_19:
  sub_6F69D0(v11, 0);
  *(_QWORD *)(v5 + 16) = sub_6F6F40(v11, 0);
LABEL_6:
  if ( !i )
    goto LABEL_14;
LABEL_7:
  v6 = sub_726700(23);
  *(_QWORD *)v6 = i;
  v7 = v6;
  *(_QWORD *)(v6 + 64) = v5;
  *(_BYTE *)(v6 + 56) = a1;
  if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 2) != 0 && *(_BYTE *)(v6 + 24) )
  {
    sub_6E70E0(v6, v19);
    LODWORD(v20) = v16;
    WORD2(v20) = WORD2(v16);
    *(_QWORD *)dword_4F07508 = v20;
    LODWORD(v21) = v14;
    WORD2(v21) = v13;
    *(_QWORD *)&dword_4F061D8 = v21;
    sub_6E3280(v19, &dword_4F077C8);
    sub_6F6F40(v19, 0);
  }
  sub_6E70E0(v7, a2);
LABEL_11:
  *(_DWORD *)(a2 + 68) = v16;
  *(_WORD *)(a2 + 72) = WORD2(v16);
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(a2 + 68);
  *(_DWORD *)(a2 + 76) = v14;
  *(_WORD *)(a2 + 80) = v13;
  *(_QWORD *)&dword_4F061D8 = *(_QWORD *)(a2 + 76);
  return sub_6E3280(a2, &v16);
}
