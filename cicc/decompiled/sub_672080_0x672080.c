// Function: sub_672080
// Address: 0x672080
//
__int64 __fastcall sub_672080(__int64 **a1)
{
  __int64 *v2; // r14
  __int64 v3; // rsi
  unsigned __int8 v4; // dl
  __int64 v5; // rdi
  __int64 v6; // r8
  __int64 v7; // r13
  __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rcx
  unsigned __int16 v12; // ax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 **v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r15
  char v19; // al
  __int64 **v20; // rax
  __int64 *v21; // rax
  __int64 i; // rax
  __int64 *v23; // rax
  __int64 v24; // [rsp+10h] [rbp-60h] BYREF
  __int64 v25; // [rsp+18h] [rbp-58h] BYREF
  _BYTE v26[80]; // [rsp+20h] [rbp-50h] BYREF

  v2 = *a1;
  v3 = **a1;
  if ( unk_4D04A18 )
  {
    v4 = *(_BYTE *)(unk_4D04A18 + 80LL);
    v5 = unk_4D04A18;
    v6 = v4;
    if ( v4 == 16 )
    {
      v5 = **(_QWORD **)(unk_4D04A18 + 88LL);
      v6 = *(unsigned __int8 *)(v5 + 80);
      if ( (_BYTE)v6 != 24 )
      {
LABEL_4:
        if ( v3 != qword_4D04A00 )
        {
LABEL_5:
          LODWORD(v7) = 0;
          return (unsigned int)v7;
        }
        goto LABEL_11;
      }
    }
    else if ( v4 != 24 )
    {
      goto LABEL_4;
    }
    v5 = *(_QWORD *)(v5 + 88);
    if ( v3 != qword_4D04A00 )
      goto LABEL_5;
LABEL_11:
    LODWORD(v7) = 0;
    if ( (__int64 *)unk_4D04A18 != v2 )
    {
      if ( v4 != 3
        || !*(_BYTE *)(unk_4D04A18 + 104LL)
        || (v20 = *(__int64 ***)(unk_4D04A18 + 88LL), v20 != a1)
        && (!v20 || (v6 = dword_4F07588) == 0 || (v21 = v20[4], a1[4] != v21) || !v21) )
      {
        LODWORD(v7) = 1;
        if ( (*((_BYTE *)a1 + 177) & 0x40) != 0 && *(_BYTE *)(v5 + 80) == 4 )
        {
          v15 = *(__int64 ***)(v5 + 88);
          LODWORD(v7) = 0;
          if ( v15 != a1 )
            v7 = (unsigned int)sub_8D97D0(a1, v15, 0, qword_4D04A00, v6) == 0;
        }
      }
    }
    goto LABEL_15;
  }
  if ( v3 != qword_4D04A00 )
    goto LABEL_5;
  LODWORD(v7) = 0;
LABEL_15:
  sub_64E7D0();
  if ( (unk_4D04A10 & 0x19) != 0 )
    goto LABEL_5;
  v9 = 0;
  sub_7ADF70(v26, 0);
LABEL_17:
  sub_7AE360(v26);
  sub_7B8B50(v26, v9, v10, v11);
  v12 = word_4F06418[0];
  while ( 1 )
  {
    if ( v12 != 25 )
      goto LABEL_27;
    if ( !dword_4D043F8 )
      goto LABEL_29;
    v9 = 0;
    if ( (unsigned __int16)sub_7BE840(0, 0) != 25 )
      break;
    v9 = 1;
    sub_7BBD70(v26, 1);
    v12 = word_4F06418[0];
    if ( word_4F06418[0] == 26 )
      goto LABEL_17;
  }
  while ( 1 )
  {
    v12 = word_4F06418[0];
LABEL_27:
    if ( v12 != 28 )
      break;
    sub_7AE360(v26);
    sub_7B8B50(v26, v9, v13, v14);
  }
  if ( v12 != 27 )
  {
LABEL_29:
    sub_7BC000(v26);
    goto LABEL_5;
  }
  sub_7AE360(v26);
  sub_7B8B50(v26, v9, v16, v17);
  if ( word_4F06418[0] == 28 || word_4F06418[0] == 76 )
  {
    sub_7BC000(v26);
    goto LABEL_40;
  }
  if ( unk_4D04808 )
    *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) |= 0x80u;
  if ( !(unsigned int)sub_868D90(&v24, &v25, 1, 0, 0) || (unsigned int)sub_651B00(0x12u) )
  {
    if ( unk_4D04808 )
      *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) &= ~0x80u;
    sub_7BC000(v26);
LABEL_40:
    if ( (unk_4D04A11 & 0x40) == 0 )
      unk_4D04A10 &= ~0x80u;
    unk_4D04A18 = 0;
    sub_7D2AC0(&qword_4D04A00, a1, 4096);
    v18 = unk_4D04A18;
    if ( v2 == (__int64 *)unk_4D04A18 )
      goto LABEL_50;
    if ( unk_4D04A18 && (unsigned __int8)sub_877F80(unk_4D04A18) != 1 )
    {
      v19 = *(_BYTE *)(v18 + 80);
      if ( v19 == 3 )
      {
        for ( i = *(_QWORD *)(v18 + 88); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
          ;
        if ( a1 == (__int64 **)i )
          goto LABEL_49;
        if ( dword_4F07588 )
        {
          v23 = *(__int64 **)(i + 32);
          if ( a1[4] == v23 )
          {
            if ( v23 )
              goto LABEL_49;
          }
        }
      }
      else if ( v19 == 16 && (*(_BYTE *)(v18 + 96) & 4) == 0 )
      {
        goto LABEL_49;
      }
      sub_6851C0(1081, v18 + 48);
    }
LABEL_49:
    unk_4D04A18 = v2;
LABEL_50:
    v25 = qword_4D04A08;
    sub_87A680(&qword_4D04A00, &v25, 0);
    if ( (_DWORD)v7 )
      sub_685360(960, &qword_4D04A08);
    else
      LODWORD(v7) = 1;
    return (unsigned int)v7;
  }
  sub_867030(v24);
  LODWORD(v7) = unk_4D04808;
  if ( unk_4D04808 )
  {
    LODWORD(v7) = 0;
    *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) &= ~0x80u;
  }
  sub_7BC000(v26);
  return (unsigned int)v7;
}
