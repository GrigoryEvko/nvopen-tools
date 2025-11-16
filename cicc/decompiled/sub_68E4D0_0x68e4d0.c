// Function: sub_68E4D0
// Address: 0x68e4d0
//
__int64 __fastcall sub_68E4D0(__int64 *a1, __int64 a2, int a3, unsigned int a4, unsigned int a5, char a6)
{
  _DWORD *v6; // r13
  __int64 v9; // r15
  __int64 v10; // rdi
  int v11; // eax
  char v12; // dl
  int v13; // r14d
  __int64 v14; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  char v27; // al
  _BYTE *v28; // rax

  v6 = (_DWORD *)a2;
  v9 = *a1;
  if ( dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(v9) )
    sub_8AE000(v9);
  v10 = v9;
  v11 = sub_8D23B0(v9);
  v12 = *(_BYTE *)(v9 + 140);
  v13 = v11;
  if ( v12 == 12 )
  {
    v14 = v9;
    do
    {
      v14 = *(_QWORD *)(v14 + 160);
      v12 = *(_BYTE *)(v14 + 140);
    }
    while ( v12 == 12 );
  }
  if ( !v12 )
    goto LABEL_6;
  if ( (unsigned int)sub_8D3D40(v9) )
    goto LABEL_18;
  if ( v13
    && (a6 & 1) == 0
    && !(unsigned int)sub_8D2600(v9)
    && !(unsigned int)sub_8D2690(v9)
    && !(unsigned int)sub_8D3410(v9)
    && ((*(_BYTE *)(unk_4D03C50 + 19LL) & 4) == 0 || !(unsigned int)sub_8DD3B0(v9)) )
  {
    if ( !dword_4F077BC
      || (_DWORD)qword_4F077B4
      || qword_4F077A8 > 0x18768u
      || (v27 = *(_BYTE *)(unk_4D03C50 + 17LL), (v27 & 0x40) == 0)
      && ((v27 & 1) != 0
       || (v28 = (_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64), dword_4F04C44 == -1)
       && (v28[6] & 6) == 0
       && v28[4] != 12
       || (v28[12] & 0x10) != 0) )
    {
      v10 = a2;
      sub_6E5F60(a2, v9, 8);
LABEL_6:
      *a1 = sub_72C930(v10);
      return 1;
    }
  }
  if ( !(unsigned int)sub_8D3A70(v9) )
  {
    if ( (unsigned int)sub_8D3410(v9) )
    {
      a2 = a5;
      if ( !a5 || !(unsigned int)sub_8D23E0(v9) )
      {
        v10 = v9;
        if ( (unsigned int)sub_8D23B0(v9) )
        {
          if ( (unsigned int)sub_6E5430(v9, a5, v20, v21, v18, v19) )
          {
            v10 = 2363;
            sub_685360(0x93Bu, v6, v9);
          }
          goto LABEL_6;
        }
        v17 = a4;
        if ( !a4 )
        {
          v16 = HIDWORD(qword_4D0495C);
          if ( !HIDWORD(qword_4D0495C) )
            goto LABEL_33;
          v22 = sub_8D67C0(v9);
          *a1 = v22;
          v9 = v22;
          sub_6E5D20(5, 398, v6, v22);
        }
      }
    }
    else
    {
      v10 = v9;
      if ( (unsigned int)sub_8D2310(v9) )
        goto LABEL_33;
    }
    goto LABEL_18;
  }
  if ( dword_4F077C4 != 2 )
  {
    v10 = dword_4F077C0;
    if ( dword_4F077C0 )
    {
      if ( !a3 )
        return 0;
      goto LABEL_20;
    }
LABEL_33:
    if ( (unsigned int)sub_6E5430(v10, a2, v16, v17, v18, v19) )
    {
      v10 = 119;
      sub_685360(0x77u, v6, v9);
    }
    goto LABEL_6;
  }
  if ( (*(_BYTE *)(unk_4D03C50 + 19LL) & 0x40) != 0 )
  {
    if ( (unsigned int)sub_6E5430(v9, a2, v16, v17, v18, v19) )
    {
      sub_6851C0(0x1Cu, (_DWORD *)a2);
      v10 = v9;
      if ( !(unsigned int)sub_8D5830(v9) )
        goto LABEL_6;
      goto LABEL_39;
    }
LABEL_55:
    v10 = v9;
    if ( !(unsigned int)sub_8D5830(v9) )
      goto LABEL_6;
LABEL_39:
    if ( (unsigned int)sub_6E5430(v10, a2, v23, v24, v25, v26) )
    {
      v10 = 8;
      sub_5EB950(8u, 389, v9, a2);
    }
    goto LABEL_6;
  }
  if ( word_4D04898
    && (*(_BYTE *)(unk_4D03C50 + 17LL) & 6) == 2
    && !v13
    && !(unsigned int)sub_8D4160(v9)
    && (unsigned int)sub_6E91E0(28, a2) )
  {
    goto LABEL_55;
  }
  v10 = v9;
  if ( (unsigned int)sub_8D5830(v9) )
    goto LABEL_39;
LABEL_18:
  if ( !a3 )
    return 0;
  if ( dword_4F077C4 == 2 && (unsigned int)sub_8D3A70(v9) )
    return 0;
LABEL_20:
  if ( dword_4F077BC && qword_4F077A8 <= 0x76BFu )
    return 0;
  if ( (unsigned int)sub_6E53E0(5, 191, v6) )
    sub_684B30(0xBFu, v6);
  *a1 = sub_73D4C0(v9, dword_4F077C4 == 2);
  return 0;
}
