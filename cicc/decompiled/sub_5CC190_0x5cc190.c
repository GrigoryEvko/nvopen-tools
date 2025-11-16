// Function: sub_5CC190
// Address: 0x5cc190
//
__int64 __fastcall sub_5CC190(char a1)
{
  __int64 v2; // r8
  __int64 *v3; // rax
  __int64 *i; // r13
  unsigned __int16 v6; // ax
  __int64 v7; // rax
  __int16 v8; // r15
  __int64 v9; // r12
  __int64 v10; // rdx
  _QWORD *v11; // rax
  __int64 v12; // rax
  _BYTE *v13; // r12
  unsigned int v14; // r15d
  __int16 v15; // r14
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  _QWORD *v19; // rax
  __int64 v20; // rax
  __int64 v21; // r12
  unsigned __int16 v22; // ax
  __int64 v23; // rax
  __int64 v24; // rax
  _QWORD *v25; // rdx
  __int64 v26; // rcx
  _QWORD *v27; // rax
  unsigned int v28; // [rsp+0h] [rbp-50h]
  __int64 v30; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v31[7]; // [rsp+18h] [rbp-38h] BYREF

  v30 = 0;
  v2 = unk_4D04168;
  if ( unk_4D04168 )
  {
    v30 = unk_4D04168;
    v3 = (__int64 *)unk_4D04168;
    do
    {
      *((_BYTE *)v3 + 10) = a1;
      v3 = (__int64 *)*v3;
    }
    while ( v3 );
    unk_4D04168 = 0;
    return v2;
  }
  for ( i = &v30; ; i = (__int64 *)sub_5CB9F0((_QWORD **)i) )
  {
LABEL_7:
    v6 = word_4F06418[0];
    if ( word_4F06418[0] == 25 )
      goto LABEL_21;
LABEL_8:
    if ( v6 != 248 )
      break;
LABEL_24:
    v12 = sub_727670();
    *(_BYTE *)(v12 + 9) = 4;
    v13 = (_BYTE *)v12;
    *(_QWORD *)(v12 + 56) = *(_QWORD *)&dword_4F063F8;
    if ( unk_4F077C4 != 2
      && unk_4F07778 > 202310
      && unk_4D04A00
      && *(_QWORD *)(unk_4D04A00 + 16LL) == 8
      && !memcmp(*(const void **)(unk_4D04A00 + 8LL), "_Alignas", 8u)
      && !(unsigned int)sub_729F80(dword_4F063F8) )
    {
      sub_684AA0(4 - ((unsigned int)(unk_4D04964 == 0) - 1), 3291, &dword_4F063F8);
      sub_67D850(3291, 1, 0);
    }
    v13[8] = 3;
    sub_5C6A20((__int64)v13);
    v13[10] = a1;
    v14 = dword_4F063F8;
    v15 = unk_4F063FC;
    sub_7B8B50();
    sub_5C9ED0((__int64)v13, "(?ct+)");
    v16 = sub_727710();
    *(_DWORD *)v16 = v14;
    v17 = v16;
    *(_WORD *)(v16 + 4) = v15;
    unk_4F061D8 = unk_4F063F0;
    v18 = *(_QWORD *)&dword_4F063F8;
    *(_QWORD *)(v16 + 8) = unk_4F063F0;
    unk_4D04178 = v18;
    unk_4D04180 = unk_4F06650;
    v19 = v13;
    do
    {
      v19[5] = v17;
      v19 = (_QWORD *)*v19;
    }
    while ( v19 );
    dword_4CF6E60[(unsigned __int8)v13[8]] |= 1 << v13[9];
    *i = (__int64)v13;
  }
  while ( 1 )
  {
    if ( v6 == 142 )
    {
      if ( !unk_4D043E0 )
        break;
      v20 = sub_5CC040(a1);
      *i = v20;
      if ( v20 )
        i = (__int64 *)sub_5CB9F0((_QWORD **)i);
      goto LABEL_7;
    }
    if ( v6 != 132 )
      break;
    if ( !unk_4D043DC )
      break;
    if ( (unsigned __int8)(a1 - 1) > 1u && a1 != 5 )
    {
      if ( a1 != 9 )
        break;
      v7 = *(_QWORD *)(unk_4F04C68 + 776LL * unk_4F04C64 + 624);
      if ( !v7 || (*(_BYTE *)(v7 + 131) & 8) == 0 )
        break;
    }
    v28 = dword_4F063F8;
    v8 = unk_4F063FC;
    sub_7B8B50();
    sub_7BE280(27, 125, 0, 0);
    ++*(_BYTE *)(unk_4F061C8 + 36LL);
    v9 = sub_5CBEA0(a1, 3, 28, 0);
    if ( !v9 )
    {
      sub_7BE280(28, 18, 0, 0);
      --*(_BYTE *)(unk_4F061C8 + 36LL);
      *i = 0;
      goto LABEL_7;
    }
    v10 = sub_727710();
    *(_WORD *)(v10 + 4) = v8;
    *(_DWORD *)v10 = v28;
    unk_4F061D8 = unk_4F063F0;
    *(_QWORD *)(v10 + 8) = unk_4F063F0;
    unk_4D04178 = *(_QWORD *)&dword_4F063F8;
    unk_4D04180 = unk_4F06650;
    v11 = (_QWORD *)v9;
    do
    {
      v11[5] = v10;
      v11 = (_QWORD *)*v11;
    }
    while ( v11 );
    sub_7BE280(28, 18, 0, 0);
    --*(_BYTE *)(unk_4F061C8 + 36LL);
    *i = v9;
    i = (__int64 *)sub_5CB9F0((_QWORD **)i);
    v6 = word_4F06418[0];
    if ( word_4F06418[0] != 25 )
      goto LABEL_8;
LABEL_21:
    if ( !unk_4D043F8 )
      break;
    if ( (unsigned __int16)sub_7BE840(0, 0) == 25 )
    {
      v31[0] = *(_QWORD *)&dword_4F063F8;
      if ( unk_4F077BC && (unk_4F077C4 != 2 || unk_4F07778 <= 201102 && !unk_4F07774) )
        sub_684B40(v31, 2509);
      sub_7B8B50();
      sub_7B8B50();
      ++*(_BYTE *)(unk_4F061C8 + 34LL);
      if ( word_4F06418[0] == 179 && unk_4D041EC && (v22 = sub_7BE840(0, 0), sub_5CBA20(v22)) )
      {
        sub_7B8B50();
        if ( !sub_5CBA20(word_4F06418[0]) )
        {
          sub_6851D0(40);
          goto LABEL_45;
        }
        if ( (unk_4F077C4 != 2 || unk_4F07778 <= 201702)
          && unk_4F077BC
          && !dword_4CF6E48
          && !(unsigned int)sub_729F80(dword_4F063F8) )
        {
          sub_684B30(2921, &dword_4F063F8);
          dword_4CF6E48 = 1;
        }
        v21 = sub_727670();
        v23 = *(_QWORD *)&dword_4F063F8;
        *(_WORD *)(v21 + 8) = 258;
        *(_BYTE *)(v21 + 10) = a1;
        *(_QWORD *)(v21 + 56) = v23;
        if ( word_4F06418[0] == 118 )
          v24 = sub_724840(unk_4F073B8, "restrict");
        else
          v24 = sub_7C9D40();
        *(_QWORD *)(v21 + 16) = v24;
        *(_QWORD *)(v21 + 64) = unk_4F063F0;
        sub_5C9440(v21);
        sub_7B8B50();
        sub_7BE280(55, 53, 0, 0);
        *(_QWORD *)v21 = sub_5CBEA0(a1, 1, 26, v21);
        sub_7BE280(26, 17, 0, 0);
LABEL_59:
        v25 = (_QWORD *)sub_727710();
        *v25 = v31[0];
        unk_4F061D8 = unk_4F063F0;
        v26 = *(_QWORD *)&dword_4F063F8;
        v25[1] = unk_4F063F0;
        unk_4D04178 = v26;
        unk_4D04180 = unk_4F06650;
        v27 = (_QWORD *)v21;
        do
        {
          v27[5] = v25;
          v27 = (_QWORD *)*v27;
        }
        while ( v27 );
        sub_7BE280(26, 17, 0, 0);
        --*(_BYTE *)(unk_4F061C8 + 34LL);
        *i = v21;
        i = (__int64 *)sub_5CB9F0((_QWORD **)i);
      }
      else
      {
LABEL_45:
        v21 = sub_5CBEA0(a1, 1, 26, 0);
        sub_7BE280(26, 17, 0, 0);
        if ( v21 )
          goto LABEL_59;
        sub_7BE280(26, 17, 0, 0);
        --*(_BYTE *)(unk_4F061C8 + 34LL);
        *i = 0;
      }
      goto LABEL_7;
    }
    v6 = word_4F06418[0];
    if ( word_4F06418[0] == 248 )
      goto LABEL_24;
  }
  if ( i && *i )
    sub_5CB9F0((_QWORD **)i);
  return v30;
}
