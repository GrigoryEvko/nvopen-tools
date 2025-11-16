// Function: sub_86FD00
// Address: 0x86fd00
//
_QWORD **__fastcall sub_86FD00(int a1, int a2, int a3, unsigned int a4, unsigned int a5, _QWORD *a6)
{
  char *v7; // rdx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rax
  char v11; // cl
  __int64 v12; // rsi
  unsigned __int64 v13; // rdi
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  unsigned __int16 v17; // ax
  bool v18; // cf
  bool v19; // zf
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rdi
  unsigned __int16 v28; // ax
  __int64 v29; // r13
  _BYTE *v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v34; // rax
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // r14
  __int64 v38; // rax
  __int64 v39; // r8
  __int64 v40; // r9
  __m128i *v41; // rdi
  _QWORD *v42; // rcx
  __int64 v43; // r9
  __int64 v44; // r8
  _QWORD *i; // rbx
  _BYTE *v46; // rbx
  _QWORD *v47; // rax
  __int64 *v48; // r13
  __int64 *v49; // rbx
  _BYTE *v50; // rdi
  char v53; // [rsp+1Bh] [rbp-65h]
  unsigned int v54; // [rsp+1Ch] [rbp-64h]
  unsigned int v55; // [rsp+20h] [rbp-60h]
  _QWORD **v57; // [rsp+28h] [rbp-58h]
  int v59; // [rsp+3Ch] [rbp-44h]
  __int64 v60; // [rsp+40h] [rbp-40h] BYREF
  _QWORD v61[7]; // [rsp+48h] [rbp-38h] BYREF

  v54 = dword_4F04C3C;
  if ( a1 )
  {
    dword_4F5FD80 = 0;
    qword_4F5FD78 = 0x100000001LL;
    qword_4F5FD68 = 0;
    qword_4F5FD70 = 0;
    v42 = sub_726B30(11);
    v57 = (_QWORD **)v42;
    *(_BYTE *)(v42[10] + 24LL) &= ~1u;
    *v42 = *(_QWORD *)&dword_4F063F8;
    v44 = dword_4F04C3C;
    if ( !dword_4F04C3C )
      sub_8699D0((__int64)v42, 21, 0);
    unk_4D03B90 = -1;
    sub_86D170(0, (__int64)v57, qword_4F06BC0, 0, v44, v43);
    if ( a2 )
      *(_BYTE *)(qword_4D03B98 + 176LL * unk_4D03B90 + 4) |= 0x40u;
    v55 = 0;
    if ( dword_4F077C4 != 2 )
    {
      v55 = dword_4D047EC;
      if ( dword_4D047EC )
      {
        for ( i = *(_QWORD **)(*(_QWORD *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C5C + 184) + 208LL);
              i;
              i = (_QWORD *)*i )
        {
          sub_86F7A0((__int64)i, &dword_4F063F8);
        }
        v55 = 0;
      }
    }
  }
  else
  {
    if ( !a3 )
    {
      v55 = unk_4D03B90;
      if ( a4 )
      {
        if ( unk_4D03B90 >= 0 )
        {
LABEL_5:
          v55 = 0;
          v7 = 0;
          goto LABEL_6;
        }
        v55 = 0;
        v7 = 0;
        qword_4F5FD78 = 0x100000001LL;
        dword_4F5FD80 = 0;
      }
      else
      {
        if ( unk_4D03B90 )
          goto LABEL_5;
        v7 = 0;
        if ( *(_DWORD *)qword_4D03B98 == 8 )
        {
          v55 = 1;
          if ( (unsigned __int8)(*(_BYTE *)(*(_QWORD *)(qword_4F04C50 + 32LL) + 174LL) - 1) <= 1u )
          {
            v7 = (char *)qword_4F06BC0;
            qword_4F06BC0 = *(_QWORD *)(qword_4F06BC0 + 32LL);
          }
        }
      }
LABEL_6:
      v57 = (_QWORD **)sub_86EC60(0, a4, v7);
      if ( !a2 )
        goto LABEL_7;
      goto LABEL_59;
    }
    v57 = (_QWORD **)sub_726B30(11);
    *v57 = *(_QWORD **)&dword_4F063F8;
    if ( !dword_4F04C3C )
      sub_8699D0((__int64)v57, 21, 0);
    sub_854AB0();
    v34 = sub_7340A0(qword_4F06BC0);
    sub_86D170(0, (__int64)v57, v34, 0, v35, v36);
    v55 = 0;
    if ( a2 )
LABEL_59:
      *(_BYTE *)(qword_4D03B98 + 176LL * unk_4D03B90 + 4) |= 0x40u;
  }
LABEL_7:
  v10 = qword_4F061C8;
  v11 = *(_BYTE *)(qword_4F061C8 + 94LL);
  ++*(_BYTE *)(qword_4F061C8 + 82LL);
  v19 = word_4F06418[0] == 73;
  *(_BYTE *)(v10 + 94) = 0;
  v53 = v11;
  if ( v19 )
    v60 = *(_QWORD *)&dword_4F063F8;
  else
    v60 = *(_QWORD *)&dword_4F077C8;
  v12 = 130;
  v13 = 73;
  sub_7BE280(0x49u, 130, 0, 0, v8, v9);
  if ( dword_4F077C4 == 2 )
  {
    if ( unk_4F07778 <= 201102 && !dword_4F07774 )
    {
LABEL_11:
      if ( !unk_4D04778 )
        goto LABEL_13;
    }
  }
  else if ( unk_4F07778 <= 199900 )
  {
    goto LABEL_11;
  }
  sub_857CE0();
LABEL_13:
  if ( HIDWORD(qword_4F077B4) )
  {
    while ( 1 )
    {
      v17 = word_4F06418[0];
      v18 = word_4F06418[0] == 0;
      v19 = word_4F06418[0] == 1;
      if ( word_4F06418[0] != 1 )
        break;
      v14 = 10;
      v12 = (__int64)"__label__";
      v20 = qword_4D04A00;
      v13 = *(_QWORD *)(qword_4D04A00 + 8);
      do
      {
        if ( !v14 )
          break;
        v18 = *(_BYTE *)v12 < *(_BYTE *)v13;
        v19 = *(_BYTE *)v12++ == *(_BYTE *)v13++;
        --v14;
      }
      while ( v19 );
      LOBYTE(v20) = (!v18 && !v19) - v18;
      if ( (_BYTE)v20 )
        goto LABEL_27;
      sub_7B8B50(v13, (unsigned int *)v12, v20, v14, v15, v16);
      sub_732EF0(qword_4F04C68[0] + 776LL * (int)dword_4F04C5C);
      ++*(_BYTE *)(qword_4F061C8 + 83LL);
      while ( word_4F06418[0] == 1 )
      {
        sub_64E550(0, 1);
        if ( !(unsigned int)sub_7BE800(0x43u, (unsigned int *)1, v21, v22, v23, v24) )
          goto LABEL_23;
      }
      sub_6851D0(0x28u);
LABEL_23:
      v12 = 65;
      v13 = 75;
      sub_7BE280(0x4Bu, 65, 0, 0, v25, v26);
      --*(_BYTE *)(qword_4F061C8 + 83LL);
      if ( !HIDWORD(qword_4F077B4) )
        goto LABEL_24;
    }
  }
  else
  {
LABEL_24:
    v17 = word_4F06418[0];
  }
  if ( (unsigned __int16)(v17 - 9) > 1u && v17 != 74 )
  {
LABEL_27:
    v59 = 0;
    while ( 1 )
    {
      if ( dword_4F077C4 == 2 )
      {
        v12 = a5;
        v13 = 0;
        sub_8708D0(0, a5);
LABEL_33:
        v17 = word_4F06418[0];
        if ( (unsigned __int16)(word_4F06418[0] - 9) <= 1u )
          goto LABEL_43;
        goto LABEL_34;
      }
      if ( v17 == 187 )
      {
        a5 = 1;
        v61[0] = *(_QWORD *)&dword_4F063F8;
        *(_QWORD *)(qword_4D03B98 + 176LL * unk_4D03B90 + 160) = v61;
        sub_7B8B50(v13, (unsigned int *)v12, (__int64)v61, v14, v15, v16);
      }
      v28 = word_4F06418[0];
      if ( word_4F06418[0] == 142 )
        goto LABEL_66;
      if ( word_4F06418[0] != 25 )
        goto LABEL_40;
      if ( dword_4D043F8 )
        break;
LABEL_41:
      if ( (unsigned int)sub_651B00(3u) )
      {
        if ( !unk_4D04390 )
        {
          if ( v59 )
          {
            sub_6851C0(0x10Cu, dword_4F07508);
            if ( a1 )
            {
              if ( word_4F063FC[0] == 1 )
                goto LABEL_43;
            }
          }
        }
        sub_854590(0);
        v27 = *(_QWORD *)(qword_4D03B98 + 176LL * unk_4D03B90 + 16);
        if ( v27 )
        {
          sub_5CC9F0(v27);
          *(_QWORD *)(qword_4D03B98 + 176LL * unk_4D03B90 + 16) = 0;
        }
        v12 = 0;
        v13 = a5;
        sub_86E660(a5, 0);
        goto LABEL_33;
      }
LABEL_42:
      v12 = a5;
      v13 = 0;
      sub_8708D0(0, a5);
      v17 = word_4F06418[0];
      v59 = 1;
      if ( (unsigned __int16)(word_4F06418[0] - 9) <= 1u )
        goto LABEL_43;
LABEL_34:
      if ( v17 == 74 )
        goto LABEL_43;
    }
    if ( (unsigned __int16)sub_7BE840(0, 0) == 25 )
    {
LABEL_66:
      v37 = qword_4D03B98 + 176LL * unk_4D03B90;
      *(_QWORD *)(v37 + 16) = sub_5CC190(1);
    }
    v28 = word_4F06418[0];
LABEL_40:
    if ( v28 == 1 && (unsigned __int16)sub_7BE840(0, 0) == 55 )
      goto LABEL_42;
    goto LABEL_41;
  }
LABEL_43:
  if ( sub_854590(0) )
  {
    v41 = (__m128i *)sub_854840(5u, 0, 0, 0);
    if ( v41 )
    {
      *(__int64 *)((char *)&qword_4F5FD78 + 4) = 0x100000000LL;
      sub_854000(v41);
    }
    sub_854AB0();
  }
  if ( !(_DWORD)qword_4F5FD78 )
  {
LABEL_51:
    sub_854430();
    if ( !a4 )
      goto LABEL_53;
    goto LABEL_52;
  }
  if ( a1 | v55 )
  {
    v29 = *(_QWORD *)(qword_4F04C50 + 32LL);
    goto LABEL_47;
  }
  if ( a3 && unk_4D03B90 == 1 && *(_DWORD *)qword_4D03B98 == 8 )
  {
    v29 = *(_QWORD *)(qword_4F04C50 + 32LL);
    if ( (unsigned __int8)(*(_BYTE *)(v29 + 174) - 1) > 1u )
    {
LABEL_47:
      if ( dword_4D047EC )
      {
        if ( unk_4D047E8 )
        {
          v50 = sub_86B560(qword_4F5FD68, qword_4F5FD70);
          if ( v50 )
            sub_86B010((__int64)v50, qword_4F5FD78);
        }
      }
      sub_86B690(1, v61);
      if ( *(char *)(v29 + 207) < 0 )
      {
        if ( (*(_BYTE *)(sub_71DF80(v29) + 120) & 2) != 0 )
        {
          v49 = (__int64 *)sub_86E480(0xAu, &dword_4F077C8);
          v49[6] = sub_695660(0, 0, v49);
        }
      }
      else
      {
        v30 = sub_86E480(8u, &dword_4F077C8);
        *((_QWORD *)v30 + 6) = v61[0];
      }
      goto LABEL_51;
    }
    v46 = sub_86E480(0, &dword_4F077C8);
    v47 = sub_726700(8);
    *((_QWORD *)v46 + 6) = v47;
    v47[7] = 0;
    v48 = (__int64 *)*((_QWORD *)v46 + 6);
    *v48 = sub_72CBE0();
    sub_7304E0(*((_QWORD *)v46 + 6));
    qword_4F5FD78 = 0;
    dword_4F5FD80 = 0;
  }
  sub_854430();
  if ( !a4 )
    goto LABEL_72;
LABEL_52:
  *a6 = *(_QWORD *)(qword_4D03B98 + 176LL * unk_4D03B90 + 96);
LABEL_53:
  if ( !a1 )
  {
LABEL_72:
    sub_86F430(v57);
    *(_BYTE *)(qword_4F061C8 + 94LL) = v53;
    *v57[10] = *(_QWORD *)&dword_4F063F8;
    dword_4F04C3C = v54;
    sub_869D70((__int64)v57, 21);
    v38 = qword_4F063F0;
    v57[1] = (_QWORD *)qword_4F063F0;
    *(_QWORD *)&dword_4F061D8 = v38;
    sub_7BE280(0x4Au, 67, 3196, &v60, v39, v40);
    goto LABEL_55;
  }
  sub_86F030();
  unk_4D03B90 = -1;
  v31 = *(_QWORD *)&dword_4F063F8;
  *(_BYTE *)(qword_4F061C8 + 94LL) = v53;
  *v57[10] = v31;
  dword_4F04C3C = v54;
  sub_869D70((__int64)v57, 21);
  v32 = qword_4F063F0;
  v57[1] = (_QWORD *)qword_4F063F0;
  *(_QWORD *)&dword_4F061D8 = v32;
LABEL_55:
  --*(_BYTE *)(qword_4F061C8 + 82LL);
  return v57;
}
