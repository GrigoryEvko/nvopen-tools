// Function: sub_65D0E0
// Address: 0x65d0e0
//
__int64 __fastcall sub_65D0E0(__int64 a1, _DWORD *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // rax
  __int64 v11; // rax
  int v12; // r11d
  unsigned __int16 v13; // ax
  __int64 v14; // rdx
  __int64 v15; // rdi
  __int64 v17; // rdx
  __int64 v18; // rax
  bool v19; // zf
  __int64 v20; // r14
  __int64 v21; // rdx
  int v22; // ecx
  __int64 v23; // rax
  _QWORD *v24; // rax
  __int64 v25; // rax
  __int64 v26; // rdx
  char i; // cl
  int v28; // [rsp+18h] [rbp-B8h]
  _QWORD *v29; // [rsp+18h] [rbp-B8h]
  _QWORD *v30; // [rsp+18h] [rbp-B8h]
  unsigned int v31; // [rsp+24h] [rbp-ACh] BYREF
  __int64 v32; // [rsp+28h] [rbp-A8h] BYREF
  __int64 v33; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v34; // [rsp+38h] [rbp-98h] BYREF
  _QWORD v35[18]; // [rsp+40h] [rbp-90h] BYREF

  if ( *a2 || word_4F06418[0] == 27 && (*a2 = 1, sub_7B8B50(a1, a2, a3, a4), *a2) )
    ++*(_BYTE *)(qword_4F061C8 + 36LL);
  v6 = a1;
  v7 = *(_QWORD *)&dword_4F063F8;
  memset(v35, 0, 0x58u);
  *(_QWORD *)(a1 + 24) = *(_QWORD *)&dword_4F063F8;
  *(_QWORD *)dword_4F07508 = v7;
  sub_672A20(524354, a1, v35);
  v10 = *(_QWORD *)(a1 + 8);
  if ( (v10 & 0x20) != 0 )
  {
    v6 = a1 + 24;
    sub_6851C0(255, a1 + 24);
  }
  else if ( (v10 & 1) == 0 )
  {
    v6 = *(_QWORD *)(a1 + 272);
    sub_64E990((__int64)dword_4F07508, v6, 0, 0, 0, 1);
  }
  v11 = *(_QWORD *)(a1 + 288);
  if ( v11 )
  {
    while ( *(_BYTE *)(v11 + 140) == 12 )
      v11 = *(_QWORD *)(v11 + 160);
    *(_BYTE *)(v11 + 88) |= 4u;
  }
  if ( !dword_4F077BC || qword_4F077A8 > 0x76BFu )
    goto LABEL_13;
  v9 = (unsigned int)*a2;
  if ( !(_DWORD)v9 )
  {
    v23 = sub_626600(*(_QWORD *)(a1 + 288), a1, 1, 0, 0, 0, 0, &v31, (__int64)v35);
    v19 = word_4F06418[0] == 25;
    v33 = 0;
    v34 = 0;
    v20 = v23;
    v21 = qword_4F061C8;
    v22 = *(unsigned __int8 *)(qword_4F061C8 + 33LL);
    v6 = (unsigned int)(v22 + 1);
    *(_BYTE *)(qword_4F061C8 + 33LL) = v22 + 1;
    if ( !v19 )
      goto LABEL_46;
    sub_625AB0(a1, &v32, 1, 0, 0, 0, 0, 0, (__int64)v35);
    sub_624710(v32, &v33, &v34, a1, 0);
    v24 = &qword_4F061C8;
    goto LABEL_50;
  }
  v13 = word_4F06418[0];
  if ( word_4F06418[0] == 28 )
  {
    v6 = 0;
    if ( (unsigned __int16)sub_7BE840(0, 0) == 25 )
    {
      *a2 = 0;
      sub_7B8B50(0, 0, v17, v8);
      v12 = 1;
      --*(_BYTE *)(qword_4F061C8 + 36LL);
LABEL_14:
      if ( *a2 )
      {
        v13 = word_4F06418[0];
        goto LABEL_16;
      }
      v28 = v12;
      v18 = sub_626600(*(_QWORD *)(a1 + 288), a1, 1, 0, 0, 0, 0, &v31, (__int64)v35);
      v19 = word_4F06418[0] == 25;
      v33 = 0;
      v34 = 0;
      v20 = v18;
      v21 = qword_4F061C8;
      v22 = *(unsigned __int8 *)(qword_4F061C8 + 33LL);
      v6 = (unsigned int)(v22 + 1);
      *(_BYTE *)(qword_4F061C8 + 33LL) = v22 + 1;
      if ( !v19 )
      {
LABEL_46:
        *(_BYTE *)(v21 + 33) = v22;
        v8 = v31;
        if ( v31 )
        {
          v6 = a1 + 24;
          if ( (unsigned int)sub_8DD040(v20, a1 + 24) )
            v20 = sub_72C930(v20);
        }
        *(_QWORD *)(a1 + 280) = v20;
        *(_QWORD *)(a1 + 288) = v20;
        goto LABEL_22;
      }
      sub_625AB0(a1, &v32, 1, 0, 0, 0, 0, 0, (__int64)v35);
      sub_624710(v32, &v33, &v34, a1, 0);
      v24 = &qword_4F061C8;
      if ( v28 )
      {
LABEL_57:
        v6 = v33;
        if ( v33 )
        {
          if ( v20 )
          {
            v26 = v34;
            for ( i = *(_BYTE *)(v34 + 140); i == 12; i = *(_BYTE *)(v26 + 140) )
              v26 = *(_QWORD *)(v26 + 160);
            if ( i )
            {
              v30 = v24;
              sub_624710(v20, &v33, &v34, a1, 0);
              v6 = v33;
              v24 = v30;
            }
          }
          v21 = *v24;
          v20 = v6;
          LOBYTE(v22) = *(_BYTE *)(*v24 + 33LL) - 1;
        }
        else
        {
          v21 = *v24;
          LOBYTE(v22) = *(_BYTE *)(*v24 + 33LL) - 1;
        }
        goto LABEL_46;
      }
LABEL_50:
      while ( word_4F06418[0] == 25 )
      {
        v29 = v24;
        sub_625AB0(a1, &v32, 0, 0, 0, 0, 0, 0, (__int64)v35);
        sub_624710(v32, &v33, &v34, a1, 0);
        v24 = v29;
      }
      goto LABEL_57;
    }
LABEL_13:
    v12 = 0;
    goto LABEL_14;
  }
LABEL_16:
  if ( (v13 & 0xFFFD) == 0x19
    || v13 == 34
    || dword_4F077C4 == 2
    && (v13 != 1 || (unk_4D04A11 & 2) == 0)
    && (!(unsigned int)sub_7C0F00(0, 0) && word_4F06418[0] == 15
     || word_4F06418[0] == 33
     || (v6 = dword_4D04474) != 0 && word_4F06418[0] == 52) )
  {
    v6 = a1;
    sub_626F50(0x46u, a1, 0, 0, 0, v35);
  }
  if ( (*(_BYTE *)(a1 + 17) & 1) == 0 )
  {
    if ( word_4F06418[0] == 28 )
    {
      v6 = 18;
      unk_4F061D8 = qword_4F063F0;
      sub_7BE280(28, 18, 0, 0);
      --*(_BYTE *)(qword_4F061C8 + 36LL);
      goto LABEL_24;
    }
    v6 = 18;
    sub_7BE280(28, 18, 0, 0);
    --*(_BYTE *)(qword_4F061C8 + 36LL);
  }
LABEL_22:
  if ( !LODWORD(v35[7]) )
  {
    v14 = v35[5];
    unk_4F061D8 = v35[5];
    if ( (*(_BYTE *)(a1 + 124) & 0x20) == 0 )
      goto LABEL_25;
LABEL_43:
    sub_6451E0(a1);
    goto LABEL_25;
  }
  v14 = v35[7];
  unk_4F061D8 = v35[7];
LABEL_24:
  if ( (*(_BYTE *)(a1 + 124) & 0x20) != 0 )
    goto LABEL_43;
LABEL_25:
  sub_65C470(a1, v6, v14, v8, v9);
  if ( qword_4D0495C )
  {
    v15 = *(_QWORD *)(a1 + 288);
    if ( v15 )
    {
      if ( (unsigned int)sub_6454D0(v15, a1 + 24) )
      {
        v25 = sub_72C930(v15);
        *(_QWORD *)(a1 + 272) = v25;
        *(_QWORD *)(a1 + 280) = v25;
        *(_QWORD *)(a1 + 288) = v25;
      }
    }
  }
  return sub_643EB0(a1, 0);
}
