// Function: sub_6B8C50
// Address: 0x6b8c50
//
__int64 __fastcall sub_6B8C50(__int64 a1, __int64 a2, __int64 a3)
{
  bool v3; // r15
  __int64 v4; // rbx
  unsigned int *v5; // rsi
  __int64 v6; // r12
  unsigned int *v7; // r14
  __int64 v8; // rsi
  __int64 v9; // rsi
  unsigned int *v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  unsigned int v15; // r15d
  char v16; // dl
  __int64 v17; // rax
  __int64 v19; // rax
  bool v20; // dl
  __int64 v21; // r9
  __int64 v22; // rdi
  __int64 v23; // r12
  __int64 v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  int v29; // r10d
  __int64 v30; // r12
  unsigned __int16 v31; // ax
  __int64 v32; // rax
  __int64 v33; // r8
  unsigned int *v34; // r13
  __int64 v35; // rax
  int v36; // eax
  __int64 v37; // rax
  int v38; // eax
  __int64 v39; // rax
  int v40; // eax
  __int64 v41; // rax
  __int64 v42; // rax
  int v43; // [rsp+8h] [rbp-288h]
  __int64 v44; // [rsp+8h] [rbp-288h]
  __int64 v45; // [rsp+8h] [rbp-288h]
  unsigned int v46; // [rsp+10h] [rbp-280h]
  __int64 v47; // [rsp+10h] [rbp-280h]
  __int64 v48; // [rsp+10h] [rbp-280h]
  __int64 v49; // [rsp+10h] [rbp-280h]
  unsigned __int16 v50; // [rsp+18h] [rbp-278h]
  __int64 v51; // [rsp+18h] [rbp-278h]
  unsigned __int16 v52; // [rsp+20h] [rbp-270h]
  __int64 v53; // [rsp+20h] [rbp-270h]
  bool v54; // [rsp+20h] [rbp-270h]
  __int64 v55; // [rsp+20h] [rbp-270h]
  unsigned int v57; // [rsp+3Ch] [rbp-254h] BYREF
  __int64 v58; // [rsp+40h] [rbp-250h] BYREF
  __int64 v59; // [rsp+48h] [rbp-248h] BYREF
  __int64 v60; // [rsp+50h] [rbp-240h] BYREF
  char v61[8]; // [rsp+58h] [rbp-238h] BYREF
  _BYTE v62[160]; // [rsp+60h] [rbp-230h] BYREF
  unsigned int v63[17]; // [rsp+100h] [rbp-190h] BYREF
  unsigned int v64; // [rsp+144h] [rbp-14Ch]
  unsigned __int16 v65; // [rsp+148h] [rbp-148h]

  v3 = 0;
  v4 = a1;
  v59 = 0;
  v50 = unk_4F077CC;
  v46 = dword_4F077C8;
  v52 = word_4F06418[0];
  v5 = (unsigned int *)dword_4F077BC;
  if ( dword_4F077BC )
    v3 = qword_4F077A8 > 0x76BFu;
  if ( a1 )
  {
    v6 = (unsigned int)sub_687860(a1, dword_4F077BC);
    sub_68B050(v6, (__int64)&v57, &v60);
    sub_6E1DD0(&v59);
    v7 = 0;
    sub_6E2140(5, v62, 0, qword_4F06BC0 != 0, a1);
    sub_6E2170(v59);
    *(_WORD *)(qword_4D03C50 + 17LL) |= 0x2020u;
    v8 = unk_4D03C40;
    unk_4D03C40 = 0;
    v51 = v8;
    v9 = a1;
    sub_6F8800(*(_QWORD *)a1, a1, v63);
    goto LABEL_5;
  }
  v24 = dword_4F07738;
  if ( !dword_4F07738 )
  {
    a3 = dword_4D04320;
    if ( dword_4D04320 )
    {
      v5 = &dword_4F063F8;
      a1 = 1615;
      sub_684B30(0x64Fu, &dword_4F063F8);
    }
  }
  sub_7B8B50(a1, v5, a3, v24);
  if ( word_4F06418[0] == 27 )
  {
    v46 = dword_4F063F8;
    v50 = word_4F063FC[0];
    sub_7B8B50(a1, v5, v25, dword_4F063F8);
    a1 = 5;
    v40 = sub_679C10(5u);
    v29 = 1;
    if ( v40 )
    {
      v15 = 1;
      ++*(_BYTE *)(qword_4F061C8 + 36LL);
      sub_65CD60(&v58);
      v9 = 18;
      v49 = v58;
      sub_7BE280(28, 18, 0, 0);
      --*(_BYTE *)(qword_4F061C8 + 36LL);
      sub_7296C0(v63);
      v41 = sub_869D30();
      v10 = (unsigned int *)v63[0];
      v7 = (unsigned int *)v41;
      sub_729730(v63[0]);
      v13 = v58;
      v51 = 0;
      v14 = v49;
      goto LABEL_6;
    }
  }
  else
  {
    v29 = 0;
    if ( !v3 )
    {
      v29 = sub_6E5430(a1, v5, v25, v26, v27, v28);
      if ( v29 )
      {
        a1 = 125;
        sub_6851C0(0x7Du, &dword_4F063F8);
        v29 = 0;
      }
    }
  }
  v43 = v29;
  v30 = (unsigned int)sub_687860(a1, &v57);
  sub_68B050(v30, (__int64)&v57, &v60);
  sub_6E1DD0(&v59);
  sub_6E2140(5, v62, 0, qword_4F06BC0 != 0, 0);
  sub_6E2170(v59);
  *(_WORD *)(qword_4D03C50 + 17LL) |= 0x2020u;
  sub_7296C0(v63);
  v7 = (unsigned int *)sub_869D30();
  sub_729730(v63[0]);
  if ( v3 )
  {
    if ( v43 )
    {
      v9 = 0;
      sub_69ED20((__int64)v63, 0, 18, 8);
      v64 = v46;
      v31 = v50;
      v51 = 0;
      v65 = v31;
      goto LABEL_5;
    }
  }
  else if ( v43 )
  {
    sub_69ED20((__int64)v63, 0, 0, 0);
    v9 = 18;
    sub_7BE280(28, 18, 0, 0);
    v51 = 0;
    goto LABEL_5;
  }
  v9 = 0;
  sub_69ED20((__int64)v63, 0, 18, 0);
  v51 = 0;
LABEL_5:
  sub_6F6C80(v63);
  v10 = v63;
  sub_831BB0(v63);
  v13 = *(_QWORD *)v63;
  v14 = 0;
  v15 = dword_4F077C0;
  v58 = *(_QWORD *)v63;
  if ( dword_4F077C0 )
  {
    v9 = (__int64)v61;
    v10 = v63;
    if ( !(unsigned int)sub_6E9790(v63, v61, v11, v12, *(_QWORD *)v63, 0) )
    {
      v10 = (unsigned int *)v58;
      v9 = dword_4F077C4 == 2;
      v58 = sub_73D4C0(v58, v9);
    }
    v13 = v58;
    v15 = 0;
    v14 = 0;
  }
LABEL_6:
  v16 = *(_BYTE *)(v13 + 140);
  if ( v16 == 12 )
  {
    v17 = v13;
    do
    {
      v17 = *(_QWORD *)(v17 + 160);
      v16 = *(_BYTE *)(v17 + 140);
    }
    while ( v16 == 12 );
  }
  if ( v16 )
  {
    v47 = v14;
    v19 = sub_7259C0(12);
    v20 = 0;
    v21 = v47;
    v13 = v19;
    if ( dword_4F077C4 == 2 )
    {
      if ( dword_4F04C44 != -1
        || (v37 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v37 + 6) & 6) != 0)
        || *(_BYTE *)(v37 + 4) == 12 )
      {
        v44 = v13;
        v38 = sub_8DBE70(v58);
        v13 = v44;
        v21 = v47;
        v20 = v38 != 0;
      }
    }
    v22 = v58;
    if ( v52 == 190 )
    {
      v45 = v13;
      v48 = v21;
      v54 = v20;
      v39 = sub_73D4C0(v58, 1);
      v13 = v45;
      v21 = v48;
      v58 = v39;
      v20 = v54;
      v22 = v39;
    }
    *(_QWORD *)(v13 + 160) = v22;
    *(_BYTE *)(v13 + 184) = v15 + 6;
    *(_BYTE *)(v13 + 186) = (8 * v20) | *(_BYTE *)(v13 + 186) & 0xF7;
    *(_QWORD *)(*(_QWORD *)(v13 + 168) + 40LL) = v21;
    if ( !v15 )
    {
      v53 = v13;
      if ( v20 )
      {
        sub_6F40C0(v63);
        v10 = v63;
        v9 = 0;
        v42 = sub_6F6F40(v63, 0);
        v33 = v53;
        v34 = (unsigned int *)v42;
      }
      else
      {
        v9 = 0;
        v10 = v63;
        v32 = sub_6F6F40(v63, 0);
        v33 = v53;
        v34 = (unsigned int *)v32;
        if ( dword_4F04C44 != -1
          || (v35 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v35 + 6) & 6) != 0)
          || *(_BYTE *)(v35 + 4) == 12 )
        {
          v10 = v34;
          v36 = sub_731EE0(v34);
          v33 = v53;
          if ( v36 )
            *(_BYTE *)(v53 + 186) |= 8u;
        }
      }
      if ( (*(_BYTE *)(v34 - 2) & 1) != 0 )
      {
        *(_QWORD *)(*(_QWORD *)(v33 + 168) + 24LL) = v34;
      }
      else
      {
        v9 = 1;
        v10 = v34;
        v55 = v33;
        sub_72D910(v34, 1, v33);
        v33 = v55;
      }
      v58 = v33;
      if ( !v7 )
        goto LABEL_28;
LABEL_12:
      if ( *(_QWORD *)v7 )
        goto LABEL_13;
LABEL_27:
      v10 = v7;
      v9 = (unsigned int)dword_4F04C64;
      sub_869FD0(v7, (unsigned int)dword_4F04C64);
      if ( v15 )
      {
LABEL_16:
        v13 = v58;
        goto LABEL_17;
      }
LABEL_28:
      sub_6E2B30(v10, v9);
      sub_6E1DF0(v59);
      v23 = v60;
      sub_729730(v57);
      qword_4F06BC0 = v23;
      if ( v4 )
        unk_4D03C40 = v51;
      goto LABEL_16;
    }
    v58 = v13;
  }
  else if ( !v15 )
  {
    *(_BYTE *)(qword_4D03C50 + 18LL) &= ~0x20u;
    if ( !v7 )
      goto LABEL_28;
    goto LABEL_12;
  }
  if ( v7 )
  {
    if ( *(_QWORD *)v7 )
    {
LABEL_13:
      if ( !dword_4F04C3C )
        sub_8699D0(v58, 6, v7);
      v10 = (unsigned int *)v58;
      v9 = 6;
      sub_869D70(v58, 6);
      if ( v15 )
        goto LABEL_16;
      goto LABEL_28;
    }
    goto LABEL_27;
  }
LABEL_17:
  if ( a2 )
    *(_QWORD *)(a2 + 40) = qword_4F063F0;
  return v13;
}
