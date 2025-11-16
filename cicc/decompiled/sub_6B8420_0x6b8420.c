// Function: sub_6B8420
// Address: 0x6b8420
//
__int64 __fastcall sub_6B8420(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rbx
  __int64 v5; // rax
  char v6; // dl
  __int64 v7; // rcx
  __int64 v8; // rax
  char v9; // r13
  _QWORD *v10; // r15
  __int64 v11; // rsi
  __int64 v12; // rdi
  __int64 v13; // rax
  char v14; // dl
  __int64 v15; // r13
  char v16; // dl
  __int64 v17; // rdi
  __int64 v18; // rbx
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rax
  char v25; // dl
  __int64 v26; // rdx
  __int64 v27; // rax
  char v28; // al
  __int64 v29; // r12
  __int64 v30; // rdx
  __int64 v31; // rax
  int v32; // eax
  int v33; // eax
  char v34; // dl
  __int64 v35; // rdx
  char v36; // cl
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rax
  int v40; // eax
  __int64 v41; // rcx
  __int64 v42; // rdx
  _DWORD *v43; // r13
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r9
  char v48; // al
  int v49; // [rsp+4h] [rbp-28Ch]
  unsigned int v50; // [rsp+8h] [rbp-288h]
  unsigned int v51; // [rsp+Ch] [rbp-284h]
  __int64 v52; // [rsp+10h] [rbp-280h]
  char v53; // [rsp+18h] [rbp-278h]
  __int64 v54; // [rsp+18h] [rbp-278h]
  __int64 v55; // [rsp+18h] [rbp-278h]
  __int64 v56; // [rsp+18h] [rbp-278h]
  char v57; // [rsp+18h] [rbp-278h]
  unsigned int v58; // [rsp+2Ch] [rbp-264h] BYREF
  __int64 v59; // [rsp+30h] [rbp-260h] BYREF
  __int64 v60; // [rsp+38h] [rbp-258h] BYREF
  __int64 v61; // [rsp+40h] [rbp-250h] BYREF
  __int64 v62; // [rsp+48h] [rbp-248h]
  __int64 v63; // [rsp+50h] [rbp-240h]
  char v64[160]; // [rsp+60h] [rbp-230h] BYREF
  unsigned int v65[4]; // [rsp+100h] [rbp-190h] BYREF
  char v66; // [rsp+110h] [rbp-180h]
  FILE v67; // [rsp+144h] [rbp-14Ch] BYREF
  unsigned __int8 v68; // [rsp+240h] [rbp-50h]
  __int64 v69; // [rsp+248h] [rbp-48h]
  int v70; // [rsp+250h] [rbp-40h]

  v4 = a1;
  v51 = a2;
  if ( a1 )
  {
    v52 = unk_4D03C40;
    v5 = *(_QWORD *)a1;
    if ( *(_BYTE *)(*(_QWORD *)a1 + 24LL) == 1 )
    {
      do
      {
        while ( 1 )
        {
          v6 = *(_BYTE *)(v5 + 56);
          if ( v6 != 91 )
            break;
          v5 = *(_QWORD *)(*(_QWORD *)(v5 + 72) + 16LL);
          if ( *(_BYTE *)(v5 + 24) != 1 )
            goto LABEL_8;
        }
      }
      while ( v6 == 25 && *(_BYTE *)(v5 + 24) == 1 );
    }
LABEL_8:
    unk_4D03C40 = v5;
  }
  else
  {
    sub_7B8B50(0, a2, a3, a4);
    v38 = qword_4F061C8;
    ++*(_BYTE *)(qword_4F061C8 + 63LL);
    ++*(_BYTE *)(v38 + 34);
    sub_7BE280(25, 17, 0, 0);
    a2 = 53;
    a1 = 55;
    sub_7BE280(55, 53, 0, 0);
    v39 = qword_4F061C8;
    v52 = 0;
    --*(_BYTE *)(qword_4F061C8 + 63LL);
    --*(_BYTE *)(v39 + 34);
  }
  v50 = sub_687860(a1, a2);
  sub_68B050(v50, (__int64)&v58, &v60);
  sub_6E1DD0(&v59);
  sub_6E2140(5, v64, 0, qword_4F06BC0 != 0, v4);
  sub_6E2170(v59);
  v7 = qword_4D03C50;
  v8 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  *(_WORD *)(qword_4D03C50 + 17LL) |= 0x2020u;
  v9 = *(_BYTE *)(v7 + 18) >> 7;
  v53 = *(_BYTE *)(v8 + 13) & 1;
  *(_BYTE *)(v8 + 13) |= 1u;
  if ( v4 )
  {
    v10 = 0;
    sub_6F8800(*(_QWORD *)v4, v4, v65);
  }
  else
  {
    sub_7296C0(v65);
    v10 = (_QWORD *)sub_869D30();
    sub_729730(v65[0]);
    ++*(_BYTE *)(qword_4F061C8 + 34LL);
    ++*(_QWORD *)(qword_4D03C50 + 40LL);
    sub_69ED20((__int64)v65, 0, 0, 0);
  }
  v11 = 0;
  *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 13) = v53
                                                            | *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 13)
                                                            & 0xFE;
  *(_BYTE *)(qword_4D03C50 + 18LL) = (v9 << 7) | *(_BYTE *)(qword_4D03C50 + 18LL) & 0x7F;
  sub_6F69D0(v65, 0);
  v12 = *(_QWORD *)v65;
  if ( (unsigned int)sub_8D3A70(*(_QWORD *)v65) )
  {
    LODWORD(v61) = 0;
    v37 = sub_72CD60(v12, 0);
    v12 = (__int64)v65;
    v11 = v37;
    sub_845C60(v65, v37, 0, 2048, &v61);
  }
  if ( (v66 & 0xFD) == 0 )
  {
LABEL_14:
    v13 = *(_QWORD *)v65;
    v14 = *(_BYTE *)(*(_QWORD *)v65 + 140LL);
    if ( v14 != 12 )
      goto LABEL_16;
    goto LABEL_15;
  }
  v13 = *(_QWORD *)v65;
  v34 = *(_BYTE *)(*(_QWORD *)v65 + 140LL);
  if ( v34 != 12 )
  {
    if ( !v34 )
      goto LABEL_33;
    goto LABEL_57;
  }
  v35 = *(_QWORD *)v65;
  do
  {
    v35 = *(_QWORD *)(v35 + 160);
    v36 = *(_BYTE *)(v35 + 140);
  }
  while ( v36 == 12 );
  if ( v36 )
  {
LABEL_57:
    v11 = 1;
    v12 = (__int64)v65;
    sub_6F4D20(v65, 1, 1);
    goto LABEL_14;
  }
  do
  {
LABEL_15:
    v13 = *(_QWORD *)(v13 + 160);
    v14 = *(_BYTE *)(v13 + 140);
  }
  while ( v14 == 12 );
LABEL_16:
  if ( !v14 )
  {
LABEL_33:
    v15 = sub_72C930(v12);
    goto LABEL_34;
  }
  v12 = (__int64)v65;
  if ( (unsigned int)sub_696840((__int64)v65) )
  {
    v15 = *(_QWORD *)&dword_4D03B80;
    v16 = *(_BYTE *)(*(_QWORD *)&dword_4D03B80 + 140LL);
    if ( v16 != 12 )
      goto LABEL_19;
LABEL_35:
    v24 = v15;
    do
    {
      v24 = *(_QWORD *)(v24 + 160);
      v25 = *(_BYTE *)(v24 + 140);
    }
    while ( v25 == 12 );
    if ( !v25 )
      goto LABEL_20;
    goto LABEL_38;
  }
  v12 = *(_QWORD *)v65;
  if ( !(unsigned int)sub_8D2630(*(_QWORD *)v65, v11) )
  {
    v11 = (__int64)&v67;
    v12 = 3358;
    sub_6E5E80(3358, &v67, *(_QWORD *)v65);
    v15 = sub_72C930(3358);
    goto LABEL_34;
  }
  if ( v66 != 2 )
  {
    if ( (unsigned int)sub_6E5430(v12, v11, v20, v21, v22, v23) )
    {
      v11 = (__int64)&v67;
      v12 = 3360;
      sub_6851C0(0xD20u, &v67);
    }
    goto LABEL_33;
  }
  v42 = v68;
  v15 = v69;
  if ( v68 == 48 )
  {
    v48 = *(_BYTE *)(v69 + 8);
    if ( v48 == 1 )
    {
      v15 = *(_QWORD *)(v69 + 32);
      v42 = 2;
    }
    else
    {
      if ( v48 != 2 )
      {
        if ( v48 )
          sub_721090(v12);
        v15 = *(_QWORD *)(v69 + 32);
        goto LABEL_34;
      }
      v15 = *(_QWORD *)(v69 + 32);
      v42 = 59;
    }
LABEL_70:
    v57 = v42;
    v49 = v70;
    if ( (unsigned int)sub_6E5430(v12, v11, v42, v21, v22, v23) )
    {
      LOBYTE(v61) = v57;
      v11 = (__int64)&v67;
      v62 = v15;
      LODWORD(v63) = v49;
      v43 = sub_67D610(0xD1Fu, &v67, 8u);
      sub_67F190((__int64)v43, (__int64)&v67, v44, v45, v46, v47, v61, v62, v63);
      v12 = (__int64)v43;
      sub_685910((__int64)v43, &v67);
    }
    goto LABEL_33;
  }
  if ( v68 != 6 )
    goto LABEL_70;
LABEL_34:
  v16 = *(_BYTE *)(v15 + 140);
  if ( v16 == 12 )
    goto LABEL_35;
LABEL_19:
  if ( !v16 )
  {
LABEL_20:
    *(_BYTE *)(qword_4D03C50 + 18LL) &= ~0x20u;
    goto LABEL_21;
  }
LABEL_38:
  if ( !v4 || (*(_DWORD *)(v4 + 40) & 0x86140) != 0 )
  {
    v26 = sub_7259C0(12);
    if ( dword_4F04C44 != -1
      || (v27 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v27 + 6) & 6) != 0)
      || *(_BYTE *)(v27 + 4) == 12 )
    {
      v55 = v26;
      v33 = sub_8DBE70(v15);
      v26 = v55;
      *(_QWORD *)(v55 + 160) = v15;
      if ( v33 )
      {
        *(_BYTE *)(v55 + 186) |= 0x3Cu;
        sub_6F40C0(v65);
        sub_7296C0(&v61);
        v11 = 0;
        v29 = sub_6F6F40(v65, 0);
        sub_729730((unsigned int)v61);
        v30 = v55;
LABEL_50:
        if ( (*(_BYTE *)(v29 - 8) & 1) != 0 )
        {
          *(_QWORD *)(*(_QWORD *)(v30 + 168) + 24LL) = v29;
        }
        else
        {
          v11 = 6;
          v56 = v30;
          sub_72D910(v29, 6, v30);
          v30 = v56;
          *(_QWORD *)(v56 + 48) = *(_QWORD *)(qword_4F04C68[0] + 776LL * (int)v50 + 216);
        }
        v15 = v30;
LABEL_21:
        if ( !v10 )
          goto LABEL_26;
        goto LABEL_22;
      }
      *(_BYTE *)(v55 + 186) = *(_BYTE *)(v55 + 186) & 0xC3 | 4;
    }
    else
    {
      v28 = *(_BYTE *)(v26 + 186);
      *(_QWORD *)(v26 + 160) = v15;
      *(_BYTE *)(v26 + 186) = v28 & 0xC3 | 4;
    }
    v54 = v26;
    sub_7296C0(&v61);
    v11 = 0;
    v29 = sub_6F6F40(v65, 0);
    sub_729730((unsigned int)v61);
    v30 = v54;
    if ( dword_4F04C44 != -1
      || (v11 = (__int64)qword_4F04C68, v31 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v31 + 6) & 6) != 0)
      || *(_BYTE *)(v31 + 4) == 12 )
    {
      v32 = sub_731EE0(v29);
      v30 = v54;
      if ( v32 )
        *(_BYTE *)(v54 + 186) |= 8u;
    }
    goto LABEL_50;
  }
  v15 = sub_72C930(v12);
  sub_6E4710(v65);
  if ( !v10 )
    goto LABEL_27;
LABEL_22:
  if ( *v10 )
  {
    if ( !dword_4F04C3C )
      sub_8699D0(v15, 6, v10);
    v11 = 6;
    sub_869D70(v15, 6);
  }
  else
  {
    v11 = (unsigned int)dword_4F04C64;
    sub_869FD0(v10, (unsigned int)dword_4F04C64);
  }
LABEL_26:
  if ( v4 )
  {
LABEL_27:
    v17 = v52;
    unk_4D03C40 = v52;
    goto LABEL_28;
  }
  sub_7BE280(55, 53, 0, 0);
  v11 = 17;
  v17 = 26;
  --*(_BYTE *)(qword_4F061C8 + 34LL);
  --*(_QWORD *)(qword_4D03C50 + 40LL);
  v40 = sub_7BE5B0(26, 17, 0, 0);
  if ( !v51 && v40 )
    sub_7B8B50(26, 17, v51, v41);
LABEL_28:
  sub_6E2B30(v17, v11);
  sub_6E1DF0(v59);
  v18 = v60;
  sub_729730(v58);
  qword_4F06BC0 = v18;
  return v15;
}
