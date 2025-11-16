// Function: sub_6D0CE0
// Address: 0x6d0ce0
//
__int64 __fastcall sub_6D0CE0(__int64 *a1, unsigned int a2, __int64 *a3, __int64 *a4)
{
  __int64 v4; // r14
  __int64 v5; // rbx
  __int64 i; // r15
  __int64 v7; // r13
  __int64 v8; // r8
  __int64 v9; // r9
  _QWORD *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rsi
  __int64 v13; // r14
  __int64 v14; // r12
  __int64 *v15; // rdi
  char v16; // dl
  __int64 v17; // r11
  __int64 v18; // rsi
  char v19; // dl
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v23; // rax
  __int64 v24; // r8
  char v25; // al
  __int64 v26; // rax
  char m; // dl
  unsigned int *v28; // r14
  __int64 v29; // rcx
  __int64 v30; // rcx
  int v31; // eax
  __int64 v32; // rdi
  __int64 v33; // rax
  __int64 v34; // rdi
  __int64 v35; // rax
  __int64 v36; // r12
  int v37; // eax
  __int64 v38; // rax
  unsigned int v39; // r14d
  unsigned int v40; // edi
  int v41; // eax
  _DWORD *v42; // rcx
  __int64 v43; // rax
  int v44; // eax
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // rax
  char k; // dl
  __int64 v49; // rax
  int v50; // eax
  __int64 *v51; // [rsp-10h] [rbp-2D0h]
  bool j; // [rsp+6h] [rbp-2BAh]
  bool v53; // [rsp+7h] [rbp-2B9h]
  __int64 v54; // [rsp+8h] [rbp-2B8h]
  unsigned int v55; // [rsp+10h] [rbp-2B0h]
  __int64 v56; // [rsp+18h] [rbp-2A8h]
  __int64 v57; // [rsp+18h] [rbp-2A8h]
  unsigned int v58; // [rsp+20h] [rbp-2A0h]
  char v61; // [rsp+28h] [rbp-298h]
  _DWORD *v62; // [rsp+28h] [rbp-298h]
  __int64 v63; // [rsp+30h] [rbp-290h]
  __int64 v64; // [rsp+30h] [rbp-290h]
  unsigned int v66; // [rsp+44h] [rbp-27Ch] BYREF
  __int64 v67; // [rsp+48h] [rbp-278h] BYREF
  __int64 v68; // [rsp+50h] [rbp-270h] BYREF
  __int64 v69; // [rsp+58h] [rbp-268h] BYREF
  __int64 v70; // [rsp+60h] [rbp-260h] BYREF
  __int64 v71; // [rsp+68h] [rbp-258h]
  char v72; // [rsp+89h] [rbp-237h]
  char v73; // [rsp+8Bh] [rbp-235h]
  _BYTE v74[160]; // [rsp+90h] [rbp-230h] BYREF
  __int64 v75[2]; // [rsp+130h] [rbp-190h] BYREF
  char v76; // [rsp+140h] [rbp-180h]
  __int64 v77; // [rsp+174h] [rbp-14Ch] BYREF
  __int64 v78; // [rsp+17Ch] [rbp-144h]

  v4 = (__int64)a1;
  v5 = *(_QWORD *)(unk_4F04C50 + 32LL);
  for ( i = *(_QWORD *)(v5 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  *a3 = 0;
  sub_6E1DD0(&v67);
  sub_6E1E00(4, v74, 0, 0);
  sub_6E6990(&v70);
  v73 |= 0x80u;
  if ( word_4F06418[0] != 73 )
    goto LABEL_4;
  if ( dword_4F077BC )
  {
    if ( !dword_4D04428 && (unsigned int)sub_6E53E0(5, 2068, &dword_4F063F8) )
      sub_684B30(0x814u, &dword_4F063F8);
    goto LABEL_50;
  }
  if ( dword_4D04428 )
  {
LABEL_50:
    v23 = 776LL * dword_4F04C58;
    *(_BYTE *)(qword_4F04C68[0] + v23 + 6) &= ~1u;
    v12 = 0;
    *(_QWORD *)(*(_QWORD *)(qword_4F04C68[0] + v23 + 184) + 72LL) = 0;
    v7 = sub_6BA760(0, 0);
    sub_6E6260(v75);
    v77 = *(_QWORD *)sub_6E1A20(v7);
    v78 = *(_QWORD *)sub_6E1A60(v7);
    if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 14) & 2) != 0 )
    {
      v38 = sub_6F6C90(v7, 0, qword_4F04C68, &v70);
      v21 = v7;
      v14 = sub_6E2700(v38);
      sub_6E1990(v7);
      goto LABEL_46;
    }
    v25 = *(_BYTE *)(v5 + 207);
    if ( v25 < 0 )
    {
      v21 = v7;
      v14 = 0;
      *a4 = v7;
      v12 = (*(_BYTE *)(qword_4D03C50 + 18LL) & 4) != 0;
      sub_689210(v7, v12);
      goto LABEL_46;
    }
    if ( (v25 & 0x10) == 0 || (*(_BYTE *)(v5 + 195) & 8) != 0 )
    {
      v12 = (__int64)a1;
      sub_839D30(v7, (_DWORD)a1, 0, 1, 0, 130, 0, 0, 0, 0, (__int64)&v70, 0);
      v14 = v71;
      if ( v71 )
      {
        v21 = v71;
        LOBYTE(i) = 0;
        *a3 = v71;
        v14 = 0;
        sub_6E2920(v21);
      }
      else
      {
        if ( (v72 & 2) != 0 )
          v34 = sub_7305B0(v7, a1);
        else
          v34 = sub_730690(v70);
        *(_QWORD *)(v34 + 28) = v77;
        *(_QWORD *)(v34 + 36) = v77;
        *(_QWORD *)(v34 + 44) = v78;
        i = sub_6E2700(v34);
        if ( (unsigned int)sub_8D2600(v4) )
          sub_7304E0(i);
        v21 = 3;
        v35 = sub_6EAFA0(3);
        *a3 = v35;
        *(_QWORD *)(v35 + 56) = i;
        LOBYTE(i) = 0;
        *(_BYTE *)(*a3 + 50) |= 0x40u;
      }
      goto LABEL_42;
    }
    v40 = (*(_BYTE *)(v5 + 206) & 2) == 0 ? 2678 : 2366;
    if ( (unsigned int)sub_6E5430(v40, 0, qword_4F04C68, &v70, v24) )
      sub_6851C0(v40, &v77);
    v15 = (__int64 *)v7;
    sub_6E6470(v7);
    v16 = *(_BYTE *)(v5 + 207);
    goto LABEL_21;
  }
LABEL_4:
  if ( (*(_BYTE *)(*(_QWORD *)(i + 168) + 16LL) & 0x20) != 0 )
  {
    LODWORD(i) = 1;
    *(_BYTE *)(qword_4D03C50 + 18LL) |= 4u;
    goto LABEL_6;
  }
  if ( (*(_BYTE *)(v5 + 207) & 0x10) == 0 || (*(_BYTE *)(v5 + 195) & 8) != 0 )
  {
    LODWORD(i) = 0;
LABEL_6:
    v7 = 0;
    sub_69ED20((__int64)v75, 0, 0, 0);
    if ( *(char *)(v5 + 207) >= 0 )
      goto LABEL_7;
LABEL_56:
    v14 = 0;
    v21 = sub_6E3060(v75);
    *a4 = v21;
    v12 = (*(_BYTE *)(qword_4D03C50 + 18LL) & 4) != 0;
    sub_689210(v21, v12);
    goto LABEL_46;
  }
  v15 = v75;
  *(_BYTE *)(qword_4D03C50 + 18LL) |= 4u;
  sub_69ED20((__int64)v75, 0, 0, 0);
  v16 = *(_BYTE *)(v5 + 207);
  if ( v16 < 0 )
    goto LABEL_56;
  v7 = 0;
LABEL_21:
  if ( (v16 & 0x10) != 0 )
  {
    if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 14) & 2) != 0 )
    {
      sub_68CD10(v75, 1);
      v12 = 0;
      v21 = sub_6F6F40(v75, 0);
      v14 = sub_6E2700(v21);
      goto LABEL_46;
    }
    v17 = *(_QWORD *)(v5 + 152);
    v61 = v16 & 0x20;
    for ( j = (*(_BYTE *)(v5 + 206) & 2) != 0; *(_BYTE *)(v17 + 140) == 12; v17 = *(_QWORD *)(v17 + 160) )
      ;
    if ( v61 )
      v63 = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C58 + 760);
    else
      v63 = *(_QWORD *)(v17 + 160);
    v54 = v17;
    v55 = (unsigned int)sub_8D2FB0(v63) == 0 ? 4 : 7;
    v56 = sub_8D4940(v63);
    v53 = 0;
    v58 = sub_8D3EA0(v56);
    v18 = v55;
    if ( v58 )
    {
      v53 = *(_DWORD *)(*(_QWORD *)(v56 + 168) + 24LL) == 2;
      v58 = v53;
      if ( *(_DWORD *)(*(_QWORD *)(v56 + 168) + 24LL) == 2 )
        v18 = 7;
    }
    sub_6F69D0(v75, v18);
    if ( !v61 )
      v63 = *(_QWORD *)(v54 + 160);
    v19 = *(_BYTE *)(v56 + 140);
    if ( v19 == 12 )
    {
      v20 = v56;
      do
      {
        v20 = *(_QWORD *)(v20 + 160);
        v19 = *(_BYTE *)(v20 + 140);
      }
      while ( v19 == 12 );
    }
    if ( v19 )
    {
      if ( (unsigned int)sub_8D2600(v75[0]) )
      {
        sub_695540(v5, !j, &v77);
        v4 = *(_QWORD *)(v54 + 160);
      }
      else
      {
        v31 = sub_696CB0(v58, 0, 0, 0, v63, (_BYTE *)v56, !j, (__int64)v75, 0, (__int64)&v77, &v68, &v69, &v66);
        v9 = v56;
        if ( v31 )
        {
          sub_65C2A0(v56, v69);
          sub_68A860(v68, (__int64)&v77, *(_QWORD *)(unk_4F04C50 + 32LL), v45, v46);
          v4 = *(_QWORD *)(v54 + 160);
        }
        else if ( !v66 )
        {
          v32 = !v53 ? 1587 : 2544;
          sub_6851C0(v32, &v77);
          v4 = sub_72C930(v32);
          *(_QWORD *)(v54 + 160) = v4;
        }
      }
    }
    else
    {
      v4 = sub_72C930(v75);
      *(_QWORD *)(v54 + 160) = v4;
    }
    if ( v7 || (v33 = *(_QWORD *)(i + 168), LODWORD(i) = 1, (*(_BYTE *)(v33 + 16) & 0x20) == 0) )
    {
      sub_6891A0();
      LODWORD(i) = 0;
      if ( dword_4F04C44 != -1 )
        goto LABEL_39;
      goto LABEL_8;
    }
  }
  else
  {
    LODWORD(i) = 0;
    v4 = sub_72C930(v15);
  }
LABEL_7:
  if ( dword_4F04C44 != -1 )
    goto LABEL_39;
LABEL_8:
  v10 = qword_4F04C68;
  v11 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  if ( (*(_BYTE *)(v11 + 6) & 6) != 0 || *(_BYTE *)(v11 + 4) == 12 )
  {
LABEL_39:
    if ( (unsigned int)sub_8DBE70(v75[0])
      || (unsigned int)sub_8DBE70(v4)
      || dword_4F077BC && !(_DWORD)qword_4F077B4 && (*(_BYTE *)(v5 + 195) & 8) != 0 )
    {
      sub_6F40C0(v75);
      v21 = (__int64)v75;
      v12 = 0;
      v14 = sub_6F6F40(v75, 0);
      goto LABEL_42;
    }
  }
  if ( (_DWORD)i
    || dword_4F077C4 == 2
    && (unsigned int)sub_8D3A70(v4)
    && (v75[0] == v4 || (unsigned int)sub_8D97D0(v4, v75[0], 32, v29, v8)) )
  {
    v30 = qword_4F04C68[0] + 776LL * dword_4F04C58;
    if ( (*(_BYTE *)(v30 + 6) & 1) == 0 )
    {
LABEL_79:
      v12 = v4;
      v14 = 0;
      sub_8470D0((unsigned int)v75, v4, 0, 130, a2, 0, (__int64)a3);
      v21 = *a3;
      sub_6E2920(*a3);
      sub_6891A0();
      goto LABEL_42;
    }
    v57 = qword_4F04C68[0] + 776LL * dword_4F04C58;
    v64 = *(_QWORD *)(v30 + 184);
    v41 = sub_6E9790(v75, &v69, v10, v30, v8, v9);
    v42 = (_DWORD *)v57;
    if ( v41 || dword_4F077C4 == 2 && unk_4F07778 > 202001 && (v44 = sub_6EA170(v75, &v69), v42 = (_DWORD *)v57, v44) )
    {
      v43 = *(_QWORD *)(v64 + 72);
      if ( v43 )
      {
        if ( v69 == v43 )
          goto LABEL_79;
      }
      else
      {
        v62 = v42;
        v50 = sub_695430(v69, 1, 0);
        v42 = v62;
        if ( v50
          && (*(_BYTE *)(v69 + 177) != 2 || (*(_BYTE *)(*(_QWORD *)(v69 + 184) + 51LL) & 2) == 0)
          && *(_DWORD *)(*(_QWORD *)v69 + 40LL) == *v62 )
        {
          *(_QWORD *)(v64 + 72) = v69;
          goto LABEL_79;
        }
      }
    }
    *((_BYTE *)v42 + 6) &= ~1u;
    *(_QWORD *)(v64 + 72) = 0;
    goto LABEL_79;
  }
  if ( (unsigned int)sub_8D2600(v4) )
  {
    sub_6F6BD0(v75, 0);
    v12 = 0;
    v13 = sub_6F7180(v75, 0);
    if ( dword_4F077C0 )
    {
      if ( !(unsigned int)sub_8D2600(v75[0]) )
      {
        v12 = a2;
        if ( (unsigned int)sub_6E53E0(5, a2, &v77) )
        {
          v12 = (__int64)&v77;
          sub_684B30(a2, &v77);
          v14 = sub_6E2700(v13);
LABEL_103:
          v21 = v14;
          sub_7304E0(v14);
          goto LABEL_42;
        }
      }
    }
    else if ( !(unsigned int)sub_8D2600(v75[0]) && !(unsigned int)sub_8D3D40(v75[0]) && v76 )
    {
      v47 = v75[0];
      for ( k = *(_BYTE *)(v75[0] + 140); k == 12; k = *(_BYTE *)(v47 + 140) )
        v47 = *(_QWORD *)(v47 + 160);
      if ( k )
      {
        v12 = (__int64)v75;
        sub_6E68E0(a2, v75);
        v49 = sub_7305B0(a2, v75);
        v14 = sub_6E2700(v49);
        goto LABEL_103;
      }
    }
    v14 = sub_6E2700(v13);
    goto LABEL_103;
  }
  sub_843C40((unsigned int)v75, v4, 0, 0, 1, 130, a2);
  v36 = sub_6F6F40(v75, 0);
  v37 = sub_8D32E0(v4);
  v12 = (__int64)v51;
  LOBYTE(i) = v37;
  if ( v37 )
  {
    v21 = v36;
    LOBYTE(i) = 0;
    v14 = sub_6E2700(v36);
  }
  else
  {
    v12 = (__int64)&v69;
    if ( (unsigned int)sub_6EFEE0(v36, &v69) )
    {
      v39 = 1056 - (((_DWORD)v69 == 0) - 1);
      v12 = v39;
      if ( (unsigned int)sub_6E53E0(5, v39, &v77) )
      {
        v12 = (__int64)&v77;
        sub_684B30(v39, &v77);
      }
    }
    v21 = v36;
    v14 = sub_6E2700(v36);
  }
LABEL_42:
  if ( (*(_BYTE *)(v5 + 193) & 2) != 0 && !dword_4D0488C )
  {
    if ( !word_4D04898
      || !(_DWORD)qword_4F077B4
      || qword_4F077A0 <= 0x765Bu
      || (v21 = dword_4F063F8, !(unsigned int)sub_729F80(dword_4F063F8)) )
    {
      if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x20) != 0 )
      {
        if ( v7 && (i & 1) == 0 )
        {
          v21 = v7;
          v28 = (unsigned int *)sub_6E1A20(v7);
          if ( (v72 & 2) != 0 )
            goto LABEL_45;
          goto LABEL_105;
        }
        if ( v76 )
        {
          v26 = v75[0];
          for ( m = *(_BYTE *)(v75[0] + 140); m == 12; m = *(_BYTE *)(v26 + 140) )
            v26 = *(_QWORD *)(v26 + 160);
          if ( m )
          {
            v28 = (unsigned int *)&v77;
LABEL_105:
            if ( (*(_BYTE *)(v5 + 195) & 3) == 1 )
            {
              if ( (*(_BYTE *)(v5 + 195) & 8) == 0 )
                *(_BYTE *)(v5 + 193) &= ~2u;
            }
            else
            {
              v12 = 2416;
              v21 = (unsigned int)sub_729F80(*v28) == 0 ? 7 : 5;
              sub_6E5C80(v21, 2416, v28);
            }
          }
        }
      }
    }
  }
  if ( v7 )
  {
LABEL_45:
    v21 = v7;
    sub_6E1990(v7);
  }
LABEL_46:
  sub_6E2B30(v21, v12);
  sub_6E1DF0(v67);
  *(_QWORD *)&dword_4F061D8 = v78;
  return v14;
}
