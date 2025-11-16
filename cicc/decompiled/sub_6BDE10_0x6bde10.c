// Function: sub_6BDE10
// Address: 0x6bde10
//
void __fastcall sub_6BDE10(__int64 a1, int a2)
{
  __int64 v4; // r13
  unsigned int v5; // r15d
  unsigned __int64 v6; // rbx
  char v7; // al
  __int64 v8; // rdx
  __int64 v9; // rax
  char v10; // al
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 i; // r12
  __int64 v14; // rax
  _QWORD *v15; // rsi
  unsigned int v16; // edi
  __int64 v17; // r8
  __int64 j; // rax
  _QWORD *v19; // rdx
  int v20; // r12d
  __int64 v21; // r11
  unsigned int v22; // ecx
  unsigned __int64 v23; // r9
  int v24; // ebx
  unsigned int v25; // edi
  unsigned __int64 *v26; // rdx
  unsigned __int64 *v27; // rax
  unsigned __int64 v28; // r10
  __int64 v29; // rdi
  char v30; // al
  __int64 v31; // r8
  char v32; // al
  char v33; // di
  char v34; // al
  _BYTE *v35; // r9
  __int64 *v36; // r11
  unsigned int v37; // edx
  int v38; // r10d
  char v39; // si
  __int64 v40; // rdi
  _BOOL8 v41; // rsi
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  _QWORD *v45; // rdx
  char v46; // al
  unsigned int v47; // r12d
  __int64 v48; // rax
  __int64 v49; // rdi
  __int64 v50; // rdi
  __int64 v51; // rsi
  __int64 v52; // rdi
  char v53; // al
  int v54; // eax
  __int64 v55; // rdi
  __int64 v56; // rax
  _DWORD *v57; // rax
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // rdx
  int v61; // eax
  int v62; // eax
  __int64 v63; // rdi
  __int64 v64; // rsi
  __int64 v65; // rdi
  __int64 v66; // rax
  __int64 v67; // [rsp+0h] [rbp-110h]
  __int64 v68; // [rsp+8h] [rbp-108h]
  __int64 v69; // [rsp+8h] [rbp-108h]
  __int64 v70; // [rsp+10h] [rbp-100h]
  __int64 v71; // [rsp+10h] [rbp-100h]
  __int64 v72; // [rsp+10h] [rbp-100h]
  __int64 v73; // [rsp+10h] [rbp-100h]
  __int64 v74; // [rsp+10h] [rbp-100h]
  __int64 v75; // [rsp+10h] [rbp-100h]
  __int64 v76; // [rsp+10h] [rbp-100h]
  char v77; // [rsp+1Fh] [rbp-F1h]
  unsigned int v78; // [rsp+2Ch] [rbp-E4h] BYREF
  __int64 v79; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v80; // [rsp+38h] [rbp-D8h] BYREF
  int v81[52]; // [rsp+40h] [rbp-D0h] BYREF

  v4 = *(_QWORD *)a1;
  if ( (*(_BYTE *)(a1 + 124) & 1) != 0 || (*(_BYTE *)(a1 + 131) & 4) != 0 )
  {
    v77 = 0;
    v5 = 0;
    v6 = 0;
    if ( !v4 )
      goto LABEL_15;
    v7 = *(_BYTE *)(v4 + 80);
    if ( v7 == 7 )
      goto LABEL_53;
  }
  else
  {
    if ( !v4 )
    {
      v77 = 0;
      v6 = 0;
      v5 = 0;
      goto LABEL_15;
    }
    v7 = *(_BYTE *)(v4 + 80);
    v77 = 1;
    v5 = 1;
    if ( v7 == 7 )
      goto LABEL_53;
  }
  if ( v7 != 9 )
  {
    if ( (*(_BYTE *)(a1 + 131) & 4) == 0 )
      sub_721090(a1);
    v6 = 0;
    goto LABEL_8;
  }
LABEL_53:
  v6 = *(_QWORD *)(v4 + 88);
  v81[0] = 1;
  sub_69DD00(qword_4D03BF0, v6, v81, v6 >> 3);
LABEL_8:
  v4 = 0;
  if ( v77 )
  {
    v8 = 4;
    if ( (*(_BYTE *)(a1 + 121) & 0x40) != 0 )
    {
      v8 = 1;
      if ( *(_QWORD *)a1 )
      {
        if ( *(_BYTE *)(*(_QWORD *)a1 + 80LL) == 9 && v6 )
          v8 = 3 * (unsigned int)((*(_BYTE *)(v6 + 172) & 0x28) == 32) + 1;
        else
          v8 = 1;
      }
    }
    sub_6E2250(v81, &v79, v8, 1, a1, 0);
    v9 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    v4 = *(_QWORD *)(v9 + 624);
    *(_QWORD *)(v9 + 624) = a1;
  }
LABEL_15:
  if ( !a2 )
  {
    v10 = *(_BYTE *)(a1 + 132);
    if ( (v10 & 8) == 0 )
    {
      v11 = sub_6BB5A0(v5, 0);
      goto LABEL_18;
    }
    goto LABEL_50;
  }
  v30 = *(_BYTE *)(a1 + 129);
  *(_BYTE *)(a1 + 129) = v30 | 2;
  if ( (*(_BYTE *)(a1 + 125) & 4) != 0 )
  {
    v10 = *(_BYTE *)(a1 + 132);
    if ( (v10 & 8) != 0 )
    {
LABEL_50:
      v11 = *(_QWORD *)(a1 + 328);
      *(_BYTE *)(a1 + 132) = v10 & 0xF7;
      sub_6E1BE0(a1 + 328);
      goto LABEL_18;
    }
    v11 = sub_6BDC10(0x1Cu, 0, 1u, 0);
  }
  else
  {
    *(_BYTE *)(a1 + 129) = v30 | 6;
    v11 = sub_6BB770(a1, v5, 1, 0);
  }
LABEL_18:
  if ( v11 )
    sub_6E1C20(v11, 1, a1 + 328);
  if ( dword_4F077C4 != 2 )
  {
    v12 = *(_QWORD *)(a1 + 288);
    for ( i = *(_QWORD *)(a1 + 304); *(_BYTE *)(v12 + 140) == 12; v12 = *(_QWORD *)(v12 + 160) )
      ;
    if ( i == v12 )
    {
      v80 = *(_QWORD *)(*(_QWORD *)(v11 + 24) + 8LL);
      v54 = sub_8D3410(v80);
      v55 = v80;
      if ( v54 )
        v55 = sub_8D67C0(v80);
      while ( *(_BYTE *)(v55 + 140) == 12 )
        v55 = *(_QWORD *)(v55 + 160);
      v80 = v55;
      if ( (unsigned int)sub_8D2B80(v55) )
        sub_73C7D0(&v80);
    }
    else
    {
      sub_6851C0(0xA9Du, (_DWORD *)(a1 + 104));
      v80 = sub_72C930(2717);
    }
    sub_725570(i, 12);
    v14 = v80;
    *(_BYTE *)(i + 184) = 3;
    *(_QWORD *)(i + 160) = v14;
    *(_QWORD *)(a1 + 312) = v14;
    goto LABEL_26;
  }
  v31 = *(_QWORD *)(a1 + 280);
  if ( (*(_BYTE *)(a1 + 10) & 8) != 0 && (*(_BYTE *)(a1 + 125) & 2) == 0 )
  {
    if ( (*(_BYTE *)(v31 + 140) & 0xFB) != 8
      || (v70 = *(_QWORD *)(a1 + 280), v32 = sub_8D4C10(v70, 0), v31 = v70, (v32 & 1) == 0) )
    {
      v31 = sub_73C570(v31, 1, -1);
    }
  }
  if ( !v11 )
    goto LABEL_63;
  if ( *(_BYTE *)(v11 + 8) != 1 )
    goto LABEL_62;
  v33 = *(_BYTE *)(a1 + 125);
  if ( (v33 & 4) != 0 )
  {
    v75 = v31;
    v59 = sub_8D2290(*(_QWORD *)(a1 + 288));
    v60 = *(_QWORD *)(v11 + 24);
    v31 = v75;
    if ( a2 )
      goto LABEL_62;
    if ( !v60 )
      goto LABEL_62;
    if ( *(_QWORD *)v60 )
      goto LABEL_62;
    if ( *(_BYTE *)(v60 + 8) )
      goto LABEL_62;
    v67 = *(_QWORD *)(v11 + 24);
    v69 = v75;
    v76 = v59;
    v61 = sub_8D3F60(v59);
    v31 = v69;
    if ( !v61 )
      goto LABEL_62;
    v62 = sub_8AE090(*(_QWORD *)(*(_QWORD *)(v67 + 24) + 8LL), *(_QWORD *)(*(_QWORD *)(v76 + 168) + 32LL), &v80);
    v31 = v69;
    if ( !v62 )
      goto LABEL_62;
LABEL_133:
    v56 = *(_QWORD *)(v11 + 24);
    goto LABEL_115;
  }
  v53 = *(_BYTE *)(a1 + 127);
  if ( (v53 & 8) == 0 )
  {
    if ( (*(_BYTE *)(a1 + 131) & 0x10) == 0 )
    {
      v35 = *(_BYTE **)(a1 + 304);
      v36 = (__int64 *)(a1 + 288);
      v71 = a1 + 48;
      v38 = *(_BYTE *)(a1 + 124) & 1;
LABEL_93:
      v37 = *(_BYTE *)(a1 + 176) & 1;
      goto LABEL_65;
    }
    goto LABEL_122;
  }
  if ( !(_DWORD)qword_4F077B4 )
  {
    if ( dword_4F077BC )
    {
      if ( qword_4F077A8 <= 0xC34Fu )
        goto LABEL_113;
    }
    else if ( a2 )
    {
      goto LABEL_120;
    }
    goto LABEL_133;
  }
  if ( qword_4F077A0 <= 0x784Fu )
  {
LABEL_113:
    v35 = *(_BYTE **)(a1 + 304);
    v37 = 1;
    v36 = (__int64 *)(a1 + 288);
    v71 = a1 + 48;
    v38 = *(_BYTE *)(a1 + 124) & 1;
LABEL_65:
    v39 = v33;
    v40 = (v33 & 2) != 0;
    v41 = (v39 & 4) != 0;
    v68 = v31;
    if ( !(unsigned int)sub_696CB0(v40, v41, v37, a2, v31, v35, v38, 0, v11, v71, v36, &v80, &v78) )
    {
      if ( v78
        || (v45 = (_QWORD *)dword_4F077BC, dword_4F077BC)
        && !(_DWORD)qword_4F077B4
        && qword_4F077A8
        && (v45 = qword_4F04C68, (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) != 0)
        && unk_4F04C50 )
      {
        *(_QWORD *)(a1 + 288) = v68;
        *(_QWORD *)(a1 + 312) = sub_72CBA0();
      }
      else
      {
        v46 = *(_BYTE *)(a1 + 125);
        v47 = 2958;
        if ( (v46 & 4) == 0 )
          v47 = (v46 & 2) == 0 ? 1587 : 2544;
        if ( (unsigned int)sub_6E5430(v40, v41, v45, v78, v68, v44) )
        {
          v40 = v47;
          sub_6851C0(v47, (_DWORD *)(a1 + 104));
        }
        v48 = sub_72C930(v40);
        *(_BYTE *)(a1 + 124) &= ~0x80u;
        *(_QWORD *)(a1 + 288) = v48;
        *(_QWORD *)(a1 + 312) = v48;
        *(_QWORD *)(a1 + 272) = v48;
      }
      goto LABEL_26;
    }
    v50 = *(_QWORD *)(a1 + 312);
    v51 = v80;
    if ( v50 && *(_BYTE *)(v50 + 140) != 21 && v50 != v80 )
    {
      if ( !(unsigned int)sub_8D97D0(v50, v80, 0, v42, v43) )
        sub_6E5ED0(1594, v71, v80, *(_QWORD *)(a1 + 312));
      v51 = v80;
    }
    *(_QWORD *)(a1 + 312) = v51;
    if ( (*(_BYTE *)(a1 + 10) & 8) != 0 && (*(_BYTE *)(a1 + 125) & 2) != 0 )
    {
      v52 = *(_QWORD *)(a1 + 288);
      if ( (*(_BYTE *)(v52 + 140) & 0xFB) != 8 )
      {
LABEL_87:
        *(_QWORD *)(a1 + 288) = sub_73C570(v52, 1, -1);
        goto LABEL_88;
      }
      if ( (sub_8D4C10(v52, dword_4F077C4 != 2) & 1) == 0 )
      {
        v52 = *(_QWORD *)(a1 + 288);
        goto LABEL_87;
      }
    }
LABEL_88:
    sub_65C3D0(a1);
    goto LABEL_26;
  }
  v56 = *(_QWORD *)(v11 + 24);
  if ( a2 && !dword_4F077BC )
  {
LABEL_120:
    v74 = v31;
    v58 = sub_6E1A20(v11);
    sub_6E5C80(7, 2957, v58);
    v31 = v74;
    goto LABEL_62;
  }
LABEL_115:
  if ( !v56 || *(_QWORD *)v56 )
  {
    v73 = v31;
    v57 = (_DWORD *)sub_6E1A20(v11);
    sub_684AA0(7u, 0xA67u, v57);
    v31 = v73;
    if ( (*(_BYTE *)(a1 + 131) & 0x10) == 0 )
      goto LABEL_63;
    goto LABEL_94;
  }
  if ( *(_BYTE *)(v56 + 8) != 1 )
    v11 = v56;
LABEL_62:
  if ( (*(_BYTE *)(a1 + 131) & 0x10) == 0 )
  {
LABEL_63:
    v33 = *(_BYTE *)(a1 + 125);
    v34 = *(_BYTE *)(a1 + 127) & 8;
    goto LABEL_64;
  }
LABEL_94:
  if ( *(_BYTE *)(v11 + 8) )
  {
    v33 = *(_BYTE *)(a1 + 125);
    v53 = *(_BYTE *)(a1 + 127);
LABEL_122:
    v34 = v53 & 8;
LABEL_64:
    v35 = *(_BYTE **)(a1 + 304);
    v36 = (__int64 *)(a1 + 288);
    v71 = a1 + 48;
    v37 = 1;
    v38 = *(_BYTE *)(a1 + 124) & 1;
    if ( v34 )
      goto LABEL_65;
    goto LABEL_93;
  }
  if ( *(_BYTE *)(*(_QWORD *)(a1 + 288) + 140LL) == 6 )
    goto LABEL_63;
  v72 = v31;
  if ( !(unsigned int)sub_8D3410(*(_QWORD *)(*(_QWORD *)(v11 + 24) + 8LL)) )
  {
    v33 = *(_BYTE *)(a1 + 125);
    v31 = v72;
    v34 = *(_BYTE *)(a1 + 127) & 8;
    goto LABEL_64;
  }
  v63 = *(_QWORD *)(a1 + 288);
  v64 = 0;
  if ( (*(_BYTE *)(v63 + 140) & 0xFB) == 8 )
    v64 = (unsigned int)sub_8D4C10(v63, dword_4F077C4 != 2);
  v65 = *(_QWORD *)(*(_QWORD *)(v11 + 24) + 8LL);
  *(_QWORD *)(a1 + 312) = v65;
  v66 = sub_73C570(v65, v64, -1);
  *(_QWORD *)(a1 + 288) = v66;
  *(_QWORD *)(a1 + 272) = v66;
  sub_65C3D0(a1);
LABEL_26:
  if ( v6 )
  {
    *(_QWORD *)(v6 + 120) = *(_QWORD *)(a1 + 288);
    v15 = qword_4D03BF0;
    v16 = *((_DWORD *)qword_4D03BF0 + 2);
    v17 = *qword_4D03BF0;
    for ( j = v16 & (unsigned int)(v6 >> 3); ; j = v16 & ((_DWORD)j + 1) )
    {
      v19 = (_QWORD *)(v17 + 16LL * (unsigned int)j);
      if ( *v19 == v6 )
        break;
    }
    *v19 = 0;
    if ( *(_QWORD *)(v17 + 16LL * (((_DWORD)j + 1) & v16)) )
    {
      v20 = *((_DWORD *)v15 + 2);
      v21 = *v15;
      v22 = v20 & (j + 1);
      v23 = *(_QWORD *)(*v15 + 16LL * v22);
      while ( 1 )
      {
        v25 = v20 & (v23 >> 3);
        v26 = (unsigned __int64 *)(v21 + 16LL * (v20 & (v22 + 1)));
        if ( v25 <= (unsigned int)j && (v22 < v25 || v22 > (unsigned int)j) || v22 > (unsigned int)j && v22 < v25 )
        {
          v27 = (unsigned __int64 *)(v21 + 16 * j);
          v28 = *v27;
          v29 = v21 + 16LL * v22;
          if ( *v27 )
          {
            *v27 = v23;
            v24 = *((_DWORD *)v27 + 2);
            if ( v23 )
              *((_DWORD *)v27 + 2) = *(_DWORD *)(v29 + 8);
            *(_QWORD *)v29 = v28;
            *(_DWORD *)(v29 + 8) = v24;
            v23 = *v26;
            if ( !*v26 )
              break;
          }
          else
          {
            *v27 = v23;
            if ( v23 )
              *((_DWORD *)v27 + 2) = *(_DWORD *)(v29 + 8);
            *(_QWORD *)v29 = 0;
            v23 = *v26;
            if ( !*v26 )
              break;
          }
          j = v22;
        }
        else
        {
          v23 = *v26;
          if ( !*v26 )
            break;
        }
        v22 = v20 & (v22 + 1);
      }
    }
    --*((_DWORD *)v15 + 3);
  }
  if ( v77 )
  {
    v49 = v79;
    *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 624) = v4;
    sub_6E2C70(v49, 1, a1, 0);
  }
}
