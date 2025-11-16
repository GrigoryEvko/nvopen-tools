// Function: sub_6BA760
// Address: 0x6ba760
//
__int64 __fastcall sub_6BA760(__int64 a1, __int64 a2)
{
  unsigned int v2; // r15d
  _QWORD *v3; // r14
  __int64 v4; // rbx
  __int64 *v5; // r12
  __int64 v6; // r8
  __int64 v7; // r13
  __int64 v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rdx
  unsigned int v13; // r12d
  _QWORD *v14; // r14
  __int64 v15; // rbx
  __int64 v16; // rax
  bool v17; // zf
  __int64 *v18; // r13
  __int64 v19; // rax
  __int64 *v20; // r12
  __int64 v21; // r13
  _QWORD *v22; // rbx
  __int64 v23; // rax
  _QWORD *v24; // rdi
  _DWORD *v25; // rax
  int v26; // r9d
  unsigned __int16 v27; // ax
  int v28; // eax
  int v29; // eax
  int v30; // eax
  __int64 v31; // rax
  __int64 v32; // r15
  __int16 v33; // ax
  int v34; // ebx
  __int64 v35; // r12
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // rdx
  __int64 v39; // rcx
  unsigned __int16 v40; // ax
  __int64 v41; // rdi
  __int64 v42; // rsi
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 v48; // rdx
  __int64 v49; // rcx
  int v50; // r13d
  int v51; // eax
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // rax
  __int64 v57; // rdx
  __int64 v58; // rcx
  __int64 v59; // r8
  __int64 v60; // r9
  __int64 v61; // rdx
  __int64 v62; // rcx
  __int64 v63; // rdx
  __int64 v64; // rcx
  __int64 v65; // rdx
  __int64 v66; // rcx
  unsigned int v67; // [rsp+8h] [rbp-C8h]
  int v68; // [rsp+8h] [rbp-C8h]
  __int64 v69; // [rsp+8h] [rbp-C8h]
  unsigned __int16 v70; // [rsp+12h] [rbp-BEh]
  __int64 v71; // [rsp+18h] [rbp-B8h]
  int v72; // [rsp+20h] [rbp-B0h]
  unsigned int v73; // [rsp+24h] [rbp-ACh]
  int v74; // [rsp+28h] [rbp-A8h]
  _QWORD *v75; // [rsp+28h] [rbp-A8h]
  int v76; // [rsp+30h] [rbp-A0h]
  int v77; // [rsp+34h] [rbp-9Ch]
  unsigned int v78; // [rsp+38h] [rbp-98h]
  int v79; // [rsp+3Ch] [rbp-94h]
  _QWORD *v80; // [rsp+40h] [rbp-90h]
  __int64 v81; // [rsp+50h] [rbp-80h]
  __int64 v82; // [rsp+58h] [rbp-78h]
  __int64 v83; // [rsp+68h] [rbp-68h] BYREF
  unsigned __int64 v84; // [rsp+70h] [rbp-60h] BYREF
  unsigned __int64 v85; // [rsp+78h] [rbp-58h] BYREF
  _QWORD v86[10]; // [rsp+80h] [rbp-50h] BYREF

  v2 = a1;
  v80 = (_QWORD *)a2;
  if ( a2 )
  {
    v3 = &qword_4F061C8;
    v4 = *(_QWORD *)a2;
    *(_QWORD *)a2 = 0;
    ++*(_BYTE *)(qword_4F061C8 + 82LL);
    ++*(_QWORD *)(qword_4D03C50 + 40LL);
    if ( v4 )
    {
      v5 = *(__int64 **)(v4 + 16);
      v82 = *(_QWORD *)(v4 + 8);
      v6 = *v5;
      if ( !*(_QWORD *)v4 )
      {
        *v5 = 0;
        v9 = v6;
        sub_6E1990(v6);
        v77 = 0;
        v76 = 0;
        v79 = 0;
        v78 = 0;
        goto LABEL_35;
      }
      a2 = v4;
      v81 = *v5;
      v7 = sub_6BA760(a1, v4);
      if ( *(_BYTE *)(v7 + 8) != 3 )
      {
        *v5 = 0;
        sub_6E1990(v81);
        *v5 = v7;
        v77 = 0;
        v76 = 0;
        v79 = 0;
        v78 = 0;
        v74 = 1;
        goto LABEL_8;
      }
      *v80 = v4;
      goto LABEL_6;
    }
  }
  else
  {
    v3 = &qword_4F061C8;
    ++*(_BYTE *)(qword_4F061C8 + 82LL);
    ++*(_QWORD *)(qword_4D03C50 + 40LL);
  }
  v9 = 1;
  v82 = sub_6E2F40(1);
  v12 = v2 & 1;
  *(_BYTE *)(v82 + 9) = v12 | *(_BYTE *)(v82 + 9) & 0xFE;
  *(_QWORD *)(v82 + 32) = *(_QWORD *)&dword_4F063F8;
  sub_7B8B50(1, a2, v12, v82);
  if ( word_4F06418[0] == 74 )
  {
    v79 = 0;
    goto LABEL_36;
  }
  v77 = 0;
  v4 = 0;
  v7 = 0;
  v76 = 0;
  v79 = 0;
  v78 = 0;
  v74 = 0;
  do
  {
LABEL_16:
    if ( !unk_4D04798 )
    {
      v76 = 1;
      goto LABEL_18;
    }
    v71 = v4;
    v26 = 0;
    v72 = 0;
    ++*(_BYTE *)(*v3 + 64LL);
    v73 = 0;
    v27 = word_4F06418[0];
    while ( 2 )
    {
      while ( v27 == 29 )
      {
        if ( v26 )
        {
          v34 = v73;
          v35 = v7;
LABEL_111:
          v26 = 1;
          a2 = dword_4D04790;
          if ( dword_4D04790 )
          {
            v9 = 2902;
            v7 = v35;
            a2 = (__int64)&dword_4F063F8;
            sub_6851C0(0xB56u, &dword_4F063F8);
            v73 = v34;
            v26 = 1;
            goto LABEL_113;
          }
        }
        else
        {
          a2 = 0;
          v9 = 0;
          v33 = sub_7BE840(0, 0);
          v26 = 0;
          if ( v33 != 1 )
          {
            v27 = word_4F06418[0];
            break;
          }
          v34 = v73;
          v35 = v7;
        }
        v73 = v34;
        v7 = v35;
LABEL_113:
        v68 = v26;
        sub_7B8B50(v9, a2, v10, v11);
        v52 = *v3;
        a2 = 40;
        v9 = 1;
        ++*(_BYTE *)(v52 + 33);
        ++*(_BYTE *)(v52 + 37);
        sub_7BE5B0(1, 40, 0, 0);
        v53 = *v3;
        v26 = v68;
        --*(_BYTE *)(v53 + 37);
        --*(_BYTE *)(v53 + 33);
        v27 = word_4F06418[0];
        if ( word_4F06418[0] == 1 )
        {
          v9 = 2;
          v35 = sub_6E2F40(2);
          *(_QWORD *)(v35 + 48) = *(_QWORD *)&dword_4F063F8;
          *(_QWORD *)(v35 + 24) = qword_4D04A00;
          sub_7B8B50(2, 40, v54, v55);
          if ( dword_4D04794 && word_4F06418[0] == 55 )
          {
            a2 = (__int64)&dword_4F063F8;
            v9 = 955;
            sub_6851C0(0x3BBu, &dword_4F063F8);
            sub_7B8B50(955, &dword_4F063F8, v65, v66);
            v72 = 1;
            v34 = v68;
          }
          else if ( dword_4F077C0 && qword_4F077A8 <= 0x9C3Fu )
          {
            v34 = v68;
            v10 = word_4F06418[0] & 0xFFFB;
            if ( (word_4F06418[0] & 0xFFFB) != 0x19 && word_4F06418[0] != 56 )
            {
LABEL_85:
              v72 = 1;
              goto LABEL_86;
            }
            v73 = 1;
          }
          else
          {
            v73 = 1;
            v34 = v68;
          }
LABEL_86:
          if ( !v7 )
            goto LABEL_107;
LABEL_87:
          *(_QWORD *)v7 = v35;
          goto LABEL_88;
        }
      }
      if ( v27 != 25 )
      {
LABEL_61:
        v11 = dword_4D04794;
        if ( dword_4D04794 && (v26 & 1) == 0 )
        {
          if ( word_4F06418[0] != 1 || (a2 = 0, v9 = 0, (unsigned __int16)sub_7BE840(0, 0) != 55) )
          {
            v10 = v73;
            v4 = v71;
            if ( v73 )
            {
              v73 = 0;
              goto LABEL_68;
            }
            v76 = 1;
            --*(_BYTE *)(*v3 + 64LL);
            goto LABEL_18;
          }
          v9 = 2;
          v34 = 0;
          v35 = sub_6E2F40(2);
          *(_QWORD *)(v35 + 48) = *(_QWORD *)&dword_4F063F8;
          *(_QWORD *)(v35 + 24) = qword_4D04A00;
          sub_7B8B50(2, 0, v36, v37);
          sub_7B8B50(2, 0, v38, v39);
          goto LABEL_85;
        }
        v17 = v73 == 0;
        v4 = v71;
        v73 = v26;
        if ( v17 )
          goto LABEL_71;
        goto LABEL_68;
      }
      v35 = v7;
      v34 = v26;
LABEL_93:
      v41 = dword_4D0448C;
      if ( dword_4D0448C )
      {
        sub_7ADF70(v86, 0);
        a2 = 0;
        if ( (unsigned int)sub_7C6040(v86, 0)
          || (sub_7AE360(v86), sub_7B8B50(v86, 0, v61, v62), (word_4F06418[0] & 0xFFFB) != 0x19)
          && word_4F06418[0] != 56 )
        {
          v9 = (__int64)v86;
          v7 = v35;
          sub_7BC000(v86);
          v26 = v34;
          goto LABEL_61;
        }
        v41 = (__int64)v86;
        sub_7BC000(v86);
      }
      v67 = dword_4F063F8;
      v70 = word_4F063FC[0];
      v42 = dword_4D04790;
      if ( dword_4D04790 )
      {
        v10 = dword_4F077BC;
        if ( !dword_4F077BC )
        {
          v42 = (__int64)&dword_4F063F8;
          v41 = 2901;
          sub_6851C0(0xB55u, &dword_4F063F8);
        }
      }
      sub_7B8B50(v41, v42, v10, v11);
      v43 = *v3;
      ++*(_BYTE *)(v43 + 34);
      ++*(_BYTE *)(v43 + 84);
      v50 = sub_6BA690((__int64 *)&v84, v42, v44, v45, v46, v47) & 1;
      if ( dword_4D04794 && word_4F06418[0] == 76 )
      {
        sub_7B8B50(&v84, v42, v48, v49);
        v86[0] = *(_QWORD *)&dword_4F063F8;
        if ( ((unsigned int)sub_6BA690((__int64 *)&v85, v42, v57, v58, v59, v60) & v50) != 0 )
        {
          if ( v85 >= v84 )
          {
            a2 = 17;
            --*(_BYTE *)(*v3 + 84LL);
            sub_7BE280(26, 17, 0, 0);
            --*(_BYTE *)(*v3 + 34LL);
            break;
          }
          sub_6851C0(0x3BCu, v86);
        }
        a2 = 17;
        v9 = 26;
        --*(_BYTE *)(*v3 + 84LL);
        sub_7BE280(26, 17, 0, 0);
        --*(_BYTE *)(*v3 + 34LL);
        goto LABEL_100;
      }
      a2 = 17;
      v9 = 26;
      v85 = v84;
      --*(_BYTE *)(*v3 + 84LL);
      sub_7BE280(26, 17, 0, 0);
      --*(_BYTE *)(*v3 + 34LL);
      if ( !v50 )
      {
LABEL_100:
        v10 = 0;
        v11 = dword_4D04794;
        if ( !dword_4D04794 )
        {
          v73 = 1;
LABEL_148:
          v7 = v35;
          v26 = v34;
          v27 = word_4F06418[0];
          continue;
        }
LABEL_101:
        if ( word_4F06418[0] == 55 )
        {
          a2 = (__int64)&dword_4F063F8;
          v9 = 955;
          v69 = v10;
          sub_6851C0(0x3BBu, &dword_4F063F8);
          sub_7B8B50(955, &dword_4F063F8, v63, v64);
          v10 = v69;
        }
        else
        {
          v51 = 1;
          if ( word_4F06418[0] != 56 )
            v51 = v73;
          v73 = v51;
        }
        goto LABEL_105;
      }
      break;
    }
    v9 = 2;
    v10 = sub_6E2F40(2);
    *(_DWORD *)(v10 + 48) = v67;
    *(_WORD *)(v10 + 52) = v70;
    *(_QWORD *)(v10 + 32) = v84;
    *(_QWORD *)(v10 + 40) = v85;
    if ( dword_4D04794 )
      goto LABEL_101;
    v73 = 1;
LABEL_105:
    if ( !v10 )
      goto LABEL_148;
    v7 = v35;
    v35 = v10;
    if ( v7 )
      goto LABEL_87;
LABEL_107:
    *(_QWORD *)(v82 + 24) = v35;
LABEL_88:
    *(_BYTE *)(v82 + 9) |= 4u;
    if ( v34 )
    {
      v40 = word_4F06418[0];
      if ( word_4F06418[0] == 29 )
        goto LABEL_111;
      v73 = v34;
LABEL_91:
      if ( v40 != 25 )
      {
        v4 = v71;
        v7 = v35;
        if ( !v73 )
        {
          v74 = 0;
          v79 = 1;
          --*(_BYTE *)(*v3 + 64LL);
          goto LABEL_18;
        }
LABEL_68:
        if ( word_4F06418[0] == 73 )
        {
          if ( dword_4D04428 )
          {
            if ( v7 )
              *(_BYTE *)(v7 + 10) |= 2u;
            goto LABEL_71;
          }
        }
        else if ( word_4F06418[0] == 56 )
        {
          sub_7B8B50(v9, a2, v10, v11);
          goto LABEL_71;
        }
        a2 = (__int64)&dword_4F063F8;
        sub_6851C0(0x2BEu, &dword_4F063F8);
LABEL_71:
        --*(_BYTE *)(*v3 + 64LL);
        v28 = v73;
        if ( !v73 )
          v28 = v79;
        v79 = v28;
        v29 = 0;
        if ( !v73 )
          v29 = v74;
        v74 = v29;
        v30 = 1;
        if ( v73 )
          v30 = v76;
        v76 = v30;
        goto LABEL_18;
      }
      v34 = 1;
      goto LABEL_93;
    }
    if ( !HIDWORD(qword_4F077B4) )
    {
      if ( v72 )
      {
        v4 = v71;
        goto LABEL_125;
      }
LABEL_109:
      v40 = word_4F06418[0];
      if ( word_4F06418[0] == 29 )
        goto LABEL_110;
      goto LABEL_91;
    }
    if ( !v72 )
    {
      if ( dword_4F077C4 == 2 || unk_4F07778 <= 199900 )
      {
        if ( dword_4D04320 )
        {
          v9 = 1605;
          a2 = sub_6E1A20(v35);
          sub_684B30(0x645u, (_DWORD *)a2);
          v40 = word_4F06418[0];
          if ( word_4F06418[0] == 29 )
          {
LABEL_110:
            v34 = v73;
            goto LABEL_111;
          }
        }
        else
        {
          v72 = 0;
          v40 = word_4F06418[0];
          if ( word_4F06418[0] == 29 )
            goto LABEL_110;
        }
        goto LABEL_91;
      }
      goto LABEL_109;
    }
    v4 = v71;
    if ( dword_4D04320 )
    {
      a2 = sub_6E1A20(v35);
      sub_684B30(0x646u, (_DWORD *)a2);
    }
LABEL_125:
    v9 = v73;
    if ( v73 )
    {
      v7 = v35;
      goto LABEL_68;
    }
    v74 = 0;
    v7 = v35;
    v79 = 1;
    --*(_BYTE *)(*v3 + 64LL);
LABEL_18:
    if ( !v77 )
    {
      v77 = dword_4D04790;
      if ( dword_4D04790 )
      {
        v77 = v76 & v79;
        if ( (v76 & v79) != 0 )
        {
          a2 = (__int64)&dword_4F063F8;
          sub_6851C0(0xB57u, &dword_4F063F8);
        }
      }
    }
    if ( (unsigned int)sub_869470(&v83) )
    {
      v75 = v3;
      v13 = v78;
      v14 = (_QWORD *)v4;
      v15 = v7;
      while ( 1 )
      {
        v18 = (__int64 *)v15;
        v19 = sub_6BB5A0(v2, 1);
        v15 = v19;
        if ( v18 )
        {
          *v18 = v19;
          if ( *(_BYTE *)(v19 + 8) == 3 )
            goto LABEL_31;
        }
        else
        {
          *(_QWORD *)(v82 + 24) = v19;
          if ( *(_BYTE *)(v19 + 8) == 3 )
          {
LABEL_31:
            v20 = v18;
            v21 = v19;
            v22 = v14;
            v3 = v75;
            if ( !v22 )
            {
              v56 = sub_823970(48);
              *(_BYTE *)(v56 + 40) |= 1u;
              v22 = (_QWORD *)v56;
              *(_QWORD *)v56 = 0;
              *(_QWORD *)(v56 + 8) = 0;
              *(_QWORD *)(v56 + 16) = 0;
              *(_QWORD *)(v56 + 24) = 0;
              *(_QWORD *)(v56 + 32) = 0;
            }
            v23 = *(_QWORD *)(v21 + 24);
            v22[2] = v20;
            *v22 = v23;
            v22[1] = v82;
            *(_QWORD *)(v21 + 24) = v22;
            *v80 = v22;
            goto LABEL_37;
          }
        }
        a2 = 0;
        ++v13;
        v16 = sub_867630(v83, 0);
        if ( v16 )
        {
          v17 = *(_BYTE *)(v15 + 8) == 0;
          *(_QWORD *)(v15 + 16) = v16;
          if ( v17 )
            *(_QWORD *)(*(_QWORD *)(v15 + 24) + 136LL) = v16;
        }
        if ( !(unsigned int)sub_866C00(v83) )
        {
          v7 = v15;
          v78 = v13;
          v4 = (__int64)v14;
          v3 = v75;
          v74 = 1;
          break;
        }
      }
    }
LABEL_8:
    if ( word_4F06418[0] != 67 && word_4F06418[0] != 74 )
    {
      if ( (unsigned int)sub_692B20(word_4F06418[0])
        || word_4F06418[0] == 29
        || word_4F06418[0] == 73
        && (a2 = 0, (unsigned __int16)sub_7BE840(0, 0) != 75)
        && (a2 = 0, (unsigned __int16)sub_7BE840(0, 0) != 67) )
      {
        a2 = 253;
        ++*(_BYTE *)(*v3 + 75LL);
        sub_7BE5B0(67, 253, 0, 0);
        --*(_BYTE *)(*v3 + 75LL);
      }
    }
    v9 = 67;
    if ( !(unsigned int)sub_7BE800(67) )
    {
LABEL_36:
      *(_QWORD *)(v82 + 40) = *(_QWORD *)&dword_4F063F8;
      sub_7BE280(74, 67, 0, 0);
      goto LABEL_37;
    }
  }
  while ( !v74 );
  v5 = (__int64 *)v7;
LABEL_35:
  if ( word_4F06418[0] == 74 )
    goto LABEL_36;
  v74 = 1;
  if ( v78 <= 0x3E8 || !v80 )
  {
LABEL_57:
    v7 = (__int64)v5;
    goto LABEL_16;
  }
  if ( v4 )
  {
    if ( (*(_BYTE *)(v4 + 40) & 1) == 0 )
      goto LABEL_57;
  }
  else
  {
    v31 = sub_823970(48);
    *(_BYTE *)(v31 + 40) |= 1u;
    v4 = v31;
    *(_QWORD *)v31 = 0;
    *(_QWORD *)(v31 + 8) = 0;
    *(_QWORD *)(v31 + 16) = 0;
    *(_QWORD *)(v31 + 24) = 0;
    *(_QWORD *)(v31 + 32) = 0;
  }
  v32 = v82;
  *(_QWORD *)v4 = 0;
  *(_QWORD *)(v4 + 16) = v5;
  *(_QWORD *)(v4 + 8) = v82;
  *v80 = v4;
  v82 = sub_6E2F40(3);
  *(_QWORD *)(v82 + 24) = *v80;
  *v5 = v82;
  *(_QWORD *)(v32 + 40) = *(_QWORD *)(v32 + 32);
LABEL_37:
  if ( v79 && dword_4D04408 && *(_BYTE *)(v82 + 8) == 1 )
  {
    v24 = *(_QWORD **)(v82 + 24);
    while ( v24 )
    {
      if ( v24[2] )
      {
        v25 = (_DWORD *)sub_6E1A20(v24);
        sub_6851C0(0xB61u, v25);
        break;
      }
      if ( !*v24 )
        break;
      if ( *(_BYTE *)(*v24 + 8LL) == 3 )
        v24 = (_QWORD *)sub_6BBB10(v24);
      else
        v24 = (_QWORD *)*v24;
    }
  }
LABEL_6:
  --*(_BYTE *)(*v3 + 82LL);
  --*(_QWORD *)(qword_4D03C50 + 40LL);
  return v82;
}
