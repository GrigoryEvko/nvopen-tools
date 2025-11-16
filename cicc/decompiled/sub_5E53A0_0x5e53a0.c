// Function: sub_5E53A0
// Address: 0x5e53a0
//
__int64 __fastcall sub_5E53A0(__int64 a1, __int64 a2, _DWORD *a3, __int64 *a4, __int64 a5)
{
  int v5; // r12d
  __int64 v6; // rbx
  __int64 v7; // r15
  __int64 i; // r13
  char v9; // al
  int v10; // r8d
  __int64 v11; // rax
  __int64 v12; // rcx
  char v13; // r14
  char v14; // al
  __int64 v15; // rax
  __int64 result; // rax
  __int64 v17; // rax
  char v18; // al
  __int64 v19; // rcx
  char v20; // al
  __int64 v21; // rsi
  __int64 v22; // r10
  __int64 v23; // r12
  __int64 v24; // r15
  __int64 v25; // rbx
  int v26; // eax
  unsigned __int64 v27; // r11
  unsigned __int64 v28; // rcx
  int v29; // edx
  int v30; // r8d
  __int64 v31; // r10
  int v32; // eax
  int v33; // eax
  int v34; // edx
  int v35; // r14d
  unsigned __int64 v36; // rcx
  unsigned __int64 v37; // rax
  __int64 v38; // rax
  char j; // dl
  __int64 v40; // rax
  int v41; // esi
  __int64 v42; // [rsp+8h] [rbp-A8h]
  int v43; // [rsp+8h] [rbp-A8h]
  int v44; // [rsp+8h] [rbp-A8h]
  int v45; // [rsp+10h] [rbp-A0h]
  __int64 v46; // [rsp+10h] [rbp-A0h]
  int v47; // [rsp+10h] [rbp-A0h]
  int v48; // [rsp+10h] [rbp-A0h]
  int v49; // [rsp+18h] [rbp-98h]
  __int64 v50; // [rsp+18h] [rbp-98h]
  int v51; // [rsp+18h] [rbp-98h]
  int v52; // [rsp+18h] [rbp-98h]
  __int64 v53; // [rsp+20h] [rbp-90h]
  unsigned __int64 v54; // [rsp+20h] [rbp-90h]
  unsigned __int64 v55; // [rsp+20h] [rbp-90h]
  unsigned __int64 v56; // [rsp+20h] [rbp-90h]
  int v57; // [rsp+20h] [rbp-90h]
  int v58; // [rsp+20h] [rbp-90h]
  __int64 v59; // [rsp+28h] [rbp-88h]
  unsigned __int64 v60; // [rsp+28h] [rbp-88h]
  unsigned __int64 v61; // [rsp+28h] [rbp-88h]
  unsigned __int64 v62; // [rsp+28h] [rbp-88h]
  unsigned __int64 v63; // [rsp+28h] [rbp-88h]
  unsigned __int64 v64; // [rsp+28h] [rbp-88h]
  int v65; // [rsp+28h] [rbp-88h]
  unsigned __int64 v66; // [rsp+28h] [rbp-88h]
  unsigned __int64 v70; // [rsp+48h] [rbp-68h]
  int v71; // [rsp+5Ch] [rbp-54h] BYREF
  char s[80]; // [rsp+60h] [rbp-50h] BYREF

  v5 = a2;
  v6 = a1;
  v7 = *a4;
  for ( i = *a4; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  if ( !(unsigned int)sub_8D3D40(i) )
  {
    if ( !(unsigned int)sub_8D2930(i) )
      i = sub_72BA30(5);
    v17 = sub_73A460(a2);
    v10 = 0;
    *(_QWORD *)(a1 + 168) = v17;
    v18 = *(_BYTE *)(a2 + 173);
    if ( !v18 )
    {
      v19 = dword_4F06BA0;
      v20 = *(_BYTE *)(i + 161);
      v13 = dword_4F06BA0;
      if ( (v20 & 8) != 0 )
      {
        v70 = dword_4F06BA0;
        v10 = 1;
        goto LABEL_33;
      }
      if ( (v20 & 1) != 0 )
      {
        v70 = dword_4F06BA0;
        v5 = 1;
        goto LABEL_12;
      }
      v70 = dword_4F06BA0;
      v10 = 1;
      v40 = *(unsigned __int8 *)(i + 160);
      v41 = unk_4F077C4;
      LOBYTE(v5) = unk_4F077C4 != 1 && (_BYTE)v40 == 1;
      if ( (_BYTE)v5 )
        goto LABEL_12;
      goto LABEL_94;
    }
    if ( v18 == 1 )
    {
      v71 = 0;
      v12 = dword_4F06BA0;
      goto LABEL_92;
    }
    if ( unk_4F04C44 != -1 )
    {
      v19 = dword_4F06BA0;
      v70 = dword_4F06BA0;
      goto LABEL_64;
    }
    goto LABEL_7;
  }
  *(_QWORD *)(a1 + 168) = sub_73A460(a2);
  v9 = *(_BYTE *)(a2 + 173);
  if ( !v9 )
  {
    v5 = 0;
    v70 = dword_4F06BA0;
    v13 = dword_4F06BA0;
    goto LABEL_12;
  }
  if ( v9 != 1 )
  {
    v10 = 1;
    if ( unk_4F04C44 != -1 )
    {
      v70 = dword_4F06BA0;
      goto LABEL_28;
    }
LABEL_7:
    v11 = unk_4F04C68 + 776LL * unk_4F04C64;
    if ( (*(_BYTE *)(v11 + 6) & 6) == 0 && *(_BYTE *)(v11 + 4) != 12 )
    {
      v71 = 0;
      v12 = dword_4F06BA0;
      if ( v10 )
        goto LABEL_74;
LABEL_92:
      v36 = *(_QWORD *)(i + 128) * v12;
      v35 = 0;
      goto LABEL_75;
    }
    v19 = dword_4F06BA0;
    v70 = dword_4F06BA0;
    if ( !v10 )
    {
LABEL_64:
      v20 = *(_BYTE *)(i + 161);
      if ( (v20 & 8) != 0 )
      {
        v13 = v19;
        if ( *(_BYTE *)(a2 + 173) == 12 )
          goto LABEL_66;
LABEL_33:
        v13 = v19;
        v5 = 0;
        if ( (**(_BYTE **)(i + 176) & 1) != 0 )
        {
          v22 = *(_QWORD *)(i + 168);
          if ( (v20 & 0x10) != 0 )
            v22 = *(_QWORD *)(v22 + 96);
          v5 = 0;
          if ( v22 )
          {
            v23 = v22;
            if ( *(_QWORD *)(v22 + 120) )
            {
              v59 = v19;
              v49 = v10;
              v53 = v7;
              v24 = v22;
              v25 = *(_QWORD *)(v22 + 120);
              do
              {
                if ( (int)sub_621060(v25, v24) < 0 )
                  v24 = v25;
                if ( (int)sub_621060(v25, v23) > 0 )
                  v23 = v25;
                v25 = *(_QWORD *)(v25 + 120);
              }
              while ( v25 );
              v22 = v24;
              v19 = v59;
              v7 = v53;
              v10 = v49;
              v6 = a1;
            }
            v45 = v10;
            v54 = v19;
            v50 = v22;
            v60 = (int)sub_621DD0(v23);
            v42 = v50;
            v26 = sub_6210B0(v50, 0);
            v27 = v60;
            v51 = v26;
            v28 = v54;
            v29 = unk_4F06AA0;
            v30 = v45;
            if ( unk_4F06AA0 )
            {
              v34 = 0;
            }
            else
            {
              v31 = v42;
              if ( v26 < 0 || v60 < v54 && !unk_4F06A9C )
              {
                v43 = v45;
                v46 = v31;
                v55 = v60;
                v61 = v28;
                v32 = sub_6210B0(v23, 0);
                v30 = v43;
                v29 = 1;
                v31 = v46;
                v28 = v61;
                v27 = (v32 > 0) + v55;
              }
              v44 = v30;
              v47 = v29;
              v56 = v27;
              v62 = v28;
              v33 = sub_621DD0(v31);
              v27 = v56;
              v28 = v62;
              v34 = v47;
              v30 = v44;
              if ( v56 < v33 )
                v27 = v33;
            }
            if ( v27 > v28 )
            {
              v48 = v30;
              v57 = v34;
              v64 = v28;
              sub_684B30(229, dword_4F07508);
              v30 = v48;
              v34 = v57;
              v28 = v64;
            }
            if ( unk_4F06AA0 && v51 < 0 )
            {
              v52 = v30;
              v58 = v34;
              v66 = v28;
              sub_685350(942, i);
              v28 = v66;
              v34 = v58;
              v30 = v52;
            }
            if ( !v30 && !*a3 )
            {
              if ( !v34 )
              {
LABEL_66:
                v5 = 0;
                goto LABEL_12;
              }
              v5 = 1;
              if ( v28 != 1 )
                goto LABEL_12;
              goto LABEL_61;
            }
            v5 = v34 & 1;
          }
        }
        goto LABEL_12;
      }
      if ( (v20 & 1) != 0 )
      {
        v13 = v19;
        if ( v10 || *a3 )
          goto LABEL_72;
        goto LABEL_70;
      }
      v40 = *(unsigned __int8 *)(i + 160);
      v41 = unk_4F077C4;
      LOBYTE(v5) = (_BYTE)v40 == 1 && unk_4F077C4 != 1;
      if ( (_BYTE)v5 )
      {
        v13 = v19;
        if ( v10 || *a3 )
          goto LABEL_12;
LABEL_70:
        if ( v19 == 1 )
        {
LABEL_62:
          v5 = 1;
          v13 = 1;
          sub_684B30(108, a5);
          goto LABEL_12;
        }
LABEL_71:
        v13 = v19;
LABEL_72:
        v5 = 1;
        goto LABEL_12;
      }
LABEL_94:
      if ( !byte_4B6DF90[v40] || qword_4D0495C )
        goto LABEL_89;
      if ( unk_4F06AA8 )
      {
        v13 = v19;
        v5 = 0;
        if ( v41 == 1 )
          i = sub_72BA30(byte_4B6DF80[v40]);
        goto LABEL_12;
      }
      if ( v19 == 1 )
      {
        v5 = 0;
        v13 = 1;
        if ( !unk_4F06AA4 )
        {
          v65 = v10;
          if ( !(unsigned int)sub_8D29A0(i) )
          {
            v5 = 1;
            if ( !v65 && !*a3 )
            {
LABEL_61:
              v5 = 1;
              v13 = 1;
              if ( (*(_BYTE *)(i + 161) & 8) != 0 )
                goto LABEL_12;
              goto LABEL_62;
            }
          }
        }
LABEL_12:
        v14 = *(_BYTE *)(v7 + 140);
        if ( v14 == 12 )
        {
LABEL_13:
          v15 = v7;
          do
            v15 = *(_QWORD *)(v15 + 160);
          while ( *(_BYTE *)(v15 + 140) == 12 );
          if ( v15 == i )
            goto LABEL_16;
          goto LABEL_23;
        }
LABEL_29:
        if ( v7 == i )
        {
LABEL_16:
          *a4 = v7;
          *(_BYTE *)(v6 + 137) = v13;
          *(_QWORD *)(v6 + 176) = v70;
          *(_BYTE *)(v6 + 144) = (8 * v5) | *(_BYTE *)(v6 + 144) & 0xF7;
          return (unsigned int)(8 * v5);
        }
        if ( (v14 & 0xFB) != 8 )
        {
          v21 = 0;
          goto LABEL_25;
        }
LABEL_23:
        v21 = (unsigned int)sub_8D4C10(v7, unk_4F077C4 != 2);
LABEL_25:
        while ( *(_BYTE *)(i + 140) == 12 )
          i = *(_QWORD *)(i + 160);
        v7 = sub_73C570(i, v21, -1);
        goto LABEL_16;
      }
      goto LABEL_71;
    }
LABEL_28:
    v14 = *(_BYTE *)(v7 + 140);
    v13 = v70;
    v5 = 0;
    if ( v14 == 12 )
      goto LABEL_13;
    goto LABEL_29;
  }
  v71 = 0;
  v12 = dword_4F06BA0;
LABEL_74:
  v35 = 1;
  v36 = unk_4F06AE0 * v12;
LABEL_75:
  v63 = v36;
  v37 = sub_620FD0(a2, &v71);
  v19 = v63;
  v70 = v37;
  if ( v63 >= v37 && !v71 )
  {
    if ( v37 )
    {
      v19 = v37;
      v10 = 0;
    }
    else
    {
      if ( *a3 )
      {
        v19 = 0;
LABEL_87:
        v10 = 0;
        goto LABEL_88;
      }
      if ( qword_4D0495C )
      {
        sub_684B30(107, a5);
        v10 = 0;
        v19 = 0;
        *a3 = 1;
      }
      else
      {
        sub_6851C0(107, a5);
        v10 = 1;
        v19 = 1;
      }
    }
LABEL_88:
    if ( v35 )
    {
LABEL_89:
      v13 = v19;
      v5 = 0;
      goto LABEL_12;
    }
    goto LABEL_64;
  }
  v38 = *a4;
  for ( j = *(_BYTE *)(*a4 + 140); j == 12; j = *(_BYTE *)(v38 + 140) )
    v38 = *(_QWORD *)(v38 + 160);
  if ( !j )
    goto LABEL_87;
  if ( v71 || unk_4F077C4 != 2 && (!unk_4F077C0 || qword_4F077A8 > 0x76BFu) )
  {
    v70 = v63;
    sub_6851C0(105, dword_4F07508);
    v19 = v63;
    v10 = 1;
    goto LABEL_88;
  }
  sprintf(s, "%lu", v63);
  result = sub_684B10(959, dword_4F07508, s);
  v19 = v63;
  v10 = unk_4F077C0;
  if ( !unk_4F077C0 )
    goto LABEL_88;
  *(_BYTE *)(a1 + 144) &= ~4u;
  return result;
}
