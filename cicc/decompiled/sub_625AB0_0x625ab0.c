// Function: sub_625AB0
// Address: 0x625ab0
//
__int64 __fastcall sub_625AB0(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        unsigned int a8,
        __int64 a9)
{
  int v9; // r15d
  int v10; // r13d
  int v12; // ebx
  __int64 v13; // rdx
  __int64 v14; // rcx
  unsigned __int16 v15; // ax
  __int64 v16; // rdi
  _QWORD *v17; // rsi
  int v18; // eax
  __int64 v19; // rdi
  unsigned __int16 v20; // ax
  unsigned __int8 v21; // dl
  __int64 v22; // rsi
  __int64 v23; // rax
  char v24; // al
  __int64 v25; // rdx
  __int64 v26; // rcx
  int v27; // r15d
  __int64 v28; // r14
  __int64 v29; // rax
  __int64 v30; // rcx
  __int64 v32; // rsi
  __int64 v33; // rax
  __int64 v34; // r14
  __int64 v35; // rdi
  __int64 v36; // rcx
  _DWORD *v37; // rsi
  _QWORD *v38; // rdx
  int v39; // r11d
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rax
  __int64 v43; // rbx
  __int64 v44; // rax
  char v45; // r15
  bool v46; // r14
  _QWORD *v47; // rdi
  int v48; // [rsp+8h] [rbp-78h]
  char v49; // [rsp+Ch] [rbp-74h]
  unsigned int v50; // [rsp+10h] [rbp-70h]
  __int64 v51; // [rsp+10h] [rbp-70h]
  __int64 v52; // [rsp+18h] [rbp-68h]
  int v53; // [rsp+24h] [rbp-5Ch] BYREF
  int v54; // [rsp+28h] [rbp-58h] BYREF
  unsigned int v55; // [rsp+2Ch] [rbp-54h] BYREF
  __int64 v56; // [rsp+30h] [rbp-50h] BYREF
  __int64 v57; // [rsp+38h] [rbp-48h] BYREF
  __int64 v58; // [rsp+40h] [rbp-40h] BYREF
  _QWORD v59[7]; // [rsp+48h] [rbp-38h] BYREF

  v9 = a3;
  v10 = a5;
  v12 = a4;
  v52 = a1;
  v53 = 0;
  v56 = sub_724DC0(a1, a2, a3, a4, a5, a6);
  v54 = 0;
  v57 = 0;
  v58 = *(_QWORD *)&dword_4F063F8;
  sub_7B8B50(a1, a2, v13, v14);
  v15 = word_4F06418[0];
  if ( word_4F06418[0] == 25 && dword_4D043F8 )
  {
    a1 = 7;
    sub_684AA0(7, 2791, &v58);
    v15 = word_4F06418[0];
  }
  ++*(_BYTE *)(qword_4F061C8 + 34LL);
  if ( dword_4F077C4 == 2 )
  {
    if ( dword_4F04C5C == -1 || *(_BYTE *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C5C + 4) == 1 )
    {
      v10 = 0;
      v12 = 0;
      goto LABEL_8;
    }
  }
  else if ( unk_4F07778 > 199900 && a8 && v15 == 100 )
  {
    sub_7B8B50(a1, &dword_4F077C4, a8, &qword_4F061C8);
    if ( word_4F06418[0] == 81
      || word_4F06418[0] == 107
      || (unsigned __int16)(word_4F06418[0] - 263) <= 3u
      || (unsigned __int16)(word_4F06418[0] - 118) <= 1u )
    {
      v16 = a9;
      v17 = 0;
      v59[0] = *(_QWORD *)&dword_4F063F8;
      v18 = sub_624060(a9);
      v50 = 1;
      goto LABEL_32;
    }
    v50 = 1;
LABEL_87:
    v49 = 0;
    v20 = word_4F06418[0];
    goto LABEL_16;
  }
  if ( unk_4F04C50 && (*(_BYTE *)(*(_QWORD *)(unk_4F04C50 + 32LL) + 198LL) & 0x10) != 0 )
  {
    v10 = 0;
    v12 = 0;
  }
LABEL_8:
  if ( v15 != 81 && v15 != 107 && (unsigned __int16)(v15 - 118) > 1u && (unsigned __int16)(v15 - 263) > 3u )
  {
    v50 = 0;
    goto LABEL_87;
  }
  v16 = a9;
  v17 = 0;
  v59[0] = *(_QWORD *)&dword_4F063F8;
  v18 = sub_624060(a9);
  if ( !a8 )
  {
    v19 = 749;
    if ( v18 == 4 )
      v19 = 643;
    sub_6851C0(v19, v59);
    v50 = 0;
    v49 = 0;
    v20 = word_4F06418[0];
    goto LABEL_16;
  }
  v50 = 0;
LABEL_32:
  if ( dword_4F077C4 == 2 )
  {
    v49 = v18 & 0x7C;
    if ( (v18 & 0xFFFFFF83) == 0 )
      goto LABEL_89;
  }
  else
  {
    v25 = unk_4F07778;
    if ( unk_4F07778 > 199900 )
    {
      v26 = v18 & 0x7F;
      v49 = v18 & 0x7F;
      if ( (v18 & 0xFFFFFF80) == 0 )
        goto LABEL_37;
    }
    else
    {
      v49 = v18 & 0x7C;
      if ( (v18 & 0xFFFFFF83) == 0 )
      {
LABEL_89:
        v20 = word_4F06418[0];
        goto LABEL_16;
      }
    }
  }
  v17 = v59;
  v16 = 749;
  sub_6851C0(749, v59);
  if ( dword_4F077C4 == 2 )
    goto LABEL_89;
  v25 = unk_4F07778;
LABEL_37:
  v20 = word_4F06418[0];
  if ( (int)v25 > 199900 && word_4F06418[0] == 100 )
  {
    if ( v50 )
    {
      v59[0] = *(_QWORD *)&dword_4F063F8;
      goto LABEL_19;
    }
    sub_7B8B50(v16, v17, v25, v26);
    v50 = 1;
    v59[0] = *(_QWORD *)&dword_4F063F8;
    v20 = word_4F06418[0];
    goto LABEL_17;
  }
LABEL_16:
  v59[0] = *(_QWORD *)&dword_4F063F8;
  if ( v20 != 26 )
  {
LABEL_17:
    if ( unk_4D047EC )
    {
      if ( v20 == 34 )
      {
        v35 = 0;
        if ( (unsigned __int16)sub_7BE840(0, 0) == 26 )
        {
          if ( v10
            && (v37 = (_DWORD *)v50, !v50)
            && dword_4F04C5C != -1
            && (v38 = qword_4F04C68, *(_BYTE *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C5C + 4) == 1) )
          {
            v27 = 1;
          }
          else
          {
            v35 = 890;
            v27 = 0;
            v37 = dword_4F07508;
            sub_6851C0(890, dword_4F07508);
            v54 = 1;
          }
          sub_7B8B50(v35, v37, v38, v36);
          v28 = 0;
          v48 = 0;
          goto LABEL_44;
        }
      }
    }
    goto LABEL_19;
  }
  if ( !v50 )
    goto LABEL_43;
LABEL_19:
  if ( v12 | v9 )
  {
    if ( v12 )
    {
      v21 = *(_BYTE *)(v52 + 124);
      v22 = ((v21 >> 3) ^ 1) & 1;
    }
    else
    {
      v22 = 0;
      v21 = *(_BYTE *)(v52 + 124);
    }
    sub_6D4F40(v12 == 0, v22, (v21 & 2) != 0, &v53, &v57, v56);
    v23 = v57;
    if ( dword_4F077C0 && a7 )
    {
      if ( !v57 )
        goto LABEL_27;
      if ( dword_4F04C58 != -1 )
      {
        sub_684B30(1155, dword_4F07508);
        sub_72BAF0(v56, 0, 5);
        v57 = 0;
        v53 = 1;
        *(_BYTE *)(v52 + 130) |= 0x10u;
        goto LABEL_27;
      }
LABEL_43:
      v48 = 0;
      v27 = 0;
      v28 = 0;
      goto LABEL_44;
    }
  }
  else
  {
    sub_6C9F20(v56);
    v53 = 1;
    v23 = v57;
  }
  if ( v23 )
    goto LABEL_43;
LABEL_27:
  v24 = *(_BYTE *)(v56 + 173);
  if ( v24 == 1 )
  {
    if ( (int)sub_6210B0(v56, 0) > 0 )
    {
      v28 = sub_620FD0(v56, &v54);
      if ( !v54 )
      {
        v40 = sub_7259C0(8);
        v41 = v58;
        *a2 = v40;
        *(_QWORD *)(v40 + 64) = v41;
        *(_BYTE *)(*a2 + 169) = ((_BYTE)v50 << 6) | *(_BYTE *)(*a2 + 169) & 0xBF;
        v32 = v57;
        *(_BYTE *)(*a2 + 168) = v49 | *(_BYTE *)(*a2 + 168) & 0x80;
        if ( v32 )
          goto LABEL_67;
        sub_7296C0(&v55);
        goto LABEL_98;
      }
      v27 = 0;
      sub_6851C0(95, dword_4F07508);
      v48 = 0;
    }
    else
    {
      if ( !HIDWORD(qword_4F077B4) || (v27 = sub_6210B0(v56, 0)) != 0 )
      {
        sub_6851C0(94, dword_4F07508);
        v54 = 1;
        goto LABEL_57;
      }
      v48 = 0;
      v28 = 0;
    }
  }
  else
  {
    if ( v24 != 12 )
    {
      if ( v24 )
        sub_721090(v56);
      v54 = 1;
      goto LABEL_57;
    }
    v48 = 1;
    v27 = 0;
    v28 = 0;
  }
LABEL_44:
  if ( v54 )
  {
LABEL_57:
    *a2 = sub_72C930();
    goto LABEL_53;
  }
  v29 = sub_7259C0(8);
  v30 = v58;
  *a2 = v29;
  *(_QWORD *)(v29 + 64) = v30;
  *(_BYTE *)(*a2 + 169) = ((_BYTE)v50 << 6) | *(_BYTE *)(*a2 + 169) & 0xBF;
  *(_BYTE *)(*a2 + 168) = v49 | *(_BYTE *)(*a2 + 168) & 0x80;
  if ( v27 )
  {
    *(_BYTE *)(*a2 + 169) |= 2u;
    *(_BYTE *)(*a2 + 169) |= 1u;
    unk_4F072F4 = 1;
    goto LABEL_47;
  }
  v32 = v57;
  if ( v57 )
  {
LABEL_67:
    *(_BYTE *)(*a2 + 169) |= 1u;
    v33 = *a2;
    if ( v12 )
    {
      *(_BYTE *)(v33 + 169) |= 2u;
      unk_4F072F4 = 1;
      if ( *(_BYTE *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C5C + 4) == 1 )
      {
        sub_87E130(*a2, v32, 0, v59);
      }
      else
      {
        v34 = sub_733470(*a2, v32, 0, v59);
        if ( !(unsigned int)sub_696820() )
          sub_86F7A0(v34, &v58);
      }
    }
    else
    {
      *(_QWORD *)(v33 + 176) = v32;
    }
    goto LABEL_47;
  }
  sub_7296C0(&v55);
  if ( v48 )
  {
    if ( (unsigned int)sub_73A390(v56) )
    {
      v42 = sub_73A460(v56);
LABEL_119:
      *(_QWORD *)(*a2 + 176) = v42;
      *(_BYTE *)(*a2 + 168) |= 0x80u;
      goto LABEL_102;
    }
    v45 = *(_BYTE *)(v56 + 176);
    v46 = v45 == 1 || (unsigned __int8)(v45 - 5) <= 5u;
    v42 = sub_740540(v56, 0, v46);
    v47 = (_QWORD *)(v42 + 184);
    if ( v45 != 1 )
    {
      if ( !v46 )
        goto LABEL_119;
      v47 = (_QWORD *)(v42 + 192);
    }
    v51 = v42;
    sub_6235F0(v47, *a2, 1);
    v42 = v51;
    goto LABEL_119;
  }
LABEL_98:
  if ( v53 )
  {
    v43 = sub_73A460(v56);
    if ( (*(_BYTE *)(v43 + 170) & 0x10) == 0 )
    {
      v44 = *(_QWORD *)(v56 + 144);
      if ( (!v44
         || *(_BYTE *)(v44 + 24) != 1
         || *(_BYTE *)(v44 + 56) != 106
         || *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v44 + 72) + 16LL) + 24LL) != 6)
        && dword_4F04C58 != -1 )
      {
        *(_QWORD *)(v43 + 144) = v44;
        sub_6235F0((_QWORD *)(v43 + 144), *a2, 0);
      }
    }
    *(_QWORD *)(*a2 + 184) = v43;
  }
  v39 = HIDWORD(qword_4F077B4);
  *(_QWORD *)(*a2 + 176) = v28;
  if ( !v39 || !v53 )
  {
    if ( !dword_4F077BC )
      goto LABEL_102;
    if ( (_DWORD)qword_4F077B4 )
      goto LABEL_102;
    if ( qword_4F077A8 > 0xEA5Fu )
      goto LABEL_102;
    if ( !a7 )
      goto LABEL_102;
    if ( v28 )
      goto LABEL_102;
    *(_BYTE *)(*a2 + 169) |= 0x20u;
    if ( !v53 )
      goto LABEL_102;
LABEL_115:
    if ( unk_4D04320 )
      sub_684B30(1617, v59);
    goto LABEL_102;
  }
  if ( !v28 )
  {
    *(_BYTE *)(*a2 + 169) |= 0x20u;
    goto LABEL_115;
  }
LABEL_102:
  sub_729730(v55);
LABEL_47:
  if ( HIDWORD(qword_4F077B4)
    && (dword_4F077C4 == 2 || unk_4F07778 <= 199900)
    && (*(_BYTE *)(*a2 + 169) & 2) != 0
    && unk_4D04320 )
  {
    sub_684B30(1604, v59);
  }
LABEL_53:
  if ( a9 )
    *(_QWORD *)(a9 + 56) = qword_4F063F0;
  unk_4F061D8 = qword_4F063F0;
  sub_7BE280(26, 17, 0, 0);
  --*(_BYTE *)(qword_4F061C8 + 34LL);
  *(_QWORD *)dword_4F07508 = v58;
  sub_622ED0(v52, a2);
  return sub_724E30(&v56);
}
