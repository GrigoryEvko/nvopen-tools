// Function: sub_5CDE00
// Address: 0x5cde00
//
__int64 __fastcall sub_5CDE00(__int64 a1, __int64 a2, char a3)
{
  char v6; // cl
  bool v7; // al
  char v8; // dl
  int v9; // r14d
  __int64 v10; // r12
  char *v11; // r8
  __int64 v12; // rsi
  __int64 i; // rdi
  unsigned __int64 v14; // rdx
  __int64 v15; // r15
  char v16; // al
  __int64 v17; // rcx
  __int64 v18; // rax
  unsigned int v20; // r14d
  unsigned int v21; // eax
  __int64 v22; // r8
  unsigned int v23; // eax
  __int64 v24; // rax
  unsigned int v25; // eax
  char v26; // al
  __int64 v27; // rdi
  unsigned int v28; // eax
  __int64 v29; // [rsp+0h] [rbp-60h]
  char *v30; // [rsp+8h] [rbp-58h]
  __int64 v31; // [rsp+8h] [rbp-58h]
  __int64 v32; // [rsp+8h] [rbp-58h]
  __int64 v33; // [rsp+8h] [rbp-58h]
  int v34; // [rsp+10h] [rbp-50h]
  bool v35; // [rsp+17h] [rbp-49h]
  __int64 v36; // [rsp+18h] [rbp-48h] BYREF
  unsigned int v37; // [rsp+24h] [rbp-3Ch] BYREF
  __int64 v38[7]; // [rsp+28h] [rbp-38h] BYREF

  v6 = *(_BYTE *)(a1 + 9);
  v36 = a2;
  v7 = v6 == 1 || v6 == 4;
  if ( v7 )
  {
    v8 = *(_BYTE *)(a1 + 11) & 0x10;
    v9 = v8 == 0;
    if ( v8 )
      v7 = 0;
  }
  else
  {
    v9 = 0;
  }
  v35 = 0;
  if ( unk_4F077C4 == 2 && dword_4F077B8 && !dword_4F077B4 )
    v35 = a3 == 6 || qword_4F077A8 <= 0x9F5Fu;
  v10 = *(_QWORD *)(a1 + 48);
  if ( v6 == 2 || (*(_BYTE *)(a1 + 11) & 0x10) != 0 )
  {
    v11 = "c|e|t|v:-r!|d|p|r";
    if ( qword_4F077A8 <= 0x9D6Bu )
      v11 = "c|e|t|v:-r!|d|p";
    goto LABEL_9;
  }
  if ( !v7 )
  {
    v11 = "c|e|t|v|d|r";
    goto LABEL_9;
  }
  if ( unk_4F077BC && !dword_4F077B4 )
  {
    v11 = "c|e|t|v:-r!-h!|d:-b!";
    goto LABEL_111;
  }
  if ( !v10 || !*(_QWORD *)(v10 + 192) )
  {
    v11 = "c|e|v:-r!-h!|d:-b!";
LABEL_111:
    if ( unk_4F077C4 == 2 || unk_4F07778 <= 201111 )
      goto LABEL_9;
    goto LABEL_57;
  }
  if ( unk_4F077C4 == 2 || unk_4F07778 <= 201111 )
  {
    sub_6851C0(1098, a1 + 56);
    *(_BYTE *)(a1 + 8) = 0;
    if ( unk_4F077C4 != 2 && unk_4F07778 > 201111 && *(_BYTE *)(a1 + 9) == 4 && (*(_BYTE *)(a1 + 10) & 0xFB) != 1 )
      sub_684AA0(7, 1098, a1 + 56);
    return v36;
  }
  v11 = "T";
LABEL_57:
  if ( v6 == 4 && (*(_BYTE *)(a1 + 10) & 0xFB) != 1 )
  {
    v30 = v11;
    sub_684AA0(7, 1098, a1 + 56);
    v11 = v30;
  }
LABEL_9:
  v12 = a1;
  i = (__int64)v11;
  if ( !(unsigned int)sub_5CCB50(v11, a1, v36, a3) || !*(_BYTE *)(a1 + 8) )
    return v36;
  v15 = *(_QWORD *)(a1 + 32);
  v34 = 0;
  v29 = a1 + 56;
  v37 = 0;
  if ( !v15 )
    goto LABEL_28;
LABEL_12:
  v16 = *(_BYTE *)(v15 + 10);
  if ( !v16 )
    goto LABEL_43;
  if ( v16 == 4 )
  {
    i = *(_QWORD *)(v15 + 40);
    if ( (unsigned int)sub_8D32E0() )
      i = sub_8D46C0(i);
    if ( (unsigned int)sub_8D3410(i) )
      i = sub_8D40F0(i);
    if ( (unsigned int)sub_8D2310(i) )
    {
      v12 = v15 + 24;
      i = 168;
      sub_6851C0(168, v15 + 24);
      *(_BYTE *)(a1 + 8) = 0;
    }
    else if ( !(unsigned int)sub_8DBE70(i) )
    {
      if ( unk_4F077C4 == 2 && (unsigned int)sub_8D23B0(i) )
        sub_8AE000(i);
      v32 = i;
      if ( (unsigned int)sub_8D23B0(i) )
      {
        v12 = v15 + 24;
        i = (unsigned int)sub_67F240(i);
        sub_685A50(i, v15 + 24, v32, 8);
        *(_BYTE *)(a1 + 8) = 0;
      }
      else
      {
        if ( !(unsigned int)sub_8D2BE0() )
        {
          i = *(_QWORD *)(v15 + 40);
          if ( *(char *)(i + 142) >= 0 && *(_BYTE *)(i + 140) == 12 )
            v21 = ((__int64 (*)(void))sub_8D4AB0)();
          else
            v21 = *(_DWORD *)(i + 136);
          v37 = v21;
          goto LABEL_15;
        }
        v12 = v15 + 24;
        i = 3414;
        sub_685360(3414, v15 + 24);
        *(_BYTE *)(a1 + 8) = 0;
      }
    }
LABEL_75:
    if ( !unk_4F077BC )
      goto LABEL_26;
    goto LABEL_76;
  }
  if ( v16 != 3 )
  {
LABEL_15:
    if ( !unk_4F077BC )
      goto LABEL_29;
LABEL_16:
    if ( dword_4F077B4 )
    {
      v14 = dword_4F077B8;
      if ( !dword_4F077B8 )
        goto LABEL_65;
      goto LABEL_33;
    }
    if ( a3 != 6 || (v9 & 1) == 0 )
      goto LABEL_29;
    v17 = v36;
    if ( *(_BYTE *)(v36 + 140) == 12 && *(_QWORD *)(v36 + 8) )
    {
      v12 = a1 + 56;
      i = 2470;
      sub_684B30(2470, v29);
      v14 = (unsigned __int64)&dword_4F077B8;
      if ( !dword_4F077B8 )
        goto LABEL_37;
      v9 = dword_4F077B4;
      if ( dword_4F077B4 )
        goto LABEL_34;
      v14 = (unsigned __int64)&qword_4F077A8;
      if ( qword_4F077A8 > 0xEA5Fu )
        goto LABEL_35;
      v17 = v36;
      if ( *(_BYTE *)(v36 + 140) != 2 )
        goto LABEL_35;
    }
    else
    {
      v14 = dword_4F077B8;
      if ( !dword_4F077B8 )
        goto LABEL_21;
      v12 = dword_4F077B4;
      if ( dword_4F077B4 )
        goto LABEL_21;
      v14 = (unsigned __int64)&qword_4F077A8;
      if ( qword_4F077A8 > 0xEA5Fu || *(_BYTE *)(v36 + 140) != 2 )
        goto LABEL_21;
      v9 = 1;
    }
LABEL_124:
    if ( (*(_BYTE *)(v17 + 161) & 8) != 0 )
    {
      v12 = a1 + 56;
      i = 1865;
      sub_684B30(1865, v29);
      *(_BYTE *)(a1 + 8) = 0;
      v34 = 1;
      goto LABEL_25;
    }
LABEL_33:
    if ( !v9 )
    {
LABEL_34:
      v14 = (unsigned __int64)&qword_4F077A8;
LABEL_35:
      if ( qword_4F077A8 <= 0x9F5Fu )
        goto LABEL_66;
      if ( (unsigned __int8)(a3 - 7) > 1u )
        goto LABEL_37;
      v9 = 0;
      goto LABEL_22;
    }
LABEL_21:
    v9 = 1;
LABEL_22:
    if ( *(_DWORD *)(v10 + 376) < v37 || v35 )
    {
      *(_DWORD *)(v10 + 376) = v37;
      *(_QWORD *)(v10 + 384) = a1;
    }
    goto LABEL_25;
  }
  v12 = a1;
  i = v15;
  v38[0] = 0;
  if ( !(unsigned int)sub_5CACA0(v15, a1, 0, 0x7FFFFFFFFFFFFFFFLL, v38) )
    goto LABEL_75;
  i = v38[0];
  if ( !v38[0] )
    goto LABEL_75;
  v12 = (__int64)&v37;
  if ( (unsigned int)sub_7A7520(v38[0], &v37) )
    goto LABEL_15;
  v12 = v15 + 24;
  i = 1090;
  sub_6851C0(1090, v15 + 24);
  *(_BYTE *)(a1 + 8) = 0;
  if ( !unk_4F077BC )
    goto LABEL_26;
LABEL_76:
  i = dword_4F077B4;
  if ( !dword_4F077B4 && a3 == 6 && (v9 & 1) != 0 )
  {
    v9 = 1;
    if ( *(_BYTE *)(v36 + 140) == 12 )
    {
      if ( *(_QWORD *)(v36 + 8) )
      {
        v12 = a1 + 56;
        i = 2470;
        v9 = 0;
        sub_684B30(2470, v29);
      }
    }
  }
  while ( 1 )
  {
LABEL_26:
    v15 = *(_QWORD *)v15;
    if ( !v15 )
      goto LABEL_43;
    v18 = *(_QWORD *)(a1 + 32);
    v37 = 0;
    if ( v18 )
      goto LABEL_12;
LABEL_28:
    v37 = unk_4F06984;
    if ( unk_4F077BC )
      goto LABEL_16;
LABEL_29:
    if ( dword_4F077B8 )
    {
      if ( dword_4F077B4 )
        goto LABEL_33;
      v14 = (unsigned __int64)&qword_4F077A8;
      if ( qword_4F077A8 > 0xEA5Fu )
        goto LABEL_33;
      if ( a3 != 6 )
        goto LABEL_33;
      v17 = v36;
      if ( *(_BYTE *)(v36 + 140) != 2 )
        goto LABEL_33;
      goto LABEL_124;
    }
LABEL_65:
    if ( v9 )
      goto LABEL_22;
LABEL_66:
    if ( a3 == 8 )
    {
      v20 = v37;
      v31 = v36;
      if ( (unsigned int)sub_7A7D20(i, v12, v14) )
      {
        v25 = sub_7A7D20(i, v12, v14);
        if ( v25 < v37 )
          v20 = sub_7A7D20(i, v12, v14);
      }
      *(_DWORD *)(v31 + 140) = v20;
      v9 = 0;
      goto LABEL_25;
    }
LABEL_37:
    if ( a3 == 7 )
    {
      v14 = v37;
      if ( v35 || *(_BYTE *)(a1 + 9) == 2 || (*(_BYTE *)(a1 + 11) & 0x10) != 0 )
      {
        *(_DWORD *)(v36 + 152) = v37;
        v9 = 0;
      }
      else
      {
        v9 = 0;
        if ( *(_DWORD *)(v36 + 152) < v37 )
          *(_DWORD *)(v36 + 152) = v37;
      }
      goto LABEL_25;
    }
    if ( a3 == 6 )
      break;
    if ( a3 == 11 )
    {
      v12 = (__int64)&v36;
      i = a1;
      v9 = 0;
      v24 = sub_5C7B50(a1, (__int64)&v36, 11);
      if ( v24 )
      {
        v14 = v37;
        *(_BYTE *)(v24 + 142) |= 0x80u;
        *(_DWORD *)(v24 + 136) = v14;
      }
      goto LABEL_25;
    }
    if ( a3 != 3 )
      sub_721090(i);
    v9 = dword_4F077B4;
    if ( !dword_4F077B4 )
    {
      for ( i = *(_QWORD *)(v36 + 8); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      if ( (unsigned int)sub_8D2E30(i) )
        goto LABEL_25;
      v9 = sub_8D2FB0(i);
      if ( !v9 )
      {
        v12 = a1;
        i = 8;
        sub_5CCAE0(8u, a1);
        *(_BYTE *)(a1 + 8) = 0;
        goto LABEL_25;
      }
    }
    v9 = 0;
    if ( !v15 )
      goto LABEL_43;
  }
  v22 = v36;
  if ( *(_BYTE *)(a1 + 9) != 3 )
  {
    v23 = v37;
    if ( *(char *)(v36 + 142) >= 0 || *(_DWORD *)(v36 + 136) < v37 || (v9 = 0, v35) )
    {
      *(_BYTE *)(v36 + 142) |= 0x80u;
      v9 = 0;
      *(_DWORD *)(v22 + 136) = v23;
    }
    goto LABEL_25;
  }
  v26 = *(_BYTE *)(v36 + 140);
  if ( v26 == 12 )
  {
    if ( !*(_QWORD *)(v36 + 8) )
      goto LABEL_145;
    v27 = *(_QWORD *)(v36 + 160);
    if ( *(char *)(v27 + 142) >= 0 && *(_BYTE *)(v27 + 140) == 12 )
    {
      v33 = v36;
      v28 = sub_8D4AB0(v27, v12, v14);
      if ( v37 >= v28 )
      {
        v22 = v33;
        v26 = *(_BYTE *)(v33 + 140);
        goto LABEL_143;
      }
    }
    else
    {
      v12 = v37;
      if ( v37 >= *(_DWORD *)(v27 + 136) )
        goto LABEL_146;
    }
    v12 = a1 + 56;
    i = 1286;
    v9 = 0;
    sub_684B30(1286, v29);
    *(_BYTE *)(a1 + 8) = 0;
  }
  else
  {
LABEL_143:
    if ( v26 != 2 || (*(_BYTE *)(v22 + 161) & 8) == 0 )
    {
LABEL_145:
      v12 = v37;
LABEL_146:
      i = v22;
      v9 = 0;
      sub_8E3480(v22, v12, v29);
      goto LABEL_25;
    }
    v12 = a1 + 56;
    i = 1723;
    v9 = 0;
    sub_684B30(1723, v29);
    *(_BYTE *)(a1 + 8) = 0;
  }
LABEL_25:
  if ( v15 )
    goto LABEL_26;
LABEL_43:
  if ( !*(_BYTE *)(a1 + 8) && ((unsigned __int8)v9 & (v10 != 0)) != 0 && !v34 )
    *(_DWORD *)(v10 + 376) = 0;
  return v36;
}
